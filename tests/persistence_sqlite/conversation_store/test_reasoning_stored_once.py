"""Regression for #125273: an assistant turn's reasoning is stored once when it repeats ``reasoning_content``.

The row keeps the text in ``reasoning_content`` and stores ``reasoning`` as NULL; every reader gets both fields
back, so a request replays the same bytes. Rows written before the change hold the text in both columns, and
this build writes its copies of them once: every check that compares a stored row (with a message, or with
another row) must see one version, or resuming, flushing and compacting such a session stores a message again
or loses an edit. The single-row write/read cases live in ``test_reasoning_roundtrip.py``.
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict

import pytest

from persistence.sqlite.conversation._store.rules import RESTORED_FROM_STORE
from agent.message_sanitization import apply_reasoning_content_policy, reasoning_replay_route
from agent.transcript_repair import sync_flushed_message_markers
from persistence.sqlite.conversation._store.transcript_repair import _active_message_row
from persistence.sqlite._holder import StoreHolder
from persistence.sqlite.conversation._store.messages import (
    append_message, append_messages_batch, archive_and_compact, conversation_revision,
    get_active_message_watermark, get_messages, get_messages_as_conversation, get_resume_conversations,
)
from persistence.sqlite.conversation._store.sessions import create_session
from persistence.sqlite.housekeeping._portability import SessionPortability
from persistence.sqlite.conversation._store.messages import _decoded_repair_row, get_messages_around
from persistence.sqlite.conversation._store.search import get_anchored_view
from persistence.sqlite.conversation._store.stored_reads import display_generations, segment_page_messages
from persistence.sqlite.conversation._store.timeline import get_session_messages_around
from persistence.sqlite.conversation.messages import _message_record

SID = "s"
TEXT = "compare both branches, then take the cheaper one"
SUMMARY = "[CONTEXT COMPACTION] summary of the earlier turns"
EDITED = "the cheaper one, measured"
DETAILS = [{"type": "thinking", "thinking": TEXT, "signature": "c2lnbmVkIHRoaW5raW5n"}]
ITEMS = [{"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": TEXT}],
          "encrypted_content": "gAAAAB-opaque"}]
# (api_mode, provider, model, base_url): the default carrier, an OpenRouter-format gateway (every carrier) and
# the two must-echo families, Kimi on the native Anthropic wire whose pad sends reasoning_content verbatim.
ROUTES = {
    "default": ("chat_completions", "openai-api", "gpt-5", "https://api.openai.com/v1"),
    "openrouter": ("chat_completions", "openrouter", "anthropic/claude-sonnet-4", "https://openrouter.ai/api/v1"),
    "deepseek": ("chat_completions", "deepseek", "deepseek-v4-flash", "https://api.deepseek.com/v1"),
    "kimi": ("anthropic_messages", "kimi-coding", "kimi-k2", "https://api.kimi.com/coding"),
}


@pytest.fixture
def db(tmp_path):
    handle = StoreHolder(tmp_path / "state.db")
    yield handle
    handle.close()


def _seed(db, **reasoning):
    create_session(db, SID, source="cli")
    append_message(db, SID, role="user", content="which branch?")
    append_message(db, SID, role="assistant", content="the cheaper one", **reasoning)


def _seed_older_session(db) -> int:
    """A session an older build wrote (the text in both columns), with one turn before the compaction watermark."""
    create_session(db, SID, source="cli")
    append_message(db, SID, role="user", content="earlier turn")
    append_message(db, SID, role="assistant", content="earlier reply")
    watermark = get_active_message_watermark(db, SID)
    append_message(db, SID, role="user", content="which branch?")
    append_message(db, SID, role="assistant", content="the cheaper one", reasoning=TEXT, reasoning_content=TEXT)
    _as_an_older_build_wrote_it(db)
    return watermark


def _as_an_older_build_wrote_it(db):
    db._execute_write(lambda conn: conn.execute(
        "UPDATE messages SET reasoning = reasoning_content WHERE session_id = ? AND role = 'assistant'", (SID,)))


def _compact_tail(db, watermark):
    """Another surface compacts the session, carrying its last two rows (the writer stores its copies once)."""
    held = get_messages_as_conversation(db, SID, include_row_ids=True)[-2:]
    archive_and_compact(db, SID, [{"role": "user", "content": SUMMARY}, *held], watermark=watermark, tail_count=2)


def _columns(db):
    """``(reasoning, reasoning_content)`` of each assistant row that carries reasoning, archived copies included."""
    with db._read_ctx() as conn:
        return [tuple(row) for row in conn.execute(
            "SELECT reasoning, reasoning_content FROM messages WHERE session_id = ? AND role = 'assistant' "
            "AND reasoning_content IS NOT NULL ORDER BY id", (SID,))]


def _assistant(messages):
    return next(m for m in messages if m["role"] == "assistant")


def _wire(msg, route):
    """The assistant turn as *route* replays it: the request builder clones the stored dict, then the policy
    shapes its reasoning carriers (``turn_context.build_api_messages``)."""
    api_msg = copy.deepcopy(msg)
    apply_reasoning_content_policy(msg, api_msg, *reasoning_replay_route(*ROUTES[route]))
    return api_msg


def _read_paths(db):
    """The assistant dict from every decoder a caller reaches: conversation replay and resume (model and
    display), the live-replay load that carries the row-version digest, and ``get_messages`` (REST, export)."""
    model, display = get_resume_conversations(db, SID)
    messages = get_messages(db, SID)
    prompt, reply = messages[0]["id"], messages[-1]["id"]
    with db._read_ctx() as conn:
        raw = conn.execute("SELECT * FROM messages WHERE id = ?", (reply,)).fetchone()
        adopted = _decoded_repair_row(db, raw)
    page, _, _, _ = segment_page_messages(db, SID, latest=False, after=None, count=10)
    anchored = get_anchored_view(db, SID, reply)
    return {
        "repair": adopted,
        "display_generations": _assistant([m for m, _ in display_generations(db, SID, [SID])]),
        "segment_page": _assistant([m for m, _ in page]),
        "timeline_jump": _assistant(get_session_messages_around(db, SID, prompt)["messages"]),
        "search_around": _assistant(get_messages_around(db, SID, reply)["window"]),
        "search_anchored": _assistant(anchored["window"]),
        "search_bookend": _assistant(get_anchored_view(db, SID, prompt, window=0)["bookend_end"]),
        "typed_record": asdict(_message_record(messages[-1])),
        "conversation": _assistant(get_messages_as_conversation(db, SID)),
        "live_replay": _assistant(get_messages_as_conversation(db, SID, repair_alternation=True)),
        "resume_model": _assistant(model),
        "resume_display": _assistant(display),
        "messages": _assistant(get_messages(db, SID)),
        "export": _assistant([asdict(row) for row in SessionPortability._export_session(db, SID).messages]),
    }


class TestStoredOnce:
    def test_a_row_stored_once_reads_and_replays_as_a_row_holding_both_columns(self, db):
        create_session(db, SID, source="cli")
        append_message(db, SID, role="user", content="which branch?")
        append_messages_batch(db, SID, [{"role": "assistant", "content": "the cheaper one", "reasoning": TEXT,
                                        "reasoning_content": TEXT, "reasoning_details": DETAILS,
                                        "codex_reasoning_items": ITEMS}])
        assert _columns(db) == [(None, TEXT)]

        stored_once = _read_paths(db)
        _as_an_older_build_wrote_it(db)
        held_twice = _read_paths(db)
        for path, msg in stored_once.items():
            # Same keys, order and row-version digest: requests are built from clones of these dicts.
            assert json.dumps(msg, default=str) == json.dumps(held_twice[path], default=str), path
        for route in ROUTES:
            replayed = _wire(stored_once["conversation"], route)
            assert replayed["reasoning_content"] == TEXT, route
            assert json.dumps(replayed) == json.dumps(_wire(held_twice["conversation"], route)), route

    def test_a_reader_without_the_restore_sends_the_same_text_in_another_key_order(self, db):
        """NULL beside a non-blank ``reasoning_content`` is also what an assistant merge leaves on a turn with no
        reasoning of its own. This build replays it as a row holding both columns (above). A reader without the
        restore (an older build that opens the store) sends the same values on every route; on a route that
        carries ``reasoning`` its policy appends that key after ``reasoning_content``, so those bytes differ."""
        _seed(db, reasoning_content=TEXT)
        assert _columns(db) == [(None, TEXT)]
        msg = _assistant(get_messages_as_conversation(db, SID))
        without = {key: value for key, value in msg.items() if key != "reasoning"}
        for route in ROUTES:
            ours, older = _wire(msg, route), _wire(without, route)
            assert ours == older, route  # same keys and values
            assert (json.dumps(ours) == json.dumps(older)) is ("reasoning" not in ours), route

    def test_adopting_another_writers_version_hands_both_fields_to_the_live_dict(self, db):
        _seed(db, reasoning=TEXT, reasoning_content=TEXT)
        live = _assistant(get_messages_as_conversation(db, SID, repair_alternation=True))
        db._execute_write(lambda conn: conn.execute(  # another writer rewrites the row after our load
            "UPDATE messages SET content = ? WHERE session_id = ? AND role = 'assistant'", (EDITED, SID)))

        append_messages_batch(db, SID, [live])
        sync_flushed_message_markers([live], [live])

        assert (live["content"], live["reasoning"], live["reasoning_content"]) == (EDITED, TEXT, TEXT)


class TestSessionAnOlderBuildWrote:
    def test_resuming_flushing_and_compacting_it_adds_and_rewrites_no_row(self, db):
        watermark = _seed_older_session(db)
        restored = get_messages_as_conversation(db, SID)
        for msg in restored:  # handed back to the flush with only its uid: the wire strip removed the rest
            msg.pop("_db_persisted", None)
            msg[RESTORED_FROM_STORE] = True
        assert append_messages_batch(db, SID, restored) == 0

        loaded = get_messages_as_conversation(db, SID, repair_alternation=True)  # uid + row-version digest
        revision = conversation_revision(db, SID)
        assert append_messages_batch(db, SID, loaded) == 0
        assert conversation_revision(db, SID) == revision
        assert _columns(db) == [(TEXT, TEXT)]

        _compact_tail(db, watermark)
        with db._read_ctx() as conn:
            copies = conn.execute("SELECT active FROM messages WHERE session_id = ? AND role = 'assistant' "
                                  "AND reasoning_content IS NOT NULL ORDER BY id", (SID,)).fetchall()
        assert [row["active"] for row in copies] == [0, 1], "the archived original and one carried copy"

    def test_an_edit_after_another_surface_compacted_it_reaches_the_live_row(self, db):
        """The live dict holds the original row's version digest; compaction archived that row and carried a
        copy, which this build stores once."""
        watermark = _seed_older_session(db)
        live = get_messages_as_conversation(db, SID, repair_alternation=True)
        _compact_tail(db, watermark)

        assistant = live[-1]
        assistant["content"] = EDITED
        assistant.pop("_db_persisted", None)
        append_messages_batch(db, SID, [assistant])
        sync_flushed_message_markers([assistant], [assistant])

        assert assistant["content"] == EDITED
        assert [m["content"] for m in get_messages_as_conversation(db, SID) if m["role"] == "assistant"] == [EDITED]

    def test_a_row_id_resolves_to_the_carried_copy_only_when_its_reasoning_matches(self, db):
        """A flush addressed by the archived original's row id lands on its one active clone: the carried copy,
        which stores the reasoning once, but never a row whose reasoning differs."""
        _compact_tail(db, _seed_older_session(db))
        with db._read_ctx() as conn:
            archived, carried = (row["id"] for row in conn.execute(
                "SELECT id FROM messages WHERE session_id = ? AND role = 'assistant' "
                "AND reasoning_content IS NOT NULL ORDER BY id", (SID,)))
            assert _active_message_row(conn, SID, archived, "assistant")["id"] == carried
        db._execute_write(lambda conn: conn.execute(
            "UPDATE messages SET reasoning = ? WHERE id = ?", (EDITED, carried)))
        with db._read_ctx() as conn:
            assert _active_message_row(conn, SID, archived, "assistant")["id"] == archived
