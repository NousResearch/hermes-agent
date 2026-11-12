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

import pytest

from agent.message_metadata import RESTORED_FROM_STORE
from agent.message_sanitization import apply_reasoning_content_policy, reasoning_replay_route
from agent.transcript_repair import sync_flushed_message_markers
from hermes_state import SessionDB

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
    handle = SessionDB(tmp_path / "state.db")
    yield handle
    handle.close()


def _seed(db, **reasoning):
    db.create_session(SID, source="cli")
    db.append_message(SID, role="user", content="which branch?")
    db.append_message(SID, role="assistant", content="the cheaper one", **reasoning)


def _seed_older_session(db) -> int:
    """A session an older build wrote (the text in both columns), with one turn before the compaction watermark."""
    db.create_session(SID, source="cli")
    db.append_message(SID, role="user", content="earlier turn")
    db.append_message(SID, role="assistant", content="earlier reply")
    watermark = db.get_active_message_watermark(SID)
    db.append_message(SID, role="user", content="which branch?")
    db.append_message(SID, role="assistant", content="the cheaper one", reasoning=TEXT, reasoning_content=TEXT)
    _as_an_older_build_wrote_it(db)
    return watermark


def _as_an_older_build_wrote_it(db):
    db._execute_write(lambda conn: conn.execute(
        "UPDATE messages SET reasoning = reasoning_content WHERE session_id = ? AND role = 'assistant'", (SID,)))


def _compact_tail(db, watermark):
    """Another surface compacts the session, carrying its last two rows (the writer stores its copies once)."""
    held = db.get_messages_as_conversation(SID, include_row_ids=True)[-2:]
    db.archive_and_compact(SID, [{"role": "user", "content": SUMMARY}, *held], watermark=watermark, tail_count=2)


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
    model, display = db.get_resume_conversations(SID)
    return {
        "conversation": _assistant(db.get_messages_as_conversation(SID)),
        "live_replay": _assistant(db.get_messages_as_conversation(SID, repair_alternation=True)),
        "resume_model": _assistant(model),
        "resume_display": _assistant(display),
        "messages": _assistant(db.get_messages(SID)),
        "export": _assistant(db.export_session(SID)["messages"]),
    }


class TestStoredOnce:
    def test_a_row_stored_once_reads_and_replays_as_a_row_holding_both_columns(self, db):
        db.create_session(SID, source="cli")
        db.append_message(SID, role="user", content="which branch?")
        db.append_messages_batch(SID, [{"role": "assistant", "content": "the cheaper one", "reasoning": TEXT,
                                        "reasoning_content": TEXT, "reasoning_details": DETAILS,
                                        "codex_reasoning_items": ITEMS}])
        assert _columns(db) == [(None, TEXT)]
        with db._read_ctx() as conn:
            sidecars = conn.execute("SELECT reasoning_details, codex_reasoning_items FROM messages "
                                    "WHERE role = 'assistant'").fetchone()
        assert tuple(sidecars) == (json.dumps(DETAILS), json.dumps(ITEMS))

        stored_once = _read_paths(db)
        _as_an_older_build_wrote_it(db)
        held_twice = _read_paths(db)
        for path, msg in stored_once.items():
            # Same keys, order and row-version digest: requests are built from clones of these dicts.
            assert json.dumps(msg) == json.dumps(held_twice[path]), path
        for route in ROUTES:
            replayed = _wire(stored_once["conversation"], route)
            assert replayed["reasoning_content"] == TEXT, route
            assert json.dumps(replayed) == json.dumps(_wire(held_twice["conversation"], route)), route

    def test_without_the_restored_reasoning_every_route_replays_the_same(self, db):
        """NULL beside a non-blank ``reasoning_content`` is also what an assistant merge leaves on a turn with no
        reasoning of its own, and what an older build that opens the store reads, with no restore. Read back,
        the row carries both fields; without ``reasoning`` every route still sends the same request."""
        _seed(db, reasoning_content=TEXT)
        assert _columns(db) == [(None, TEXT)]
        msg = _assistant(db.get_messages_as_conversation(SID))
        without = {key: value for key, value in msg.items() if key != "reasoning"}
        for route in ROUTES:
            assert _wire(msg, route) == _wire(without, route), route

    @pytest.mark.parametrize("pad", [" ", "\n\n"], ids=["space", "blank_join"])
    def test_a_blank_pad_in_both_fields_is_stored_and_read_as_given(self, db, pad):
        _seed(db, reasoning=pad, reasoning_content=pad)
        assert _columns(db) == [(pad, pad)]
        for path, msg in _read_paths(db).items():
            assert (msg["reasoning"], msg["reasoning_content"]) == (pad, pad), path

    def test_adopting_another_writers_version_hands_both_fields_to_the_live_dict(self, db):
        _seed(db, reasoning=TEXT, reasoning_content=TEXT)
        live = _assistant(db.get_messages_as_conversation(SID, repair_alternation=True))
        db._execute_write(lambda conn: conn.execute(  # another writer rewrites the row after our load
            "UPDATE messages SET content = ? WHERE session_id = ? AND role = 'assistant'", (EDITED, SID)))

        db.append_messages_batch(SID, [live])
        sync_flushed_message_markers([live], [live])

        assert (live["content"], live["reasoning"], live["reasoning_content"]) == (EDITED, TEXT, TEXT)

    def test_export_then_import_of_an_older_builds_session_stores_one_copy(self, db, tmp_path):
        _seed(db, reasoning=TEXT, reasoning_content=TEXT)
        _as_an_older_build_wrote_it(db)
        payload = json.loads(json.dumps(db.export_session(SID)))  # through a JSON backup file

        other = SessionDB(tmp_path / "other.db")
        try:
            assert other.import_sessions([payload])["ok"]
            assert _columns(other) == [(None, TEXT)]
            msg = _assistant(other.get_messages_as_conversation(SID))
            assert (msg["reasoning"], msg["reasoning_content"]) == (TEXT, TEXT)
        finally:
            other.close()


class TestSessionAnOlderBuildWrote:
    def test_resuming_flushing_and_compacting_it_adds_and_rewrites_no_row(self, db):
        watermark = _seed_older_session(db)
        restored = db.get_messages_as_conversation(SID)
        for msg in restored:  # handed back to the flush with only its uid: the wire strip removed the rest
            msg.pop("_db_persisted", None)
            msg[RESTORED_FROM_STORE] = True
        assert db.append_messages_batch(SID, restored) == 0

        loaded = db.get_messages_as_conversation(SID, repair_alternation=True)  # uid + row-version digest
        revision = db.conversation_revision(SID)
        assert db.append_messages_batch(SID, loaded) == 0
        assert db.conversation_revision(SID) == revision
        assert _columns(db) == [(TEXT, TEXT)]

        _compact_tail(db, watermark)
        with db._read_ctx() as conn:
            copies = conn.execute("SELECT active FROM messages WHERE session_id = ? AND role = 'assistant' "
                                  "AND reasoning_content IS NOT NULL ORDER BY id", (SID,)).fetchall()
        assert [row["active"] for row in copies] == [0, 1], "the archived original and one carried copy"

    @pytest.mark.parametrize("holder", ["digest", "row_id"])
    def test_an_edit_after_another_surface_compacted_it_reaches_the_live_row(self, db, holder):
        """The holder's dict names the original row (its row-version digest, or the row id an earlier flush
        stamped); compaction archived that row and carried a copy, which this build stores once."""
        watermark = _seed_older_session(db)
        live = db.get_messages_as_conversation(SID, repair_alternation=True)
        if holder == "row_id":
            db.append_messages_batch(SID, live)
            sync_flushed_message_markers(live, live)
        _compact_tail(db, watermark)

        assistant = live[-1]
        assistant["content"] = EDITED
        assistant.pop("_db_persisted", None)
        db.append_messages_batch(SID, [assistant])
        sync_flushed_message_markers([assistant], [assistant])

        assert assistant["content"] == EDITED
        assert [m["content"] for m in db.get_messages_as_conversation(SID) if m["role"] == "assistant"] == [EDITED]
