"""Restart-safety regressions for proactive tool-result pruning."""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.context_compressor import _estimate_msg_budget_tokens
from agent.conversation_compression_archive import held_message
from persistence.conversation.messages import (
    ConversationRequest,
    NewMessage,
    PrunedRowChange,
    TranscriptRead,
    TranscriptView,
)
from persistence.core.errors import PruneRowStale, PruneRowUnresolved
from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_holder import raw_holder
from tests.fixtures.storage_read import conversation, session_row, transcript_rows
from tests.fixtures.storage_seed import seed_messages, seed_session

_REARM_KEY = "_proactive_prune_rearm_tokens"
_WITH_ROW_IDS = ConversationRequest(include_row_ids=True)
_ALL_ROWS = TranscriptRead(view=TranscriptView.ALL)
_DISPLAY = TranscriptRead(view=TranscriptView.DISPLAY)

@pytest.fixture()
def store(tmp_path: Path):
    """A storage interface handle on a temp state.db, closed after the test."""
    handle = open_persistence(profile_scope(home=tmp_path))
    yield handle
    handle.close()

def _assistant_call(call_id: str) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {"name": "terminal", "arguments": '{"cmd":"ls"}'},
        }],
    }

def _tool_result(call_id: str, content: str) -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": content}

def _history(*, large_chars: int = 24_000) -> list[dict]:
    messages: list[dict] = [{"role": "user", "content": "start"}]
    for index in range(8):
        call_id = f"call_{index}"
        messages.append(_assistant_call(call_id))
        content = chr(65 + index) * large_chars if index < 3 else "ok"
        messages.append(_tool_result(call_id, content))
    return messages

def _build_agent(store, session_id: str, *, platform: str = "telegram"):
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        return AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            session_db=store,
            persistence=store,
            session_id=session_id,
            platform=platform,
            skip_context_files=True,
            skip_memory=True,
        )

def _configure_pruning(agent) -> None:
    compressor = agent.context_compressor
    compressor.proactive_prune_tokens = 48_000
    compressor.proactive_prune_min_result_chars = 8_000
    compressor.proactive_prune_min_reclaim_tokens = 4_096
    compressor.protect_first_n = 2
    compressor.protect_last_n = 4

def _model_config(store, session_id: str) -> dict:
    raw = session_row(store, session_id)["model_config"]
    return json.loads(raw) if raw else {}

def _compact(store, session_id: str, messages: list[dict], **kwargs) -> None:
    """Commit *messages* as the session's new active generation through the full writer."""
    store.messages.compact(session_id, [NewMessage.from_live(m) for m in messages], **kwargs)

def _rewrite_pruned_rows(store, session_id: str, changes: list[tuple[dict, dict]]):
    """The in-place writer over ``(held, replacement)`` dict pairs, each change built as the prune's commit builds it."""
    return store.messages.rewrite_pruned_rows(session_id, [
        PrunedRowChange(held_message(original), NewMessage.from_live(replacement), NewMessage.from_live(original))
        for original, replacement in changes])

def test_gateway_eviction_reload_keeps_prune_and_durable_runway(store) -> None:
    """A fresh gateway agent must reload both the pruned body and its runway."""
    session_id = "GATEWAY_PRUNE_RESTART"
    seed_session(
        store, session_id, "telegram", model_config={"keep": "value"},
    )
    seed_messages(store, session_id, *_history())

    first_agent = _build_agent(store, session_id)
    _configure_pruning(first_agent)
    before = conversation(store, session_id)
    pruned, count = first_agent.context_compressor.prune_tool_results_only(
        before, current_tokens=120_000,
    )

    assert count >= 1
    durable = conversation(store, session_id)
    assert [message["content"] for message in durable] == [
        message["content"] for message in pruned
    ]
    assert len(durable[2]["content"]) < 24_000
    stored_runway = _model_config(store, session_id)[_REARM_KEY]
    assert _model_config(store, session_id)["keep"] == "value"
    assert stored_runway > sum(map(_estimate_msg_budget_tokens, durable))

    # Simulate gateway cache eviction / process restart: construct a wholly
    # new AIAgent and load the active transcript from SQLite.
    resumed_agent = _build_agent(store, session_id)
    _configure_pruning(resumed_agent)
    assert resumed_agent.context_compressor._proactive_prune_rearm_tokens == stored_runway
    reloaded = conversation(store, session_id)
    archived_before = len(transcript_rows(store, session_id, _ALL_ROWS))
    result, second_count = resumed_agent.context_compressor.prune_tool_results_only(
        reloaded, current_tokens=1_000_000,
    )

    assert result is reloaded
    assert second_count == 0
    assert len(transcript_rows(store, session_id, _ALL_ROWS)) == archived_before

def test_fresh_agent_rearms_after_durable_history_regrowth_once(store) -> None:
    session_id = "PRUNE_DURABLE_REGROWTH"
    seed_session(store, session_id, "telegram")
    seed_messages(store, session_id, *_history())
    first_agent = _build_agent(store, session_id)
    _configure_pruning(first_agent)
    _first, first_count = first_agent.context_compressor.prune_tool_results_only(
        conversation(store, session_id), current_tokens=120_000,
    )
    assert first_count >= 1
    first_runway = _model_config(store, session_id)[_REARM_KEY]

    growth = [
        _assistant_call("regrown_large"),
        _tool_result("regrown_large", "z" * 240_000),
        _assistant_call("tail_1"),
        _tool_result("tail_1", "ok"),
        _assistant_call("tail_2"),
        _tool_result("tail_2", "ok"),
    ]
    seed_messages(store, session_id, *growth)

    resumed = _build_agent(store, session_id)
    _configure_pruning(resumed)
    grown = conversation(store, session_id)
    assert sum(map(_estimate_msg_budget_tokens, grown)) >= first_runway
    _second, second_count = resumed.context_compressor.prune_tool_results_only(
        grown, current_tokens=1_000_000,
    )

    assert second_count >= 1
    second_runway = _model_config(store, session_id)[_REARM_KEY]
    assert second_runway > first_runway

    restarted = _build_agent(store, session_id)
    _configure_pruning(restarted)
    durable = conversation(store, session_id)
    result, third_count = restarted.context_compressor.prune_tool_results_only(
        durable, current_tokens=1_000_000,
    )
    assert result is durable
    assert third_count == 0
    assert restarted.context_compressor._proactive_prune_rearm_tokens == second_runway

@pytest.mark.parametrize("writer", ["rewrite_pruned_rows", "compact"])
def test_prune_persistence_failure_is_a_noop(store, writer: str) -> None:
    session_id = "PRUNE_PERSISTENCE_FAILURE"
    seed_session(store, session_id, "telegram")
    seed_messages(store, session_id, *_history())
    agent = _build_agent(store, session_id)
    _configure_pruning(agent)
    messages = conversation(store, session_id)
    original_contents = [message["content"] for message in messages]

    # The full rewrite runs when the in-place writer cannot name the changed rows.
    in_place = (patch.object(store.messages, "rewrite_pruned_rows", side_effect=PruneRowUnresolved(session_id))
                if writer == "compact" else contextlib.nullcontext())
    with in_place, patch.object(
        store.messages, writer, side_effect=RuntimeError("disk full"),
    ):
        result, count = agent.context_compressor.prune_tool_results_only(
            messages, current_tokens=120_000,
        )

    assert result is messages
    assert count == 0
    assert agent.context_compressor._proactive_prune_rearm_tokens == 0
    assert [message["content"] for message in messages] == original_contents
    assert [message["content"] for message in conversation(store, session_id)] == original_contents
    assert _REARM_KEY not in _model_config(store, session_id)

def test_archive_model_config_patch_rolls_back_with_transcript(store) -> None:
    session_id = "PRUNE_ATOMIC_ARCHIVE_FAILURE"
    seed_session(
        store,
        session_id,
        "telegram",
        model_config={"keep": "value", _REARM_KEY: 120_000},
    )
    original = [{"role": "user", "content": "original"}]
    seed_messages(store, session_id, *original)

    # The backend's row insert inside the compaction transaction is the failure point.
    with patch(
        "persistence.sqlite.conversation._store.messages._insert_message_rows",
        side_effect=RuntimeError("insert failed"),
    ):
        with pytest.raises(RuntimeError, match="insert failed"):
            _compact(
                store,
                session_id,
                [{"role": "user", "content": "replacement"}],
                model_config_patch={_REARM_KEY: None},
            )

    assert conversation(store, session_id)[0]["content"] == "original"
    assert _model_config(store, session_id) == {"keep": "value", _REARM_KEY: 120_000}

def test_model_switch_clears_durable_runway(store) -> None:
    """update_model must clear BOTH the in-memory and the durable runway."""
    session_id = "MODEL_SWITCH_CLEARS_RUNWAY"
    seed_session(
        store,
        session_id,
        "telegram",
        model_config={"keep": "value", _REARM_KEY: 120_000},
    )
    agent = _build_agent(store, session_id)
    compressor = agent.context_compressor
    assert compressor._proactive_prune_rearm_tokens == 120_000

    compressor.update_model("other/model", 200_000)

    assert compressor._proactive_prune_rearm_tokens == 0
    assert _REARM_KEY not in _model_config(store, session_id)
    assert _model_config(store, session_id)["keep"] == "value"

def test_patch_session_model_config_merge_and_delete(store) -> None:
    session_id = "PATCH_MODEL_CONFIG"
    seed_session(
        store, session_id, "cli", model_config={"keep": "value", "drop": 1},
    )

    store.sessions.patch_model_config(session_id, {"drop": None, "added": 7})
    assert _model_config(store, session_id) == {"keep": "value", "added": 7}

    # Missing rows and empty patches are no-ops, never errors.
    store.sessions.patch_model_config("NO_SUCH_SESSION", {"x": 1})
    store.sessions.patch_model_config(session_id, {})


def _rows(store, session_id: str) -> list[tuple]:
    with raw_holder(store)._read_ctx() as conn:
        return conn.execute(
            "SELECT id, role, content, active, compacted FROM messages WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()


def test_in_place_prune_grows_by_changed_rows_only(store) -> None:
    """#124102: N messages with K demoted leave exactly N + K rows, and every live row keeps its id."""
    session_id = "PRUNE_IN_PLACE_GROWTH"
    seed_session(store, session_id, "telegram")
    seed_messages(store, session_id, *_history())
    agent = _build_agent(store, session_id)
    _configure_pruning(agent)
    before_rows = _rows(store, session_id)
    messages = conversation(store, session_id)

    pruned, count = agent.context_compressor.prune_tool_results_only(messages, current_tokens=120_000)

    changed = [i for i, (old, new) in enumerate(zip(messages, pruned)) if old["content"] != new["content"]]
    assert count >= 1 and changed
    after_rows = _rows(store, session_id)
    live = [row for row in after_rows if row[3] == 1]
    twins = [row for row in after_rows if row[3] == 0]
    assert len(after_rows) == len(before_rows) + len(changed)
    assert [row[0] for row in live] == [row[0] for row in before_rows]
    assert sorted(row[2] for row in twins) == sorted(before_rows[i][2] for i in changed)
    assert all(row[4] == 1 for row in twins)  # the demoted full bodies stay searchable
    assert [m["content"] for m in conversation(store, session_id)] == [m["content"] for m in pruned]
    assert all(m.get("_db_persisted") for m in pruned)
    # The twin keeps the live row's display slot: the transcript page shows one row per message, in place.
    assert len(transcript_rows(store, session_id, _DISPLAY)) == len(pruned)


def _page(store, session_id: str) -> list:
    return [m["content"] for m in transcript_rows(store, session_id, _DISPLAY)]


@pytest.mark.parametrize("shape", ["first-generation", "carried", "carried-legacy-slots"])
def test_in_place_prune_display_survives_later_compaction_and_backfill(store, shape: str) -> None:
    """The transcript page shows each message once, in its own slot, as the stub and never the archived
    full body: right after the prune, after a later full compaction, and after a later display backfill."""
    session_id = "PRUNE_IN_PLACE_DISPLAY"
    seed_session(store, session_id, "telegram")
    seed_messages(store, session_id, *_history())
    if shape != "first-generation":  # every message already has an archived copy in its display slot
        _compact(store, session_id, conversation(store, session_id))
    if shape == "carried-legacy-slots":  # rows written before the display index existed
        raw_holder(store)._execute_write(lambda conn: conn.execute(
            "UPDATE messages SET display_order = NULL, display_identity = NULL WHERE session_id = ?",
            (session_id,)))
    agent = _build_agent(store, session_id)
    _configure_pruning(agent)
    pruned, count = agent.context_compressor.prune_tool_results_only(
        conversation(store, session_id), current_tokens=120_000)
    assert count >= 1
    # The prune leaves every visible row its slot, so no whole-session backfill follows it.
    with raw_holder(store)._read_ctx() as conn:
        unslotted = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = ? AND (active = 1 OR compacted = 1) "
            "AND (display_order IS NULL OR display_identity IS NULL)", (session_id,)).fetchone()[0]
    assert unslotted == 0
    expected = [m["content"] for m in pruned]
    assert _page(store, session_id) == expected

    _compact(store, session_id, conversation(store, session_id))
    assert _page(store, session_id) == expected

    # A later content rewrite elsewhere (the turn prologue's) re-derives the session's display slots.
    (row_id,) = seed_messages(store, session_id, {"role": "user", "content": "@notes.md"})
    assert store.messages.set_user_message_content(session_id, row_id, "expanded notes") is True
    assert _page(store, session_id) == expected + ["expanded notes"]


def test_in_place_prune_twice_keeps_one_search_hit_per_unchanged_turn(store) -> None:
    session_id = "PRUNE_IN_PLACE_SEARCH"
    seed_session(store, session_id, "telegram")
    seed_messages(store, session_id, *_history())
    agent = _build_agent(store, session_id)
    _configure_pruning(agent)
    compressor = agent.context_compressor
    _, first_count = compressor.prune_tool_results_only(
        conversation(store, session_id), current_tokens=120_000)
    assert first_count >= 1
    grown = []
    for index in range(8, 16):
        call_id = f"call_{index}"
        grown += [_assistant_call(call_id), _tool_result(call_id, chr(65 + index % 26) * 24_000)]
    seed_messages(store, session_id, *grown)
    rows_before_second = len(_rows(store, session_id))

    _, second_count = compressor.prune_tool_results_only(
        conversation(store, session_id), current_tokens=1_000_000)

    assert second_count >= 1
    assert len(_rows(store, session_id)) - rows_before_second == second_count
    hits = [hit for hit in store.search.search_messages("start") if hit.session_id == session_id]
    assert len(hits) == 1


def test_in_place_prune_skips_when_another_writer_changed_the_stored_body(store) -> None:
    """A stored body that moved on belongs to a newer generation: the prune must not commit at all.

    The in-place rewrite keeps row ids, so the held-history watermark cannot see the race; only the
    stored-body mismatch can. Falling back to the full writer here archives rows it never compared and
    republishes the stale held text over the newer one (#124102).
    """
    session_id = "PRUNE_IN_PLACE_STALE"
    seed_session(store, session_id, "telegram")
    seed_messages(store, session_id, *_history())
    agent = _build_agent(store, session_id)
    _configure_pruning(agent)
    messages = conversation(store, session_id)
    # Another surface rewrites the row this agent still holds the pre-edit body of.
    raw_holder(store)._execute_write(lambda conn: conn.execute(
        "UPDATE messages SET content = 'edited elsewhere' WHERE session_id = ? AND tool_call_id = 'call_0'",
        (session_id,)))
    rows_before = _rows(store, session_id)

    with patch.object(store.messages, "compact", wraps=store.messages.compact) as full_rewrite:
        result, count = agent.context_compressor.prune_tool_results_only(messages, current_tokens=120_000)

    assert (result, count) == (messages, 0)
    assert full_rewrite.call_count == 0
    assert _rows(store, session_id) == rows_before
    live = [message["content"] for message in conversation(store, session_id)]
    assert live.count("edited elsewhere") == 1


def test_in_place_prune_falls_back_to_full_rewrite_when_a_row_cannot_be_named(store) -> None:
    """Genuine ambiguity over *current* content still commits through the full writer."""
    session_id = "PRUNE_IN_PLACE_UNRESOLVED"
    seed_session(store, session_id, "telegram")
    seed_messages(store, session_id, *_history())
    agent = _build_agent(store, session_id)
    _configure_pruning(agent)
    messages = conversation(store, session_id)

    # Two live rows now carry call_0's current body, so no single row can be named.
    def _duplicate_call_0(conn):
        columns = ", ".join(
            name for name in (row[1] for row in conn.execute("PRAGMA table_info(messages)"))
            if name not in ("id", "display_order"))
        conn.execute(
            f"INSERT INTO messages ({columns}) SELECT {columns} FROM messages "
            "WHERE session_id = ? AND tool_call_id = 'call_0' AND active = 1", (session_id,))

    raw_holder(store)._execute_write(_duplicate_call_0)

    compact = agent.persistence.messages.compact
    with patch.object(agent.persistence.messages, "compact", wraps=compact) as full_rewrite:
        pruned, count = agent.context_compressor.prune_tool_results_only(messages, current_tokens=120_000)

    assert count >= 1
    assert full_rewrite.call_count == 1
    assert [m["content"] for m in conversation(store, session_id)] == [
        m["content"] for m in pruned]


def test_in_place_prune_leaves_unwritten_suffix_to_the_flush(store) -> None:
    session_id = "PRUNE_IN_PLACE_SUFFIX"
    seed_session(store, session_id, "telegram")
    seed_messages(store, session_id, *_history())
    agent = _build_agent(store, session_id)
    _configure_pruning(agent)
    messages = conversation(store, session_id) + [{"role": "user", "content": "new turn"}]

    pruned, count = agent.context_compressor.prune_tool_results_only(messages, current_tokens=120_000)

    assert count >= 1
    assert not pruned[-1].get("_db_persisted")
    assert "new turn" not in [m["content"] for m in conversation(store, session_id)]
    agent._flush_messages_to_session_db(pruned)
    durable = [m["content"] for m in conversation(store, session_id)]
    assert durable == [m["content"] for m in pruned]
    assert durable.count("new turn") == 1


def test_rewrite_pruned_rows_refuses_user_rows(store) -> None:
    session_id = "PRUNE_IN_PLACE_USER"
    seed_session(store, session_id, "telegram")
    seed_messages(store, session_id, *_history())
    messages = conversation(store, session_id)
    rows_before = _rows(store, session_id)

    with pytest.raises(PruneRowUnresolved):
        _rewrite_pruned_rows(store, session_id, [(messages[0], {**messages[0], "content": "rewritten"})])
    assert _rows(store, session_id) == rows_before


def test_a_stale_generation_with_an_unstamped_trailing_row_still_aborts(store) -> None:
    """The stale check must run for the durable prefix an unstamped trailing row cannot suppress.

    A lease-less writer can hold an exact persisted prefix plus a trailing message with no stamps (a live
    turn this process authored and has not flushed yet). The early return that caps the watermark for that
    trailing row must not also skip the version check for the rows the caller does hold exactly (#124102).
    """
    session_id = "PRUNE_STALE_PREFIX"
    seed_session(store, session_id, "telegram")
    seed_messages(store, session_id, *_history())
    agent = _build_agent(store, session_id)
    _configure_pruning(agent)
    messages = conversation(store, session_id) + [{"role": "user", "content": "live turn"}]
    raw_holder(store)._execute_write(lambda conn: conn.execute(
        "UPDATE messages SET content = 'edited elsewhere' WHERE session_id = ? AND tool_call_id = 'call_0'",
        (session_id,)))
    rows_before = _rows(store, session_id)

    with patch.object(store.messages, "compact", wraps=store.messages.compact) as full_rewrite:
        result, count = agent.context_compressor.prune_tool_results_only(messages, current_tokens=120_000)

    assert (result, count) == (messages, 0)
    assert full_rewrite.call_count == 0
    assert _rows(store, session_id) == rows_before
    live = [message["content"] for message in conversation(store, session_id)]
    assert live.count("edited elsewhere") == 1


def _history_with_padded_text() -> list[dict]:
    """A history whose user/assistant rows carry surrounding whitespace, the resume projection's lens."""
    messages: list[dict] = [{"role": "user", "content": "  start \n"}]
    for index in range(8):
        call_id = f"call_{index}"
        messages.append(_assistant_call(call_id))
        content = chr(65 + index) * 24_000 if index < 3 else "ok"
        messages.append(_tool_result(call_id, content))
    messages.append({"role": "assistant", "content": "  done \n"})
    messages.append({"role": "user", "content": "wrap up"})
    return messages


@pytest.mark.parametrize("loader", ["plain", "resume"], ids=["reload", "resume_projection"])
def test_a_freshly_resumed_unchanged_transcript_is_not_stale(store, loader: str) -> None:
    """The loader lens (sanitize + strip on user/assistant strings) is not a rewrite (#124102).

    Stored user/assistant content with surrounding whitespace loads stripped; the pre-repair head compared
    raw storage bytes against that loaded view and falsely reported every such unchanged row as rewritten,
    silently disabling the prune. Through the lens it must pass, and the prune must still reclaim.
    """
    session_id = "PRUNE_NOT_STALE_WHITESPACE"
    seed_session(store, session_id, "telegram")
    seed_messages(store, session_id, *_history_with_padded_text())
    agent = _build_agent(store, session_id)
    _configure_pruning(agent)
    if loader == "resume":
        messages = [m.as_openai_message() for m in store.messages.read_resume_conversations(session_id).model]
    else:
        messages = conversation(store, session_id, _WITH_ROW_IDS)

    assert store.messages.rewritten_held_row_ids(session_id, [held_message(m) for m in messages]) == ()

    with patch.object(store.messages, "compact", wraps=store.messages.compact) as full_rewrite:
        result, count = agent.context_compressor.prune_tool_results_only(messages, current_tokens=120_000)

    assert (result, count) != (messages, 0)
    assert full_rewrite.call_count == 0  # committed in place, not through the generation writer


def test_an_assistant_tool_calls_rewrite_is_stale_with_and_without_a_snapshot(store) -> None:
    """The old guard compared ids and tool content only; changed ``tool_calls`` arguments slipped through.

    A named assistant row whose stored call arguments moved on is a lost race. It must fail closed in the
    row-local writer through the loader-lens identity compare (which covers the arguments), and equally
    through the lease-less gate — aborting the whole prune, not just the row (#124102).
    """
    session_id = "PRUNE_STALE_TOOL_CALLS"
    seed_session(store, session_id, "telegram")
    seed_messages(store, session_id, *_history())
    messages = conversation(store, session_id, _WITH_ROW_IDS)
    held_assistant = next(m for m in messages if m.get("role") == "assistant" and m.get("tool_calls"))
    # Another writer changes the stored arguments of that row; the content (empty) is untouched.
    raw_holder(store)._execute_write(lambda conn: conn.execute(
        "UPDATE messages SET tool_calls = ? WHERE session_id = ? AND id = ?",
        (json.dumps([{**held_assistant["tool_calls"][0],
                      "function": {**held_assistant["tool_calls"][0]["function"],
                                   "arguments": '{"cmd":"winner"}'}}]), session_id,
         held_assistant["_row_id"])))

    # The row-local writer names the row exactly and fails closed through the lens compare.
    stale = dict(held_assistant)
    with pytest.raises(PruneRowStale):
        _rewrite_pruned_rows(store, session_id, [(stale, {**stale, "content": "pruned stub"})])
    # The lease-less gate sees the same race: the prune must be a true no-op.
    agent = _build_agent(store, session_id)
    _configure_pruning(agent)
    rows_before = _rows(store, session_id)
    with patch.object(store.messages, "compact", wraps=store.messages.compact) as full_rewrite:
        result, count = agent.context_compressor.prune_tool_results_only(messages, current_tokens=120_000)
    assert (result, count) == (messages, 0)
    assert full_rewrite.call_count == 0
    assert _rows(store, session_id) == rows_before


def test_a_concurrent_user_edit_stays_active_when_the_prune_falls_back(store) -> None:
    """The fallback full writer republishes every held row, user rows included.

    A lease-less prune can hold an unwritten message BEFORE later durable rows, which sends the commit
    down the full-writer fallback. Its staleness gate must then cover the USER rows the fallback
    re-inserts verbatim, not just the tool/assistant rows the in-place writer edits: a user row another
    view rewrote in place (``set_user_message_content``) is the same lost race. Republishing the held
    pre-edit body would leave the newer edit with zero active copies (#124102).
    """
    session_id = "PRUNE_STALE_USER_EDIT_FALLBACK"
    seed_session(store, session_id, "telegram")
    seed_messages(store, session_id, *_history())
    agent = _build_agent(store, session_id)
    _configure_pruning(agent)
    messages = conversation(store, session_id, _WITH_ROW_IDS)
    user_row = next(m for m in messages if m.get("role") == "user")
    # Another view rewrites that durable user row in place (a newer user edit).
    assert store.messages.set_user_message_content(
        session_id, user_row["_row_id"], "edited by the user elsewhere") is True
    # One unwritten live-turn message placed BEFORE later durable rows forces the fallback branch:
    # _rewrite_pruned_rows_in_place refuses (row order would break) and returns None.
    splice = next(i for i, m in enumerate(messages)
                  if m.get("_row_id") and m["_row_id"] > user_row["_row_id"])
    held = messages[:splice] + [{"role": "user", "content": "live turn not yet flushed"}] + messages[splice:]
    rows_before = _rows(store, session_id)

    with patch.object(store.messages, "compact", wraps=store.messages.compact) as full_rewrite:
        result, count = agent.context_compressor.prune_tool_results_only(held, current_tokens=120_000)

    assert (result, count) == (held, 0)  # a true no-op: the stale generation is never published
    assert full_rewrite.call_count == 0
    assert _rows(store, session_id) == rows_before
    live = [message["content"] for message in conversation(store, session_id)]
    assert live.count("edited by the user elsewhere") == 1  # the newer edit keeps its active copy
    assert "start" not in live  # the stale pre-edit body is not republished


def test_the_fallback_still_commits_when_no_held_row_was_rewritten(store) -> None:
    """Widening the stale check to user rows must not disable the fallback itself: with no concurrent
    rewrite, the same interleaved-unwritten shape still commits through the full writer (#124102)."""
    session_id = "PRUNE_FALLBACK_CLEAN_COMMIT"
    seed_session(store, session_id, "telegram")
    seed_messages(store, session_id, *_history())
    agent = _build_agent(store, session_id)
    _configure_pruning(agent)
    messages = conversation(store, session_id, _WITH_ROW_IDS)
    first_user = next(i for i, m in enumerate(messages) if m.get("role") == "user")
    splice = next(i for i, m in enumerate(messages) if i > first_user and m.get("_row_id"))
    held = messages[:splice] + [{"role": "user", "content": "live turn not yet flushed"}] + messages[splice:]

    with patch.object(store.messages, "compact", wraps=store.messages.compact) as full_rewrite:
        result, count = agent.context_compressor.prune_tool_results_only(held, current_tokens=120_000)

    assert (result, count) != (held, 0)
    assert full_rewrite.call_count == 1
    live = [message["content"] for message in conversation(store, session_id)]
    assert live.count("start") == 1  # the held user row is republished exactly once
    assert live.count("live turn not yet flushed") == 1


def test_a_user_edit_landing_after_the_precheck_stays_active_in_the_commit(store, tmp_path: Path) -> None:
    """The lease-less stale gate runs OUTSIDE the full writer's write transaction.

    A second handle that rewrites a covered user row AFTER the precheck but BEFORE the commit's
    ``BEGIN IMMEDIATE`` is invisible to the precheck; republishing the held text would leave the newer
    edit with zero active copies (#124102). The commit must re-verify the held rows against the live
    store inside its own transaction and abort as a no-op when one moved: the held row is never
    published and the edit keeps its row.
    """
    session_id = "PRUNE_CAS_COMMIT_WINDOW"
    seed_session(store, session_id, "telegram")
    seed_messages(store, session_id, *_history())
    agent = _build_agent(store, session_id)
    _configure_pruning(agent)
    messages = conversation(store, session_id, _WITH_ROW_IDS)
    user_row = next(m for m in messages if m.get("role") == "user")
    splice = next(i for i, m in enumerate(messages)
                  if m.get("_row_id") and m["_row_id"] > user_row["_row_id"])
    held = messages[:splice] + [{"role": "user", "content": "live turn not yet flushed"}] + messages[splice:]

    other = open_persistence(profile_scope(home=tmp_path))
    original = store.messages.compact

    def _edit_then_commit(*args, **kwargs):
        # The call boundary is after the caller's precheck and before BEGIN IMMEDIATE: a second
        # writer can take the write lock first and rewrite the covered user row in place.
        assert other.messages.set_user_message_content(
            session_id, user_row["_row_id"], "edited inside the window") is True
        return original(*args, **kwargs)

    with patch.object(store.messages, "compact", side_effect=_edit_then_commit):
        result, count = agent.context_compressor.prune_tool_results_only(held, current_tokens=120_000)

    assert (result, count) == (held, 0)  # the commit aborted inside its transaction: a true no-op
    live = [message["content"] for message in conversation(store, session_id)]
    assert live.count("edited inside the window") == 1  # the newer edit keeps its single active row
    assert "start" not in live  # the stale pre-edit body was never republished
    other.close()
