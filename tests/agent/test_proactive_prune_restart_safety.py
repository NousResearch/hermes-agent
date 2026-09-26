"""Restart-safety regressions for proactive tool-result pruning."""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.context_compressor import _estimate_msg_budget_tokens
from hermes_state import SessionDB

_REARM_KEY = "_proactive_prune_rearm_tokens"

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

def _build_agent(db: SessionDB, session_id: str, *, platform: str = "telegram"):
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        return AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            session_db=db,
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

def _model_config(db: SessionDB, session_id: str) -> dict:
    raw = db.get_session(session_id)["model_config"]
    return json.loads(raw) if raw else {}

def test_gateway_eviction_reload_keeps_prune_and_durable_runway(tmp_path: Path) -> None:
    """A fresh gateway agent must reload both the pruned body and its runway."""
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "GATEWAY_PRUNE_RESTART"
    db.create_session(
        session_id, source="telegram", model_config={"keep": "value"},
    )
    db.append_messages_batch(session_id, _history())

    first_agent = _build_agent(db, session_id)
    _configure_pruning(first_agent)
    before = db.get_messages_as_conversation(session_id)
    pruned, count = first_agent.context_compressor.prune_tool_results_only(
        before, current_tokens=120_000,
    )

    assert count >= 1
    durable = db.get_messages_as_conversation(session_id)
    assert [message["content"] for message in durable] == [
        message["content"] for message in pruned
    ]
    assert len(durable[2]["content"]) < 24_000
    stored_runway = _model_config(db, session_id)[_REARM_KEY]
    assert _model_config(db, session_id)["keep"] == "value"
    assert stored_runway > sum(map(_estimate_msg_budget_tokens, durable))

    # Simulate gateway cache eviction / process restart: construct a wholly
    # new AIAgent and load the active transcript from SQLite.
    resumed_agent = _build_agent(db, session_id)
    _configure_pruning(resumed_agent)
    assert resumed_agent.context_compressor._proactive_prune_rearm_tokens == stored_runway
    reloaded = db.get_messages_as_conversation(session_id)
    archived_before = len(db.get_messages(session_id, include_inactive=True))
    result, second_count = resumed_agent.context_compressor.prune_tool_results_only(
        reloaded, current_tokens=1_000_000,
    )

    assert result is reloaded
    assert second_count == 0
    assert len(db.get_messages(session_id, include_inactive=True)) == archived_before

def test_fresh_agent_rearms_after_durable_history_regrowth_once(tmp_path: Path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "PRUNE_DURABLE_REGROWTH"
    db.create_session(session_id, source="telegram")
    db.append_messages_batch(session_id, _history())
    first_agent = _build_agent(db, session_id)
    _configure_pruning(first_agent)
    first, first_count = first_agent.context_compressor.prune_tool_results_only(
        db.get_messages_as_conversation(session_id), current_tokens=120_000,
    )
    assert first_count >= 1
    first_runway = _model_config(db, session_id)[_REARM_KEY]

    growth = [
        _assistant_call("regrown_large"),
        _tool_result("regrown_large", "z" * 240_000),
        _assistant_call("tail_1"),
        _tool_result("tail_1", "ok"),
        _assistant_call("tail_2"),
        _tool_result("tail_2", "ok"),
    ]
    db.append_messages_batch(session_id, growth)

    resumed = _build_agent(db, session_id)
    _configure_pruning(resumed)
    grown = db.get_messages_as_conversation(session_id)
    assert sum(map(_estimate_msg_budget_tokens, grown)) >= first_runway
    second, second_count = resumed.context_compressor.prune_tool_results_only(
        grown, current_tokens=1_000_000,
    )

    assert second_count >= 1
    second_runway = _model_config(db, session_id)[_REARM_KEY]
    assert second_runway > first_runway

    restarted = _build_agent(db, session_id)
    _configure_pruning(restarted)
    durable = db.get_messages_as_conversation(session_id)
    result, third_count = restarted.context_compressor.prune_tool_results_only(
        durable, current_tokens=1_000_000,
    )
    assert result is durable
    assert third_count == 0
    assert restarted.context_compressor._proactive_prune_rearm_tokens == second_runway

@pytest.mark.parametrize("writer", ["rewrite_pruned_rows", "archive_and_compact"])
def test_prune_persistence_failure_is_a_noop(tmp_path: Path, writer: str) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "PRUNE_PERSISTENCE_FAILURE"
    db.create_session(session_id, source="telegram")
    db.append_messages_batch(session_id, _history())
    agent = _build_agent(db, session_id)
    _configure_pruning(agent)
    messages = db.get_messages_as_conversation(session_id)
    original_contents = [message["content"] for message in messages]

    # The full rewrite runs when the store has no in-place writer.
    in_place = patch.object(db, "rewrite_pruned_rows", None) if writer == "archive_and_compact" else contextlib.nullcontext()
    with in_place, patch.object(
        db, writer, side_effect=RuntimeError("disk full"),
    ):
        result, count = agent.context_compressor.prune_tool_results_only(
            messages, current_tokens=120_000,
        )

    assert result is messages
    assert count == 0
    assert agent.context_compressor._proactive_prune_rearm_tokens == 0
    assert [message["content"] for message in messages] == original_contents
    assert [message["content"] for message in db.get_messages_as_conversation(session_id)] == original_contents
    assert _REARM_KEY not in _model_config(db, session_id)

def test_archive_model_config_patch_rolls_back_with_transcript(tmp_path: Path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "PRUNE_ATOMIC_ARCHIVE_FAILURE"
    db.create_session(
        session_id,
        source="telegram",
        model_config={"keep": "value", _REARM_KEY: 120_000},
    )
    original = [{"role": "user", "content": "original"}]
    db.append_messages_batch(session_id, original)

    with patch.object(
        db, "_insert_message_rows", side_effect=RuntimeError("insert failed"),
    ):
        with pytest.raises(RuntimeError, match="insert failed"):
            db.archive_and_compact(
                session_id,
                [{"role": "user", "content": "replacement"}],
                model_config_patch={_REARM_KEY: None},
            )

    assert db.get_messages_as_conversation(session_id)[0]["content"] == "original"
    assert _model_config(db, session_id) == {"keep": "value", _REARM_KEY: 120_000}

def test_model_switch_clears_durable_runway(tmp_path: Path) -> None:
    """update_model must clear BOTH the in-memory and the durable runway."""
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "MODEL_SWITCH_CLEARS_RUNWAY"
    db.create_session(
        session_id,
        source="telegram",
        model_config={"keep": "value", _REARM_KEY: 120_000},
    )
    agent = _build_agent(db, session_id)
    compressor = agent.context_compressor
    assert compressor._proactive_prune_rearm_tokens == 120_000

    compressor.update_model("other/model", 200_000)

    assert compressor._proactive_prune_rearm_tokens == 0
    assert _REARM_KEY not in _model_config(db, session_id)
    assert _model_config(db, session_id)["keep"] == "value"

def test_patch_session_model_config_merge_and_delete(tmp_path: Path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "PATCH_MODEL_CONFIG"
    db.create_session(
        session_id, source="cli", model_config={"keep": "value", "drop": 1},
    )

    db.patch_session_model_config(session_id, {"drop": None, "added": 7})
    assert _model_config(db, session_id) == {"keep": "value", "added": 7}

    # Missing rows and empty patches are no-ops, never errors.
    db.patch_session_model_config("NO_SUCH_SESSION", {"x": 1})
    db.patch_session_model_config(session_id, {})


def _rows(db: SessionDB, session_id: str) -> list[tuple]:
    return db._conn.execute(
        "SELECT id, role, content, active, compacted FROM messages WHERE session_id = ? ORDER BY id",
        (session_id,),
    ).fetchall()


def test_in_place_prune_grows_by_changed_rows_only(tmp_path: Path) -> None:
    """#124102: N messages with K demoted leave exactly N + K rows, and every live row keeps its id."""
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "PRUNE_IN_PLACE_GROWTH"
    db.create_session(session_id, source="telegram")
    db.append_messages_batch(session_id, _history())
    agent = _build_agent(db, session_id)
    _configure_pruning(agent)
    before_rows = _rows(db, session_id)
    messages = db.get_messages_as_conversation(session_id)

    pruned, count = agent.context_compressor.prune_tool_results_only(messages, current_tokens=120_000)

    changed = [i for i, (old, new) in enumerate(zip(messages, pruned)) if old["content"] != new["content"]]
    assert count >= 1 and changed
    after_rows = _rows(db, session_id)
    live = [row for row in after_rows if row[3] == 1]
    twins = [row for row in after_rows if row[3] == 0]
    assert len(after_rows) == len(before_rows) + len(changed)
    assert [row[0] for row in live] == [row[0] for row in before_rows]
    assert sorted(row[2] for row in twins) == sorted(before_rows[i][2] for i in changed)
    assert all(row[4] == 1 for row in twins)  # the demoted full bodies stay searchable
    assert [m["content"] for m in db.get_messages_as_conversation(session_id)] == [m["content"] for m in pruned]
    assert all(m.get("_db_persisted") for m in pruned)
    # The twin keeps the live row's display slot: the transcript page shows one row per message, in place.
    assert len(db.get_messages(session_id, include_compacted=True)) == len(pruned)


def test_in_place_prune_twice_keeps_one_search_hit_per_unchanged_turn(tmp_path: Path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "PRUNE_IN_PLACE_SEARCH"
    db.create_session(session_id, source="telegram")
    db.append_messages_batch(session_id, _history())
    agent = _build_agent(db, session_id)
    _configure_pruning(agent)
    compressor = agent.context_compressor
    _, first_count = compressor.prune_tool_results_only(
        db.get_messages_as_conversation(session_id), current_tokens=120_000)
    assert first_count >= 1
    grown = []
    for index in range(8, 16):
        call_id = f"call_{index}"
        grown += [_assistant_call(call_id), _tool_result(call_id, chr(65 + index % 26) * 24_000)]
    db.append_messages_batch(session_id, grown)
    rows_before_second = len(_rows(db, session_id))

    _, second_count = compressor.prune_tool_results_only(
        db.get_messages_as_conversation(session_id), current_tokens=1_000_000)

    assert second_count >= 1
    assert len(_rows(db, session_id)) - rows_before_second == second_count
    hits = [hit for hit in db.search_messages("start") if hit.get("session_id") == session_id]
    assert len(hits) == 1


def test_in_place_prune_falls_back_to_full_rewrite_when_a_row_cannot_be_named(tmp_path: Path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "PRUNE_IN_PLACE_UNRESOLVED"
    db.create_session(session_id, source="telegram")
    db.append_messages_batch(session_id, _history())
    agent = _build_agent(db, session_id)
    _configure_pruning(agent)
    messages = db.get_messages_as_conversation(session_id)
    # The stored body no longer matches what the agent holds, so the row cannot be named exactly.
    db._conn.execute(
        "UPDATE messages SET content = 'edited elsewhere' WHERE session_id = ? AND tool_call_id = 'call_0'",
        (session_id,))
    db._conn.commit()

    with patch.object(db, "archive_and_compact", wraps=db.archive_and_compact) as full_rewrite:
        _, count = agent.context_compressor.prune_tool_results_only(messages, current_tokens=120_000)

    assert count >= 1
    assert full_rewrite.call_count == 1


def test_in_place_prune_leaves_unwritten_suffix_to_the_flush(tmp_path: Path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "PRUNE_IN_PLACE_SUFFIX"
    db.create_session(session_id, source="telegram")
    db.append_messages_batch(session_id, _history())
    agent = _build_agent(db, session_id)
    _configure_pruning(agent)
    messages = db.get_messages_as_conversation(session_id) + [{"role": "user", "content": "new turn"}]

    pruned, count = agent.context_compressor.prune_tool_results_only(messages, current_tokens=120_000)

    assert count >= 1
    assert not pruned[-1].get("_db_persisted")
    assert "new turn" not in [m["content"] for m in db.get_messages_as_conversation(session_id)]
    agent._flush_messages_to_session_db(pruned)
    durable = [m["content"] for m in db.get_messages_as_conversation(session_id)]
    assert durable == [m["content"] for m in pruned]
    assert durable.count("new turn") == 1


def test_rewrite_pruned_rows_refuses_user_rows(tmp_path: Path) -> None:
    from hermes_state_errors import PruneRowUnresolvedError

    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "PRUNE_IN_PLACE_USER"
    db.create_session(session_id, source="telegram")
    db.append_messages_batch(session_id, _history())
    messages = db.get_messages_as_conversation(session_id)
    rows_before = _rows(db, session_id)

    with pytest.raises(PruneRowUnresolvedError):
        db.rewrite_pruned_rows(session_id, [(messages[0], {**messages[0], "content": "rewritten"})])
    assert _rows(db, session_id) == rows_before
