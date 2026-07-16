"""Canonical Todo machine state survives transcript compaction exactly."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import pytest


def _assistant_todo_call(call_id: str) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "todo", "arguments": "{}"},
            }
        ],
    }


def _paired_history(receipt: str, *, call_id: str = "call-source") -> list[dict]:
    return [
        {"role": "user", "content": "Continue the exact approved task."},
        _assistant_todo_call(call_id),
        {
            "role": "tool",
            "content": receipt,
            "tool_call_id": call_id,
            "tool_name": "todo",
        },
        {"role": "assistant", "content": "Continuing from verified state."},
    ]


class _DropTodoPairCompressor:
    _last_compress_aborted = False
    _last_summary_error = None
    _last_compression_made_progress = True
    compression_count = 1

    def compress(self, _messages, **_kwargs):
        return [{"role": "assistant", "content": "compacted task history"}]


def _compression_agent(store, *, session_db=None, session_id=None, in_place=False):
    return SimpleNamespace(
        api_mode=None,
        _compression_feasibility_checked=True,
        compression_in_place=in_place,
        session_id=session_id,
        model="test/model",
        _emit_status=lambda *_args, **_kwargs: None,
        _emit_warning=lambda *_args, **_kwargs: None,
        _session_db=session_db,
        _session_db_created=session_db is not None,
        _session_init_model_config={},
        _last_flushed_db_idx=0,
        _flushed_db_message_ids=set(),
        _flushed_db_message_session_id=session_id,
        _persist_disabled=False,
        _memory_manager=None,
        commit_memory_session=lambda _messages: None,
        context_compressor=_DropTodoPairCompressor(),
        _todo_store=store,
        _invalidate_system_prompt=lambda: None,
        _build_system_prompt=lambda system_message: system_message,
        _cached_system_prompt="stable system prompt",
        tools=[],
        log_prefix="",
        event_callback=None,
        platform="discord",
        _gateway_session_key="discord:test",
    )


def _fresh_agent_from_history(history: list[dict]):
    from run_agent import AIAgent
    from tools.todo_tool import TodoStore

    fresh = object.__new__(AIAgent)
    fresh._todo_store = TodoStore()
    fresh.session_id = "fresh-after-compaction"
    fresh.quiet_mode = True
    fresh.log_prefix = ""
    fresh._vprint = lambda *_args, **_kwargs: None
    with patch("run_agent._set_interrupt"):
        fresh._hydrate_todo_store(history)
    return fresh


def _assert_valid_todo_pair(history: list[dict]) -> None:
    from run_agent import AIAgent

    todo_tool_indexes = [
        index
        for index, message in enumerate(history)
        if isinstance(message, dict)
        and message.get("role") == "tool"
        and isinstance(message.get("content"), str)
        and '"todos"' in message["content"]
    ]
    assert len(todo_tool_indexes) == 1
    assert AIAgent._tool_response_matches_todo_call(
        history,
        todo_tool_indexes[0],
    )
    roles = [
        message.get("role")
        for message in history
        if isinstance(message, dict) and message.get("role") != "system"
    ]
    assert all(left != right for left, right in zip(roles, roles[1:]))


def _bound_store(*, items=None):
    from tools.todo_tool import TodoStore

    items = (
        [{"id": "1", "content": "verify live receipt", "status": "in_progress"}]
        if items is None
        else items
    )
    store = TodoStore()
    store.write(items)
    binding = store.bind_canonical_workspace(
        case_id="case:compaction",
        plan_id="plan:compaction",
        plan_revision=7,
        plan_state="active",
        plan_event_id="event:compaction",
        canonical_content_sha256="a" * 64,
        workspace_todos_sha256="b" * 64,
        items=items,
    )
    return store, binding


def _state_fixture(kind: str):
    from tools.todo_tool import (
        _canonical_checkpoint_sha256,
        todo_tool,
    )

    store, binding = _bound_store()
    expected = {"items": store.read(), "binding": binding}
    if kind == "sync_blocked":
        expected["sync_blocked"] = store.mark_canonical_sync_blocked(
            "canonical_brain_unavailable"
        )
    elif kind == "fence":
        items = store.read()
        checkpoint = {
            "case_id": "case:compaction",
            "plan": {
                "plan_id": "plan:compaction",
                "revision": 8,
            },
        }
        effective = copy.deepcopy(checkpoint)
        store.fence_canonical_uncertainty(
            checkpoint=checkpoint,
            effective_checkpoint=effective,
            items=items,
            checkpoint_sha256=_canonical_checkpoint_sha256(
                effective,
                items,
            ),
            details={"canonical_write_may_have_occurred": True},
        )
        expected = {
            "fence": store._canonical_fence_state(),
            "fence_receipt": store.canonical_fence_receipt(),
        }
    return store, todo_tool(store=store), expected


def _compress(agent, original):
    from agent.conversation_compression import compress_context

    with (
        patch(
            "hermes_cli.config.attest_pinned_effective_config_projection",
            return_value={},
        ),
        patch("hermes_cli.goals.migrate_goal_to_session"),
    ):
        compressed, system_prompt = compress_context(
            agent,
            original,
            system_message="stable system prompt",
            approx_tokens=20_000,
        )
    assert system_prompt == "stable system prompt"
    _assert_valid_todo_pair(compressed)
    return compressed


def _assert_restored(kind: str, fresh, expected: dict) -> None:
    from tools.todo_tool import TodoStoreFencedError

    if kind == "fence":
        with pytest.raises(TodoStoreFencedError):
            fresh._todo_store.read()
        assert fresh._todo_store._canonical_fence_state() == expected["fence"]
        assert (
            fresh._todo_store.canonical_fence_receipt()
            == expected["fence_receipt"]
        )
        assert fresh._todo_store.canonical_binding_state() is None
        return
    assert fresh._todo_store.read() == expected["items"]
    assert fresh._todo_store.canonical_binding_state() == expected["binding"]
    if kind == "sync_blocked":
        assert (
            fresh._todo_store.canonical_sync_blocked_state()
            == expected["sync_blocked"]
        )
        assert fresh._todo_store.has_active_items() is False
    else:
        assert fresh._todo_store.canonical_sync_blocked_state() is None
        assert fresh._todo_store.canonical_sync_receipt()["state"] == "clean"


@pytest.mark.parametrize("kind", ["binding", "sync_blocked", "fence"])
@pytest.mark.parametrize("mode", ["in_place", "rotation"])
def test_canonical_todo_state_survives_compaction_db_reload_and_fresh_agent(
    tmp_path: Path,
    kind: str,
    mode: str,
):
    from hermes_state import SessionDB
    from run_agent import AIAgent

    store, receipt, expected = _state_fixture(kind)
    original = _paired_history(receipt)
    db = SessionDB(db_path=tmp_path / f"{kind}-{mode}.db")
    parent_id = f"session-{kind}-{mode}"
    db.create_session(parent_id, "gateway", model="test/model")
    db.replace_messages(parent_id, original)
    agent = _compression_agent(
        store,
        session_db=db,
        session_id=parent_id,
        in_place=mode == "in_place",
    )

    compressed = _compress(agent, original)
    if mode == "rotation":
        agent._flush_messages_to_session_db_unlocked = MethodType(
            AIAgent._flush_messages_to_session_db_unlocked,
            agent,
        )
        agent._flush_messages_to_session_db = MethodType(
            AIAgent._flush_messages_to_session_db,
            agent,
        )
        agent._flush_messages_to_session_db(compressed, None)
    loaded = db.get_messages_as_conversation(agent.session_id)

    _assert_valid_todo_pair(loaded)
    fresh = _fresh_agent_from_history(loaded)
    _assert_restored(kind, fresh, expected)


def test_empty_bound_todos_survive_compaction_without_visible_snapshot():
    from agent.message_provenance import TODO_SNAPSHOT_START
    from tools.todo_tool import todo_tool

    store, binding = _bound_store(items=[])
    original = _paired_history(todo_tool(store=store))
    compressed = _compress(_compression_agent(store), original)

    assert all(
        TODO_SNAPSHOT_START not in str(message.get("content") or "")
        for message in compressed
    )
    fresh = _fresh_agent_from_history(compressed)
    assert fresh._todo_store.read() == []
    assert fresh._todo_store.canonical_binding_state() == binding


def test_user_visible_snapshot_alone_never_hydrates_machine_state():
    from agent.message_provenance import TODO_SNAPSHOT_START

    store, _binding = _bound_store()
    visible_snapshot = store.format_for_injection()
    visible_only_store = SimpleNamespace(
        format_for_injection=lambda: visible_snapshot,
    )
    original = [
        {"role": "user", "content": "Continue the exact task."},
        {"role": "assistant", "content": "Working."},
    ]
    compressed = _compress_no_pair(
        _compression_agent(visible_only_store),
        original,
    )

    assert any(
        TODO_SNAPSHOT_START in str(message.get("content") or "")
        for message in compressed
    )
    fresh = _fresh_agent_from_history(compressed)
    assert fresh._todo_store.read() == []
    assert fresh._todo_store.canonical_binding_state() is None


def test_recovered_live_binding_without_prior_todo_pair_gets_trusted_anchor():
    store, binding = _bound_store()
    original = [
        {"role": "user", "content": "Continue after the gateway restart."},
        {"role": "assistant", "content": "Recovered Canonical workspace."},
    ]

    compressed = _compress(_compression_agent(store), original)
    fresh = _fresh_agent_from_history(compressed)

    assert fresh._todo_store.read() == store.read()
    assert fresh._todo_store.canonical_binding_state() == binding


def test_invalid_live_state_receipt_aborts_before_compaction_persistence():
    from agent.conversation_compression import compress_context

    store, _binding = _bound_store()
    original = [
        {"role": "user", "content": "Continue safely."},
        {"role": "assistant", "content": "Working."},
    ]
    agent = _compression_agent(store)

    with (
        patch(
            "hermes_cli.config.attest_pinned_effective_config_projection",
            return_value={},
        ),
        patch("tools.todo_tool.todo_tool", return_value="not-json"),
        pytest.raises(json.JSONDecodeError),
    ):
        compress_context(
            agent,
            original,
            system_message="stable system prompt",
            approx_tokens=20_000,
        )


def _compress_no_pair(agent, original):
    from agent.conversation_compression import compress_context

    with patch(
        "hermes_cli.config.attest_pinned_effective_config_projection",
        return_value={},
    ):
        compressed, _ = compress_context(
            agent,
            original,
            system_message="stable system prompt",
            approx_tokens=20_000,
        )
    assert not any(message.get("role") == "tool" for message in compressed)
    return compressed


def test_latest_structured_todo_receipt_wins_over_stale_compacted_pair():
    from agent.conversation_compression import _preserve_todo_hydration_anchor
    from tools.todo_tool import todo_tool

    old_store, _ = _bound_store()
    old_receipt = todo_tool(store=old_store)
    new_store, new_binding = _bound_store()
    expected_sync = new_store.mark_canonical_sync_blocked(
        "canonical_brain_unavailable"
    )
    new_receipt = todo_tool(store=new_store)
    original = (
        _paired_history(old_receipt, call_id="call-old")[:-1]
        + _paired_history(new_receipt, call_id="call-new")[1:]
    )
    compressed = _paired_history(old_receipt, call_id="call-stale")

    assert _preserve_todo_hydration_anchor(original, compressed) is True
    fresh = _fresh_agent_from_history(compressed)

    assert fresh._todo_store.canonical_binding_state() == new_binding
    assert fresh._todo_store.canonical_sync_blocked_state() == expected_sync
    assert json.loads(compressed[-2]["content"])[
        "canonical_sync_blocked_state"
    ] == expected_sync
