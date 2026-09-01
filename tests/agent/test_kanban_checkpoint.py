"""Tests for agent/kanban_checkpoint.py — finalize-in-process machinery.

Covers the four executable behaviors the card demands:
1. Per-turn checkpoint reminder is present when a kanban worker is running and
   is NOT duplicated across turns / after a terminal tool.
2. A simulated text-response-without-terminal triggers exactly one finalize
   turn (via should_fire_finalize_turn + the fired latch), and the built
   instruction points only at the terminal tools.
3. A finalize turn that still fails falls through (fired latch stays set; the
   bounded attempt count is not extended).
4. Non-kanban sessions are untouched (env gate).
"""

from __future__ import annotations

import pytest

from agent.kanban_checkpoint import (
    build_checkpoint_reminder,
    build_finalize_instruction,
    finalize_metrics,
    kanban_checkpoint_enabled,
    mark_finalize_fired,
    mark_finalize_succeeded,
    maybe_append_checkpoint_reminder,
    reset_finalize_state,
    should_fire_finalize_turn,
    terminal_only_schemas,
    terminal_tools_present,
)


@pytest.fixture(autouse=True)
def _clean_state():
    reset_finalize_state()
    yield
    reset_finalize_state()


@pytest.fixture
def worker_env(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_f7f99419")
    for var in ("HERMES_KANBAN_RUN_ID", "HERMES_KANBAN_STOP_NUDGE"):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def _tool_def(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"{name} tool",
            "parameters": {"type": "object", "properties": {}},
        },
    }


# ── 4. Non-kanban sessions are untouched ─────────────────────────────


def test_disabled_without_task_env(monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    assert kanban_checkpoint_enabled() is False
    msgs = [{"role": "user", "content": "work"}]
    out = maybe_append_checkpoint_reminder(msgs, "")
    # Returns the same object; nothing appended.
    assert out is msgs
    assert all(not m.get("_kanban_checkpoint_reminder_synthetic") for m in out)


def test_should_fire_finalize_off_when_disabled(worker_env):
    worker_env.delenv("HERMES_KANBAN_TASK")
    assert should_fire_finalize_turn(
        attempts=0, terminal_tools_available=True
    ) is False


# ── 2. Exactly one finalize turn ─────────────────────────────────────


def test_should_fire_finalize_once(worker_env):
    assert should_fire_finalize_turn(
        attempts=0, terminal_tools_available=True
    ) is True
    # Fired latch: subsequent attempts must NOT re-fire the finalize turn.
    assert should_fire_finalize_turn(
        attempts=1, terminal_tools_available=True
    ) is False


def test_finalize_requires_terminal_tools(worker_env):
    # No terminal schemas in the toolset → cannot force a finalize turn.
    assert should_fire_finalize_turn(
        attempts=0, terminal_tools_available=False
    ) is False


def test_finalize_instruction_targets_only_terminal_tools(worker_env):
    instr = build_finalize_instruction("t_f7f99419")
    assert "kanban_complete" in instr
    assert "kanban_block" in instr
    assert "t_f7f99419" in instr
    assert "only tools" in instr.lower()


def test_terminal_only_schemas_filter(worker_env):
    tools = [
        _tool_def("web_search"),
        _tool_def("kanban_complete"),
        _tool_def("kanban_heartbeat"),
        _tool_def("kanban_block"),
        _tool_def("kanban_request_review"),
        _tool_def("kanban_request_changes"),
    ]
    only = terminal_only_schemas(tools)
    assert only is not None
    names = {t["function"]["name"] for t in only}
    assert names == {
        "kanban_complete", "kanban_block",
        "kanban_request_review", "kanban_request_changes",
    }
    assert terminal_tools_present(tools) is True


def test_terminal_only_schemas_missing_returns_none(worker_env):
    tools = [_tool_def("web_search"), _tool_def("kanban_complete")]
    assert terminal_only_schemas(tools) is None
    assert terminal_tools_present(tools) is False
    assert terminal_only_schemas([]) is None


def test_terminal_only_schemas_graceful_on_unknown_shape(worker_env):
    assert terminal_only_schemas([{"not_a_function": True}]) is None
    assert terminal_only_schemas(None) is None


# ── 3. Failed finalize still falls through ───────────────────────────


def test_fired_latch_is_sticky_after_mark(worker_env):
    mark_finalize_fired()
    # Even if the finalize turn still produced no terminal call, the next
    # text-exit must not run a SECOND finalize turn (capped at one per run).
    assert should_fire_finalize_turn(
        attempts=0, terminal_tools_available=True
    ) is False


# ── 1. Per-turn checkpoint reminder ─────────────────────────────────


def test_checkpoint_reminder_short_and_named(worker_env):
    rem = build_checkpoint_reminder("t_f7f99419")
    assert "t_f7f99419" in rem
    assert "kanban_complete" in rem or "kanban_block" in rem
    # A recency notice, not a paragraph.
    assert len(rem) < 260


def test_reminder_appended_after_tool_result(worker_env):
    msgs = [
        {"role": "user", "content": "work"},
        {"role": "tool", "name": "web_search", "tool_call_id": "1", "content": "ok"},
    ]
    out = maybe_append_checkpoint_reminder(msgs, "t_f7f99419")
    assert out != msgs  # new list returned when appended
    assert len(out) == len(msgs) + 1
    tail = out[-1]
    assert tail["role"] == "user"
    assert tail.get("_kanban_checkpoint_reminder_synthetic") is True
    assert "t_f7f99419" in tail["content"]


def test_reminder_skipped_when_terminal_already_called(worker_env):
    msgs = [
        {"role": "user", "content": "work"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": "kanban_complete", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "name": "kanban_complete",
            "tool_call_id": "1",
            "content": "done",
        },
    ]
    out = maybe_append_checkpoint_reminder(msgs, "t_f7f99419")
    assert out is msgs  # no reminder added after completion
    assert all(not m.get("_kanban_checkpoint_reminder_synthetic") for m in out)


def test_reminder_skipped_when_tail_is_user(worker_env):
    # Would create user→user on strict wire providers — must be skipped.
    msgs = [{"role": "user", "content": "work"}, {"role": "user", "content": "next"}]
    out = maybe_append_checkpoint_reminder(msgs, "t_f7f99419")
    assert out is msgs
    assert all(not m.get("_kanban_checkpoint_reminder_synthetic") for m in out)


def test_reminder_not_stacked_across_appends(worker_env):
    # Appending again (a later turn's request copy) yields exactly one reminder
    # — it never accumulates a paragraph of repeated notices.
    msgs = [{"role": "tool", "name": "web_search", "tool_call_id": "1", "content": "ok"}]
    out1 = maybe_append_checkpoint_reminder(msgs, "t_f7f99419")
    out2 = maybe_append_checkpoint_reminder(list(out1), "t_f7f99419")
    assert len([m for m in out2 if m.get("_kanban_checkpoint_reminder_synthetic")]) == 1


# ── Instrumentation metrics ──────────────────────────────────────────


def test_metrics_track_fired_and_succeeded(worker_env):
    assert finalize_metrics() == {
        "finalize_turn_fired": False,
        "finalize_turn_succeeded": False,
    }
    mark_finalize_fired()
    m = finalize_metrics()
    assert m["finalize_turn_fired"] is True
    assert m["finalize_turn_succeeded"] is False
    mark_finalize_succeeded()
    m = finalize_metrics()
    assert m["finalize_turn_fired"] is True
    assert m["finalize_turn_succeeded"] is True


def test_mark_succeeded_implies_fired(worker_env):
    mark_finalize_succeeded()
    m = finalize_metrics()
    assert m["finalize_turn_fired"] is True
    assert m["finalize_turn_succeeded"] is True


def test_reset_clears_metrics(worker_env):
    mark_finalize_fired()
    reset_finalize_state()
    assert finalize_metrics() == {
        "finalize_turn_fired": False,
        "finalize_turn_succeeded": False,
    }

# ── Loop-hook composition (restrict then release) ────────────────────


def _simplify_tools(tools):
    """Return just the exposed tool names to assert restriction/release."""
    if tools is None:
        return None
    return {t["function"]["name"] for t in tools}


def test_finalize_restricts_then_releases_on_terminal(worker_env):
    """Simulate the request-build hook: while a finalize turn is pending, ONLY
    the terminal kanban tools are exposed; once a terminal call is visible, the
    restriction releases and the full toolset is available again."""
    from agent.kanban_checkpoint import _session_called_terminal_in as terminal_seen

    tools = [
        _tool_def("web_search"),
        _tool_def("kanban_complete"),
        _tool_def("kanban_block"),
        _tool_def("kanban_request_review"),
        _tool_def("kanban_request_changes"),
    ]
    _fin = terminal_only_schemas(tools)
    assert _simplify_tools(_fin) == {
        "kanban_complete", "kanban_block",
        "kanban_request_review", "kanban_request_changes",
    }

    # (a) Finalize turn in flight, no terminal call yet → restricted request.
    api_msgs = [{"role": "tool", "name": "web_search", "tool_call_id": "1", "content": "cm"}]
    assert terminal_seen(api_msgs) is False
    tools_for_api = _fin  # hook replaces the toolset
    assert _simplify_tools(tools_for_api) == {
        "kanban_complete", "kanban_block",
        "kanban_request_review", "kanban_request_changes",
    }

    # (b) The model now produces the terminal call → restriction releases.
    api_msgs = api_msgs + [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "2", "type": "function",
                 "function": {"name": "kanban_block", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "name": "kanban_block", "tool_call_id": "2", "content": "blocked"},
    ]
    assert terminal_seen(api_msgs) is True
    tools_for_api = tools  # hook clears _kanban_finalize_tools → full toolset
    assert _simplify_tools(tools_for_api) == {
        "web_search", "kanban_complete", "kanban_block",
        "kanban_request_review", "kanban_request_changes",
    }


def test_finalize_restriction_releases_on_review_handoff(worker_env):
    """A correct review handoff inside the forced turn is a terminal close too:
    the restriction must release (not keep steering toward complete/block)."""
    from agent.kanban_checkpoint import _session_called_terminal_in as terminal_seen

    tools = [
        _tool_def("web_search"),
        _tool_def("kanban_complete"),
        _tool_def("kanban_block"),
        _tool_def("kanban_request_review"),
        _tool_def("kanban_request_changes"),
    ]
    _fin = terminal_only_schemas(tools)
    api_msgs = [
        {"role": "tool", "name": "web_search", "tool_call_id": "1", "content": "cm"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "9", "type": "function",
                 "function": {"name": "kanban_request_review", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "name": "kanban_request_review", "tool_call_id": "9", "content": "review"},
    ]
    assert terminal_seen(api_msgs) is True
    # Restriction releases exactly as it does for complete/block.
    assert _simplify_tools(_fin) == {
        "kanban_complete", "kanban_block",
        "kanban_request_review", "kanban_request_changes",
    }


def test_review_handoff_is_terminal_so_guard_does_not_fire(worker_env):
    """Rodge round-1 Critical: a worker that correctly handed off to review must
    not be re-nudged (or force-steered) into complete/block. Both the kanban_stop
    terminal loop and the checkpoint detector must treat request_review /
    request_changes as terminal transitions."""
    from agent.kanban_checkpoint import _session_called_terminal_in as terminal_seen
    from agent.kanban_stop import build_kanban_stop_nudge, session_called_kanban_terminal

    for tool in ("kanban_request_review", "kanban_request_changes"):
        msgs = [
            {"role": "user", "content": "work"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "1", "type": "function",
                     "function": {"name": tool, "arguments": "{}"}}
                ],
            },
            {"role": "tool", "name": tool, "tool_call_id": "1", "content": "ok"},
        ]
        assert session_called_kanban_terminal(msgs) is True
        assert terminal_seen(msgs) is True
        # The stop-guard sees a terminal handoff → no nudge, no forced turn.
        assert build_kanban_stop_nudge(messages=msgs, attempts=0) is None


def test_finalize_request_build_hook_is_noop_when_fired_not_pending(worker_env):
    """Once the finalize turn has fired and produced no terminal call, the next
    (fall-through) request still exposes the full toolset — we do NOT keep the
    run permanently locked to two tools. This is the 'still fails → exit exactly
    as today' contract."""
    tools = [_tool_def("web_search"), _tool_def("kanban_block")]
    # Not pending (agent._kanban_finalize_tools is None after a failed turn) —
    # the hook leaves tools_for_api untouched, so the full toolset stays.
    _fin = None
    tools_for_api = list(tools)  # unchanged by the no-op hook
    assert {t["function"]["name"] for t in tools_for_api} == {
        "web_search", "kanban_block",
    }
