"""Quiet one-shot chat suppresses callbacks that can write presentation output (#93220)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import cli


def test_quiet_single_query_neutralizes_callbacks_before_running_turn(monkeypatch):
    monkeypatch.delenv("HERMES_SINGLE_QUERY_SESSION", raising=False)
    monkeypatch.setattr(cli, "_should_seed_interactive", lambda *args, **kwargs: False)
    monkeypatch.setattr(cli, "_collect_query_images", lambda query, image: (query, []))
    monkeypatch.setattr(cli, "_collect_kanban_task_images", lambda images: [])
    monkeypatch.setattr(cli, "_route_single_query_images", lambda *_args: "routed query")
    monkeypatch.setattr(
        cli,
        "_single_query_exit_code",
        lambda *_args, **_kwargs: 0,
    )
    monkeypatch.setattr(cli, "_finalize_single_query", lambda _cli: None)

    callbacks = SimpleNamespace(
        reasoning_callback=object(),
        tool_start_callback=object(),
        tool_complete_callback=object(),
        tool_progress_callback=object(),
        tool_progress_mode="all",
        stream_delta_callback=object(),
        tool_gen_callback=object(),
        quiet_mode=False,
        suppress_status_output=False,
    )
    stub = SimpleNamespace(
        session_id="session-1",
        model="test-model",
        tool_progress_mode="all",
        _active_agent_route_signature="route-1",
        _claim_active_session=lambda *_args, **_kwargs: True,
        _ensure_runtime_credentials=lambda: True,
        _resolve_turn_agent_config=lambda _query: {
            "signature": "route-1",
            "model": "test-model",
            "runtime": None,
            "request_overrides": None,
        },
    )

    def init_agent(**_kwargs):
        stub.agent = callbacks
        return True

    stub._init_agent = init_agent
    observed = {}

    def run_quiet_turn(_cli, _query, emitter=None):
        observed.update(vars(_cli.agent))

    monkeypatch.setattr(cli, "_run_quiet_single_query", run_quiet_turn)
    from hermes_cli import quiet_single_query

    def exit_single_query(code):
        raise SystemExit(code)

    monkeypatch.setattr(quiet_single_query, "exit_single_query", exit_single_query)

    with pytest.raises(SystemExit) as exc:
        cli._run_single_query_mode(stub, "question", None, True, True)

    assert exc.value.code == 0
    assert observed["quiet_mode"] is True
    assert observed["suppress_status_output"] is True
    assert observed["tool_progress_mode"] == "off"
    for key in (
        "reasoning_callback",
        "tool_start_callback",
        "tool_complete_callback",
        "tool_progress_callback",
        "stream_delta_callback",
        "tool_gen_callback",
    ):
        assert observed[key] is None


def test_suppress_status_output_gates_quiet_tool_messages():
    """The executor's fallback [tool]/[done] messages stay silent under -Q."""
    from run_agent import AIAgent

    agent = object.__new__(AIAgent)
    agent.quiet_mode = True
    agent.tool_progress_callback = None
    agent.platform = "cli"

    agent.suppress_status_output = False
    assert agent._should_emit_quiet_tool_messages() is True

    agent.suppress_status_output = True
    assert agent._should_emit_quiet_tool_messages() is False
