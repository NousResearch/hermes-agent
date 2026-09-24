"""Acceptance notices must use the actual CLI-to-agent observer wiring.

The provider constructor is isolated; CLI setup, steering, local routing and
stream-safe presentation are production methods. No model request is made.
"""
from __future__ import annotations

import re
import threading

import pytest


def _plain(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


@pytest.fixture
def wired_cli(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import cli as cli_mod
    import run_agent
    from hermes_cli import mcp_startup

    core = run_agent.AIAgent.__new__(run_agent.AIAgent)
    core._pending_steer = None
    core._pending_steer_lock = threading.Lock()
    core._pending_redirect = None
    core._pending_redirect_lock = threading.Lock()
    core._interrupt_requested = False
    core._interrupt_message = None
    core._tool_interrupt_reason = None
    core._hard_interrupt_requested = threading.Event()
    core._execution_thread_id = None
    core._interrupt_thread_signal_pending = False
    core.api_mode = "chat_completions"
    core.quiet_mode = True

    # Only provider construction/startup work is isolated. In particular, do not
    # install _steer_accepted_callback in the fixture: _init_agent must own it.
    monkeypatch.setattr(run_agent, "AIAgent", lambda **_kwargs: core)
    monkeypatch.setattr(cli_mod, "_prepare_deferred_agent_startup", lambda: None)
    monkeypatch.setattr(cli_mod, "_active_agent_ref", None)
    monkeypatch.setattr(mcp_startup, "ensure_mcp_discovery_before_agent_build", lambda **_kwargs: None)
    shell = cli_mod.HermesCLI(compact=True, max_turns=1)
    shell.agent = None
    shell._session_db = object()
    shell._resumed = False
    shell.conversation_history = []
    shell._install_tool_callbacks = lambda: None
    shell._ensure_tirith_security = lambda: None
    shell._ensure_runtime_credentials = lambda: True
    shell._pending_title = None
    shell.show_reasoning = False
    shell.final_response_markdown = "raw"
    shell.show_timestamps = False
    shell._reset_stream_state()
    emitted = []
    monkeypatch.setattr(cli_mod, "_cprint", lambda text: emitted.append(_plain(text)))
    monkeypatch.setattr(cli_mod, "_terminal_width_for_streaming", lambda: 74)
    monkeypatch.setattr(type(shell), "_scrollback_box_width", lambda self: 74)
    assert shell._init_agent(
        model_override="test/model",
        runtime_override={"provider": "openrouter", "api_mode": "chat_completions", "api_key": "test-key", "base_url": "https://example.invalid/v1"},
    ) is True
    assert shell.agent is core
    shell._agent_running = True
    emitted.clear()
    yield shell, core, emitted
    cli_mod._active_agent_ref = None


def _notices(emitted):
    return [line for line in emitted if "Steer queued:" in line]


def test_external_acceptance_is_rendered_by_cli_setup(wired_cli):
    shell, core, emitted = wired_cli
    outcomes = []
    worker = threading.Thread(target=lambda: outcomes.append(core.steer("  focus on errors  ")))
    worker.start()
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert outcomes == [True]
    assert core._pending_steer == "focus on errors"
    notices = _notices(emitted)
    assert len(notices) == 1, "accepted external steer must render through _init_agent wiring"
    assert "focus on errors" in notices[0]
    assert shell.conversation_history == [], "queued is not yet a model-consumed message"


def test_local_slash_steer_is_not_echoed_twice(wired_cli):
    shell, core, emitted = wired_cli
    shell.process_command("/steer local instruction")
    assert core._pending_steer == "local instruction"
    assert len(_notices(emitted)) == 1
    assert sum("local instruction" in line for line in emitted) == 1
    assert shell._pending_input.empty()


def test_busy_enter_uses_the_same_notice(wired_cli, monkeypatch):
    shell, core, emitted = wired_cli
    import agent.onboarding
    monkeypatch.setattr(agent.onboarding, "is_seen", lambda *_a: True)
    shell.busy_input_mode = "steer"
    shell._tui_enter_while_busy("busy instruction", [], "busy instruction")
    assert core._pending_steer == "busy instruction"
    assert len(_notices(emitted)) == 1
    assert sum("busy instruction" in line for line in emitted) == 1
    assert shell._pending_input.empty()


def test_empty_and_internal_requeue_have_no_new_notice(wired_cli):
    _shell, core, emitted = wired_cli
    assert core.steer("  ") is False
    assert core.steer("recovered instruction", _notify=False) is True
    assert core._pending_steer == "recovered instruction"
    assert emitted == []


def test_soft_clear_preserves_accepted_steer_but_hard_cancel_drops_it(wired_cli):
    _shell, core, emitted = wired_cli
    assert core.steer("keep this across recovery") is True
    core.clear_interrupt()
    assert core._pending_steer == "keep this across recovery"
    assert len(_notices(emitted)) == 1
    core.clear_interrupt(hard_cancel=True)
    assert core._pending_steer is None
    assert len(_notices(emitted)) == 1


def test_external_notice_waits_until_stream_box_closes(wired_cli):
    shell, core, emitted = wired_cli
    shell._stream_delta("First paragraph.\n")
    assert core.steer("external guidance") is True
    assert not _notices(emitted)
    shell._stream_delta("Second paragraph.\n")
    shell._flush_stream()
    notices = _notices(emitted)
    assert len(notices) == 1
    footer = next(index for index, line in enumerate(emitted) if line.startswith("╰"))
    second = next(index for index, line in enumerate(emitted) if "Second paragraph" in line)
    assert second < footer < emitted.index(notices[0])
