"""Acceptance notices must use the real CLI wiring, not a manually attached observer."""
from __future__ import annotations

import queue
import re
import threading
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def wired_cli(monkeypatch):
    import cli as cli_mod
    import run_agent
    from cli import HermesCLI

    # Keep the production steer/redirect methods. Only agent construction and
    # unrelated startup I/O are substituted; _init_agent owns callback wiring.
    agent = run_agent.AIAgent.__new__(run_agent.AIAgent)
    agent._pending_steer = None
    agent._pending_steer_lock = threading.Lock()
    agent._pending_redirect = None
    agent._pending_redirect_lock = threading.Lock()
    agent._interrupt_requested = False
    agent._execution_thread_id = None

    cli = HermesCLI.__new__(HermesCLI)
    settings = {
        "agent": None, "_session_db": MagicMock(), "_resumed": False,
        "max_turns": 3, "enabled_toolsets": [], "disabled_toolsets": [],
        "verbose": False, "system_prompt": None, "prefill_messages": [],
        "reasoning_config": None, "service_tier": None,
        "_providers_only": None, "_providers_ignore": None,
        "_providers_order": None, "_provider_sort": None,
        "_provider_require_params": None, "_provider_data_collection": None,
        "_openrouter_min_coding_score": None, "_fallback_model": None,
        "session_id": "steer-wiring-test", "checkpoints_enabled": False,
        "checkpoint_max_snapshots": 1, "checkpoint_max_total_size_mb": 1,
        "checkpoint_max_file_size_mb": 1, "pass_session_id": False,
        "ignore_rules": True, "_inline_diffs_enabled": False,
        "streaming_enabled": False, "_pending_title": None,
        "show_reasoning": False, "final_response_markdown": "raw",
        "show_timestamps": False, "_agent_running": True,
        "busy_input_mode": "steer", "_pending_input": queue.Queue(),
        "_interrupt_queue": queue.Queue(),
    }
    for name, value in settings.items():
        setattr(cli, name, value)
    for name in ("finalize_preloaded_skills", "_install_tool_callbacks", "_ensure_tirith_security"):
        monkeypatch.setattr(cli, name, lambda: None)
    monkeypatch.setattr(cli, "_ensure_runtime_credentials", lambda: True)
    monkeypatch.setattr(cli, "_current_reasoning_callback", lambda: None)
    monkeypatch.setattr(cli_mod, "_prepare_deferred_agent_startup", lambda: None)
    monkeypatch.setattr("hermes_cli.mcp_startup.ensure_mcp_discovery_before_agent_build", lambda **_: None)
    monkeypatch.setattr("agent.credits_tracker.seed_credits_at_session_start", lambda _: None)
    monkeypatch.setattr(run_agent, "AIAgent", lambda **_: agent)
    # _init_agent updates this global; undo it when the fixture leaves.
    monkeypatch.setattr(cli_mod, "_active_agent_ref", None)

    emitted = []
    monkeypatch.setattr(cli_mod, "_cprint", lambda text, **_: emitted.append(text))
    monkeypatch.setattr(cli_mod, "_terminal_width_for_streaming", lambda: 74)
    monkeypatch.setattr(HermesCLI, "_scrollback_box_width", lambda self: 74)
    cli._reset_stream_state()
    assert cli._init_agent(
        model_override="test/model",
        runtime_override={"api_key": "test-key", "base_url": "https://example.invalid/v1", "provider": "custom"},
    )
    assert cli.agent is agent
    return cli, emitted


def _notices(emitted, text):
    return [line for line in emitted if text in line]


def test_external_steer_uses_initialized_cli_observer(wired_cli):
    cli, emitted = wired_cli
    assert cli.agent.steer("  external guidance  ") is True
    assert len(_notices(emitted, "external guidance")) == 1
    # Acceptance is not consumption. The full text still awaits the agent loop.
    assert cli.agent._pending_steer == "external guidance"
    assert cli.agent._drain_pending_steer() == "external guidance"
    assert cli.agent._drain_pending_steer() is None


def test_local_slash_steer_has_one_notice(wired_cli):
    cli, emitted = wired_cli
    cli._cmd_steer("/steer local guidance")
    assert len(_notices(emitted, "local guidance")) == 1
    assert cli.agent._pending_steer == "local guidance"
    assert cli._pending_input.empty()


def test_busy_enter_and_external_steer_have_identical_notices(wired_cli):
    cli, emitted = wired_cli
    cli._tui_enter_while_busy("same guidance", [], "same guidance")
    local = _notices(emitted, "same guidance")
    assert len(local) == 1
    emitted.clear()
    cli.agent.steer("same guidance")
    assert _notices(emitted, "same guidance") == local
    assert cli.agent._pending_steer == "same guidance\nsame guidance"
    assert cli._pending_input.empty()


def test_rejected_and_internal_requeues_do_not_render_acceptance(wired_cli):
    cli, emitted = wired_cli
    assert cli.agent.steer("   ") is False
    assert emitted == []
    assert cli.agent.steer("internal recovery", _notify=False) is True
    assert emitted == []
    assert cli.agent._pending_steer == "internal recovery"


def test_redirect_tool_fallback_does_not_render_fresh_steer(wired_cli):
    cli, emitted = wired_cli
    cli.agent._executing_tools = True
    assert cli.agent.redirect("redirected guidance") is True
    assert emitted == []
    assert cli.agent._pending_steer == "redirected guidance"


def test_external_thread_notice_waits_for_response_footer(wired_cli):
    cli, emitted = wired_cli
    cli._stream_delta("First paragraph.\n")
    results = []
    worker = threading.Thread(target=lambda: results.append(cli.agent.steer("thread guidance")), daemon=True)
    worker.start()
    worker.join(timeout=5)
    assert not worker.is_alive()
    assert results == [True]
    assert _notices(emitted, "thread guidance") == []
    cli._stream_delta("Second paragraph.\n")
    cli._flush_stream()
    lines = [re.sub(r"\x1b\[[0-9;]*m", "", line) for line in emitted]
    second = next(i for i, line in enumerate(lines) if "Second paragraph" in line)
    footer = next(i for i, line in enumerate(lines) if line.startswith("╰"))
    notice = next(i for i, line in enumerate(lines) if "thread guidance" in line)
    assert second < footer < notice
    assert len(_notices(lines, "thread guidance")) == 1


def test_render_failure_does_not_lose_accepted_guidance(wired_cli, monkeypatch):
    cli, _ = wired_cli
    def failed_output(*args, **kwargs):
        raise RuntimeError("test renderer failure")
    monkeypatch.setattr(cli, "_agent_status_print", failed_output)
    assert cli.agent.steer("keep this guidance") is True
    assert cli.agent._pending_steer == "keep this guidance"
