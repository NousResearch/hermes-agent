"""A down managed local model server is reported as itself, never as a credential problem.

With the main model on the managed llama.cpp runtime and no server serving, the readiness probe
and a turn's resolution failure must say "local server down" in a structured way, so clients keep
the app open with that error instead of opening provider sign-in or offering "Update API key".
"""
from __future__ import annotations

import json

import pytest

from hermes_constants import get_hermes_home


@pytest.fixture
def llamacpp_home(monkeypatch):
    from hermes_cli.local_runtime import detect

    def write(local_runtime: dict) -> None:
        home = get_hermes_home()
        home.mkdir(parents=True, exist_ok=True)
        config = {"model": {"provider": "llamacpp", "default": "local-qwen"}, "local_runtime": local_runtime}
        (home / "config.yaml").write_text(json.dumps(config), encoding="utf-8")

    # No external llama-server on this machine may answer for the managed one.
    monkeypatch.setattr(detect, "detect_server", lambda extra_ports=(): None)
    return write


def _resolution_error():
    from hermes_cli.runtime_provider import resolve_runtime_provider

    with pytest.raises(ValueError) as caught:
        resolve_runtime_provider(requested="llamacpp", target_model="local-qwen")
    return caught.value


@pytest.mark.parametrize("enabled", [False, True])
def test_runtime_check_names_a_down_local_server(llamacpp_home, enabled):
    from hermes_cli.runtime_readiness import readiness_failure

    llamacpp_home({"enabled": enabled})
    answer = readiness_failure(_resolution_error())

    assert answer["ok"] is False
    assert answer["error_code"] == "local_runtime_unavailable"
    assert "local model server" in answer["error"]


def test_runtime_check_leaves_other_failures_uncoded():
    from hermes_cli.runtime_readiness import readiness_failure

    assert readiness_failure(ValueError("No usable credentials found for openrouter.")) == {
        "ok": False, "error": "No usable credentials found for openrouter."}


@pytest.mark.parametrize("enabled", [False, True])
def test_turn_failure_is_an_endpoint_failure_not_a_rejected_key(llamacpp_home, enabled):
    from agent.error_surface import build_error_surface_from_result
    from gateway.config import Platform
    from gateway.run_turn_runner import _resolution_failure_reply

    llamacpp_home({"enabled": enabled})
    exc = _resolution_error()
    text, reason, retryable = _resolution_failure_reply(exc, Platform.TELEGRAM)

    # Switched on but not serving yet is worth a retry; switched off is not.
    assert (reason, retryable) == ("local_runtime_unavailable", enabled)
    assert str(exc) in text
    surface = build_error_surface_from_result(
        {"failed": True, "error": str(exc), "failure_reason": reason, "failure_retryable": retryable},
        provider="llamacpp")
    assert surface["layer"] == "endpoint"
    assert surface["code"] == "local_runtime_unavailable"
    assert "auth_kind" not in surface


@pytest.mark.parametrize("enabled", [False, True])
def test_chat_text_names_the_local_server_and_no_desktop_only_pane(llamacpp_home, enabled):
    """The messaging reply and the TUI/dashboard turn text both name the local model server, and
    neither points at Settings → Providers → Local models, a pane only the Desktop app has."""
    from agent.error_surface import build_error_surface_from_result
    from gateway.config import Platform
    from gateway.run_turn_runner import _resolution_failure_reply
    from tui_gateway.user_messages import turn_error_text

    llamacpp_home({"enabled": enabled})
    exc = _resolution_error()
    reply, reason, retryable = _resolution_failure_reply(exc, Platform.TELEGRAM)
    surface = build_error_surface_from_result(
        {"failed": True, "error": str(exc), "failure_reason": reason, "failure_retryable": retryable},
        provider="llamacpp")
    tui = turn_error_text(str(exc), surface, recoverable=retryable)

    title = tui.splitlines()[0]
    assert "local model server" in title
    assert "custom model endpoint" not in tui
    for text in (reply, tui):
        assert "local model server" in text
        assert "Settings" not in text
        assert "/model" in text


@pytest.mark.parametrize("enabled", [False, True])
def test_chat_text_suggests_retry_only_while_the_server_can_come_up(llamacpp_home, enabled):
    """A switched-off server never comes up on its own: the TUI/dashboard text (built from the
    error surface, as a turn builds it) names where to start it, never /retry."""
    from agent.error_surface import build_error_surface_from_result
    from tui_gateway.user_messages import turn_error_text

    llamacpp_home({"enabled": enabled})
    exc = _resolution_error()
    surface = build_error_surface_from_result(
        {"failed": True, "error": str(exc), "failure_reason": exc.error_code, "failure_retryable": exc.retryable},
        provider="llamacpp")
    hint = turn_error_text(str(exc), surface).splitlines()[-1]

    assert ("/retry" in hint) is enabled
    if not enabled:
        assert "hermes model" in hint and "/model" in hint
        # Surface-neutral: the backend sentence names the terminal path too, not only the Desktop app.
        assert "hermes model" in str(exc)
