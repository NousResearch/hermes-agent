"""on_human_input_request / on_human_input_resolved fire wherever the agent blocks for a human (#132333).

Each surface's real choke point is driven end to end — sudo password prompt, clarify, dangerous-command
approval on the CLI callback path and the gateway poll path — with only the plugin hook delivery captured.
Every request pairs with exactly one resolution carrying the same ``request_id``, and no typed secret (the
sudo password, a clarify answer) or unredacted credential ever appears in any payload.
"""

import threading
from unittest.mock import patch

import pytest

from tools import approval as approval_module
from tools import approval_context, approval_gateway_wait, terminal_tool, terminal_tool_sudo
from tools.approval_prompt import prompt_dangerous_approval
from tools.clarify_tool import clarify_tool

PASSWORD = "hunter2-do-not-leak"
TOKEN = "ghp_" + "Z" * 36
SECRET_COMMAND = f"sudo env GITHUB_TOKEN={TOKEN} apt-get install -y jq"
HOOKS = ("on_human_input_request", "on_human_input_resolved")


@pytest.fixture
def captured(monkeypatch):
    calls: list[tuple[str, dict]] = []

    def fake_invoke_hook(name, **kwargs):
        if name in HOOKS:
            calls.append((name, kwargs))
        return []

    monkeypatch.setattr(terminal_tool, "_callback_tls", threading.local())
    monkeypatch.setattr(terminal_tool_sudo, "_sudo_password_cache", {})
    with patch("hermes_cli.plugins.invoke_hook", side_effect=fake_invoke_hook):
        yield calls


def _single_pair(calls, kind):
    assert [name for name, _ in calls] == list(HOOKS), calls
    request, resolved = calls[0][1], calls[1][1]
    assert request["kind"] == resolved["kind"] == kind
    assert request["request_id"] and request["request_id"] == resolved["request_id"]
    assert "outcome" not in request
    for key in ("session_id", "session_key", "platform", "prompt"):
        assert key in request and request[key] == resolved[key]
    return request, resolved


def _assert_no_secret(calls, *secrets):
    blob = repr(calls)
    for secret in secrets:
        assert secret not in blob


def test_hooks_are_registered():
    from hermes_cli.plugins import VALID_HOOKS

    assert set(HOOKS) <= VALID_HOOKS


class TestSudo:
    def test_callback_prompt_fires_pair_without_password(self, captured):
        terminal_tool.set_sudo_password_callback(lambda: PASSWORD)

        assert terminal_tool_sudo._prompt_for_sudo_password(command=SECRET_COMMAND) == PASSWORD

        request, resolved = _single_pair(captured, "sudo")
        assert resolved["outcome"] == "provided"
        assert "apt-get install" in request["prompt"]
        _assert_no_secret(captured, PASSWORD, TOKEN)

    def test_empty_answer_is_skipped(self, captured):
        terminal_tool.set_sudo_password_callback(lambda: "")

        assert terminal_tool_sudo._prompt_for_sudo_password(command="sudo true") == ""

        assert _single_pair(captured, "sudo")[1]["outcome"] == "skipped"

    def test_tty_fallback_timeout_resolves_as_timeout(self, captured, monkeypatch):
        monkeypatch.setattr(terminal_tool_sudo, "_read_hidden_password", lambda result: None)

        assert terminal_tool_sudo._prompt_for_sudo_password(timeout_seconds=0, command="sudo true") == ""

        assert _single_pair(captured, "sudo")[1]["outcome"] == "timeout"


class TestClarify:
    QUESTIONS = [{"question": "Which database?", "choices": ["postgres", "sqlite"]}]

    @pytest.mark.parametrize("outcome", ["submitted", "timed_out", "cancelled"])
    def test_callback_outcome_is_reported_without_answers(self, captured, outcome):
        def callback(normalized):
            return {"answers": {normalized[0]["qid"]: PASSWORD}, "outcome": outcome}

        clarify_tool(self.QUESTIONS, callback=callback)

        request, resolved = _single_pair(captured, "clarify")
        assert resolved["outcome"] == outcome
        assert "Which database?" in request["prompt"]
        _assert_no_secret(captured, PASSWORD)

    def test_callback_failure_resolves_as_error(self, captured):
        def callback(normalized):
            raise RuntimeError("ui gone")

        clarify_tool(self.QUESTIONS, callback=callback)

        assert _single_pair(captured, "clarify")[1]["outcome"] == "error"


class TestApproval:
    def test_cli_callback_prompt_redacts_command(self, captured):
        choice = prompt_dangerous_approval(SECRET_COMMAND, "privileged install",
                                           approval_callback=lambda *a, **k: "once")

        assert choice == "once"
        request, resolved = _single_pair(captured, "approval")
        assert resolved["outcome"] == "once"
        assert "apt-get install" in request["prompt"]
        _assert_no_secret(captured, TOKEN)

    def test_gateway_decision_fires_pair(self, captured, monkeypatch):
        session_key = "human-input-gateway"
        monkeypatch.setattr(approval_context, "_get_approval_timeout", lambda: 5)
        approval_module._gateway_queues.pop(session_key, None)

        def notify(data):
            threading.Timer(0.05, approval_module.resolve_gateway_approval, (session_key, "session")).start()

        approval_data = {"command": SECRET_COMMAND, "description": "privileged install",
                         "pattern_key": "sudo", "pattern_keys": ["sudo"]}
        decision = approval_gateway_wait._await_gateway_decision(session_key, notify, approval_data)

        assert decision["choice"] == "session"
        request, resolved = _single_pair(captured, "approval")
        assert request["session_key"] == session_key
        assert resolved["outcome"] == "session"
        _assert_no_secret(captured, TOKEN)

    def test_gateway_notify_failure_still_resolves(self, captured):
        def notify(data):
            raise ConnectionError("adapter offline")

        approval_data = {"command": "rm -rf build", "description": "d", "pattern_key": "rm", "pattern_keys": ["rm"]}
        decision = approval_gateway_wait._await_gateway_decision("human-input-notify-fail", notify, approval_data)

        assert decision.get("notify_failed") is True
        assert _single_pair(captured, "approval")[1]["outcome"] == "notify_failed"

    def test_plugin_transport_fires_pair(self, captured, monkeypatch):
        from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
        from tools import approval_prompt

        manager = PluginManager()
        manifest = PluginManifest(name="phone-fixture", version="1.0.0", description="", source="user", key="phone")
        PluginContext(manifest, manager).register_approval_transport("phone", lambda req: req.respond("deny"))
        monkeypatch.setattr(approval_prompt, "get_plugin_manager", lambda: manager)
        monkeypatch.setattr(approval_context, "_get_approval_transport_config", lambda: ("phone", None))

        attempt = approval_prompt._present_with_selected_transport(
            command=SECRET_COMMAND, description="d", pattern_key="sudo", pattern_keys=["sudo"],
            session_key="human-input-transport", surface="cli", allow_session=True, allow_permanent=True)

        assert attempt["choice"] == "deny"
        request, resolved = _single_pair(captured, "approval")
        assert request["session_key"] == "human-input-transport"
        assert resolved["outcome"] == "deny"
        _assert_no_secret(captured, TOKEN)
