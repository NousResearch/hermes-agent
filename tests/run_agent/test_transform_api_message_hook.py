"""Tests for the per-request transform_api_message plugin hook."""

from run_agent import AIAgent


def _agent() -> AIAgent:
    agent = object.__new__(AIAgent)
    agent.session_id = "session-1"
    agent.model = "gpt-test"
    agent.platform = "cli"
    return agent


def test_transform_api_message_mutates_api_copy_only(monkeypatch):
    calls = []

    def fake_invoke_hook(hook_name, **kwargs):
        calls.append((hook_name, kwargs))
        kwargs["api_msg"]["content"] = "[redacted for api]"
        kwargs["api_msg"]["ephemeral_only"] = True
        return ["ignored"]

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", fake_invoke_hook)

    msg = {"role": "tool", "content": "full local result"}
    api_msg = msg.copy()

    _agent()._apply_transform_api_message_hooks(msg, api_msg, 3)

    assert msg == {"role": "tool", "content": "full local result"}
    assert api_msg == {
        "role": "tool",
        "content": "[redacted for api]",
        "ephemeral_only": True,
    }
    assert calls[0][0] == "transform_api_message"
    assert calls[0][1]["msg"] is msg
    assert calls[0][1]["api_msg"] is api_msg
    assert calls[0][1]["idx"] == 3
    assert calls[0][1]["session_id"] == "session-1"
    assert calls[0][1]["model"] == "gpt-test"
    assert calls[0][1]["platform"] == "cli"


def test_transform_api_message_hook_errors_leave_api_message_unchanged(monkeypatch, caplog):
    def fake_invoke_hook(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", fake_invoke_hook)
    api_msg = {"role": "user", "content": "hello"}

    _agent()._apply_transform_api_message_hooks({"role": "user", "content": "hello"}, api_msg, 0)

    assert api_msg == {"role": "user", "content": "hello"}
    assert "transform_api_message hook failed" in caplog.text
