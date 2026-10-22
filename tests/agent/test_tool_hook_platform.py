"""Tool-path hooks carry ``platform`` — the same value ``pre_llm_call`` already gets.

Contract under test: a plugin that gates tool calls can tell an attended turn from an
unattended one (``cron`` / ``subagent`` / ``curator``) without guessing from session ids.
Covers all three tool hooks (pre/post/transform) plus the shell-hook ``extra`` dict.
"""

from types import SimpleNamespace

from agent.inline_tool_executors import tool_hook_ids
from model_tools import _CallIds, _apply_transform_tool_result_hook, _emit_post_tool_call_hook


def _agent(platform="cron"):
    return SimpleNamespace(
        platform=platform,
        session_id="sess-1",
        _current_turn_id="turn-1",
        _current_api_request_id="api-1",
    )


def test_tool_hook_ids_carries_agent_platform():
    ids = tool_hook_ids(_agent("cron"), "task-1", "call-1")
    assert ids["platform"] == "cron"
    # Unknown platform (older agent object, programmatic caller) coerces to "" like the other ids.
    assert tool_hook_ids(SimpleNamespace(), "t", "c")["platform"] == ""


def test_call_ids_round_trips_platform_into_hook_kwargs():
    ids = _CallIds(**tool_hook_ids(_agent("subagent"), "task-1", "call-1"))
    assert ids.hook_kwargs()["platform"] == "subagent"


def test_pre_tool_call_hook_receives_platform(monkeypatch):
    from hermes_cli import lifecycle
    from hermes_cli.plugins import _dispatch_pre_tool_call_hooks

    seen = {}

    def fake_invoke_hook(hook_name, **kwargs):
        seen.update(kwargs)
        return []

    monkeypatch.setattr(lifecycle, "invoke_hook", fake_invoke_hook)
    _dispatch_pre_tool_call_hooks("terminal", {"command": "echo hi"}, task_id="t", platform="cron")
    assert seen["platform"] == "cron"


def test_post_tool_call_hook_receives_platform(monkeypatch):
    from hermes_cli import lifecycle

    seen = {}

    def fake_invoke_hook(hook_name, **kwargs):
        seen.update(kwargs)
        return []

    monkeypatch.setattr(lifecycle, "has_hook", lambda _name: True)
    monkeypatch.setattr(lifecycle, "invoke_hook", fake_invoke_hook)
    _emit_post_tool_call_hook(function_name="terminal", function_args={}, result="ok",
                              platform="curator")
    assert seen["platform"] == "curator"


def test_transform_tool_result_hook_receives_platform(monkeypatch):
    from hermes_cli import lifecycle

    seen = {}

    def fake_invoke_hook(hook_name, **kwargs):
        seen.update(kwargs)
        return ["rewritten"]

    monkeypatch.setattr(lifecycle, "has_hook", lambda _name: True)
    monkeypatch.setattr(lifecycle, "invoke_hook", fake_invoke_hook)
    out = _apply_transform_tool_result_hook(
        "terminal", {}, "raw", 1, _CallIds(**tool_hook_ids(_agent("cron"), "t", "c")))
    assert out == "rewritten"
    assert seen["platform"] == "cron"


def test_shell_hook_payload_carries_tool_hook_identity_in_extra():
    """Shell hooks: identity fields the payload does not promote land in ``extra``."""
    from agent.shell_hooks import _payload_fields

    payload = _payload_fields({"tool_name": "terminal", "args": {},
                               **tool_hook_ids(_agent("cron"), "task-1", "call-1")})
    assert "platform" not in payload
    assert payload["extra"]["platform"] == "cron"
