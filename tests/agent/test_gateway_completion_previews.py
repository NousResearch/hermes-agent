"""Gateway completion previews are opt-in, bounded and redacted on every path."""

import logging
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.log_previews import PREVIEW_CHARS, completion_preview, gateway_previews_enabled
from agent.redact import REDACTION_UNAVAILABLE
from agent.tool_executor import _ToolCallRef, _commit_tool_result, _ConcurrentBatch
from gateway.run_turn import GatewayTurnMixin
from tests.gateway.test_display_null_turn_wiring import _wire


def test_disabled_by_default_and_for_malformed_configuration():
    for config in ({}, {"logging": None}, {"logging": "true"}, {"logging": {"tool_previews": "true"}}):
        assert not gateway_previews_enabled(config)
        assert not _wire(config)._gateway_completion_previews
    assert _wire({"logging": {"tool_previews": True}})._gateway_completion_previews


def test_preview_redacts_before_truncation_and_escapes_newlines():
    secret = "ghp_0123456789abcdefghijklmnopqrst"
    text = "a" * (PREVIEW_CHARS - 8) + " " + secret + "\n" + "b" * 80
    result = completion_preview(text)
    assert len(result) <= PREVIEW_CHARS + 8
    assert secret not in result and "0123456789" not in result
    assert "\\n" not in result  # newline falls beyond the bound
    assert "\n" not in completion_preview("a\nb")
    assert "\\n" in completion_preview("a\nb")


@pytest.mark.parametrize("enabled", [False, True])
def test_sequential_success_log_preview_gate(caplog, enabled):
    # Stop after logging, before the unrelated persistence machinery.
    class StopAfterLog(Exception):
        pass

    args = {"command": "echo test", "token": "ghp_0123456789abcdefghijklmnopqrst"}
    agent = SimpleNamespace(
        _gateway_completion_previews=enabled,
        _append_guardrail_observation=lambda name, a, result, **kw: result,
        _record_file_mutation_result=lambda *a, **kw: None,
        verbose_logging=False,
        _current_tool=None,
        _touch_activity=lambda *a: (_ for _ in ()).throw(StopAfterLog()),
    )
    ref = _ToolCallRef("terminal", args, "id", "task", None)
    with caplog.at_level(logging.INFO, logger="agent.tool_executor"), pytest.raises(StopAfterLog):
        _commit_tool_result(agent, [], ref, "hello world", budget=None, tool_duration=0.1,
                            is_error=False, blocked=False, effect_disposition=None,
                            observed=True, success_log_chars=11)
    assert ("args=" in caplog.text) is enabled
    assert ("output_preview=" in caplog.text) is enabled
    assert "ghp_012345" not in caplog.text


@pytest.mark.parametrize("enabled", [False, True])
def test_concurrent_worker_log_preview_gate(caplog, enabled):
    agent = SimpleNamespace(_gateway_completion_previews=enabled)
    ref = SimpleNamespace(name="terminal", args={"command": "ls"},
                          middleware_kwargs=lambda: {})
    batch = object.__new__(_ConcurrentBatch)
    batch.agent = agent
    batch.authorization_gate = None
    managed = SimpleNamespace(result="listed", args=ref.args, middleware_trace=None,
                              blocked=False, dispatched=True)
    with patch("agent.tool_executor._run_agent_tool_execution_middleware", return_value=managed), \
         patch("agent.tool_executor._detect_tool_failure", return_value=(False, None)), \
         caplog.at_level(logging.INFO, logger="agent.tool_executor"):
        batch._dispatch_worker(0, ref, None, SimpleNamespace(advance=lambda: None))
    assert ("args=" in caplog.text) is enabled
    assert ("output_preview=" in caplog.text) is enabled


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
async def test_final_response_preview_gate(caplog, enabled):
    class Runner(GatewayTurnMixin):
        async def _clear_restart_failure_count(self, *args):
            pass
    runner = Runner()
    runner.async_session_store = SimpleNamespace(clear_resume_pending=lambda *a: None)
    source = SimpleNamespace(chat_id="chat", platform=SimpleNamespace(value="telegram"))
    with patch("gateway.run._load_gateway_config", return_value={"logging": {"tool_previews": enabled}}), \
         caplog.at_level(logging.INFO, logger="gateway.run"):
        await runner._hmwa_shape_agent_response(
            {"final_response": "hello world", "messages": [], "api_calls": 1},
            source, [], SimpleNamespace(session_id="s"), None, None, 0, "s", "telegram", 0.0)
    assert ("response_preview=" in caplog.text) is enabled


def test_preview_fail_closed_if_redaction_raises():
    with patch("agent.log_previews.redact_for_egress", side_effect=ValueError("unsafe")):
        assert REDACTION_UNAVAILABLE in completion_preview("private material")
        assert "private material" not in completion_preview("private material")


def test_cached_agent_rewire_clears_previews():
    config = {"logging": {"tool_previews": True}}
    agent = _wire(config)
    assert agent._gateway_completion_previews
    config["logging"]["tool_previews"] = False
    # The same object is reused on the next gateway turn.
    assert not _wire(config)._gateway_completion_previews
