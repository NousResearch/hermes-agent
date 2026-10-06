"""Diagnostic presentation does not rewrite execution evidence or next-turn callbacks."""
import copy
from unittest.mock import MagicMock, patch

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter, _ProviderAuthResolutionError


@pytest.mark.asyncio
@pytest.mark.parametrize("setting", [None, False, True])
@pytest.mark.parametrize("category", ["diagnostic", "result"])
async def test_notification_projection_preserves_source_outcome(tmp_path, monkeypatch, setting, category):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("{}" if setting is None else
        f"display: {{suppress_warning_notifications: {str(setting).lower()}}}\n")
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    source = {"final_response": "execution evidence", "completed": False, "failed": True,
              "error": "failure details", "messages": [{"role": "assistant", "content": "execution evidence"}]}
    expected = copy.deepcopy(source)
    agent = MagicMock()
    agent.session_id = "session"
    agent._last_compaction_in_place = False
    agent.session_prompt_tokens = 3
    agent.session_completion_tokens = 4
    agent.session_total_tokens = 7
    delivered = []
    callback = lambda text: delivered.append(text)
    agent.stream_delta_callback = callback

    def run(**kwargs):
        if agent.stream_delta_callback:
            agent.stream_delta_callback("execution evidence")
        return source

    agent.run_conversation.side_effect = run
    finished = []
    finish = adapter._finish_turn_result

    def record_finish(agent, result, *args, **kwargs):
        finished.append(copy.deepcopy(result))
        return finish(agent, result, *args, **kwargs)

    with patch.object(adapter, "_create_agent", return_value=agent), patch.object(adapter, "_finish_turn_result", side_effect=record_finish):
        result, usage = await adapter._run_agent("internal event", [], session_id="session", notification_category=category)
    muted = setting is True and category == "diagnostic"
    assert finished == [expected]
    assert source["final_response"] == expected["final_response"]
    assert result["final_response"] == "execution evidence"
    assert bool(result.get("_notification_presentation_suppressed")) is muted
    assert result["failed"] is True and result["error"] == "failure details"
    assert result["messages"] == expected["messages"]
    assert usage == {"input_tokens": 3, "output_tokens": 4, "total_tokens": 7}
    assert delivered == ([] if muted else ["execution evidence"])
    assert agent.stream_delta_callback is callback
    assert adapter._inflight_agent_runs == 0
    assert adapter._shutdown_interruptible_agents == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("setting", [None, False, True])
async def test_pre_agent_auth_diagnostic_obeys_policy_without_losing_logs(tmp_path, monkeypatch, caplog, setting):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("{}" if setting is None else
        f"display: {{suppress_warning_notifications: {str(setting).lower()}}}\n")
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    with patch.object(adapter, "_create_agent", side_effect=_ProviderAuthResolutionError("fixture unavailable")):
        result, usage = await adapter._run_agent("internal event", [], session_id="session", notification_category="diagnostic")
    assert "fixture unavailable" in result["final_response"]
    assert bool(result.get("_notification_presentation_suppressed")) is (setting is True)
    assert "fixture unavailable" in caplog.text
    assert result["api_calls"] == 0 and usage["total_tokens"] == 0
    assert adapter._inflight_agent_runs == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("setting", [None, False, True])
async def test_http_diagnostic_projection_keeps_source_and_terminal_flags(tmp_path, monkeypatch, stream, setting):
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer
    import json

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("{}" if setting is None else
        f"display: {{suppress_warning_notifications: {str(setting).lower()}}}\n")
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "test-local-key"}))
    app = web.Application()
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    source = {"final_response": "private diagnostic evidence", "completed": False, "partial": True,
              "failed": False, "error": "private diagnostic detail", "messages": []}
    agent = MagicMock()
    agent.session_id = "session"
    agent._last_compaction_in_place = False
    agent.session_prompt_tokens = 3
    agent.session_completion_tokens = 4
    agent.session_total_tokens = 7

    def create(**kwargs):
        agent.stream_delta_callback = kwargs.get("stream_delta_callback")
        return agent

    def run(**kwargs):
        if agent.stream_delta_callback:
            agent.stream_delta_callback("private diagnostic evidence")
        return source

    agent.run_conversation.side_effect = run
    with patch.object(adapter, "_create_agent", side_effect=create):
        async with TestClient(TestServer(app)) as client:
            response = await client.post("/v1/chat/completions", headers={
                "Authorization": "Bearer test-local-key", "X-Hermes-Session-Id": "session"}, json={
                "messages": [{"role": "user", "content": "background diagnostic"}],
                "stream": stream, "hermes_notification_category": "diagnostic"})
            wire = await response.text()
    assert response.status == 200, wire
    assert source["final_response"] == "private diagnostic evidence"
    assert source["error"] == "private diagnostic detail"
    if setting is True:
        assert "private diagnostic" not in wire
    else:
        assert "private diagnostic evidence" in wire
    if stream:
        frames = [json.loads(line[6:]) for line in wire.splitlines()
                  if line.startswith("data: ") and line != "data: [DONE]"]
        terminal = frames[-1]
        assert terminal["choices"][0]["finish_reason"] == "error"
        assert terminal["hermes"]["partial"] is True
        assert terminal["usage"]["total_tokens"] == 7
    else:
        body = json.loads(wire)
        assert body["choices"][0]["finish_reason"] == "error"
        assert body["hermes"]["partial"] is True
        assert body["usage"]["total_tokens"] == 7


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("setting", [None, False, True])
async def test_http_unhandled_diagnostic_error_is_quiet_but_failed(tmp_path, monkeypatch, caplog, stream, setting):
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer
    import json

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("{}" if setting is None else
        f"display: {{suppress_warning_notifications: {str(setting).lower()}}}\n")
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "test-local-key"}))
    app = web.Application()
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    with patch.object(adapter, "_create_agent", side_effect=RuntimeError("private constructor failure")):
        async with TestClient(TestServer(app)) as client:
            response = await client.post("/v1/chat/completions", headers={
                "Authorization": "Bearer test-local-key", "X-Hermes-Session-Id": "session"}, json={
                "messages": [{"role": "user", "content": "background diagnostic"}],
                "stream": stream, "hermes_notification_category": "diagnostic"})
            wire = await response.text()
    assert response.status == (200 if stream else 500), wire
    assert "private constructor failure" in caplog.text
    assert ("private constructor failure" in wire) is (setting is not True)
    if stream:
        frames = [json.loads(line[6:]) for line in wire.splitlines()
                  if line.startswith("data: ") and line != "data: [DONE]"]
        assert frames[-1]["choices"][0]["finish_reason"] == "error"
        assert frames[-1]["hermes"]["failed"] is True
    assert adapter._inflight_agent_runs == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("categories", [("diagnostic", "result"), ("result", "diagnostic")])
async def test_http_idempotency_does_not_replay_opposite_presentation(tmp_path, monkeypatch, categories):
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer
    from gateway.platforms.api_server import _IdempotencyCache

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("gateway.platforms.api_server._idem_cache", _IdempotencyCache())
    (tmp_path / "config.yaml").write_text("display: {suppress_warning_notifications: true}\n")
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "test-local-key"}))
    app = web.Application()
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    agent = MagicMock()
    agent.session_id = "session"
    agent._last_compaction_in_place = False
    agent.session_prompt_tokens = agent.session_completion_tokens = agent.session_total_tokens = 0
    agent.run_conversation.side_effect = lambda **kwargs: {"final_response": "required result", "messages": []}
    with patch.object(adapter, "_create_agent", return_value=agent):
        async with TestClient(TestServer(app)) as client:
            for category in categories:
                body = {"messages": [{"role": "user", "content": "event"}],
                        "hermes_notification_category": category}
                for repeat in range(2):
                    response = await client.post("/v1/chat/completions", headers={
                        "Authorization": "Bearer test-local-key", "X-Hermes-Session-Id": "session",
                        "Idempotency-Key": "same-key"}, json=body)
                    wire = await response.json()
                    assert response.status == 200, wire
                    assert wire["choices"][0]["message"]["content"] == ("" if category == "diagnostic" else "required result")
    assert agent.run_conversation.call_count == 2
