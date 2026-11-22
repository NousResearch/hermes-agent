"""Config -> gateway turn -> real AIAgent -> SDK-serialized regional Responses request.

Only HTTP is replaced. This is composed offline proof, not an OpenAI entitlement test.
"""

import json

import httpx
import pytest

from gateway.config import Platform, StreamingConfig
from gateway.session import SessionSource
from hermes_constants import get_hermes_home
from tests.gateway.test_run_progress_topics import ProgressCaptureAdapter, _make_runner


@pytest.mark.asyncio
@pytest.mark.parametrize("host", ["api.openai.com", "us.api.openai.com", "eu.api.openai.com"])
@pytest.mark.parametrize("tier,window,with_history,expected", [
    ("fast", 60, False, "priority"),
    ("priority", 60, False, "priority"),
    ("ultrafast", 60, False, "ultrafast"),
    ("auto", 60, True, "priority"),
    ("auto", 0, False, None),
    ("cold", 60, False, "priority"),
    ("cold", 60, True, None),
    ("normal", 60, False, None),
])
async def test_configured_tier_reaches_regional_sdk_request(
    monkeypatch, tmp_path, record_property, host, tier, window, with_history, expected,
):
    import gateway.run as gateway_run

    monkeypatch.chdir(tmp_path)
    home = get_hermes_home()
    (home / ".env").write_text("OPENAI_API_KEY=test-key\n")
    (home / "config.yaml").write_text(json.dumps({
        "model": {"default": "gpt-6.1-sol", "provider": "openai-api",
                  "base_url": f"https://{host}/v1"},
        "agent": {"service_tier": tier, "fast_auto_seconds": window, "max_turns": 2},
        "display": {"tool_progress": "off"},
        "compression": {"enabled": False},
        "memory": {"memory_enabled": False, "user_profile_enabled": False},
    }))
    requests = []

    def respond(_transport, request):
        assert request.url.host == host, str(request.url)
        assert request.url.path == "/v1/responses", str(request.url)
        body = json.loads(request.content)
        requests.append(body)
        response = {"id": "resp_test", "object": "response", "status": "completed",
                    "model": "gpt-6.1-sol", "output": [{"id": "msg_test", "type": "message",
                    "role": "assistant", "status": "completed",
                    "content": [{"type": "output_text", "text": "Regional request recorded.", "annotations": []}]}],
                    "usage": {"input_tokens": 10, "output_tokens": 4, "total_tokens": 14}}
        if body.get("stream"):
            events = [
                {"type": "response.output_text.delta", "delta": "Regional request recorded."},
                {"type": "response.completed", "response": response},
            ]
            return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                  content="".join(f"data: {json.dumps(event)}\n\n" for event in events).encode())
        return httpx.Response(200, json=response)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", respond)
    monkeypatch.setattr(gateway_run, "_hermes_home", home)
    runner = _make_runner(ProgressCaptureAdapter())
    runner.config.streaming = StreamingConfig(enabled=False)
    history = [{"role": "user", "content": "Earlier question."},
               {"role": "assistant", "content": "Earlier answer."}] if with_history else []
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="tier-test", chat_type="dm", user_id="test")
    result = await runner._run_agent(message="Record this request.", context_prompt="", history=history,
                                     source=source, session_id="tier-test", session_key="agent:main:telegram:dm:tier-test")
    assert result["final_response"] == "Regional request recorded.", result
    assert len(requests) == 1, requests
    assert requests[0]["model"] == "gpt-6.1-sol"
    record_property("sdk_request", json.dumps({"url": f"https://{host}/v1/responses", "body": requests[0]}))
    assert requests[0].get("service_tier") == expected
    if expected is None:
        assert "service_tier" not in requests[0]
