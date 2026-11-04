"""The complete xAI turn adopts a stored rotation after HTTP 403, preserving the request (#82052/#108122)."""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path

import httpx
import pytest

from agent.client_lifecycle import ClientLifecycleMixin
from run_agent import AIAgent


def _token(generation: str, expires_at: float) -> str:
    claims = {"sub": "synthetic-account", "jti": generation, "exp": expires_at}
    encoded = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    return f"header.{encoded}.signature"


def _successful_stream() -> bytes:
    item = {
        "type": "message", "id": "message-fixture", "role": "assistant", "status": "completed",
        "content": [{"type": "output_text", "text": "Recovered.", "annotations": []}],
    }
    response = {
        "id": "response-fixture", "object": "response", "created_at": 1, "status": "completed",
        "model": "grok-fixture", "output": [item],
        "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
    }
    events = [
        {"type": "response.created", "response": {**response, "status": "in_progress", "output": []}},
        {"type": "response.output_item.done", "output_index": 0, "item": item},
        {"type": "response.completed", "response": response},
    ]
    return "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events).encode()


@pytest.mark.parametrize("still_rejected", [False, True])
def test_full_turn_catches_up_once_without_rewriting_the_request(tmp_path, monkeypatch, still_rejected):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    now = time.time()
    old_token = _token("expired", now - 3600)
    current_token = _token("rotated", now + 7200)
    requests, transports, exchanges = [], [], []

    def respond(request):
        requests.append({
            "bearer": request.headers["Authorization"], "body": json.loads(request.content),
            "conversation": request.headers.get("x-grok-conv-id"), "url": str(request.url),
        })
        assert len(requests) <= 2, "one rejected-bearer recovery must not become an unbounded retry loop"
        if len(requests) == 1 or still_rejected:
            return httpx.Response(403, json={
                "code": "unauthenticated:bad-credentials",
                "error": "The OAuth2 access token could not be validated.",
            })
        return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, content=_successful_stream())

    def transport(*_args, **_kwargs):
        client = httpx.Client(transport=httpx.MockTransport(respond))
        transports.append(client)
        return client

    def exchange(*args, **_kwargs):
        exchanges.append(args)
        raise AssertionError("a stored live token must be adopted without refreshing its grant")

    monkeypatch.setattr(ClientLifecycleMixin, "_build_keepalive_http_client", staticmethod(transport))
    monkeypatch.setattr("hermes_cli.auth.refresh_xai_oauth_pure", exchange)
    agent = AIAgent(
        model="grok-fixture", provider="xai-oauth", api_mode="codex_responses",
        base_url="https://api.x.ai/v1", api_key=old_token, quiet_mode=True, max_iterations=3,
        enabled_toolsets=[], disabled_toolsets=["all"], skip_context_files=True,
        skip_memory=True, skip_background_review=True,
    )
    # Rotation happens after the long-lived agent was constructed, as it does in another process.
    (home / "auth.json").write_text(json.dumps({
        "version": 1,
        "providers": {"xai-oauth": {"tokens": {
            "access_token": current_token, "refresh_token": "synthetic-current-refresh",
        }}},
    }), encoding="utf-8")

    try:
        result = agent.run_conversation("Please reply.", system_message="Keep this system instruction unchanged.")
        assert [request["bearer"] for request in requests] == [f"Bearer {old_token}", f"Bearer {current_token}"]
        assert exchanges == []
        assert requests[0]["url"] == requests[1]["url"] == "https://api.x.ai/v1/responses"
        assert requests[0]["body"] == requests[1]["body"]
        assert "Keep this system instruction unchanged." in requests[0]["body"]["instructions"]
        assert requests[0]["conversation"] == requests[1]["conversation"]
        assert agent.api_key == current_token
        assert result["completed"] is not still_rejected
        if not still_rejected:
            assert result["final_response"] == "Recovered."
    finally:
        agent.close()
        for client in transports:
            client.close()
