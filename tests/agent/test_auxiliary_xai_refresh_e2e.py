"""Exercise xAI refresh through config, auth storage, SDK, and the aux cache."""

import asyncio
import base64
import json
from pathlib import Path
import time
from urllib.parse import parse_qs

import httpx
import pytest
import yaml


def _token(label):
    payload = json.dumps({"exp": int(time.time()) + 86400, "sub": label}).encode()
    return "e30." + base64.urlsafe_b64encode(payload).decode().rstrip("=") + ".sig"


@pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
def test_auto_xai_refresh_rebuilds_both_cached_routes(tmp_path, monkeypatch, async_mode):
    from agent import auxiliary_client as aux
    from hermes_cli import auth_xai as auth

    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text(yaml.safe_dump({
        "model": {"provider": "xai-oauth", "default": "grok-4.6"},
    }))
    old, current, fresh = (_token(label) for label in ("old", "current", "fresh"))
    auth._save_xai_oauth_tokens(
        {"access_token": old, "refresh_token": "refresh-original"},
        discovery={"token_endpoint": "https://auth.x.ai/token"},
    )
    calls = []
    refreshes = []
    rejected = set()

    def handle_request(_transport, request):
        if request.url.host == "auth.x.ai" and request.url.path == "/token":
            refreshes.append(parse_qs(request.content.decode()))
            return httpx.Response(200, json={
                "access_token": fresh, "refresh_token": "refresh-next", "expires_in": 86400,
            })
        assert request.url.host == "api.x.ai", request.url
        assert request.url.path == "/v1/responses", request.url
        bearer = request.headers["Authorization"].removeprefix("Bearer ")
        calls.append(bearer)
        if bearer in rejected:
            return httpx.Response(403, json={
                "code": "unauthenticated:bad-credentials",
                "error": "The OAuth2 access token could not be validated.",
            })
        payload = json.loads(request.content)
        assert payload["model"] == "grok-4.6"
        item = {
            "id": "msg-test", "type": "message", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": "continue", "annotations": []}],
        }
        events = [
            {"type": "response.output_item.done", "output_index": 0, "item": item},
            {"type": "response.completed", "response": {
                "id": "resp-test", "status": "completed", "output": [item],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            }},
        ]
        content = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
        return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, content=content)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", handle_request)
    aux.shutdown_cached_clients()
    aux.clear_runtime_main()
    aux._reset_aux_unhealthy_cache()

    async def exercise():
        async def invoke(**overrides):
            kwargs = {"task": "goal_judge", "messages": [{"role": "user", "content": "judge"}]}
            kwargs.update(overrides)
            result = await aux.async_call_llm(**kwargs) if async_mode else aux.call_llm(**kwargs)
            assert result.choices[0].message.content == "continue"

        await invoke(provider="xai-oauth", model="grok-4.6")
        await invoke()
        # Another caller rotated the login while both route labels retained the old client.
        auth._save_xai_oauth_tokens({"access_token": current, "refresh_token": "refresh-current"})
        rejected.add(old)
        await invoke()
        await invoke()
        await invoke(provider="xai-oauth", model="grok-4.6")

    try:
        asyncio.run(exercise())
        assert calls == [old, old, old, fresh, fresh, fresh]
        assert len(refreshes) == 1
        assert refreshes[0]["grant_type"] == ["refresh_token"]
        assert refreshes[0]["refresh_token"] == ["refresh-current"]
        tokens = auth._read_xai_oauth_tokens()["tokens"]
        assert tokens["access_token"] == fresh
        assert tokens["refresh_token"] == "refresh-next"
    finally:
        aux.shutdown_cached_clients()
        aux.clear_runtime_main()
        aux._reset_aux_unhealthy_cache()


def test_credential_eviction_preserves_other_profile_clients(tmp_path):
    from agent import auxiliary_client as aux
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    clients = []
    homes = [tmp_path / "first", tmp_path / "second"]
    kwargs = {"model": "fixture", "base_url": "https://example.invalid/v1", "api_key": "fixture"}
    aux.shutdown_cached_clients()
    try:
        for home in homes:
            home.mkdir()
            token = set_hermes_home_override(str(home))
            try:
                client, _ = aux._get_cached_client("custom", **kwargs)
                clients.append(client)
            finally:
                reset_hermes_home_override(token)
        token = set_hermes_home_override(str(homes[0]))
        try:
            aux._evict_cached_clients("custom")
        finally:
            reset_hermes_home_override(token)
        assert clients[0].is_closed()
        assert not clients[1].is_closed()
        token = set_hermes_home_override(str(homes[1]))
        try:
            cached, _ = aux._get_cached_client("custom", **kwargs)
            assert cached is clients[1]
        finally:
            reset_hermes_home_override(token)
    finally:
        aux.shutdown_cached_clients()
