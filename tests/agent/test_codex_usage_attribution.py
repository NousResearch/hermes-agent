"""Exercise required Codex attribution through real SDK request construction."""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import yaml

from hermes_cli import __version__
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


CODEX_URL = "https://chatgpt.com/backend-api/codex"
MODEL = "gpt-5.4"


def _jwt(account_id="acct-attribution-test"):
    payload = json.dumps({
        "https://api.openai.com/auth": {"chatgpt_account_id": account_id},
    }).encode()
    encoded = base64.urlsafe_b64encode(payload).rstrip(b"=").decode()
    return f"e30.{encoded}.test-signature"


@pytest.fixture
def profile(tmp_path, monkeypatch):
    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    token = set_hermes_home_override(home)
    try:
        yield home
    finally:
        reset_hermes_home_override(token)


def _set_legacy_attribution(profile, enabled):
    """Old draft settings must not disable required harness identification."""
    if enabled is not None:
        (profile / "config.yaml").write_text(
            yaml.safe_dump({
                "telemetry": {"usage_attribution": {"enabled": enabled}},
            }),
            encoding="utf-8",
        )


@pytest.fixture
def wire(profile, monkeypatch):
    """Replace only HTTP transports; use Hermes routing and the real SDK."""
    from agent import auxiliary_client
    from run_agent import AIAgent

    requests = []
    response = {
        "id": "resp_attribution_test",
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "model": MODEL,
        "output": [{
            "type": "message",
            "id": "msg_test",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "ok", "annotations": []}],
        }],
    }

    def respond(request):
        requests.append(request)
        if json.loads(request.content).get("stream"):
            events = [
                {
                    "type": "response.output_item.done",
                    "output_index": 0,
                    "item": response["output"][0],
                },
                {"type": "response.completed", "response": response},
            ]
            content = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
            return httpx.Response(
                200,
                headers={"Content-Type": "text/event-stream"},
                content=content + "data: [DONE]\n\n",
            )
        return httpx.Response(200, json=response)

    def http_client(*_args, async_mode=False, **_kwargs):
        cls = httpx.AsyncClient if async_mode else httpx.Client
        return cls(transport=httpx.MockTransport(respond))

    monkeypatch.setattr(
        auxiliary_client, "_openai_http_client_kwargs",
        lambda _url, *, async_mode=False: {
            "http_client": http_client(async_mode=async_mode),
        },
    )
    monkeypatch.setattr(AIAgent, "_build_keepalive_http_client", staticmethod(http_client))
    return requests


def _assert_identity(request, account_id="acct-attribution-test"):
    assert request.headers["originator"] == "hermes-agent"
    assert request.headers["user-agent"] == f"HermesAgent/{__version__}"
    assert request.headers["chatgpt-account-id"] == account_id
    assert "extra_headers" not in json.loads(request.content)


@pytest.mark.parametrize("legacy_enabled", [None, False, True])
def test_required_identity_preserves_account_id(profile, legacy_enabled):
    from agent.auxiliary_client import _codex_cloudflare_headers

    _set_legacy_attribution(profile, legacy_enabled)
    headers = _codex_cloudflare_headers(_jwt())

    assert headers["originator"] == "hermes-agent"
    assert headers["User-Agent"] == f"HermesAgent/{__version__}"
    assert headers["ChatGPT-Account-ID"] == "acct-attribution-test"
    assert "ChatGPT-Account-ID" not in _codex_cloudflare_headers("not-a-jwt")


@pytest.mark.parametrize(
    ("base_url", "attributed"),
    [
        (CODEX_URL, True),
        (CODEX_URL + "/", True),
        (CODEX_URL + "/responses", True),
        ("https://CHATGPT.COM:443/backend-api/codex", True),
        ("http://chatgpt.com/backend-api/codex", False),
        ("https://chatgpt.com:8443/backend-api/codex", False),
        ("https://api.openai.com/v1", False),
        ("https://proxy.example/backend-api/codex", False),
        ("https://chatgpt.com.example/backend-api/codex", False),
        ("https://subdomain.chatgpt.com/backend-api/codex", False),
        ("https://chatgpt.com/backend-api/codex-other", False),
        ("https://chatgpt.com/backend-api/other", False),
        ("https://chatgpt.com:invalid/backend-api/codex", False),
    ],
)
def test_new_identity_is_limited_to_the_official_endpoint(base_url, attributed):
    from agent.auxiliary_client import _codex_cloudflare_headers

    headers = _codex_cloudflare_headers(_jwt(), base_url=base_url)

    assert headers["originator"] == ("hermes-agent" if attributed else "codex_cli_rs")
    assert headers["User-Agent"] == (
        f"HermesAgent/{__version__}"
        if attributed else "codex_cli_rs/0.0.0 (Hermes Agent)"
    )


@pytest.mark.parametrize("legacy_enabled", [None, False, True])
def test_primary_client_and_credential_rebuild_send_expected_headers(
    profile, wire, legacy_enabled,
):
    from run_agent import AIAgent

    _set_legacy_attribution(profile, legacy_enabled)
    agent = AIAgent(
        api_key=_jwt(),
        base_url=CODEX_URL,
        provider="openai-codex",
        model=MODEL,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    clients = [agent.client]
    try:
        agent.client.responses.create(model=MODEL, input="test")
        _assert_identity(wire[-1])

        agent._client_kwargs["api_key"] = _jwt("acct-rotated")
        agent._apply_client_headers_for_base_url(CODEX_URL)
        assert agent._replace_primary_openai_client(reason="attribution-test")
        clients.append(agent.client)
        agent.client.responses.create(model=MODEL, input="test")
        _assert_identity(wire[-1], "acct-rotated")

        direct_url = "https://api.openai.com/v1"
        agent._client_kwargs.update(api_key="test-direct-key", base_url=direct_url)
        agent._apply_client_headers_for_base_url(direct_url)
        assert agent._replace_primary_openai_client(reason="attribution-route-change")
        clients.append(agent.client)
        agent.client.responses.create(model=MODEL, input="test")
        assert "originator" not in wire[-1].headers
        assert "chatgpt-account-id" not in wire[-1].headers
        assert not wire[-1].headers["user-agent"].startswith("HermesAgent/")
    finally:
        for client in clients:
            client.close()


@pytest.mark.parametrize("legacy_enabled", [None, False, True])
def test_auxiliary_raw_and_async_clients_send_expected_headers(
    profile, wire, monkeypatch, legacy_enabled,
):
    from agent import auxiliary_client

    _set_legacy_attribution(profile, legacy_enabled)
    monkeypatch.setattr(auxiliary_client, "_select_pool_entry", lambda _p: (False, None))
    monkeypatch.setattr(auxiliary_client, "_read_codex_access_token", _jwt)

    wrapped, model = auxiliary_client._build_codex_client(MODEL)
    raw, raw_model = auxiliary_client.resolve_provider_client(
        "openai-codex", model=MODEL, raw_codex=True,
    )
    try:
        result = wrapped.chat.completions.create(
            model=model, messages=[{"role": "user", "content": "test"}],
        )
        assert result.choices[0].message.content == "ok"
        _assert_identity(wire[-1])

        raw.responses.create(model=raw_model, input="test")
        _assert_identity(wire[-1])

        async def send_async():
            async_wrapped, _ = auxiliary_client._to_async_client(wrapped, model)
            result = await async_wrapped.chat.completions.create(
                model=model, messages=[{"role": "user", "content": "test"}],
            )
            assert result.choices[0].message.content == "ok"
            _assert_identity(wire[-1])

            async_raw, _ = auxiliary_client._to_async_client(raw, raw_model)
            try:
                await async_raw.responses.create(model=raw_model, input="test")
                _assert_identity(wire[-1])
            finally:
                await async_raw.close()

        asyncio.run(send_async())
    finally:
        wrapped.close()
        raw.close()


def test_credential_pool_custom_endpoint_keeps_existing_identity(
    wire, monkeypatch,
):
    from agent import auxiliary_client

    entry = SimpleNamespace(
        runtime_api_key=_jwt(),
        runtime_base_url="https://proxy.example/backend-api/codex",
    )
    monkeypatch.setattr(auxiliary_client, "_select_pool_entry", lambda _p: (True, entry))

    client, model = auxiliary_client._build_codex_client(MODEL)
    try:
        client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": "test"}],
        )
        assert wire[-1].url.host == "proxy.example"
        assert wire[-1].headers["originator"] == "codex_cli_rs"
        assert wire[-1].headers["user-agent"] == "codex_cli_rs/0.0.0 (Hermes Agent)"
        assert wire[-1].headers["chatgpt-account-id"] == "acct-attribution-test"
    finally:
        client.close()


def test_legacy_disabled_setting_cannot_disable_attribution_for_new_clients(
    profile, wire, monkeypatch,
):
    from agent import auxiliary_client

    monkeypatch.setattr(auxiliary_client, "_read_codex_access_token", _jwt)
    _set_legacy_attribution(profile, True)
    old, _ = auxiliary_client.resolve_provider_client(
        "openai-codex", model=MODEL, raw_codex=True,
    )
    _set_legacy_attribution(profile, False)
    new, _ = auxiliary_client.resolve_provider_client(
        "openai-codex", model=MODEL, raw_codex=True,
    )
    try:
        old.responses.create(model=MODEL, input="test")
        _assert_identity(wire[-1])
        new.responses.create(model=MODEL, input="test")
        _assert_identity(wire[-1])
    finally:
        old.close()
        new.close()


def test_required_identity_wins_over_configured_header_defaults(
    profile, wire,
):
    from agent import auxiliary_client
    from run_agent import AIAgent

    overrides = {
        "Originator": "codex_cli_rs",
        "user-agent": "custom-client",
        "X-Test-Header": "preserved",
    }
    (profile / "config.yaml").write_text(
        yaml.safe_dump({"model": {"default_headers": overrides}}),
        encoding="utf-8",
    )
    agent = AIAgent(
        api_key=_jwt(),
        base_url=CODEX_URL,
        provider="openai-codex",
        model=MODEL,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    raw = auxiliary_client._create_openai_client(
        api_key=_jwt(), base_url=CODEX_URL, default_headers=overrides,
    )
    proxy = auxiliary_client._create_openai_client(
        api_key="test-proxy-key",
        base_url="https://proxy.example/v1",
        default_headers=overrides,
    )
    clients = [agent.client, raw, proxy]
    try:
        for client in (agent.client, raw):
            client.responses.create(model=MODEL, input="test")
            _assert_identity(wire[-1])
            assert wire[-1].headers["x-test-header"] == "preserved"

        agent._apply_client_headers_for_base_url(CODEX_URL)
        assert agent._replace_primary_openai_client(reason="required-identity-test")
        clients.append(agent.client)
        agent.client.responses.create(model=MODEL, input="test")
        _assert_identity(wire[-1])
        assert wire[-1].headers["x-test-header"] == "preserved"

        async def send_async():
            async_raw, _ = auxiliary_client._to_async_client(raw, MODEL)
            try:
                await async_raw.responses.create(model=MODEL, input="test")
                _assert_identity(wire[-1])
                assert wire[-1].headers["x-test-header"] == "preserved"
            finally:
                await async_raw.close()

        asyncio.run(send_async())

        proxy.responses.create(model=MODEL, input="test")
        assert wire[-1].headers["originator"] == "codex_cli_rs"
        assert "custom-client" in wire[-1].headers.get_list("user-agent")
        assert "HermesAgent/" not in wire[-1].headers["user-agent"]
        assert wire[-1].headers["x-test-header"] == "preserved"
        assert "chatgpt-account-id" not in wire[-1].headers
    finally:
        for client in clients:
            client.close()
