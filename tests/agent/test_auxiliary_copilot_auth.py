"""Behavioural regression tests for Copilot auxiliary credential freshness."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

import agent.auxiliary_client as ac
import hermes_cli.copilot_auth as copilot_auth


RAW_TOKEN = "ghu_synthetic_auxiliary_test"
OLD_BEARER = "tid=old;exp=1300;sku=copilot_individual"
FRESH_BEARER = "tid=fresh;exp=5000;sku=copilot_individual"
COPILOT_URL = "https://api.githubcopilot.com"
MESSAGES = [{"role": "user", "content": "Summarise this."}]


class _ExchangeResponse:
    def __init__(self, token: str, expires_at: float):
        self._body = json.dumps({"token": token, "expires_at": expires_at}).encode()

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


@dataclass
class _Boundary:
    responses: list[int | str]
    authorisations: list[str] = field(default_factory=list)
    requests: list[dict[str, Any]] = field(default_factory=list)

    def _response(self, request: httpx.Request) -> httpx.Response:
        self.authorisations.append(request.headers.get("authorization", ""))
        self.requests.append({
            "url": str(request.url),
            "headers": dict(request.headers),
            "body": json.loads(request.content or b"{}"),
        })
        status = self.responses.pop(0) if self.responses else 200
        if status == "stream_error":
            body = (
                'data: {"type":"error","sequence_number":0,"code":"unauthorized",'
                '"message":"IDE token expired: unauthorized: token expired","param":null}\n\n'
            ).encode()
            return httpx.Response(
                200, request=request, content=body,
                headers={"content-type": "text/event-stream"},
            )
        if status == 401:
            return httpx.Response(
                401,
                request=request,
                json={"error": {"message": "IDE token expired: unauthorized: token expired"}},
            )
        return httpx.Response(
            200,
            request=request,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    def http_client_kwargs(self, _base_url: str | None, *, async_mode: bool = False) -> dict[str, Any]:
        if async_mode:
            async def handler(request: httpx.Request) -> httpx.Response:
                return self._response(request)

            return {"http_client": httpx.AsyncClient(transport=httpx.MockTransport(handler))}
        return {"http_client": httpx.Client(transport=httpx.MockTransport(self._response))}


@pytest.fixture(autouse=True)
def _isolated_copilot_state(monkeypatch):
    ac.shutdown_cached_clients()
    copilot_auth._jwt_cache.clear()
    copilot_auth._exchange_failure_cache.clear()
    monkeypatch.setattr(copilot_auth, "resolve_copilot_token", lambda: (RAW_TOKEN, "synthetic"))
    yield
    ac.shutdown_cached_clients()
    copilot_auth._jwt_cache.clear()
    copilot_auth._exchange_failure_cache.clear()


def _install_boundary(monkeypatch, boundary: _Boundary, exchange_tokens: list[tuple[str, float]]):
    exchange_calls: list[str] = []

    def exchange(req, _timeout):
        exchange_calls.append(req.get_header("Authorization"))
        token, expires_at = exchange_tokens.pop(0)
        return _ExchangeResponse(token, expires_at)

    monkeypatch.setattr(ac, "_openai_http_client_kwargs", boundary.http_client_kwargs)
    monkeypatch.setattr(copilot_auth, "_urlopen_bounded", exchange)
    return exchange_calls


def _sync_call(task: str):
    return ac.call_llm(
        task=task,
        provider="copilot",
        model="gpt-4o-mini",
        messages=MESSAGES,
    )


async def _invoke(mode: str, task: str):
    if mode == "sync":
        return _sync_call(task)
    return await ac.async_call_llm(
        task=task, provider="copilot", model="gpt-4o-mini", messages=MESSAGES,
    )


@pytest.mark.parametrize(
    "mode,task",
    [("sync", "compression"), ("async", "approval"), ("sync", "vision"), ("async", "vision")],
)
@pytest.mark.asyncio
async def test_expiring_cached_client_is_rebuilt_before_its_next_request(monkeypatch, mode, task):
    """A cached client must never send an exchanged bearer inside the refresh margin."""
    now = [1000.0]
    monkeypatch.setattr(copilot_auth.time, "time", lambda: now[0])
    boundary = _Boundary([200, 200])
    exchanges = _install_boundary(
        monkeypatch,
        boundary,
        [(OLD_BEARER, 1400.0), (FRESH_BEARER, 5000.0)],
    )

    await _invoke(mode, task)
    now[0] = 1300.0
    await _invoke(mode, task)

    assert boundary.authorisations == [f"Bearer {OLD_BEARER}", f"Bearer {FRESH_BEARER}"]
    assert exchanges == [f"token {RAW_TOKEN}", f"token {RAW_TOKEN}"]


@pytest.mark.parametrize(
    "model,api_mode,expected_type",
    [
        ("gpt-5.4", None, ac.AsyncCodexAuxiliaryClient),
        ("gpt-4o-mini", "anthropic_messages", ac.AsyncAnthropicAuxiliaryClient),
    ],
)
def test_async_wrappers_preserve_copilot_auth(monkeypatch, model, api_mode, expected_type):
    monkeypatch.setattr(copilot_auth.time, "time", lambda: 1000.0)
    boundary = _Boundary([])
    _install_boundary(monkeypatch, boundary, [(FRESH_BEARER, 5000.0)])

    client, _ = ac.resolve_provider_client(
        "copilot", model, async_mode=True, api_mode=api_mode)

    assert client is not None and isinstance(client, expected_type)
    assert client._hermes_copilot_auth == (RAW_TOKEN, FRESH_BEARER, 5000.0)


def test_direct_cache_lookup_checks_freshness_without_request_state(monkeypatch):
    """Freshness is a cache invariant, not something callers must enable."""
    now = [1000.0]
    monkeypatch.setattr(copilot_auth.time, "time", lambda: now[0])
    boundary = _Boundary([200, 200])
    exchanges = _install_boundary(
        monkeypatch,
        boundary,
        [(OLD_BEARER, 1400.0), (FRESH_BEARER, 5000.0)],
    )

    for current in (1000.0, 1300.0):
        now[0] = current
        client, model = ac._get_cached_client("copilot", model="gpt-4o-mini")
        assert client is not None
        client.chat.completions.create(model=model, messages=MESSAGES)

    assert boundary.authorisations == [f"Bearer {OLD_BEARER}", f"Bearer {FRESH_BEARER}"]
    assert exchanges == [f"token {RAW_TOKEN}", f"token {RAW_TOKEN}"]


@pytest.mark.parametrize("mode,task", [("sync", "title_generation"), ("async", "compression")])
@pytest.mark.asyncio
async def test_401_evicts_persisted_exchange_and_replays_once_with_fresh_bearer(
    monkeypatch, mode, task
):
    """Server rejection must evict disk state before one fresh exchange and replay."""
    monkeypatch.setattr(copilot_auth.time, "time", lambda: 1000.0)
    fingerprint = copilot_auth._token_fingerprint(RAW_TOKEN)
    copilot_auth._save_jwt_to_disk(fingerprint, OLD_BEARER, 5000.0, None)
    boundary = _Boundary([401, 200])
    exchanges = _install_boundary(monkeypatch, boundary, [(FRESH_BEARER, 5000.0)])

    result = await _invoke(mode, task)

    assert result.choices[0].message.content == "ok"
    assert boundary.authorisations == [f"Bearer {OLD_BEARER}", f"Bearer {FRESH_BEARER}"]
    assert exchanges == [f"token {RAW_TOKEN}"]


@pytest.mark.parametrize(
    "provider,base_url,api_key",
    [
        ("copilot", COPILOT_URL, "tid=unknown-expiry;sku=copilot_individual"),
        ("custom", "https://non-copilot.invalid/v1", "synthetic-non-copilot-key"),
    ],
)
def test_unknown_expiry_and_non_copilot_clients_keep_existing_cache_behaviour(
    monkeypatch, provider, base_url, api_key
):
    """Unknown Copilot expiry and unrelated providers are not proactively renewed."""
    boundary = _Boundary([200, 200])
    monkeypatch.setattr(ac, "_openai_http_client_kwargs", boundary.http_client_kwargs)

    exchange_calls = []

    def counted_exchange(*args, **kwargs):
        exchange_calls.append((args, kwargs))
        raise AssertionError("synthetic exchange unavailable")

    monkeypatch.setattr(copilot_auth, "_urlopen_bounded", counted_exchange)
    ac.call_llm(
        task="compression", provider=provider, model="gpt-4o-mini",
        base_url=base_url, api_key=api_key, messages=MESSAGES,
    )
    construction_exchanges = len(exchange_calls)
    ac.call_llm(
        task="compression", provider=provider, model="gpt-4o-mini",
        base_url=base_url, api_key=api_key, messages=MESSAGES,
    )

    assert boundary.authorisations == [f"Bearer {api_key}", f"Bearer {api_key}"]
    assert len(exchange_calls) == construction_exchanges


@pytest.mark.parametrize(
    "mode,task",
    [("sync", "compression"), ("async", "approval"), ("sync", "vision"), ("async", "vision")],
)
@pytest.mark.asyncio
async def test_proactive_refresh_consumes_the_operations_only_full_auth_recovery(monkeypatch, mode, task):
    """A rejection after proactive renewal must not start another full-auth cycle."""
    now = [1000.0]
    monkeypatch.setattr(copilot_auth.time, "time", lambda: now[0])
    monkeypatch.setattr(ac, "_pool_cache_hint", lambda *_args, **_kwargs: "")
    boundary = _Boundary([200, 401, 200])
    exchanges = _install_boundary(
        monkeypatch,
        boundary,
        [(OLD_BEARER, 1400.0), (FRESH_BEARER, 5000.0)],
    )

    await _invoke(mode, task)
    now[0] = 1300.0

    with pytest.raises(Exception, match="IDE token expired"):
        await _invoke(mode, task)

    assert boundary.authorisations == [f"Bearer {OLD_BEARER}", f"Bearer {FRESH_BEARER}"]
    assert exchanges == [f"token {RAW_TOKEN}", f"token {RAW_TOKEN}"]


def test_adopting_concurrent_fresh_exchange_preserves_full_auth_recovery(monkeypatch):
    """Adopting another caller's bearer must not spend this request's refresh allowance."""
    now = [1000.0]
    monkeypatch.setattr(copilot_auth.time, "time", lambda: now[0])
    concurrent_bearer = "tid=concurrent;exp=5000;sku=copilot_individual"
    recovered_bearer = "tid=recovered;exp=6000;sku=copilot_individual"
    boundary = _Boundary([200, 401, 200])
    exchanges = _install_boundary(
        monkeypatch, boundary, [(OLD_BEARER, 1400.0), (recovered_bearer, 6000.0)])

    _sync_call("compression")
    now[0] = 1300.0
    copilot_auth._jwt_cache[copilot_auth._token_fingerprint(RAW_TOKEN)] = (
        concurrent_bearer, 5000.0, None)

    result = _sync_call("compression")

    assert result.choices[0].message.content == "ok"
    assert boundary.authorisations == [
        f"Bearer {OLD_BEARER}",
        f"Bearer {concurrent_bearer}",
        f"Bearer {recovered_bearer}",
    ]
    assert exchanges == [f"token {RAW_TOKEN}", f"token {RAW_TOKEN}"]


def test_spent_copilot_allowance_does_not_block_other_provider_refresh():
    """The Copilot-only allowance must not gate another provider's recovery rung."""
    client = MagicMock(api_key="expired-anthropic-token")
    route = ac._LadderRoute(
        client, "compression", "", False, "https://api.anthropic.com", "anthropic",
        "claude-haiku-4-5-20251001", None, None, None, "claude-haiku-4-5-20251001",
        None, None, True,
    )
    class AuthError(RuntimeError):
        status_code = 401

    error = AuthError("Invalid bearer token")

    with patch("agent.auxiliary_client._refresh_provider_credentials", return_value=True) as refresh:
        rung = ac._ladder_credential_rungs(error, route, {}, False)
        step = next(rung)

    assert step == ac._LadderStep(
        "retry_same_provider", ("anthropic", "claude-haiku-4-5-20251001"))
    refresh.assert_called_once_with("anthropic", failed_api_key="expired-anthropic-token")


def test_responses_error_event_refreshes_once_and_preserves_request(monkeypatch):
    """An internally consumed Responses error event uses the shared auth recovery path."""
    monkeypatch.setattr(copilot_auth.time, "time", lambda: 1000.0)
    boundary = _Boundary(["stream_error", "stream_error"])
    exchanges = _install_boundary(
        monkeypatch, boundary, [(OLD_BEARER, 5000.0), (FRESH_BEARER, 5000.0)])

    with pytest.raises(Exception, match="IDE token expired"):
        ac.call_llm(
            task="approval", provider="copilot", model="gpt-5.4",
            messages=MESSAGES, extra_body={"service_tier": "priority"},
            extra_headers={"x-test-preserved": "yes"},
        )

    assert boundary.authorisations == [f"Bearer {OLD_BEARER}", f"Bearer {FRESH_BEARER}"]
    assert exchanges == [f"token {RAW_TOKEN}", f"token {RAW_TOKEN}"]
    assert [request["url"] for request in boundary.requests] == [
        f"{COPILOT_URL}/responses", f"{COPILOT_URL}/responses"]
    for request in boundary.requests:
        assert request["body"]["model"] == "gpt-5.4"
        assert request["body"]["service_tier"] == "priority"
        assert request["headers"]["x-test-preserved"] == "yes"


def test_failed_full_auth_does_not_start_a_second_recovery_cycle(monkeypatch):
    """A failed token exchange terminates after one real full-auth recovery attempt."""
    monkeypatch.setattr(copilot_auth.time, "time", lambda: 1000.0)
    fingerprint = copilot_auth._token_fingerprint(RAW_TOKEN)
    copilot_auth._save_jwt_to_disk(fingerprint, OLD_BEARER, 5000.0, None)
    boundary = _Boundary([401])
    exchanges = _install_boundary(monkeypatch, boundary, [])
    refresh_calls = []
    original_refresh = ac._refresh_copilot_credentials

    def traced_refresh(*args, **kwargs):
        refresh_calls.append(True)
        return original_refresh(*args, **kwargs)

    monkeypatch.setattr(ac, "_refresh_copilot_credentials", traced_refresh)

    with pytest.raises(Exception, match="IDE token expired"):
        _sync_call("title_generation")

    assert boundary.authorisations == [f"Bearer {OLD_BEARER}"]
    assert refresh_calls == [True]
    assert exchanges


def test_preflight_refresh_failure_uses_configured_fallback(monkeypatch):
    """A stale Copilot client becomes unavailable when renewal fails, allowing task fallback."""
    now = [1000.0]
    monkeypatch.setattr(copilot_auth.time, "time", lambda: now[0])
    boundary = _Boundary([200, 200])
    _install_boundary(monkeypatch, boundary, [(OLD_BEARER, 1400.0)])
    task_config = {"fallback_chain": [{
        "provider": "custom",
        "model": "gpt-4o-mini",
        "base_url": "https://fallback.invalid/v1",
        "api_key": "synthetic-fallback-key",
    }]}
    monkeypatch.setattr(ac, "_get_auxiliary_task_config", lambda task: task_config)

    _sync_call("approval")
    now[0] = 1300.0
    failed_exchanges = []

    def rejected_exchange(*_args, **_kwargs):
        failed_exchanges.append(True)
        raise ValueError("synthetic token exchange failure")

    monkeypatch.setattr(copilot_auth, "_urlopen_bounded", rejected_exchange)
    result = _sync_call("approval")

    assert failed_exchanges
    assert result.choices[0].message.content == "ok"
    assert boundary.authorisations == [
        f"Bearer {OLD_BEARER}",
        "Bearer synthetic-fallback-key",
    ]
    assert boundary.requests[-1]["url"].startswith("https://fallback.invalid/")
