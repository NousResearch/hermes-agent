"""Regression coverage: the auxiliary client cache key follows the pooled credential (#113022).

Points 1/2/5 of the issue: a token rotated or refreshed by another process must yield a new
client instead of a 401 round trip on the cached one, including while a per-model cooldown
hides the entry from ``peek()``.
"""

import asyncio
import base64
import json
import socket
import subprocess
import time
from dataclasses import replace
from unittest.mock import patch

import httpx
import pytest

import agent.anthropic_credentials as anth_cred
import agent.auxiliary_client as aux
from hermes_cli import auth
from hermes_cli import models
from agent.credential_pool import CredentialPool, PooledCredential
from hermes_cli.auth import write_credential_pool

MODEL = "claude-sonnet-4-5"


def _seed(provider: str, token: str, *, model_cooldown: str | None = None) -> None:
    row = {
        "id": "entry-1", "label": "pooled", "auth_type": "oauth", "priority": 1, "source": "manual",
        "access_token": token, "refresh_token": f"rt-{token}", "expires_at": time.time() + 3600,
    }
    if model_cooldown:
        row["model_cooldowns"] = {model_cooldown: time.time() + 600}
    write_credential_pool(provider, [row])


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setattr(aux, "_client_cache", {})
    # HERMES_HOME only redirects Hermes-owned state; the borrowed Claude Code
    # reader still consults ~/.claude/.credentials.json and the macOS Keychain,
    # so an ambient login on the host would seed a second un-cooled-down pool
    # entry and break the cooldown assertions below (#114424).
    monkeypatch.setattr(anth_cred, "read_claude_code_credentials", lambda: None)
    # Every request goes through httpx.MockTransport; any real connect or child process is a test bug.
    def blocked(*_a, **_k):
        raise AssertionError("network or subprocess use in a cache-key test")
    for target, name in ((socket.socket, "connect"), (socket, "create_connection"), (socket, "getaddrinfo"),
                         (subprocess, "Popen")):
        monkeypatch.setattr(target, name, blocked)
    return tmp_path / "hermes"




@pytest.mark.parametrize("provider", ["anthropic", "openai-codex"])
def test_pool_token_rotation_changes_cache_key_and_hides_the_secret(isolated_home, provider):
    """Same entry id, new token -> different key; the token itself never enters the key."""
    _seed(provider, "tok-old")
    before = aux._client_cache_key(provider, async_mode=False, model=MODEL)
    _seed(provider, "tok-new")
    after = aux._client_cache_key(provider, async_mode=False, model=MODEL)

    assert before != after
    flat = repr(before) + repr(after)
    assert "tok-old" not in flat and "tok-new" not in flat
    # unchanged credentials still hit the same entry (control)
    assert aux._client_cache_key(provider, async_mode=False, model=MODEL) == after


def test_pool_cache_hint_follows_rotation_during_per_model_cooldown(isolated_home):
    """A per-model cooldown hides the entry from peek(); the hint must still track the token."""
    _seed("anthropic", "tok-old", model_cooldown=MODEL)
    before = aux._pool_cache_hint("anthropic")
    _seed("anthropic", "tok-new", model_cooldown=MODEL)
    after = aux._pool_cache_hint("anthropic")

    assert before.startswith("anthropic::") and after.startswith("anthropic::")
    assert before != after
    assert "tok-" not in before + after


def _seed_two(provider: str, first_token: str, second_token: str) -> None:
    rows = []
    for idx, token in ((1, first_token), (2, second_token)):
        rows.append({
            "id": f"entry-{idx}", "label": f"pooled-{idx}", "auth_type": "oauth", "priority": idx,
            "source": "manual", "access_token": token, "refresh_token": f"rt-{token}",
            "expires_at": time.time() + 3600,
        })
    write_credential_pool(provider, rows)


def test_sibling_entry_rotation_does_not_churn_selected_entry_key(isolated_home):
    """Only the peeked entry's key is digested: rotating entry-2 keeps entry-1's client cached."""
    _seed_two("anthropic", "tok-a1", "tok-b1")
    before = aux._pool_cache_hint("anthropic")
    assert before.startswith("anthropic:entry-1:")
    _seed_two("anthropic", "tok-a1", "tok-b2")  # sibling rotated, selection unchanged
    assert aux._pool_cache_hint("anthropic") == before
    _seed_two("anthropic", "tok-a2", "tok-b2")  # the selected entry itself rotated
    after = aux._pool_cache_hint("anthropic")
    assert after.startswith("anthropic:entry-1:") and after != before
    assert "tok-" not in before + after


OR_MODEL = "openai/gpt-target"


def _completion(request: httpx.Request) -> httpx.Response:
    if request.url.path.endswith("/responses"):  # the Codex wire streams
        done = {"type": "response.completed", "sequence_number": 0, "response": {
            "id": "r", "object": "response", "created_at": 0, "model": OR_MODEL, "status": "completed",
            "output": [{"type": "message", "id": "m", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": "ok", "annotations": []}]}]}}
        return httpx.Response(200, request=request, headers={"content-type": "text/event-stream"},
                              content=f"event: response.completed\ndata: {json.dumps(done)}\n\n".encode())
    return httpx.Response(200, request=request, json={
        "id": "x", "object": "chat.completion", "created": 0, "model": OR_MODEL,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
    })


@pytest.fixture
def no_side_lookups():
    """Nous: tokens are taken as valid invoke JWTs and the Portal model lookup is skipped; Codex: no git."""
    with patch.object(auth, "_nous_invoke_jwt_is_usable", return_value=True), \
         patch.object(auth, "_agent_key_is_usable", return_value=True), \
         patch.object(models, "get_nous_recommended_aux_model", return_value=OR_MODEL), \
         patch.object(aux, "_codex_cloudflare_headers", return_value={}):
        yield


def _two_key_pool_calls(change, *, provider="openrouter", canonical=None, async_mode=False, strategy="fill_first"):
    """Two ``_get_cached_client`` calls on a real two-key pool; returns (same client?, tokens sent).

    A fresh pool has no current entry, so k0 is the first entry available for the requested model.
    k0 carries a cooldown for another id: the vendor-prefixed spelling where the provider's wire id
    drops the prefix (an alias normalized with its raw name would skip k0), else an unrelated model.
    ``least_used`` (k0 used more) makes ``select()`` pick k1, which the pool-sending builders (Nous,
    Codex, xAI) send; the explicit-key routes send the peeked k0. ``change(rows, sent_index)`` edits the
    rows between the calls the way another process's refresh or a 429 on the requested model would.
    """
    canonical = canonical or provider
    other = OR_MODEL if aux._normalize_resolved_model(OR_MODEL, canonical) != OR_MODEL else "openai/other"
    rows = [
        PooledCredential(canonical, "k0", "k0", "api_key", 0, "manual", "tok-k0", request_count=5,
                         model_cooldowns={other: time.time() + 600}),
        PooledCredential(canonical, "k1", "k1", "api_key", 1, "manual", "tok-k1"),
    ]
    sent: list[str] = []

    def handler(request):
        sent.append(request.headers["authorization"].removeprefix("Bearer "))
        return _completion(request)

    async def call(client, model):
        result = client.chat.completions.create(model=model, messages=[{"role": "user", "content": "hi"}])
        if async_mode:
            await result

    transport = httpx.MockTransport(handler)
    # load_pool re-reads the store on every call: a fresh pool over the current rows.
    with patch("agent.credential_pool.get_pool_strategy", return_value=strategy), \
         patch.object(aux, "load_pool", side_effect=lambda _p: CredentialPool(canonical, [replace(k) for k in rows])), \
         patch.object(aux, "_openai_http_client_kwargs", side_effect=lambda *_a, async_mode=False, **_k: {
             "http_client": (httpx.AsyncClient if async_mode else httpx.Client)(transport=transport)}):
        async def run():
            first, model = aux._get_cached_client(provider, OR_MODEL, async_mode=async_mode)
            assert first is not None
            await call(first, model)
            rows[:] = change(rows, ["tok-k0", "tok-k1"].index(sent[0]))  # the entry actually sent
            second, model = aux._get_cached_client(provider, OR_MODEL, async_mode=async_mode)
            assert second is not None
            await call(second, model)
            return first is second
        reused = asyncio.run(run())
    return reused, sent


def _refresh_sent(rows, i):
    return [replace(r, access_token=r.access_token + "-new") if j == i else r for j, r in enumerate(rows)]


def _cool_sent(rows, i):
    wire = aux._normalize_resolved_model(OR_MODEL, rows[i].provider)  # Codex's wire id has no vendor prefix
    return [replace(r, model_cooldowns={wire: time.time() + 600}) if j == i else r for j, r in enumerate(rows)]


_ROUTES = [
    pytest.param({}, id="openrouter"),
    pytest.param({"provider": "nous"}, id="nous"),
    pytest.param({"provider": "nous", "async_mode": True}, id="nous-async"),
    pytest.param({"provider": "nous", "strategy": "least_used"}, id="nous-least-used"),
    pytest.param({"provider": "codex", "canonical": "openai-codex", "strategy": "least_used"}, id="codex-alias"),
    pytest.param({"provider": "xai-oauth", "strategy": "least_used"}, id="xai-oauth-least-used"),
]


def _first_sent(route):
    """The entry the first build sends: the pool strategy's pick (``least_used`` prefers the idle k1)."""
    return ("tok-k1", "tok-k0") if route.get("strategy") == "least_used" else ("tok-k0", "tok-k1")


@pytest.mark.parametrize("route", _ROUTES)
def test_cached_pool_client_rebuilds_when_its_model_aware_key_is_refreshed(isolated_home, no_side_lookups, route):
    reused, sent = _two_key_pool_calls(_refresh_sent, **route)
    first, _other = _first_sent(route)
    assert not reused
    assert sent == [first, first + "-new"]


@pytest.mark.parametrize("route", _ROUTES)
def test_cached_pool_client_rebuilds_when_its_key_cools_down_for_the_model(isolated_home, no_side_lookups, route):
    reused, sent = _two_key_pool_calls(_cool_sent, **route)
    assert not reused
    assert sent == list(_first_sent(route))


@pytest.mark.parametrize("route", _ROUTES)
def test_cached_pool_client_is_reused_while_its_key_is_unchanged(isolated_home, no_side_lookups, route):
    reused, sent = _two_key_pool_calls(lambda rows, _i: rows, **route)
    first, _other = _first_sent(route)
    assert reused
    assert sent == [first, first]


def _expired_jwt() -> str:
    claims = base64.urlsafe_b64encode(json.dumps({"exp": 1}).encode()).decode().rstrip("=")
    return f"e30.{claims}.sig"


def _key(canonical, i, **fields):
    return PooledCredential(canonical, f"k{i}", f"k{i}", "api_key", i, "manual", f"tok-k{i}", **fields)


def _pool_route_calls(provider, rows, *, strategy="fill_first", calls=1, cold=False, model=OR_MODEL, current=None,
                      between=None):
    """``calls`` sequential ``_get_cached_client`` calls plus one request each, on ONE long-lived
    pool over ``rows`` (``current``: its current entry). ``cold`` clears the cache before each call;
    ``between(pool, sent_id)`` edits the pool after the first. Returns (tokens sent, builds,
    ``select()`` calls, request count per entry)."""
    pool = None
    sent, builds, selects = [], [], []
    real_select, real_resolve = CredentialPool.select, aux.resolve_provider_client

    def handler(request):
        sent.append(request.headers["authorization"].removeprefix("Bearer "))
        return _completion(request)

    def counted_select(self, **kw):
        selects.append(1)
        return real_select(self, **kw)

    transport = httpx.MockTransport(handler)
    with patch("agent.credential_pool.get_pool_strategy", return_value=strategy), \
         patch.object(CredentialPool, "_persist", lambda self, **_k: None), \
         patch.object(CredentialPool, "select", counted_select), \
         patch.object(aux, "resolve_provider_client", side_effect=lambda *a, **k: builds.append(1) or real_resolve(*a, **k)), \
         patch.object(aux, "_read_codex_singleton_token", return_value=None), \
         patch.object(auth, "resolve_xai_oauth_runtime_credentials", side_effect=auth.AuthError("no singleton")), \
         patch.object(aux, "load_pool", side_effect=lambda _p: pool), \
         patch.object(aux, "_openai_http_client_kwargs", side_effect=lambda *_a, async_mode=False, **_k: {
             "http_client": httpx.Client(transport=transport)}):
        pool = CredentialPool(rows[0].provider, rows)
        pool._current_id = current
        for i in range(calls):
            if between and i == 1:
                between(pool, sent[0].removeprefix("tok-"))
            if cold:
                aux._client_cache.clear()
            client, wire_model = aux._get_cached_client(provider, model)
            assert client is not None, "no client although a pooled key is available"
            client.chat.completions.create(model=wire_model, messages=[{"role": "user", "content": "hi"}])
        counts = {e.id: e.request_count for e in pool.entries()}
    return sent, len(builds), len(selects), counts


_SENDING_ROUTES = [pytest.param("nous", id="nous"), pytest.param("openai-codex", id="codex"),
                   pytest.param("xai-oauth", id="xai-oauth")]


@pytest.mark.parametrize("provider", [p for p in _SENDING_ROUTES if p.id != "nous"])
def test_failed_refresh_of_the_first_entry_sends_the_healthy_sibling(isolated_home, no_side_lookups, provider):
    """k0's proactive OAuth refresh fails inside select(), which then picks k1: k1 is sent."""
    rows = [_key(provider, 0, refresh_token="rt-k0"), _key(provider, 1)]
    rows[0] = replace(rows[0], auth_type="oauth", access_token=_expired_jwt())
    with patch.object(CredentialPool, "_refresh_entry", return_value=None):
        sent, _builds, _selects, _counts = _pool_route_calls(provider, rows)
    assert sent == ["tok-k1"]


@pytest.mark.parametrize("provider", [p for p in _SENDING_ROUTES if p.id != "nous"])
def test_default_model_cooldown_on_the_current_entry_sends_the_sibling(isolated_home, no_side_lookups, provider):
    """A caller without a model gets the default; the current k0 cools down for it, so k1 is sent."""
    wire = aux._normalize_resolved_model(OR_MODEL, provider)
    rows = [_key(provider, 0, model_cooldowns={wire: time.time() + 600}), _key(provider, 1)]
    with patch.object(aux, "_get_aux_model_for_provider", return_value=""), \
         patch.object(aux, "_read_main_model_for_aux", return_value=OR_MODEL):
        sent, _builds, _selects, _counts = _pool_route_calls(provider, rows, model=None, current="k0")
    assert sent == ["tok-k1"]


@pytest.mark.parametrize("strategy, calls, cold, counts, expected, counted", [
    pytest.param("least_used", 1, True, (50, 0), ["tok-k1"], (50, 1), id="least-used"),
    pytest.param("random", 1, True, (0, 0), ["tok-k1"], (0, 1), id="random"),
    pytest.param("round_robin", 2, True, (0, 0), ["tok-k0", "tok-k1"], (1, 1), id="round-robin-two-builds"),
    pytest.param("least_used", 5, False, (50, 0), ["tok-k1"] * 5, (50, 1), id="steady-pool"),
])
@pytest.mark.parametrize("provider", _SENDING_ROUTES)
def test_pool_strategy_picks_the_entry_sent_and_counts_only_it(
        isolated_home, no_side_lookups, provider, strategy, calls, cold, counts, expected, counted):
    """select() runs once per build, its pick is sent, and only it is counted (``random`` is pinned
    to the last available entry). A warm cache neither builds nor selects again."""
    rows = [_key(provider, i, request_count=n) for i, n in enumerate(counts)]
    with patch("agent.credential_pool.random.choice", side_effect=lambda available: available[-1]):
        sent, builds, selects, after = _pool_route_calls(provider, rows, strategy=strategy, calls=calls, cold=cold)
    assert sent == expected
    assert builds == selects == (calls if cold else 1)
    assert after == {f"k{i}": n for i, n in enumerate(counted)}


def _refresh_entry_in(pool, entry_id):
    pool._entries = [replace(e, access_token=e.access_token + "-new") if e.id == entry_id else e for e in pool._entries]


def _cool_entry_in(pool, entry_id):
    wire = aux._normalize_resolved_model(OR_MODEL, pool.provider)
    pool._entries = [replace(e, model_cooldowns={wire: time.time() + 600}) if e.id == entry_id else e
                     for e in pool._entries]


@pytest.mark.parametrize("between, expected", [
    pytest.param(_refresh_entry_in, ["tok-k0", "tok-k0-new"], id="refresh"),
    pytest.param(_cool_entry_in, ["tok-k0", "tok-k1"], id="cooldown"),
])
@pytest.mark.parametrize("provider", _SENDING_ROUTES)
def test_cached_client_rebuilds_when_the_entry_it_sends_changes_on_a_hit(
        isolated_home, no_side_lookups, provider, between, expected):
    """A refresh or cooldown of the cached client's own entry, on a steady pool, rebuilds it."""
    rows = [_key(provider, 0), _key(provider, 1)]
    sent, builds, selects, _counts = _pool_route_calls(provider, rows, calls=2, between=between)
    assert sent == expected
    assert builds == selects == 2
