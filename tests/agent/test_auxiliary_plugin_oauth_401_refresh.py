"""Auxiliary calls on a plugin OAuth provider (Solstice) refresh the pooled sign-in after a 401.

Title generation and approval inherit the main route as ``resolved_provider == "auto"``; vision and a
pinned ``auxiliary.<task>.provider`` resolve to ``solstice``. Every one of them used to raise the raw
401 without spending the refresh token: the auto route found no host row for
generativelanguage.googleapis.com (shared with the Gemini API-key provider), and the named route asked
the profile's generic ``refresh_runtime_credentials``, which cannot reach a pooled row. The pinned
route then benched the only credential, and the next call reported that 401 as a 429 cooldown.

Real ``call_llm`` path; only Google's HTTP and the token broker are faked.
"""

from __future__ import annotations

import contextlib
import json
import time

import httpx
import pytest

import agent.auxiliary_client as aux
from agent.auxiliary_unavailable import pool_cooldown_message
from hermes_cli import auth_oauth_pkce_plugin as pkce

GOOGLE = "generativelanguage.googleapis.com"
BASE_URL = "https://generativelanguage.googleapis.com/v1alpha"
OK_BODY = {"candidates": [{"content": {"role": "model", "parts": [{"text": "A Title"}]}, "finishReason": "STOP"}],
           "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 2, "totalTokenCount": 3}}


def _now_ms() -> int:
    return int(time.time() * 1000)


def _row(access, expires_at_ms, **extra):
    return {"id": "sol1", "label": "solstice", "auth_type": "oauth", "priority": 0, "source": pkce.POOL_SOURCE,
            "access_token": access, "refresh_token": "1//rt-0", "expires_at_ms": expires_at_ms, **extra}


@pytest.fixture
def env(monkeypatch, tmp_path):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    for k in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    (home / "config.yaml").write_text("model:\n  provider: solstice\n  default: gemini-3.1-pro\n", encoding="utf-8")
    grants, seen = [], []

    def fake_post_token(provider, cfg, data, *, code):
        grants.append(dict(data))
        return {"access_token": f"ya29.fresh-{len(grants)}", "refresh_token": "1//rt-0",
                "expires_at_ms": _now_ms() + 3_600_000, "last_refresh": "now"}

    monkeypatch.setattr(pkce, "_post_token", fake_post_token)

    def respond(url, headers):
        bearer = headers.get("Authorization", "")
        seen.append(bearer)
        req = httpx.Request("POST", str(url))
        if GOOGLE in str(url) and "fresh" in bearer:
            return httpx.Response(200, json=OK_BODY, request=req)
        return httpx.Response(401, request=req, json={"error": {
            "code": 401, "status": "UNAUTHENTICATED", "message": "Request had invalid authentication credentials."}})

    monkeypatch.setattr(httpx.Client, "post", lambda self, url, **kw: respond(url, kw.get("headers") or {}))
    monkeypatch.setattr(aux, "_client_cache", {})
    return home, grants, seen


def _write(home, row):
    (home / "auth.json").write_text(json.dumps({"version": 1, "credential_pool": {"solstice": [row]}}),
                                    encoding="utf-8")


def _disk_row(home):
    return json.loads((home / "auth.json").read_text(encoding="utf-8-sig"))["credential_pool"]["solstice"][0]


def _text(resp):
    return resp.choices[0].message.content


def _title(runtime):
    return aux.call_llm(task="title_generation", main_runtime=runtime,
                        messages=[{"role": "user", "content": "name this chat"}], max_tokens=20)


@pytest.mark.parametrize("session_key", [None, "ya29.revoked"])
def test_auto_route_401_refreshes_the_pooled_sign_in_and_retries(env, session_key):
    """Title generation inherits the main Solstice route ("auto"); a 401 on a token the pool still
    believes valid refreshes that row and retries, whether or not the session's key is pinned."""
    home, grants, seen = env
    _write(home, _row("ya29.revoked", _now_ms() + 3_000_000))
    runtime = {"provider": "solstice", "model": "gemini-3.1-pro", "base_url": BASE_URL, "api_mode": "chat_completions"}
    if session_key:
        runtime["api_key"] = session_key

    assert _text(_title(runtime)) == "A Title"
    assert len(grants) == 1
    assert seen == ["Bearer ya29.revoked", "Bearer ya29.fresh-1"]
    row = _disk_row(home)
    assert row["access_token"] == "ya29.fresh-1"
    assert row.get("last_status") != "exhausted"


@pytest.mark.parametrize("pinned", [False, True])
def test_a_cached_client_is_not_reused_past_its_rows_stored_expiry(env, monkeypatch, pinned):
    """A client cached while its token was valid keyed only on the token, so once the row's stored expiry
    passed every cached title call still sent the dead bearer and needed a 401 before refreshing."""
    home, grants, seen = env
    if pinned:
        (home / "config.yaml").write_text(
            "model:\n  provider: solstice\n  default: gemini-3.1-pro\n"
            "auxiliary:\n  title_generation:\n    provider: solstice\n    model: gemini-3.1-pro\n", encoding="utf-8")
    _write(home, _row("ya29.fresh-warm", _now_ms() + 3_600_000))
    runtime = {"provider": "solstice", "model": "gemini-3.1-pro", "api_key": "ya29.fresh-warm",
               "base_url": BASE_URL, "api_mode": "chat_completions"}
    assert _text(_title(runtime)) == "A Title"
    assert seen == ["Bearer ya29.fresh-warm"]

    data = json.loads((home / "auth.json").read_text(encoding="utf-8"))
    data["credential_pool"]["solstice"][0]["expires_at_ms"] = _now_ms() - 1000
    (home / "auth.json").write_text(json.dumps(data), encoding="utf-8")

    def respond(self, url, **kw):
        bearer = (kw.get("headers") or {}).get("Authorization", "")
        seen.append(bearer)
        req = httpx.Request("POST", str(url))
        if GOOGLE in str(url) and bearer.startswith("Bearer ya29.fresh-") and bearer != "Bearer ya29.fresh-warm":
            return httpx.Response(200, json=OK_BODY, request=req)
        return httpx.Response(401, request=req, json={"error": {"code": 401, "status": "UNAUTHENTICATED"}})

    monkeypatch.setattr(httpx.Client, "post", respond)
    seen.clear()
    assert _text(_title(runtime)) == "A Title"
    # A pinned aux provider leases the refreshed row; an inherited route keeps the session's key (as on main)
    # and pays one 401 before the refresh.
    assert seen == (["Bearer ya29.fresh-1"] if pinned else ["Bearer ya29.fresh-warm", "Bearer ya29.fresh-1"])
    assert len(grants) == 1


def test_pinned_aux_provider_401_refreshes_instead_of_benching(env):
    """``auxiliary.title_generation.provider: solstice`` used to bench the only credential on a 401
    (no refresh), so the next call failed with a cooldown message. It refreshes and keeps working."""
    home, grants, _seen = env
    (home / "config.yaml").write_text(
        "model:\n  provider: solstice\n  default: gemini-3.1-pro\n"
        "auxiliary:\n  title_generation:\n    provider: solstice\n    model: gemini-3.1-pro\n", encoding="utf-8")
    _write(home, _row("ya29.revoked", _now_ms() + 3_000_000))

    assert _text(_title(None)) == "A Title"
    assert _disk_row(home).get("last_status") != "exhausted"
    assert pool_cooldown_message("solstice") is None
    assert _text(_title(None)) == "A Title"
    assert len(grants) == 1


@pytest.mark.parametrize("task", ["vision", "approval"])
def test_vision_and_approval_401_refresh(env, task):
    home, grants, _seen = env
    _write(home, _row("ya29.revoked", _now_ms() + 3_000_000))
    content = ([{"type": "text", "text": "describe"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}]
               if task == "vision" else "approve?")

    resp = aux.call_llm(task=task, messages=[{"role": "user", "content": content}], max_tokens=20)

    assert _text(resp) == "A Title"
    assert len(grants) == 1
    assert _disk_row(home).get("last_status") != "exhausted"


@pytest.mark.parametrize(("code", "expected", "absent"), [
    (401, "rejected its sign-in (401)", "429"),
    (403, "rejected its sign-in (403)", "429"),
    (402, "billing error (402)", "429"),
    (429, "rate limit / quota error (429)", "sign-in"),
    (None, "failed request or token refresh", "429"),
])
def test_cooldown_message_names_the_recorded_error(env, code, expected, absent):
    """A benched credential's message comes from its own ``last_error_code``: a sign-in the provider
    rejected used to be reported as "cooling down after a rate limit / quota error (429)"."""
    home, _, _ = env
    _write(home, _row("ya29.dead", _now_ms() + 3_000_000, last_status="exhausted",
                      last_status_at=time.time(), last_error_code=code))

    message = pool_cooldown_message("solstice")

    assert message is not None and "cooling down" in message
    assert expected in message
    assert absent not in message
    if code in (401, 403):
        assert "Sign in again with `hermes auth add solstice`" in message


@pytest.mark.parametrize("session_key", [None, "ya29.revoked"])
def test_a_token_rejected_again_after_refresh_is_benched_not_refreshed_on_every_call(env, monkeypatch, session_key):
    """When Google rejects the freshly refreshed token too, the row is benched with the 401, so the
    next auxiliary call does not spend another refresh (the broker allows 30 per 5 minutes per user)."""
    home, grants, _seen = env
    monkeypatch.setattr(httpx.Client, "post", lambda self, url, **kw: httpx.Response(
        401, request=httpx.Request("POST", str(url)),
        json={"error": {"code": 401, "status": "UNAUTHENTICATED", "message": "invalid credentials"}}))
    _write(home, _row("ya29.revoked", _now_ms() + 3_000_000))
    runtime = {"provider": "solstice", "model": "gemini-3.1-pro", "base_url": BASE_URL, "api_mode": "chat_completions"}
    if session_key:
        runtime["api_key"] = session_key

    for _ in range(3):
        with pytest.raises(Exception):
            _title(runtime)

    assert len(grants) == 1
    row = _disk_row(home)
    assert row["last_status"] == "exhausted" and row["last_error_code"] == 401
    assert "rejected its sign-in (401)" in pool_cooldown_message("solstice")


def _rows(home):
    return {r["id"]: r for r in json.loads((home / "auth.json").read_text(encoding="utf-8-sig"))["credential_pool"]["solstice"]}


def test_a_newer_token_minted_by_another_process_during_the_retry_is_not_benched(env, monkeypatch):
    """The bench names the bearer the retry actually sent. If another process refreshed the row while
    that retry was in flight, the newer token is left alone and the next call uses it."""
    home, _grants, seen = env
    _write(home, _row("ya29.revoked", _now_ms() + 3_000_000))

    def respond(self, url, **kw):
        bearer = (kw.get("headers") or {}).get("Authorization", "")
        seen.append(bearer)
        req = httpx.Request("POST", str(url))
        if bearer == "Bearer ya29.fresh-1":  # our refreshed token: a peer re-mints while it is rejected
            _write(home, _row("ya29.peer", _now_ms() + 3_000_000))
        if bearer == "Bearer ya29.peer":
            return httpx.Response(200, json=OK_BODY, request=req)
        return httpx.Response(401, request=req, json={"error": {"code": 401, "status": "UNAUTHENTICATED"}})

    monkeypatch.setattr(httpx.Client, "post", respond)
    runtime = {"provider": "solstice", "model": "gemini-3.1-pro", "base_url": BASE_URL, "api_mode": "chat_completions"}

    with pytest.raises(Exception):
        _title(runtime)
    assert _disk_row(home).get("last_status") != "exhausted"
    monkeypatch.setattr(aux, "_client_cache", {})
    assert _text(_title(runtime)) == "A Title"
    assert seen[-1] == "Bearer ya29.peer"


def test_only_the_row_the_retry_used_is_benched(env, monkeypatch):
    """Session on row A; its refresh works, but the retry leases higher-priority row B, which is rejected.
    Only B is benched; A keeps its refreshed token. The call itself still fails: the inherited route
    re-sends the pinned session key, as on main (repeated-call recovery is a follow-up)."""
    home, _grants, seen = env
    row_a = _row("ya29.a-old", _now_ms() + 3_000_000, id="rowA", priority=1)
    row_b = {**_row("ya29.b-bad", _now_ms() + 3_000_000, id="rowB", priority=0), "refresh_token": "1//rt-b"}
    (home / "auth.json").write_text(json.dumps({"version": 1, "credential_pool": {"solstice": [row_b, row_a]}}),
                                    encoding="utf-8")
    runtime = {"provider": "solstice", "model": "gemini-3.1-pro", "api_key": "ya29.a-old",
               "base_url": BASE_URL, "api_mode": "chat_completions"}

    with pytest.raises(Exception):  # the session's own (stale) key stays pinned, as on main
        _title(runtime)

    rows = _rows(home)
    assert rows["rowA"].get("last_status") != "exhausted" and rows["rowA"]["access_token"] == "ya29.fresh-1"
    assert seen[1] == "Bearer ya29.b-bad"  # the retry leased B
    assert rows["rowB"].get("last_status") == "exhausted"


def test_a_peer_refresh_racing_the_bench_never_inherits_its_cooldown(env, monkeypatch):
    """Another process persists a newer token for the same row after the bench has read the pool. A 401
    bench is a verdict on the bearer that was sent, so the newer token ends up on disk without it."""
    from agent.credential_pool import CredentialPool

    home, _grants, _seen = env
    monkeypatch.setattr(httpx.Client, "post", lambda self, url, **kw: httpx.Response(
        401, request=httpx.Request("POST", str(url)),
        json={"error": {"code": 401, "status": "UNAUTHENTICATED", "message": "invalid credentials"}}))
    _write(home, _row("ya29.revoked", _now_ms() + 3_000_000))
    bench = CredentialPool.mark_exhausted_and_rotate
    raced = []

    def peer_refreshes_first(self, **kwargs):
        if not raced:  # the peer lands its refresh between our read of the pool and our bench write
            raced.append(kwargs.get("api_key_hint"))
            _write(home, _row("ya29.peer", _now_ms() + 3_000_000))
        return bench(self, **kwargs)

    monkeypatch.setattr(CredentialPool, "mark_exhausted_and_rotate", peer_refreshes_first)
    runtime = {"provider": "solstice", "model": "gemini-3.1-pro", "base_url": BASE_URL, "api_mode": "chat_completions"}

    with pytest.raises(Exception):
        _title(runtime)
    assert raced == ["ya29.fresh-1"], "the bench must name the bearer the refreshed retry sent"
    row = _disk_row(home)
    assert row["access_token"] == "ya29.peer"
    assert row.get("last_status") != "exhausted"


def test_an_availability_probe_never_spends_a_refresh(env):
    """``aux_probe_mode`` answers "is this provider resolvable?": an expired row stays as it is."""
    home, grants, _seen = env
    _write(home, _row("ya29.expired", _now_ms() - 1000))
    with aux.aux_probe_mode():
        aux._get_cached_client("solstice", "gemini-3.1-pro")
    assert grants == []
    assert _disk_row(home)["access_token"] == "ya29.expired"


def test_a_backoff_retry_benches_the_bearer_it_sent_not_the_refreshed_one(env, monkeypatch):
    """The refreshed retry hits a 429; its same-key backoff then re-sends on the first client and gets the
    401. Recovery must bench that request's bearer: the refreshed token never failed a sign-in."""
    from agent.credential_pool import CredentialPool

    home, _grants, seen = env

    def respond(self, url, **kw):
        bearer = (kw.get("headers") or {}).get("Authorization", "")
        seen.append(bearer)
        req = httpx.Request("POST", str(url))
        if "fresh" in bearer:
            return httpx.Response(429, request=req, headers={"retry-after": "0.1"},
                                  json={"error": {"code": 429, "status": "TOO_MANY", "message": "too many requests"}})
        return httpx.Response(401, request=req, json={"error": {"code": 401, "status": "UNAUTHENTICATED"}})

    monkeypatch.setattr(httpx.Client, "post", respond)
    benched = []
    bench = CredentialPool.mark_exhausted_and_rotate
    monkeypatch.setattr(CredentialPool, "mark_exhausted_and_rotate",
                        lambda self, **kw: benched.append(kw.get("api_key_hint")) or bench(self, **kw))
    _write(home, _row("ya29.revoked", _now_ms() + 3_000_000))
    runtime = {"provider": "solstice", "model": "gemini-3.1-pro", "base_url": BASE_URL, "api_mode": "chat_completions"}

    with pytest.raises(Exception):
        _title(runtime)
    assert seen == ["Bearer ya29.revoked", "Bearer ya29.fresh-1", "Bearer ya29.revoked"]
    assert "ya29.fresh-1" not in benched


def test_a_proxys_own_credential_is_never_swapped_for_the_pools_bearer(env):
    """``model.base_url`` names a proxy with its own key: the main route keeps that key, and the pooled
    Google bearer (issued for the provider's registered origin) is never sent to the proxy host."""
    home, _grants, seen = env
    proxy = "https://proxy.example.invalid/v1alpha"
    (home / "config.yaml").write_text(
        f"model:\n  provider: solstice\n  default: gemini-3.1-pro\n  base_url: {proxy}\n", encoding="utf-8")
    _write(home, _row("ya29.pooled", _now_ms() + 3_000_000))
    runtime = {"provider": "solstice", "model": "gemini-3.1-pro", "base_url": proxy, "api_key": "proxy-key",
               "api_mode": "chat_completions"}

    for rt in (runtime, {k: v for k, v in runtime.items() if k != "base_url"}):  # an omitted base_url too
        seen.clear()
        with pytest.raises(Exception):
            _title(rt)
        assert seen and all(b == "Bearer proxy-key" for b in seen), seen


def test_a_spending_limit_403_is_benched_as_billing_not_a_rejected_sign_in(env, monkeypatch):
    """A 403 spending limit is account-wide: benched as billing (kept across a peer's refresh) and never
    reported as a rejected sign-in the user should fix by signing in again."""
    home, _grants, _seen = env
    monkeypatch.setattr(httpx.Client, "post", lambda self, url, **kw: httpx.Response(
        403, request=httpx.Request("POST", str(url)),
        json={"error": {"code": 403, "status": "PERMISSION_DENIED", "message": "Your spending limit has been reached: check your billing."}}))
    _write(home, _row("ya29.ok", _now_ms() + 3_000_000))
    runtime = {"provider": "solstice", "model": "gemini-3.1-pro", "base_url": BASE_URL, "api_mode": "chat_completions"}

    with pytest.raises(Exception):
        _title(runtime)
    row = _disk_row(home)
    assert row["last_status"] == "exhausted" and row["failure_reason"] == "billing"
    assert "Sign in again" not in pool_cooldown_message("solstice")


def test_a_per_minute_quota_429_stays_a_rate_limit_not_billing(env, monkeypatch):
    """Google's per-minute ``RESOURCE_EXHAUSTED`` matches the payment keywords, but the main-agent
    classifier calls it a rate limit: benching it as billing would lock a sole sign-in out for an hour
    and tell the user to add credits."""
    from hermes_cli.auth import is_rate_limited_auth_error
    from hermes_cli.runtime_provider import resolve_runtime_provider

    home, _grants, _seen = env
    monkeypatch.setattr(httpx.Client, "post", lambda self, url, **kw: httpx.Response(
        429, request=httpx.Request("POST", str(url)), json={"error": {
            "code": 429, "status": "RESOURCE_EXHAUSTED", "message": "Resource has been exhausted (e.g. check quota)."}}))
    _write(home, _row("ya29.ok", _now_ms() + 3_000_000))
    runtime = {"provider": "solstice", "model": "gemini-3.1-pro", "base_url": BASE_URL, "api_mode": "chat_completions"}

    with pytest.raises(Exception):
        _title(runtime)
    row = _disk_row(home)
    assert row.get("failure_reason") != "billing"
    with pytest.raises(Exception) as excinfo:
        resolve_runtime_provider(requested="solstice", target_model="gemini-3.1-pro")
    assert is_rate_limited_auth_error(excinfo.value)


@pytest.mark.parametrize("async_mode", [False, True])
def test_the_refreshed_bearer_is_retried_only_at_the_endpoint_that_rejected_it(env, monkeypatch, async_mode):
    """The session's frozen runtime targets Google while config.yaml now names a proxy: the refresh retry
    goes back to the endpoint that sent the 401, never to the newly configured host."""
    import asyncio

    home, _grants, _seen = env
    sent = []
    respond = httpx.Client.post
    monkeypatch.setattr(httpx.Client, "post", lambda self, url, **kw: sent.append(str(url)) or respond(self, url, **kw))
    (home / "config.yaml").write_text("model:\n  provider: solstice\n  default: gemini-3.1-pro\n"
                                      "  base_url: https://proxy.example.invalid/v1alpha\n", encoding="utf-8")
    _write(home, _row("ya29.revoked", _now_ms() + 3_000_000))
    runtime = {"provider": "solstice", "model": "gemini-3.1-pro", "base_url": BASE_URL, "api_mode": "chat_completions"}
    kwargs = dict(task="title_generation", main_runtime=runtime,
                  messages=[{"role": "user", "content": "name this chat"}], max_tokens=20)
    with contextlib.suppress(Exception):
        asyncio.run(aux.async_call_llm(**kwargs)) if async_mode else aux.call_llm(**kwargs)
    assert len(sent) >= 2 and all(url.startswith(BASE_URL) for url in sent), sent
