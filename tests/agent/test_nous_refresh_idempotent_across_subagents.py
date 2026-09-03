"""Nous OAuth 401 recovery must be idempotent across concurrent subagents.

Many ``delegate_task`` children share the parent's credential pool (and its
lock). When the Nous access token expires mid-flight every child 401s at the
same instant. Before this change each child's recovery forced a refresh-token
rotation against ``POST /api/oauth/token`` even though a sibling had rotated
the grant milliseconds earlier: N children + parent = N+1 sequential POSTs.
Several Hermes processes on one egress IP then trip the portal's WAF rule
(25 POSTs/IP/60s), the WAF 429 body is HTML, and the client turned any
non-JSON refresh error into ``relogin_required=True`` — users were told to
re-authenticate over a perfectly valid credential.

Invariants pinned here:

* One pool, 1 parent + 10 children, all 401 on the same bearer → exactly ONE
  POST, and all eleven end up on the new token.
* Two processes (two pools, one auth.json + shared store): the loser of the
  lock race sees the winner's rotation and skips its POST.
* The stored token is still the failed one → exactly one POST (no more, no
  fewer).
* 429 ``slow_down`` / 429 HTML+``Retry-After`` / 5xx are transient: retried
  with backoff, never ``relogin_required``, never quarantined.
* 400 ``invalid_grant`` stays terminal: one POST, quarantined,
  ``relogin_required=True`` — unchanged.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import httpx
import pytest

import hermes_cli.auth as auth_mod
from agent.credential_pool import load_pool
from tests.hermes_cli.test_auth_nous_provider import _invoke_jwt, _setup_nous_auth


# ── Fixtures / helpers ───────────────────────────────────────────────────────


def _iso_in(seconds: float) -> str:
    return datetime.fromtimestamp(
        auth_mod.time.time() + seconds, tz=timezone.utc
    ).isoformat()


@pytest.fixture
def nous_home(tmp_path, monkeypatch):
    """A HERMES_HOME + shared auth dir holding one EXPIRED Nous grant."""
    hermes_home = tmp_path / "hermes"
    shared_dir = tmp_path / "shared"
    shared_dir.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HERMES_SHARED_AUTH_DIR", str(shared_dir))
    # Retry delays are exercised by asserting on the sleep calls, not by
    # actually sleeping.
    monkeypatch.setattr(auth_mod, "_nous_refresh_backoff_sleep", lambda _s: None)

    expired = _invoke_jwt(seconds=-30)
    _setup_nous_auth(
        hermes_home,
        access_token=expired,
        refresh_token="rt-0",
        scope=auth_mod.DEFAULT_NOUS_SCOPE,
        expires_at=_iso_in(-30),
        expires_in=0,
    )
    return SimpleNamespace(home=hermes_home, shared=shared_dir, expired=expired)


class _TokenEndpoint:
    """Scripted stand-in for ``POST /api/oauth/token``.

    ``script`` is a list of responses handed out in order; the last one is
    repeated once the script is exhausted. Each entry is either
    ``("ok", n)`` → 200 with a fresh rotated pair, ``("json", status, body,
    headers)`` or ``("raw", status, text, headers)``.
    """

    def __init__(self, script: List[tuple]):
        self.script = list(script)
        self.posts: List[str] = []
        self.issued: List[str] = []
        self._lock = threading.Lock()

    def install(self, monkeypatch) -> "_TokenEndpoint":
        endpoint = self

        def _fake_refresh(*, client, portal_base_url, client_id, refresh_token):
            with endpoint._lock:
                endpoint.posts.append(refresh_token)
                spec = endpoint.script.pop(0) if len(endpoint.script) > 1 else endpoint.script[0]
                idx = len(endpoint.posts)
            kind = spec[0]
            if kind == "ok":
                token = _invoke_jwt(seconds=3600)
                with endpoint._lock:
                    endpoint.issued.append(token)
                return {
                    "access_token": token,
                    "refresh_token": f"rt-{idx}",
                    "expires_in": 3600,
                    "token_type": "Bearer",
                    "scope": auth_mod.DEFAULT_NOUS_SCOPE,
                }
            status = spec[1]
            headers = spec[3] if len(spec) > 3 else {}
            request = httpx.Request("POST", f"{portal_base_url}/api/oauth/token")
            if kind == "json":
                response = httpx.Response(status, json=spec[2], headers=headers, request=request)
            else:
                response = httpx.Response(status, text=spec[2], headers=headers, request=request)
            # Route through the REAL classifier so the test exercises the
            # status/body → AuthError mapping, not a hand-built exception.
            return _real_refresh_access_token_from_response(response)

        monkeypatch.setattr(auth_mod, "_refresh_access_token", _fake_refresh)
        return self


def _real_refresh_access_token_from_response(response: httpx.Response) -> Dict[str, Any]:
    class _Client:
        def post(self, *_a, **_k):
            return response

    return _ORIGINAL_REFRESH(
        client=_Client(),
        portal_base_url="https://portal.example.com",
        client_id="hermes-cli",
        refresh_token="rt-irrelevant",
    )


_ORIGINAL_REFRESH = auth_mod._refresh_access_token


def _stored_nous_state(home: Path) -> Dict[str, Any]:
    return json.loads((home / "auth.json").read_text())["providers"]["nous"]


def _make_agent(pool, api_key: str, *, bind_entry_id: bool = True):
    """Minimal agent double for ``recover_with_credential_pool``.

    ``bind_entry_id`` mirrors production: the parent binds
    ``_credential_pool_entry_id`` via ``sync_credential_pool_entry_id`` at
    init and every child via ``_swap_credential`` when it leases the pool.
    """
    entry_id = None
    if bind_entry_id:
        entry = next((e for e in pool.entries() if e.access_token == api_key), None)
        entry_id = entry.id if entry is not None else None
    agent = SimpleNamespace(
        provider="nous",
        api_mode="chat_completions",
        base_url="https://inference.example.com/v1",
        api_key=api_key,
        _credential_pool=pool,
        _credential_pool_entry_id=entry_id,
        _auth_pool_refresh_counts={},
        _fallback_activated=False,
    )
    agent._is_entitlement_failure = lambda *_a, **_k: False

    def _swap(entry):
        agent.api_key = entry.runtime_api_key or entry.access_token
        agent._credential_pool_entry_id = entry.id

    agent._swap_credential = _swap
    return agent


def _recover_401(agent):
    from agent.agent_runtime_helpers import recover_with_credential_pool

    recovered, _ = recover_with_credential_pool(
        agent, status_code=401, has_retried_429=False
    )
    return recovered


# ── 1 parent + 10 children, one pool ─────────────────────────────────────────


def test_eleven_agents_one_pool_one_post(nous_home, monkeypatch, caplog):
    endpoint = _TokenEndpoint([("ok",)]).install(monkeypatch)
    pool = load_pool("nous")
    assert len(pool.entries()) == 1

    agents = [_make_agent(pool, nous_home.expired) for _ in range(11)]
    results: Dict[int, bool] = {}
    errors: List[BaseException] = []
    start = threading.Barrier(len(agents))

    def _run(idx: int):
        try:
            start.wait(timeout=10)
            results[idx] = _recover_401(agents[idx])
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    with caplog.at_level(logging.INFO):
        threads = [threading.Thread(target=_run, args=(i,)) for i in range(len(agents))]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

    assert not errors, errors
    assert endpoint.posts == ["rt-0"], f"expected exactly one POST, got {endpoint.posts}"
    assert all(results[i] for i in range(len(agents))), results
    new_token = endpoint.issued[0]
    assert {a.api_key for a in agents} == {new_token}
    assert _stored_nous_state(nous_home.home)["refresh_token"] == "rt-1"
    assert "adopting token rotated by concurrent refresh, skipping POST" in caplog.text


# ── Two processes sharing auth.json ──────────────────────────────────────────


def test_second_process_adopts_rotation_after_lock(nous_home, monkeypatch):
    """Two pools loaded from the same auth.json model two Hermes processes.

    The first refresh POSTs and persists. The second pool still carries the
    expired bearer in memory; its forced refresh must re-read the store
    under the lock, see the rotated token and skip its POST.
    """
    endpoint = _TokenEndpoint([("ok",)]).install(monkeypatch)
    pool_a = load_pool("nous")
    pool_b = load_pool("nous")

    refreshed_a = pool_a.try_refresh_matching(api_key_hint=nous_home.expired)
    assert refreshed_a is not None
    assert endpoint.posts == ["rt-0"]

    # Process B is still holding the pre-rotation entry.
    assert pool_b.entries()[0].access_token == nous_home.expired
    refreshed_b = pool_b.try_refresh_matching(api_key_hint=nous_home.expired)

    assert endpoint.posts == ["rt-0"], "process B must adopt, not re-rotate"
    assert refreshed_b is not None
    assert refreshed_b.runtime_api_key == refreshed_a.runtime_api_key == endpoint.issued[0]


def test_second_process_races_for_lock_and_skips_post(nous_home, monkeypatch):
    """Same as above but genuinely concurrent: B blocks on the flock while A
    is mid-refresh, then adopts under the lock."""
    endpoint = _TokenEndpoint([("ok",)]).install(monkeypatch)
    pools = [load_pool("nous") for _ in range(4)]
    barrier = threading.Barrier(len(pools))
    out: Dict[int, Any] = {}

    def _run(i):
        barrier.wait(timeout=10)
        out[i] = pools[i].try_refresh_matching(api_key_hint=nous_home.expired)

    threads = [threading.Thread(target=_run, args=(i,)) for i in range(len(pools))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    assert endpoint.posts == ["rt-0"]
    assert {getattr(out[i], "runtime_api_key", None) for i in out} == {endpoint.issued[0]}


# ── Stored token is STILL the failed one → exactly one POST ──────────────────


def test_stale_stored_token_posts_exactly_once(nous_home, monkeypatch):
    endpoint = _TokenEndpoint([("ok",)]).install(monkeypatch)
    pool = load_pool("nous")
    agent = _make_agent(pool, nous_home.expired)

    assert _recover_401(agent) is True
    assert endpoint.posts == ["rt-0"]
    assert agent.api_key == endpoint.issued[0]


def test_expired_bearer_is_attributed_without_entry_id(nous_home, monkeypatch):
    """``runtime_api_key`` masks an expired invoke JWT as ``""``; the 401'd
    bearer must still resolve to its pool row when no entry id is bound,
    or the refresh never runs and the single-entry pool "rotates" to nothing."""
    endpoint = _TokenEndpoint([("ok",)]).install(monkeypatch)
    pool = load_pool("nous")
    assert pool.entries()[0].runtime_api_key == ""
    agent = _make_agent(pool, nous_home.expired, bind_entry_id=False)

    assert _recover_401(agent) is True
    assert endpoint.posts == ["rt-0"]
    assert agent.api_key == endpoint.issued[0]


def test_resolver_still_posts_when_store_holds_failed_token(nous_home, monkeypatch):
    endpoint = _TokenEndpoint([("ok",)]).install(monkeypatch)
    creds = auth_mod.resolve_nous_runtime_credentials(
        force_refresh=True, stale_access_token=nous_home.expired
    )
    assert endpoint.posts == ["rt-0"]
    assert creds["api_key"] == endpoint.issued[0]


# ── Transient token-endpoint failures ────────────────────────────────────────


def test_slow_down_then_ok_is_retried_without_relogin(nous_home, monkeypatch):
    endpoint = _TokenEndpoint(
        [
            ("json", 429, {"error": "slow_down", "error_description": "Refresh already in progress"}),
            ("ok",),
        ]
    ).install(monkeypatch)

    creds = auth_mod.resolve_nous_runtime_credentials(
        force_refresh=True, stale_access_token=nous_home.expired
    )

    assert endpoint.posts == ["rt-0", "rt-0"]
    assert creds["api_key"] == endpoint.issued[0]
    state = _stored_nous_state(nous_home.home)
    assert state["refresh_token"] == "rt-2", "rotated pair must be persisted"
    assert "last_auth_error" not in state, "transient failure must not quarantine"


def test_slow_down_retry_adopts_sibling_rotation_instead_of_reposting(nous_home, monkeypatch):
    """Between the 429 and the retry a sibling persisted a rotation: the
    retry must re-run the stale check and adopt, not POST again."""
    fresh = _invoke_jwt(seconds=3600)
    calls: List[str] = []

    def _refresh(*, client, portal_base_url, client_id, refresh_token):
        calls.append(refresh_token)
        # Simulate the sibling winning: rewrite auth.json with a rotated pair
        # while we are "waiting" for our own retry.
        state = _stored_nous_state(nous_home.home)
        store = json.loads((nous_home.home / "auth.json").read_text())
        state.update(
            access_token=fresh,
            refresh_token="rt-sibling",
            expires_at=_iso_in(3600),
            expires_in=3600,
        )
        store["providers"]["nous"] = state
        (nous_home.home / "auth.json").write_text(json.dumps(store))
        request = httpx.Request("POST", "https://portal.example.com/api/oauth/token")
        return _real_refresh_access_token_from_response(
            httpx.Response(
                429,
                json={"error": "slow_down", "error_description": "Refresh already in progress"},
                request=request,
            )
        )

    monkeypatch.setattr(auth_mod, "_refresh_access_token", _refresh)

    creds = auth_mod.resolve_nous_runtime_credentials(
        force_refresh=True, stale_access_token=nous_home.expired
    )

    assert calls == ["rt-0"], "second attempt must adopt the sibling's token, not POST"
    assert creds["api_key"] == fresh


def test_waf_html_429_with_retry_after_is_transient(nous_home, monkeypatch):
    endpoint = _TokenEndpoint(
        [("raw", 429, "<html><body>Too Many Requests</body></html>", {"Retry-After": "3"})]
    ).install(monkeypatch)
    sleeps: List[float] = []
    monkeypatch.setattr(auth_mod, "_nous_refresh_backoff_sleep", sleeps.append)

    with pytest.raises(auth_mod.AuthError) as exc_info:
        auth_mod.resolve_nous_runtime_credentials(
            force_refresh=True, stale_access_token=nous_home.expired
        )

    exc = exc_info.value
    assert exc.relogin_required is False
    assert exc.retry_after == 3
    assert exc.code == auth_mod.NOUS_REFRESH_TRANSIENT_CODE
    assert not auth_mod._is_terminal_nous_refresh_error(exc)
    assert auth_mod._is_transient_nous_refresh_error(exc)
    # Bounded retries, honouring Retry-After for the delay.
    assert len(endpoint.posts) == auth_mod._NOUS_REFRESH_TRANSIENT_MAX_ATTEMPTS
    assert sleeps == [3.0] * (auth_mod._NOUS_REFRESH_TRANSIENT_MAX_ATTEMPTS - 1)
    state = _stored_nous_state(nous_home.home)
    assert state["refresh_token"] == "rt-0", "grant must be intact"
    assert "last_auth_error" not in state


def test_waf_retry_after_beyond_budget_surfaces_without_stalling(nous_home, monkeypatch):
    endpoint = _TokenEndpoint(
        [("raw", 429, "<html>blocked</html>", {"Retry-After": "60"})]
    ).install(monkeypatch)
    sleeps: List[float] = []
    monkeypatch.setattr(auth_mod, "_nous_refresh_backoff_sleep", sleeps.append)

    with pytest.raises(auth_mod.AuthError) as exc_info:
        auth_mod.resolve_nous_runtime_credentials(force_refresh=True)

    assert exc_info.value.retry_after == 60
    assert exc_info.value.relogin_required is False
    assert endpoint.posts == ["rt-0"], "a 60s Retry-After must not be waited out in-turn"
    assert sleeps == []


def test_500_then_ok_is_retried_and_not_quarantined(nous_home, monkeypatch):
    endpoint = _TokenEndpoint(
        [("json", 500, {"error": "server_error", "error_description": "db blip"}), ("ok",)]
    ).install(monkeypatch)
    sleeps: List[float] = []
    monkeypatch.setattr(auth_mod, "_nous_refresh_backoff_sleep", sleeps.append)

    creds = auth_mod.resolve_nous_runtime_credentials(force_refresh=True)

    assert endpoint.posts == ["rt-0", "rt-0"]
    assert sleeps == [1.0]
    assert creds["api_key"] == endpoint.issued[0]
    assert "last_auth_error" not in _stored_nous_state(nous_home.home)


def test_transient_failure_does_not_bench_pool_entry(nous_home, monkeypatch):
    endpoint = _TokenEndpoint([("raw", 503, "Service Unavailable")]).install(monkeypatch)
    pool = load_pool("nous")
    entry_before = pool.entries()[0]

    result = pool.try_refresh_matching(api_key_hint=nous_home.expired)

    assert len(endpoint.posts) == auth_mod._NOUS_REFRESH_TRANSIENT_MAX_ATTEMPTS
    assert result is not None and result.id == entry_before.id
    assert pool.entries()[0].last_status is None, "transient failure must not exhaust the entry"
    assert pool.entries()[0].refresh_token == "rt-0"
    assert len(pool.entries()) == 1, "singleton entry must not be quarantined"


# ── Terminal errors are unchanged ────────────────────────────────────────────


def test_invalid_grant_is_terminal_one_post_quarantined(nous_home, monkeypatch):
    endpoint = _TokenEndpoint(
        [("json", 400, {"error": "invalid_grant", "error_description": "Refresh token expired"})]
    ).install(monkeypatch)
    sleeps: List[float] = []
    monkeypatch.setattr(auth_mod, "_nous_refresh_backoff_sleep", sleeps.append)

    with pytest.raises(auth_mod.AuthError) as exc_info:
        auth_mod.resolve_nous_runtime_credentials(
            force_refresh=True, stale_access_token=nous_home.expired
        )

    exc = exc_info.value
    assert exc.code == "invalid_grant"
    assert exc.relogin_required is True
    assert auth_mod._is_terminal_nous_refresh_error(exc)
    assert endpoint.posts == ["rt-0"], "terminal errors are never retried"
    assert sleeps == []
    state = _stored_nous_state(nous_home.home)
    assert not state.get("refresh_token"), "dead grant must be quarantined"
    assert state.get("last_auth_error", {}).get("relogin_required") is True


def test_invalid_grant_via_pool_removes_singleton_entry(nous_home, monkeypatch):
    endpoint = _TokenEndpoint(
        [("json", 400, {"error": "invalid_grant", "error_description": "revoked"})]
    ).install(monkeypatch)
    pool = load_pool("nous")

    assert pool.try_refresh_matching(api_key_hint=nous_home.expired) is None
    assert endpoint.posts == ["rt-0"]
    assert pool.entries() == []


# ── Classifier unit checks (real response objects, real parser) ──────────────


@pytest.mark.parametrize(
    "status, body, headers, expected_retry_after",
    [
        (429, {"error": "slow_down", "error_description": "in progress"}, {}, None),
        (429, "<html>waf</html>", {"Retry-After": "7"}, 7),
        (502, "<html>bad gateway</html>", {}, None),
        (500, {"error": "server_error"}, {"Retry-After": "2"}, 2),
        (503, "", {}, None),
    ],
)
def test_refresh_access_token_transient_classification(status, body, headers, expected_retry_after):
    request = httpx.Request("POST", "https://portal.example.com/api/oauth/token")
    if isinstance(body, dict):
        response = httpx.Response(status, json=body, headers=headers, request=request)
    else:
        response = httpx.Response(status, text=body, headers=headers, request=request)

    with pytest.raises(auth_mod.AuthError) as exc_info:
        _real_refresh_access_token_from_response(response)

    exc = exc_info.value
    assert exc.code == auth_mod.NOUS_REFRESH_TRANSIENT_CODE
    assert exc.relogin_required is False
    assert exc.retry_after == expected_retry_after
    assert auth_mod._is_transient_nous_refresh_error(exc)
    assert not auth_mod._is_terminal_nous_refresh_error(exc)


def test_refresh_access_token_non_json_4xx_never_demands_relogin():
    """A 403 HTML page from a CDN/WAF is not the portal saying the grant is dead."""
    request = httpx.Request("POST", "https://portal.example.com/api/oauth/token")
    response = httpx.Response(403, text="<html>Forbidden</html>", request=request)

    with pytest.raises(auth_mod.AuthError) as exc_info:
        _real_refresh_access_token_from_response(response)

    assert exc_info.value.relogin_required is False
    assert not auth_mod._is_terminal_nous_refresh_error(exc_info.value)


@pytest.mark.parametrize("code", ["invalid_grant", "invalid_token", "refresh_token_reused"])
def test_refresh_access_token_terminal_codes_unchanged(code):
    request = httpx.Request("POST", "https://portal.example.com/api/oauth/token")
    response = httpx.Response(400, json={"error": code, "error_description": "nope"}, request=request)

    with pytest.raises(auth_mod.AuthError) as exc_info:
        _real_refresh_access_token_from_response(response)

    assert exc_info.value.code == code
    assert exc_info.value.relogin_required is True
    assert auth_mod._is_terminal_nous_refresh_error(exc_info.value)
    assert not auth_mod._is_transient_nous_refresh_error(exc_info.value)


# ── Proxy adapter ────────────────────────────────────────────────────────────


def test_proxy_retry_credential_adopts_rotated_token_without_post(nous_home, monkeypatch):
    from hermes_cli.proxy.adapters.base import UpstreamCredential
    from hermes_cli.proxy.adapters.nous_portal import NousPortalAdapter

    endpoint = _TokenEndpoint([("ok",)]).install(monkeypatch)
    adapter = NousPortalAdapter()

    failed = UpstreamCredential(bearer=nous_home.expired, base_url="https://inference.example.com/v1")
    first = adapter.get_retry_credential(failed_credential=failed, status_code=401)
    assert endpoint.posts == ["rt-0"]
    assert first is not None and first.bearer == endpoint.issued[0]

    # A second request that 401'd on the OLD bearer arrives after rotation.
    second = adapter.get_retry_credential(failed_credential=failed, status_code=401)
    assert endpoint.posts == ["rt-0"], "proxy must adopt the rotated token, not re-POST"
    assert second is not None and second.bearer == first.bearer


def test_proxy_retry_credential_posts_when_stored_token_is_the_failed_one(nous_home, monkeypatch):
    from hermes_cli.proxy.adapters.base import UpstreamCredential
    from hermes_cli.proxy.adapters.nous_portal import NousPortalAdapter

    endpoint = _TokenEndpoint([("ok",)]).install(monkeypatch)
    adapter = NousPortalAdapter()
    failed = UpstreamCredential(bearer=nous_home.expired, base_url="https://inference.example.com/v1")

    cred = adapter.get_retry_credential(failed_credential=failed, status_code=401)

    assert endpoint.posts == ["rt-0"]
    assert cred is not None and cred.bearer == endpoint.issued[0]
