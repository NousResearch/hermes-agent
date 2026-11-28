"""Pool-only Codex resolution refreshes the selected grant, not another account (#113694)."""

import base64
import json
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from threading import Barrier, Event, local
from urllib.parse import parse_qs

import httpx
import pytest

from agent import credential_pool
from hermes_cli import auth, auth_codex
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


def _jwt(seconds_left, subject="selected"):
    claims = {"sub": subject, "exp": int(time.time()) + seconds_left}
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).rstrip(b"=").decode()
    return f"e30.{payload}.sig"


def _pool_home(tmp_path, monkeypatch, token, base_url):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "no-codex-cli"))
    monkeypatch.delenv("HERMES_CODEX_BASE_URL", raising=False)
    monkeypatch.setattr(auth, "_codex_quota_probe_cache", {})
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "no-real-home")
    monkeypatch.setattr(auth, "_global_auth_file_path", lambda: None)
    monkeypatch.setattr(credential_pool, "_global_auth_file_path", lambda: None)
    rows = [
        {"id": "selected", "source": "manual:device_code", "auth_type": "oauth",
         "priority": 0, "access_token": token, "refresh_token": "selected-refresh",
         "base_url": base_url, "last_status": "ok"},
        {"id": "other", "source": "manual:device_code", "auth_type": "oauth",
         "priority": 1, "access_token": _jwt(86400, "other"),
         "refresh_token": "other-refresh", "base_url": auth.DEFAULT_CODEX_BASE_URL},
    ]
    path = home / "auth.json"
    path.write_text(json.dumps({"version": 1, "providers": {},
                               "credential_pool": {"openai-codex": rows}}), encoding="utf-8")
    return path, rows


def _mock_token_endpoint(monkeypatch, handler):
    client = httpx.Client
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(auth.httpx, "Client", lambda **kwargs: client(transport=transport, **kwargs))


def _overlapping_resolutions(monkeypatch, homes):
    # Instrument scheduling only: real reads, locking, POST and persistence remain in use.
    real_read = auth_codex._pool_codex_credential
    first_reads, caller = Barrier(2), local()

    def overlapping_read():
        result = real_read()
        if not getattr(caller, "read_once", False):
            caller.read_once = True
            first_reads.wait(timeout=10)
        return result

    def resolve(home):
        scope = set_hermes_home_override(home)
        try:
            return auth.resolve_codex_runtime_credentials()
        finally:
            reset_hermes_home_override(scope)

    monkeypatch.setattr(auth_codex, "_pool_codex_credential", overlapping_read)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(resolve, home) for home in homes]
        results = [future.result(timeout=15) for future in futures]
    monkeypatch.setattr(auth_codex, "_pool_codex_credential", real_read)
    return results


def _resolutions_during_post(monkeypatch, post_started, waiter_started):
    monkeypatch.setenv("HERMES_CODEX_REFRESH_TIMEOUT_SECONDS", "40")
    real_lock, caller, budgets = auth._auth_store_lock, local(), []

    @contextmanager
    def observed_lock(timeout_seconds=auth.AUTH_LOCK_TIMEOUT_SECONDS, **kwargs):
        if getattr(caller, "observe_first", False):
            caller.observe_first = False
            budgets.append(timeout_seconds)
            waiter_started.set()
        with real_lock(timeout_seconds=timeout_seconds, **kwargs):
            yield

    def late_resolve():
        caller.observe_first = True
        return auth.resolve_codex_runtime_credentials()

    monkeypatch.setattr(auth, "_auth_store_lock", observed_lock)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(auth.resolve_codex_runtime_credentials)
        assert post_started.wait(timeout=10)
        second = executor.submit(late_resolve)
        results = [future.result(timeout=15) for future in (first, second)]
    # Assert the real lock's configured budget rather than sleeping through a slow token POST.
    assert budgets == [max(float(auth.AUTH_LOCK_TIMEOUT_SECONDS), 45.0)]
    return results


@pytest.mark.parametrize("base_url", [
    auth.DEFAULT_CODEX_BASE_URL, "https://codex-gateway.example/backend-api/codex",
], ids=["direct", "gateway"])
@pytest.mark.parametrize("seconds_left,options,refresh,cooldown,mode", [
    pytest.param(60, {}, True, False, "oauth", id="expiring"),
    pytest.param(-60, {}, True, False, "oauth", id="expired"),
    pytest.param(60, {"refresh_if_expiring": False}, False, False, "oauth", id="opt-out"),
    pytest.param(86400, {}, False, False, "oauth", id="fresh"),
    pytest.param(86400, {"force_refresh": True, "refresh_if_expiring": False}, True, False,
                 "oauth", id="force-wins-over-opt-out"),
    pytest.param(60, {"read_only": True}, False, False, "oauth", id="read-only"),
    pytest.param(60, {"read_only": True, "force_refresh": True}, False, False,
                 "oauth", id="read-only-wins-over-force"),
    pytest.param(60, {"refresh_skew_seconds": 0}, False, False, "oauth", id="outside-custom-skew"),
    pytest.param(600, {"refresh_skew_seconds": 900}, True, False, "oauth", id="inside-custom-skew"),
    pytest.param(None, {}, False, False, "oauth", id="opaque-oauth-bearer"),
    pytest.param(60, {}, True, True, "oauth", id="quota-restored"),
    pytest.param(60, {}, False, False, "api-key", id="jwt-api-key"),
    pytest.param(None, {}, False, False, "api-key", id="opaque-api-key"),
    pytest.param(60, {}, False, False, "access-only", id="no-refresh-grant"),
    pytest.param(60, {}, True, False, "concurrent", id="concurrent-local"),
    pytest.param(60, {}, True, False, "borrowed-concurrent", id="concurrent-borrowed-root"),
    pytest.param(60, {}, True, False, "late-concurrent", id="arrives-during-refresh"),
])
def test_pool_only_refresh_policy_preserves_grant_and_route(
    tmp_path, monkeypatch, base_url, seconds_left, options, refresh, cooldown, mode,
):
    """Real resolution, pool selection and disk writes share the caller's refresh policy."""
    token = _jwt(seconds_left) if seconds_left is not None else "opaque-gateway-key"
    path, rows = _pool_home(tmp_path, monkeypatch, token, base_url)
    if mode in {"api-key", "access-only"}:
        rows[0].pop("refresh_token")
        if mode == "api-key":
            rows[0].update(auth_type="api_key", source="manual:api_key")
        path.write_text(json.dumps({"version": 1, "providers": {},
                                   "credential_pool": {"openai-codex": rows}}), encoding="utf-8")
    profile_paths = []
    if mode == "borrowed-concurrent":
        profile_paths = [path, tmp_path / "second-profile" / "auth.json"]
        root_path = tmp_path / "root" / "auth.json"
        root_path.parent.mkdir()
        path.rename(root_path)
        for profile_path in profile_paths:
            profile_path.parent.mkdir(exist_ok=True)
            profile_path.write_text(json.dumps({"version": 1, "providers": {}}), encoding="utf-8")
        monkeypatch.setattr(auth, "_global_auth_file_path", lambda: root_path)
        monkeypatch.setattr(credential_pool, "_global_auth_file_path", lambda: root_path)
        path = root_path
    if cooldown:
        rows[0].update(last_status="exhausted", last_error_code=429,
                       last_error_reason="quota_exhausted", last_error_reset_at=time.time() + 3600)
        # Keep the second account in its own cooldown so the first lookup reaches the quota probe.
        rows[1]["last_error_reset_at"] = time.time() + 7200
        path.write_text(json.dumps({"version": 1, "providers": {},
                                   "credential_pool": {"openai-codex": rows}}), encoding="utf-8")
    before = path.read_bytes()
    fresh = _jwt(86400)
    refresh_requests = []
    probes = []
    post_started, waiter_started = Event(), Event()

    def handle(request):
        if request.method == "GET":
            assert cooldown
            assert str(request.url).startswith(base_url.removesuffix("/codex"))
            assert request.headers["Authorization"] == f"Bearer {token}"
            probes.append(request)
            return httpx.Response(200, json={"rate_limit": {"primary_window": {"used_percent": 0}}})
        assert refresh
        assert str(request.url) == auth.CODEX_OAUTH_TOKEN_URL
        fields = parse_qs(request.content.decode())
        assert fields["refresh_token"] == [rows[0]["refresh_token"]]
        assert fields["grant_type"] == ["refresh_token"]
        if profile_paths:
            assert getattr(auth._auth_lock_holder_for(path), "depth", 0) > 0
        if mode == "late-concurrent":
            post_started.set()
            assert waiter_started.wait(timeout=10)
        refresh_requests.append(request)
        if len(refresh_requests) > 1:
            return httpx.Response(400, json={"error": "refresh_token_reused"})
        return httpx.Response(200, json={"access_token": fresh, "refresh_token": "rotated-refresh"})

    _mock_token_endpoint(monkeypatch, handle)
    if "concurrent" in mode:
        if mode == "late-concurrent":
            results = _resolutions_during_post(monkeypatch, post_started, waiter_started)
        else:
            homes = [p.parent for p in profile_paths] if profile_paths else [path.parent] * 2
            results = _overlapping_resolutions(monkeypatch, homes)
        assert results[0] == results[1]
        resolved = results[0]
    else:
        resolved = auth.resolve_codex_runtime_credentials(**options)

    assert resolved["api_key"] == (fresh if refresh else token)
    assert resolved["source"] == "credential_pool"
    assert resolved["base_url"] == base_url
    assert len(refresh_requests) == int(refresh)
    assert bool(probes) == cooldown
    stored = json.loads(path.read_text(encoding="utf-8"))
    by_id = {row["id"]: row for row in stored["credential_pool"]["openai-codex"]}
    assert by_id["other"]["access_token"] == rows[1]["access_token"]
    assert by_id["other"]["refresh_token"] == rows[1]["refresh_token"]
    assert stored["providers"] == {}
    if refresh:
        assert by_id["selected"]["access_token"] == fresh
        assert by_id["selected"]["refresh_token"] == "rotated-refresh"
        # A second resolution must reuse the committed generation without spending it again.
        assert auth.resolve_codex_runtime_credentials()["api_key"] == fresh
        assert len(refresh_requests) == 1
    else:
        assert path.read_bytes() == before
    if options.get("read_only"):
        assert not (path.parent / "auth.lock").exists()
    for profile_path in profile_paths:
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        assert not profile.get("credential_pool", {}).get("openai-codex")
        assert not profile["providers"].get("openai-codex", {}).get("tokens")


@pytest.mark.parametrize("status", [400, 503], ids=["revoked-grant", "temporary-failure"])
def test_failed_pool_only_proactive_refresh_never_returns_stale_bearer(tmp_path, monkeypatch, status):
    """A failed refresh must not silently reuse the near-expiry bearer or another grant."""
    token = _jwt(60)
    path, rows = _pool_home(tmp_path, monkeypatch, token, auth.DEFAULT_CODEX_BASE_URL)
    requests = []

    def handle(request):
        assert str(request.url) == auth.CODEX_OAUTH_TOKEN_URL
        assert parse_qs(request.content.decode())["refresh_token"] == [rows[0]["refresh_token"]]
        requests.append(request)
        error = "invalid_grant" if status == 400 else "temporarily_unavailable"
        return httpx.Response(status, json={"error": error})

    _mock_token_endpoint(monkeypatch, handle)
    with pytest.raises(auth.AuthError) as raised:
        auth.resolve_codex_runtime_credentials()

    assert raised.value.relogin_required is (status == 400)
    assert len(requests) == 1
    store = json.loads(path.read_text(encoding="utf-8"))
    other = next(row for row in store["credential_pool"]["openai-codex"] if row["id"] == "other")
    assert other["access_token"] == rows[1]["access_token"]
    assert other["refresh_token"] == rows[1]["refresh_token"]
    assert not store["providers"].get("openai-codex", {}).get("tokens")
