"""Tests for the Codex upstream-quota-restored probe and cooldown clearing.

Covers issue #43747 (externally-reset variant): Codex 429s persist a
``last_error_reset_at`` that can be days in the future, but the upstream
window can reopen early (banked reset redeemed, plan upgrade, upstream
reset).  Hermes must detect that and lift the stale local cooldown instead
of refusing requests until re-auth.
"""

import base64
import json
import time
from types import SimpleNamespace

import pytest

import hermes_cli.auth as auth_mod
import hermes_cli.auth_codex as auth_codex
from hermes_cli.auth import (
    AuthError,
    _codex_usage_probe_url,
    _is_codex_rate_limit_shaped,
    _probe_codex_quota_restored,
    clear_codex_pool_quota_cooldowns,
    resolve_codex_runtime_credentials,
)


@pytest.fixture(autouse=True)
def _clear_probe_cache():
    auth_mod._codex_quota_probe_cache.clear()
    yield
    auth_mod._codex_quota_probe_cache.clear()


def _jwt(claims: dict) -> str:
    def _part(payload: dict) -> str:
        raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    return f"{_part({'alg': 'none'})}.{_part(claims)}.sig"


class _StubResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            import httpx

            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}", request=None, response=self  # type: ignore[arg-type]
            )


class _StubClient:
    def __init__(self, calls, response):
        self._calls = calls
        self._response = response

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def get(self, url, headers=None):
        self._calls.append({"url": url, "headers": dict(headers or {})})
        return self._response


def _patch_httpx(monkeypatch, response, calls=None):
    calls = calls if calls is not None else []
    monkeypatch.setattr(
        auth_mod.httpx, "Client", lambda **kwargs: _StubClient(calls, response)
    )
    return calls


def _usage_payload(primary_used: float, secondary_used: float) -> dict:
    return {
        "rate_limit": {
            "primary_window": {"used_percent": primary_used},
            "secondary_window": {"used_percent": secondary_used},
        }
    }


# ---------------------------------------------------------------------------
# _is_codex_rate_limit_shaped
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# _codex_usage_probe_url
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# _probe_codex_quota_restored
# ---------------------------------------------------------------------------




def test_probe_sends_chatgpt_account_id_from_jwt(monkeypatch):
    calls = _patch_httpx(monkeypatch, _StubResponse(200, _usage_payload(0.0, 0.0)))
    token = _jwt(
        {
            "exp": time.time() + 3600,
            "https://api.openai.com/auth": {"chatgpt_account_id": "acct-123"},
        }
    )
    assert _probe_codex_quota_restored(token) is True
    assert calls[0]["headers"].get("ChatGPT-Account-ID") == "acct-123"


# ---------------------------------------------------------------------------
# clear_codex_pool_quota_cooldowns
# ---------------------------------------------------------------------------


def _write_auth_store(hermes_home, payload):
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(json.dumps(payload, indent=2))


def _exhausted_pool_store(now=None):
    now = now or time.time()
    return {
        "version": 1,
        "providers": {},
        "credential_pool": {
            "openai-codex": [
                {
                    "id": "cred-quota",
                    "label": "quota-frozen",
                    "auth_type": "oauth",
                    "priority": 0,
                    "source": "device_code",
                    "access_token": "tok-quota",
                    "last_status": "exhausted",
                    "last_status_at": now,
                    "last_error_code": 429,
                    "last_error_reason": "usage_limit_reached",
                    "last_error_message": "The usage limit has been reached",
                    "last_error_reset_at": now + 6 * 24 * 3600,
                },
                {
                    "id": "cred-dead",
                    "label": "revoked",
                    "auth_type": "oauth",
                    "priority": 1,
                    "source": "device_code",
                    "access_token": "tok-dead",
                    "last_status": "dead",
                    "last_status_at": now,
                    "last_error_code": 401,
                    "last_error_reason": "token_invalidated",
                },
                {
                    "id": "cred-auth",
                    "label": "auth-failure",
                    "auth_type": "oauth",
                    "priority": 2,
                    "source": "device_code",
                    "access_token": "tok-auth",
                    "last_status": "exhausted",
                    "last_status_at": now,
                    "last_error_code": 401,
                    "last_error_reason": "token_expired",
                },
            ]
        },
    }






# ---------------------------------------------------------------------------
# resolve_codex_runtime_credentials — stale cooldown lifted by live probe
# ---------------------------------------------------------------------------


def _pool_only_rate_limited_store(now=None):
    now = now or time.time()
    return {
        "version": 1,
        "providers": {},
        "credential_pool": {
            "openai-codex": [
                {
                    "id": "cred-quota",
                    "label": "quota-frozen",
                    "auth_type": "oauth",
                    "priority": 0,
                    "source": "device_code",
                    "access_token": "tok-quota",
                    "last_status": "exhausted",
                    "last_status_at": now,
                    "last_error_code": 429,
                    "last_error_reason": "usage_limit_reached",
                    "last_error_message": "The usage limit has been reached",
                    "last_error_reset_at": now + 3 * 24 * 3600,
                }
            ]
        },
    }


def test_resolver_recovers_when_probe_confirms_reset(tmp_path, monkeypatch):
    """The screenshot bug: pool-only cooldown raises `quota exhausted (429);
    retry after Ns` even though the upstream window already reset.  A positive
    probe must clear the cooldown and return the pool credential."""
    hermes_home = tmp_path / "hermes"
    _write_auth_store(hermes_home, _pool_only_rate_limited_store())
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    monkeypatch.setattr(
        auth_mod, "_probe_codex_quota_restored", lambda token, **kw: True
    )
    monkeypatch.setattr(
        auth_codex, "_probe_codex_quota_restored", lambda token, **kw: True
    )

    resolved = resolve_codex_runtime_credentials()
    assert resolved["api_key"] == "tok-quota"
    assert resolved["source"] == "credential_pool"

    store = json.loads((hermes_home / "auth.json").read_text())
    entry = store["credential_pool"]["openai-codex"][0]
    assert entry["last_status"] is None
    assert entry["last_error_reset_at"] is None


def test_resolver_selects_entry_with_expired_millisecond_reset(tmp_path, monkeypatch):
    """#103349: a millisecond ``last_error_reset_at`` that is already in the past must not
    read as far-future in selection while the rate-limit lookup reads it as elapsed."""
    now = time.time()
    store = _pool_only_rate_limited_store(now)
    main = store["credential_pool"]["openai-codex"][0]
    main["access_token"] = "tok-main"
    main["last_error_reset_at"] = (now - 3600) * 1000
    reserve = dict(main)
    reserve.update(
        {
            "id": "cred-reserve",
            "access_token": "tok-reserve",
            "priority": 1,
            "last_error_reset_at": now + 886,
        }
    )
    store["credential_pool"]["openai-codex"].append(reserve)
    hermes_home = tmp_path / "hermes"
    _write_auth_store(hermes_home, store)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(auth_mod, "_probe_codex_quota_restored", lambda token, **kw: False)
    monkeypatch.setattr(auth_codex, "_probe_codex_quota_restored", lambda token, **kw: False)

    resolved = resolve_codex_runtime_credentials()

    assert resolved["api_key"] == "tok-main"
    assert resolved["source"] == "credential_pool"




# ---------------------------------------------------------------------------
# CredentialPool._available_entries — frozen entry recovers via probe
# ---------------------------------------------------------------------------




def test_pool_probe_not_fired_for_non_quota_exhaustion(tmp_path, monkeypatch):
    """Entries frozen by auth-shaped failures must not trigger the probe."""
    now = time.time()
    store = _pool_only_rate_limited_store(now)
    entry = store["credential_pool"]["openai-codex"][0]
    entry["last_error_code"] = 401
    entry["last_error_reason"] = "token_expired"
    entry["last_error_message"] = "expired"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_auth_store(tmp_path / "hermes", store)

    from agent.credential_pool import load_pool

    pool = load_pool("openai-codex")
    probes = []

    def _spy(token, **kw):
        probes.append(token)
        return True

    monkeypatch.setattr(auth_mod, "_probe_codex_quota_restored", _spy)
    monkeypatch.setattr(auth_codex, "_probe_codex_quota_restored", _spy)
    pool._available_entries(clear_expired=True, refresh=False)
    assert probes == []




# ---------------------------------------------------------------------------
# /usage reset redemption clears persisted pool cooldowns
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# #89415 — the mid-cooldown probe must refresh an expired stored token first
# ---------------------------------------------------------------------------


def _expired_jwt_pool_store(now):
    store = _pool_only_rate_limited_store(now)
    entry = store["credential_pool"]["openai-codex"][0]
    entry["access_token"] = _jwt({"exp": now - 7200})  # expired hours ago
    entry["refresh_token"] = "rf-old"
    return store


class _ExpiryAwareClient(_StubClient):
    """Behaves like the real usage endpoint: an expired bearer gets 401 token_expired."""

    def get(self, url, headers=None):
        token = (headers or {}).get("Authorization", "").removeprefix("Bearer ")
        if auth_codex._codex_access_token_is_expiring(token, 0):
            self._calls.append({"url": url, "headers": dict(headers or {})})
            return _StubResponse(401, {"error": {"code": "token_expired"}})
        return super().get(url, headers=headers)


def _patch_expiry_aware_httpx(monkeypatch, response):
    calls: list = []
    monkeypatch.setattr(
        auth_mod.httpx, "Client", lambda **kwargs: _ExpiryAwareClient(calls, response)
    )
    return calls


def _fake_refresh(monkeypatch, fresh_token, calls):
    def _refresh(access_token, refresh_token, **kw):
        calls.append(refresh_token)
        return {"access_token": fresh_token, "refresh_token": "rf-new", "last_refresh": "now"}

    monkeypatch.setattr(auth_codex, "refresh_codex_oauth_pure", _refresh)


def test_resolver_refreshes_expired_token_before_probe(tmp_path, monkeypatch):
    """Exhausted entries are skipped by the refresh chain, so the stored access token has
    expired by the time the probe runs: /usage answers 401 -> None -> cooldown kept forever,
    even after a top-up / plan upgrade. Refresh (keeping the cooldown) and probe live."""
    now = time.time()
    hermes_home = tmp_path / "hermes"
    _write_auth_store(hermes_home, _expired_jwt_pool_store(now))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    fresh = _jwt({"exp": now + 3600})
    refresh_calls: list = []
    _fake_refresh(monkeypatch, fresh, refresh_calls)
    http_calls = _patch_expiry_aware_httpx(monkeypatch, _StubResponse(200, _usage_payload(0.0, 0.0)))

    resolved = resolve_codex_runtime_credentials()

    assert refresh_calls == ["rf-old"]
    assert http_calls[0]["headers"]["Authorization"] == f"Bearer {fresh}"
    assert resolved["api_key"] == fresh
    entry = json.loads((hermes_home / "auth.json").read_text())["credential_pool"]["openai-codex"][0]
    assert entry["refresh_token"] == "rf-new"
    assert entry["last_status"] is None


def test_pool_selection_refreshes_expired_token_before_probe(tmp_path, monkeypatch):
    """Control at the pool's hot selection path: refresh succeeds, live probe still says 100%
    -> cooldown stays, the rotated (single-use) pair is what the probe used and it is persisted
    on BOTH sides (pool row + ``providers.openai-codex`` singleton) so the next selection's
    auth-store sync cannot re-adopt the consumed pair and lift the cooldown with it."""
    now = time.time()
    hermes_home = tmp_path / "hermes"
    store = _expired_jwt_pool_store(now)
    stale = store["credential_pool"]["openai-codex"][0]
    store["providers"]["openai-codex"] = {
        "tokens": {"access_token": stale["access_token"], "refresh_token": "rf-old"}}
    _write_auth_store(hermes_home, store)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    fresh = _jwt({"exp": now + 3600})
    refresh_calls: list = []
    _fake_refresh(monkeypatch, fresh, refresh_calls)
    http_calls = _patch_expiry_aware_httpx(monkeypatch, _StubResponse(200, _usage_payload(0.0, 100.0)))
    from agent.credential_pool import load_pool

    pool = load_pool("openai-codex")

    assert pool.select() is None
    assert pool.select() is None  # second pass: auth-store sync must not resurrect rf-old

    assert refresh_calls == ["rf-old"]
    assert [c["headers"]["Authorization"] for c in http_calls] == [f"Bearer {fresh}"]
    entry = pool._entries[0]
    assert (entry.access_token, entry.refresh_token, entry.last_status) == (fresh, "rf-new", "exhausted")
    disk = json.loads((hermes_home / "auth.json").read_text())
    assert disk["credential_pool"]["openai-codex"][0]["refresh_token"] == "rf-new"
    assert disk["providers"]["openai-codex"]["tokens"]["refresh_token"] == "rf-new"


def test_pool_selection_throttles_failing_pre_probe_refresh(tmp_path, monkeypatch):
    """Regression control: a frozen entry whose refresh keeps failing (revoked grant, network
    down) must not POST to the token endpoint on every selection — at most one attempt per
    probe interval, the same budget the probe itself has (<= 1 network call per 5 min)."""
    now = time.time()
    hermes_home = tmp_path / "hermes"
    _write_auth_store(hermes_home, _expired_jwt_pool_store(now))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    attempts: list = []

    def _failing_refresh(access_token, refresh_token, **kw):
        attempts.append(refresh_token)
        raise RuntimeError("invalid_grant")

    monkeypatch.setattr(auth_codex, "refresh_codex_oauth_pure", _failing_refresh)
    http_calls = _patch_expiry_aware_httpx(monkeypatch, _StubResponse(200, _usage_payload(0.0, 0.0)))
    from agent.credential_pool import load_pool

    pool = load_pool("openai-codex")
    for _ in range(5):
        assert pool.select() is None

    assert attempts == ["rf-old"]
    assert len(http_calls) <= 1


def test_probe_counts_additional_rate_limits(monkeypatch):
    """#97315: a model-scoped allowance at 100% still 429s that model; the account-wide
    windows being open must not report the quota as restored."""
    payload = _usage_payload(0.0, 0.0)
    payload["additional_rate_limits"] = [
        {"limit_name": "codex_model_scoped",
         "rate_limit": {"primary_window": {"used_percent": 100.0}}}]
    _patch_httpx(monkeypatch, _StubResponse(200, payload))

    assert _probe_codex_quota_restored(_jwt({"exp": time.time() + 3600})) is False
