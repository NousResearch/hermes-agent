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


def test_rate_limit_shaped_variants():
    assert _is_codex_rate_limit_shaped(429, None, None)
    assert _is_codex_rate_limit_shaped(None, "usage_limit_reached", None)
    assert _is_codex_rate_limit_shaped(None, None, "The usage limit has been reached")
    assert _is_codex_rate_limit_shaped(None, "quota_exceeded", None)
    assert not _is_codex_rate_limit_shaped(401, "token_invalidated", "token revoked")
    assert not _is_codex_rate_limit_shaped(None, None, None)


# ---------------------------------------------------------------------------
# _codex_usage_probe_url
# ---------------------------------------------------------------------------


def test_probe_url_backend_api_uses_wham():
    assert (
        _codex_usage_probe_url("https://chatgpt.com/backend-api/codex")
        == "https://chatgpt.com/backend-api/wham/usage"
    )


def test_probe_url_non_backend_uses_api_codex():
    assert (
        _codex_usage_probe_url("https://example.com/codex")
        == "https://example.com/api/codex/usage"
    )


# ---------------------------------------------------------------------------
# _probe_codex_quota_restored
# ---------------------------------------------------------------------------


def test_probe_returns_true_when_windows_below_100(monkeypatch):
    calls = _patch_httpx(monkeypatch, _StubResponse(200, _usage_payload(12.0, 34.0)))
    token = _jwt({"exp": time.time() + 3600})
    assert _probe_codex_quota_restored(token) is True
    assert calls and calls[0]["url"].endswith("/usage")


def test_probe_returns_false_when_window_still_exhausted(monkeypatch):
    _patch_httpx(monkeypatch, _StubResponse(200, _usage_payload(100.0, 34.0)))
    token = _jwt({"exp": time.time() + 3600})
    assert _probe_codex_quota_restored(token) is False


def test_probe_returns_false_on_429(monkeypatch):
    _patch_httpx(monkeypatch, _StubResponse(429, {}))
    token = _jwt({"exp": time.time() + 3600})
    assert _probe_codex_quota_restored(token) is False


def test_probe_indeterminate_on_unexpected_payload(monkeypatch):
    _patch_httpx(monkeypatch, _StubResponse(200, {}))
    token = _jwt({"exp": time.time() + 3600})
    assert _probe_codex_quota_restored(token) is None


def test_probe_skips_non_jwt_tokens_without_network(monkeypatch):
    calls = _patch_httpx(monkeypatch, _StubResponse(200, _usage_payload(0.0, 0.0)))
    assert _probe_codex_quota_restored("not-a-jwt") is None
    assert _probe_codex_quota_restored("") is None
    assert calls == []


def test_probe_throttles_repeat_calls(monkeypatch):
    calls = _patch_httpx(monkeypatch, _StubResponse(200, _usage_payload(12.0, 34.0)))
    token = _jwt({"exp": time.time() + 3600})
    assert _probe_codex_quota_restored(token) is True
    assert _probe_codex_quota_restored(token) is True  # cached
    assert len(calls) == 1


def test_probe_sends_chatgpt_account_id_from_jwt(monkeypatch):
    calls = _patch_httpx(monkeypatch, _StubResponse(200, _usage_payload(0.0, 0.0)))
    token = _jwt(
        {
            "exp": time.time() + 3600,
            "https://api.openai.com/auth": {"chatgpt_account_id": "acct-123"},
        }
    )
    assert _probe_codex_quota_restored(token) is True
    assert calls[0]["headers"].get("ChatGPT-Account-Id") == "acct-123"


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


def test_clear_cooldowns_only_touches_quota_shaped_entries(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    _write_auth_store(hermes_home, _exhausted_pool_store())
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    assert clear_codex_pool_quota_cooldowns() == 1

    store = json.loads((hermes_home / "auth.json").read_text())
    entries = {e["id"]: e for e in store["credential_pool"]["openai-codex"]}
    assert entries["cred-quota"]["last_status"] is None
    assert entries["cred-quota"]["last_error_reset_at"] is None
    # DEAD (terminal auth) and non-quota exhausted entries stay untouched.
    assert entries["cred-dead"]["last_status"] == "dead"
    assert entries["cred-auth"]["last_status"] == "exhausted"


def test_clear_cooldowns_scoped_to_access_token(tmp_path, monkeypatch):
    now = time.time()
    store = _exhausted_pool_store(now)
    store["credential_pool"]["openai-codex"].append(
        {
            "id": "cred-quota-2",
            "label": "other-quota",
            "auth_type": "oauth",
            "priority": 3,
            "source": "device_code",
            "access_token": "tok-other",
            "last_status": "exhausted",
            "last_status_at": now,
            "last_error_code": 429,
            "last_error_reset_at": now + 3600,
        }
    )
    hermes_home = tmp_path / "hermes"
    _write_auth_store(hermes_home, store)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    assert clear_codex_pool_quota_cooldowns("tok-other") == 1

    persisted = json.loads((hermes_home / "auth.json").read_text())
    entries = {e["id"]: e for e in persisted["credential_pool"]["openai-codex"]}
    assert entries["cred-quota-2"]["last_status"] is None
    assert entries["cred-quota"]["last_status"] == "exhausted"


def test_clear_cooldowns_noop_without_pool(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    _write_auth_store(hermes_home, {"version": 1, "providers": {}})
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    assert clear_codex_pool_quota_cooldowns() == 0


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

    resolved = resolve_codex_runtime_credentials()
    assert resolved["api_key"] == "tok-quota"
    assert resolved["source"] == "credential_pool"

    store = json.loads((hermes_home / "auth.json").read_text())
    entry = store["credential_pool"]["openai-codex"][0]
    assert entry["last_status"] is None
    assert entry["last_error_reset_at"] is None


def test_resolver_keeps_cooldown_when_probe_negative(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    _write_auth_store(hermes_home, _pool_only_rate_limited_store())
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    monkeypatch.setattr(
        auth_mod, "_probe_codex_quota_restored", lambda token, **kw: False
    )

    with pytest.raises(AuthError) as exc:
        resolve_codex_runtime_credentials()
    assert exc.value.code == auth_mod.CODEX_RATE_LIMITED_CODE
    assert "retry after" in str(exc.value)


def test_resolver_keeps_cooldown_when_probe_indeterminate(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    _write_auth_store(hermes_home, _pool_only_rate_limited_store())
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    monkeypatch.setattr(
        auth_mod, "_probe_codex_quota_restored", lambda token, **kw: None
    )

    with pytest.raises(AuthError) as exc:
        resolve_codex_runtime_credentials()
    assert exc.value.code == auth_mod.CODEX_RATE_LIMITED_CODE


# ---------------------------------------------------------------------------
# CredentialPool._available_entries — frozen entry recovers via probe
# ---------------------------------------------------------------------------


def test_pool_entry_recovers_when_probe_confirms_reset(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_auth_store(tmp_path / "hermes", _pool_only_rate_limited_store())

    from agent.credential_pool import load_pool

    pool = load_pool("openai-codex")
    monkeypatch.setattr(
        auth_mod, "_probe_codex_quota_restored", lambda token, **kw: True
    )

    available = pool._available_entries(clear_expired=True, refresh=False)
    assert len(available) == 1
    assert available[0].last_status == "ok"
    assert available[0].last_error_reset_at is None


def test_pool_entry_stays_frozen_when_probe_negative(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_auth_store(tmp_path / "hermes", _pool_only_rate_limited_store())

    from agent.credential_pool import load_pool

    pool = load_pool("openai-codex")
    monkeypatch.setattr(
        auth_mod, "_probe_codex_quota_restored", lambda token, **kw: False
    )

    assert pool._available_entries(clear_expired=True, refresh=False) == []


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
    pool._available_entries(clear_expired=True, refresh=False)
    assert probes == []


def test_pool_readonly_enumeration_does_not_probe(tmp_path, monkeypatch):
    """clear_expired=False callers (read-only listing) must not fire probes
    or mutate persisted state."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_auth_store(tmp_path / "hermes", _pool_only_rate_limited_store())

    from agent.credential_pool import load_pool

    pool = load_pool("openai-codex")
    probes = []

    def _spy(token, **kw):
        probes.append(token)
        return True

    monkeypatch.setattr(auth_mod, "_probe_codex_quota_restored", _spy)
    assert pool._available_entries(clear_expired=False, refresh=False) == []
    assert probes == []


# ---------------------------------------------------------------------------
# /usage reset redemption clears persisted pool cooldowns
# ---------------------------------------------------------------------------


def test_redeem_reset_clears_pool_cooldowns(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    _write_auth_store(hermes_home, _pool_only_rate_limited_store())
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    from agent import account_usage

    class _FakeResetClient:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, url, headers=None):
            return _StubResponse(
                200,
                {
                    "rate_limit": {
                        "primary_window": {"used_percent": 100.0},
                        "secondary_window": {"used_percent": 40.0},
                    },
                    "rate_limit_reset_credits": {"available_count": 1},
                },
            )

        def post(self, url, headers=None, json=None):
            return _StubResponse(200, {"code": "reset", "windows_reset": 2})

    monkeypatch.setattr(
        account_usage.httpx, "Client", lambda **kwargs: _FakeResetClient(**kwargs)
    )

    result = account_usage.redeem_codex_reset_credit(
        base_url="https://chatgpt.com/backend-api/codex",
        api_key="live-agent-token",
    )
    assert result.redeemed

    store = json.loads((hermes_home / "auth.json").read_text())
    entry = store["credential_pool"]["openai-codex"][0]
    assert entry["last_status"] is None
    assert entry["last_error_reset_at"] is None
