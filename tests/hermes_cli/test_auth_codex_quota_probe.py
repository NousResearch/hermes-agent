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
    pool._available_entries(clear_expired=True, refresh=False)
    assert probes == []




# ---------------------------------------------------------------------------
# /usage reset redemption clears persisted pool cooldowns
# ---------------------------------------------------------------------------


