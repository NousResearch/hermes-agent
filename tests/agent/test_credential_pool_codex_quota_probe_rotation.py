"""The Codex quota probe's pre-probe token rotation must survive the cooldown clear.

An exhausted Codex row skips the proactive refresh chain, so by the time the selection-path quota
probe runs its access token has expired: the probe spends the single-use refresh token first and
adopts the rotated pair into the pool. When the probe then reports the quota restored, the cooldown
clear must build on that rotated row — not on the caller's pre-probe copy — or the consumed refresh
token goes back into the pool and the next refresh replays it (``refresh_token_reused``).
"""
from __future__ import annotations

import base64
import json
import time

import pytest

import hermes_cli.auth as auth_mod
import hermes_cli.auth_codex as auth_codex
from agent.credential_pool import load_pool


def _jwt(exp: float) -> str:
    def seg(obj: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(obj).encode()).rstrip(b"=").decode()

    payload = {"exp": int(exp), "sub": "user-q", "https://api.openai.com/auth": {"chatgpt_account_id": "acct-q"}}
    return f"{seg({'alg': 'none'})}.{seg(payload)}.sig"


@pytest.fixture
def home(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    auth_mod._codex_quota_probe_cache.clear()
    yield home
    auth_mod._codex_quota_probe_cache.clear()


def _write_exhausted_row(home, now: float) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "auth.json").write_text(json.dumps({
        "version": 1,
        "providers": {},
        "credential_pool": {"openai-codex": [{
            "id": "cred-quota", "label": "quota-frozen", "auth_type": "oauth", "priority": 0,
            "source": "manual:device_code",
            "access_token": _jwt(now - 7200),  # expired while the cooldown ran
            "refresh_token": "rt-0",
            "last_status": "exhausted", "last_status_at": now - 3600,
            "last_error_code": 429, "last_error_reason": "usage_limit_reached",
            "last_error_message": "The usage limit has been reached",
            "last_error_reset_at": now + 6 * 24 * 3600,
        }]},
    }), encoding="utf-8")


def _rotating_token_endpoint(monkeypatch, now: float) -> list:
    """Single-use refresh tokens: rt-N mints rt-(N+1); a spent token is rejected."""
    sent: list = []
    spent: set = set()

    def refresh(access_token, refresh_token, **kw):
        sent.append(refresh_token)
        if refresh_token in spent:
            raise RuntimeError("refresh_token_reused")
        spent.add(refresh_token)
        n = int(refresh_token.rsplit("-", 1)[1]) + 1
        return {"access_token": _jwt(now + 3600 * n), "refresh_token": f"rt-{n}", "last_refresh": "now"}

    monkeypatch.setattr(auth_codex, "refresh_codex_oauth_pure", refresh)
    monkeypatch.setattr(auth_mod, "refresh_codex_oauth_pure", refresh)
    return sent


def test_quota_restored_selection_keeps_probe_rotated_refresh_token(home, monkeypatch):
    now = time.time()
    _write_exhausted_row(home, now)
    sent = _rotating_token_endpoint(monkeypatch, now)
    monkeypatch.setattr(auth_mod, "_probe_codex_quota_restored", lambda token, **kw: True)
    monkeypatch.setattr(auth_codex, "_probe_codex_quota_restored", lambda token, **kw: True)

    pool = load_pool("openai-codex")
    selected = pool.select()

    assert sent == ["rt-0"]  # the probe's pre-probe refresh; the spent rt-0 is not replayed
    assert selected is not None and selected.last_status == "ok"
    assert selected.refresh_token == "rt-1"
    assert pool._find(lambda e: e.id == "cred-quota").refresh_token == "rt-1"
    disk = json.loads((home / "auth.json").read_text(encoding="utf-8"))
    assert disk["credential_pool"]["openai-codex"][0]["refresh_token"] == "rt-1"

    refreshed = pool.try_refresh_current()

    assert sent == ["rt-0", "rt-1"]  # rt-0 is never replayed
    assert refreshed is not None and refreshed.refresh_token == "rt-2"
    disk = json.loads((home / "auth.json").read_text(encoding="utf-8"))
    assert disk["credential_pool"]["openai-codex"][0]["refresh_token"] == "rt-2"
