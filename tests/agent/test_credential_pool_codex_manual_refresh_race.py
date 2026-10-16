"""Concurrent refreshes of a ``manual:device_code`` Codex row across pool instances.

Two Hermes processes of one profile can hit the same expiring ``manual:device_code``
entry together (e.g. two cron jobs starting in the same minute). ``manual:*`` rows
never write back to the auth.json singleton (#39236), so the loser used to never see
the winner's rotated pair: it POSTed the already-consumed refresh token, took
``refresh_token_reused`` as terminal, marked the login DEAD and persisted the stale
pair over the winner's row — killing the grant (#139727).
"""
from __future__ import annotations

import base64
import json
import time

import pytest

import hermes_cli.auth as auth_mod
from hermes_cli.auth_constants import AuthError
from agent.credential_pool import load_pool


def _jwt(account: str, sub: str, exp: float) -> str:
    def seg(obj: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(obj).encode()).rstrip(b"=").decode()

    payload = {"exp": int(exp), "sub": sub, "https://api.openai.com/auth": {"chatgpt_account_id": account}}
    return f"{seg({'alg': 'none'})}.{seg(payload)}.sig"


def _iso(ts: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))


def _write_store(home, manual_tokens: dict, manual_last_refresh: str, singleton_tokens: dict | None = None) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "auth.json").write_text(json.dumps({
        "version": 1,
        "active_provider": "openai-codex",
        "providers": {"openai-codex": {
            "tokens": singleton_tokens or {}, "last_refresh": _iso(time.time()), "auth_mode": "chatgpt"}},
        "credential_pool": {"openai-codex": [
            {"id": "manual", "label": "chatgpt", "auth_type": "oauth", "priority": 0,
             "source": "manual:device_code", **manual_tokens, "last_refresh": manual_last_refresh},
        ]},
    }), encoding="utf-8")


def _manual_row(home) -> dict:
    on_disk = json.loads((home / "auth.json").read_text(encoding="utf-8"))
    return next(e for e in on_disk["credential_pool"]["openai-codex"] if e["id"] == "manual")


@pytest.fixture
def home(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


class _FakeTokenEndpoint:
    """Accepts each refresh token exactly once, like OpenAI's rotation endpoint.

    Re-POSTing a consumed token raises the terminal ``refresh_token_reused``.
    """

    def __init__(self, mints: dict[str, tuple[str, str]]) -> None:
        self.mints = mints
        self.posted: list[str] = []
        self._spent: set[str] = set()

    def __call__(self, access_token: str, refresh_token: str) -> dict:
        self.posted.append(refresh_token)
        if refresh_token in self._spent:
            raise AuthError(
                "The refresh token has already been used.", provider="openai-codex",
                code="refresh_token_reused", relogin_required=True)
        minted = self.mints.get(refresh_token)
        if minted is None:
            raise AuthError(
                "Unknown refresh token.", provider="openai-codex",
                code="invalid_grant", relogin_required=True)
        self._spent.add(refresh_token)
        return {"access_token": minted[0], "refresh_token": minted[1], "last_refresh": _iso(time.time())}


def test_lock_waiter_adopts_the_winners_rotation_instead_of_reusing(home, monkeypatch):
    """The second pool instance must re-read the pool row under the auth-store lock and POST the
    winner's rotated token — not the consumed pair still in its memory."""
    now = time.time()
    expiring = _jwt("acct-A", "user-A", now + 60)
    fresh = _jwt("acct-A", "user-A", now + 8 * 3600)
    _write_store(home, {"access_token": expiring, "refresh_token": "rt-0"}, _iso(now - 7200))
    endpoint = _FakeTokenEndpoint({"rt-0": (fresh, "rt-1"), "rt-1": (fresh, "rt-2")})
    monkeypatch.setattr(auth_mod, "refresh_codex_oauth_pure", endpoint)

    loser = load_pool("openai-codex")  # snapshot taken before the winner runs
    winner = load_pool("openai-codex")
    winner_refreshed = winner._refresh_entry(next(e for e in winner.entries() if e.id == "manual"), force=False)
    assert winner_refreshed is not None and winner_refreshed.refresh_token == "rt-1"

    loser_refreshed = loser._refresh_entry(next(e for e in loser.entries() if e.id == "manual"), force=False)

    # The loser never re-POSTed the consumed rt-0: it adopted the winner's fresh
    # pair straight from the pool row (the reporter's patched expectation).
    assert endpoint.posted == ["rt-0"]
    assert loser_refreshed is not None
    assert loser_refreshed.refresh_token == "rt-1"
    assert loser_refreshed.last_status != "dead"
    row = _manual_row(home)
    assert row["refresh_token"] == "rt-1"
    assert row.get("last_status") != "dead"


def test_failed_refresh_adopts_a_later_pool_row_rotation_before_declaring_death(home, monkeypatch):
    """If the POST still loses the race (row re-read happened before the winner persisted), the
    recovery path must consult the pool row before quarantining: adopting the winner's pair beats
    marking the login DEAD and overwriting the row with the stale pair."""
    now = time.time()
    expiring = _jwt("acct-A", "user-A", now + 60)
    fresh = _jwt("acct-A", "user-A", now + 8 * 3600)
    _write_store(home, {"access_token": expiring, "refresh_token": "rt-0"}, _iso(now - 7200))
    endpoint = _FakeTokenEndpoint({"rt-0": (fresh, "rt-1")})
    monkeypatch.setattr(auth_mod, "refresh_codex_oauth_pure", endpoint)

    loser = load_pool("openai-codex")
    winner = load_pool("openai-codex")
    winner._refresh_entry(next(e for e in winner.entries() if e.id == "manual"), force=False)
    assert _manual_row(home)["refresh_token"] == "rt-1"

    # Simulate the race window: the loser's pre-POST re-read still sees the pre-winner row,
    # so its POST of rt-0 is rejected as reused and recovery must run.
    real_sync = loser._sync_entry_from_pool_store
    calls = {"n": 0}

    def stale_first_read(entry):
        calls["n"] += 1
        return entry if calls["n"] == 1 else real_sync(entry)

    monkeypatch.setattr(loser, "_sync_entry_from_pool_store", stale_first_read)
    entry = next(e for e in loser.entries() if e.id == "manual")
    refreshed = loser._refresh_entry(entry, force=False)

    assert refreshed is not None
    assert refreshed.refresh_token == "rt-1"
    assert refreshed.last_status != "dead"
    # The winner's rotated pair survived on disk; the stale rt-0 never overwrote it.
    assert _manual_row(home)["refresh_token"] == "rt-1"
    assert _manual_row(home).get("last_status") != "dead"


def test_unrelated_pool_row_change_is_not_adopted_by_another_provider(home, monkeypatch):
    """Guard: opening the pool-store sync to Codex must not change rows whose tokens were NOT
    rotated — a matching row re-read returns the entry unchanged (no phantom rotation)."""
    now = time.time()
    expiring = _jwt("acct-A", "user-A", now + 60)
    _write_store(home, {"access_token": expiring, "refresh_token": "rt-0"}, _iso(now - 7200))
    pool = load_pool("openai-codex")
    entry = next(e for e in pool.entries() if e.id == "manual")

    synced = pool._sync_entry_from_pool_store(entry)

    assert synced is entry  # identical on-disk row: no adoption, no log noise
