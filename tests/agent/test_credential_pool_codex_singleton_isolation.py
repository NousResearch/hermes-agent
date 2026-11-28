"""Codex pool entries and the auth.json singleton: who may adopt whose tokens.

``manual:device_code`` is ambiguous — a legacy alias of the singleton or an independent account
added with ``hermes auth add openai-codex``. Adopting the singleton into an independent account
silently turned two logins into one (both then hit the same usage limit; salvaged from #100423 and
#106788, cluster #92198 / #95297, issue #106705).
"""
from __future__ import annotations

import base64
import json
import time

import pytest

import hermes_cli.auth as auth_mod
from agent.credential_pool import STATUS_DEAD, load_pool
from hermes_cli.auth import AuthError


def _jwt(account: str, sub: str, exp: float) -> str:
    def seg(obj: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(obj).encode()).rstrip(b"=").decode()

    payload = {"exp": int(exp), "sub": sub, "https://api.openai.com/auth": {"chatgpt_account_id": account}}
    return f"{seg({'alg': 'none'})}.{seg(payload)}.sig"


def _iso(ts: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))


def _write_store(home, singleton_tokens: dict, singleton_last_refresh: str, manual: dict) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "auth.json").write_text(json.dumps({
        "version": 1,
        "active_provider": "openai-codex",
        "providers": {"openai-codex": {
            "tokens": singleton_tokens, "last_refresh": singleton_last_refresh, "auth_mode": "chatgpt"}},
        "credential_pool": {"openai-codex": [
            {"id": "seeded", "label": "device_code", "auth_type": "oauth", "priority": 0,
             "source": "device_code", **singleton_tokens, "last_refresh": singleton_last_refresh},
            {"id": "manual", "label": "second", "auth_type": "oauth", "priority": 1,
             "source": "manual:device_code", **manual},
        ]},
    }), encoding="utf-8")


@pytest.fixture
def home(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _stub_refresh(monkeypatch, minted: str, minted_rt: str, posted: list) -> None:
    def fake(access_token, refresh_token):
        posted.append(refresh_token)
        return {"access_token": minted, "refresh_token": minted_rt, "last_refresh": _iso(time.time())}

    monkeypatch.setattr(auth_mod, "refresh_codex_oauth_pure", fake)


def test_independent_manual_account_refreshes_with_its_own_pair(home, monkeypatch):
    """An independent second account never adopts the singleton: its refresh POSTs its OWN refresh
    token, and the persisted row still identifies the second principal afterwards."""
    now = time.time()
    a_at = _jwt("acct-A", "user-A", now + 8 * 3600)
    b_at = _jwt("acct-B", "user-B", now + 60)  # expiring → the pool defers it to _refresh_entry()
    _write_store(home, {"access_token": a_at, "refresh_token": "rt-A"}, _iso(now - 3600),
                 {"access_token": b_at, "refresh_token": "rt-B", "last_refresh": _iso(now - 7200)})
    posted: list = []
    b_new = _jwt("acct-B", "user-B", now + 8 * 3600)
    _stub_refresh(monkeypatch, b_new, "rt-B2", posted)

    pool = load_pool("openai-codex")
    refreshed = pool._refresh_entry(next(e for e in pool.entries() if e.id == "manual"), force=False)

    assert posted == ["rt-B"]
    assert refreshed is not None and refreshed.refresh_token == "rt-B2"
    on_disk = json.loads((home / "auth.json").read_text(encoding="utf-8"))
    manual = next(e for e in on_disk["credential_pool"]["openai-codex"] if e["id"] == "manual")
    assert (manual["access_token"], manual["refresh_token"]) == (b_new, "rt-B2")
    # The singleton (account A) is untouched by account B's rotation.
    assert on_disk["providers"]["openai-codex"]["tokens"] == {"access_token": a_at, "refresh_token": "rt-A"}


def test_same_account_alias_adopts_only_a_newer_singleton(home, monkeypatch):
    """A legacy alias (same principal) must follow a singleton that was re-authed AFTER it, but must
    not fall back onto a singleton older than its own rotation — that replays a consumed token."""
    now = time.time()
    alias_at = _jwt("acct-A", "user-A", now + 8 * 3600)
    stale_singleton_at = _jwt("acct-A", "user-A", now + 3600)
    _write_store(home, {"access_token": stale_singleton_at, "refresh_token": "rt-consumed"}, _iso(now - 7200),
                 {"access_token": alias_at, "refresh_token": "rt-alias", "last_refresh": _iso(now - 60)})
    pool = load_pool("openai-codex")
    alias = next(e for e in pool.entries() if e.id == "manual")

    synced = pool._sync_entry_from_auth_store(alias)
    assert (synced.access_token, synced.refresh_token) == (alias_at, "rt-alias")

    # The user re-authenticates the singleton (newer stamp, same principal): the alias follows.
    fresh_at = _jwt("acct-A", "user-A", now + 9 * 3600)
    _write_store(home, {"access_token": fresh_at, "refresh_token": "rt-fresh"}, _iso(now + 5),
                 {"access_token": alias_at, "refresh_token": "rt-alias", "last_refresh": _iso(now - 60)})
    synced = load_pool("openai-codex")._sync_entry_from_auth_store(alias)
    assert (synced.access_token, synced.refresh_token) == (fresh_at, "rt-fresh")


def _single_use_endpoint(monkeypatch, posted: list, on_post=None) -> dict:
    """Fake OpenAI token endpoint: each refresh token works once; a replayed one is refused."""
    live = {"rt": "rt-B", "gen": 1}

    def fake(access_token, refresh_token):
        posted.append(refresh_token)
        if on_post is not None:
            on_post()
        if refresh_token != live["rt"]:
            raise AuthError("refresh token already used", provider="openai-codex",
                            code="refresh_token_reused", relogin_required=True)
        live["gen"] += 1
        live["rt"] = f"rt-B{live['gen']}"
        return {"access_token": _jwt("acct-B", "user-B", time.time() + 8 * 3600 + live["gen"]),
                "refresh_token": live["rt"], "last_refresh": _iso(time.time())}

    monkeypatch.setattr(auth_mod, "refresh_codex_oauth_pure", fake)
    return live


def _manual_row(home) -> dict:
    on_disk = json.loads((home / "auth.json").read_text(encoding="utf-8"))
    return next(e for e in on_disk["credential_pool"]["openai-codex"] if e["id"] == "manual")


@pytest.mark.parametrize("force", [False, True])
def test_second_pool_instance_never_replays_a_peer_rotated_manual_account(home, monkeypatch, force):
    """Two pool instances (two desktop sessions, or backend + gateway) share an independent account.
    Once one rotates it, the other must refresh from the persisted row: replaying the spent token got
    ``refresh_token_reused`` and marked the row DEAD — and DEAD manual rows are pruned (account lost)."""
    now = time.time()
    _write_store(home, {"access_token": _jwt("acct-A", "user-A", now + 8 * 3600), "refresh_token": "rt-A"},
                 _iso(now - 3600),
                 {"access_token": _jwt("acct-B", "user-B", now + 60), "refresh_token": "rt-B",
                  "last_refresh": _iso(now - 7200)})
    posted: list = []
    live = _single_use_endpoint(monkeypatch, posted)
    first, second = load_pool("openai-codex"), load_pool("openai-codex")

    assert first.try_refresh_matching(credential_id="manual") is not None
    refreshed = (second.try_refresh_matching(credential_id="manual") if force
                 else second._refresh_entry(next(e for e in second.entries() if e.id == "manual"), force=False))

    assert posted.count("rt-B") == 1
    assert refreshed is not None and refreshed.refresh_token == live["rt"]
    row = _manual_row(home)
    assert row["refresh_token"] == live["rt"] and row.get("last_status") != STATUS_DEAD


def test_refused_refresh_does_not_kill_a_row_a_peer_already_rotated(home, monkeypatch):
    """A refused POST only condemns the refresh token that was sent. If the persisted row already holds
    a peer's newer pair, adopt it — the quarantine's persist would otherwise pull that fresh pair in and
    mark it DEAD."""
    now = time.time()
    _write_store(home, {"access_token": _jwt("acct-A", "user-A", now + 8 * 3600), "refresh_token": "rt-A"},
                 _iso(now - 3600),
                 {"access_token": _jwt("acct-B", "user-B", now + 60), "refresh_token": "rt-B",
                  "last_refresh": _iso(now - 7200)})
    peer_at = _jwt("acct-B", "user-B", now + 8 * 3600)

    def peer_rotates_first():
        # The peer's rotation lands after our pre-POST re-read: our rt-B is already spent.
        live["rt"] = "rt-peer"
        store = json.loads((home / "auth.json").read_text(encoding="utf-8"))
        row = next(e for e in store["credential_pool"]["openai-codex"] if e["id"] == "manual")
        row.update(access_token=peer_at, refresh_token="rt-peer")
        (home / "auth.json").write_text(json.dumps(store), encoding="utf-8")

    posted: list = []
    live = _single_use_endpoint(monkeypatch, posted, on_post=peer_rotates_first)
    pool = load_pool("openai-codex")

    refreshed = pool.try_refresh_matching(credential_id="manual")

    assert posted == ["rt-B"]
    assert refreshed is not None and (refreshed.access_token, refreshed.refresh_token) == (peer_at, "rt-peer")
    row = _manual_row(home)
    assert row["refresh_token"] == "rt-peer" and row.get("last_status") != STATUS_DEAD
