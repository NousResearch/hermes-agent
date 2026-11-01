"""#142473: the unmatched-identity rotation must not hand out an expired OAuth token.

An OAuth 401's key hint never matches a pool entry's ``runtime_api_key``, so every
retry of a long-lived session lands on ``_rotate_unmatched()``. It used to select
with ``refresh=False``: in a pool whose access tokens had all expired, each rotation
handed out the next entry with its expired token, the #70401 lap cap tripped, and
the session dropped to its fallback provider until another process refreshed
``auth.json``. ``select()`` refreshes an expiring entry before leasing it; the
unmatched rotation now does the same (outside the lock for single-use-token
providers such as openai-codex).
"""
import json

import pytest


def _seed_pool(tmp_path, monkeypatch, entries, provider="openai-codex"):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(
        json.dumps({"version": 1, "credential_pool": {provider: entries}})
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    from agent.credential_pool import load_pool

    return load_pool(provider)


def _oauth(idx, token, source=None):
    return {
        "id": f"cred-{idx}",
        "label": f"codex-{idx}",
        "auth_type": "oauth",
        "priority": idx,
        "source": source or ("manual:device_code" if idx else "device_code"),
        "access_token": token,
        "refresh_token": f"refresh-{idx}",
    }


def _wire_refresh(pool, monkeypatch, *, succeeds=True):
    """Tokens starting with ``expired`` need a refresh; a refresh swaps in ``fresh-<id>``."""
    calls = []

    def needs_refresh(entry):
        return (entry.access_token or "").startswith("expired")

    def refresh(entry, *, force):
        calls.append(entry.id)
        if not succeeds:
            return None
        with pool._lock:
            live = pool._find(lambda e: e.id == entry.id) or entry
            return pool._adopt(live, persist=False, access_token=f"fresh-{entry.id}")

    monkeypatch.setattr(pool, "_entry_needs_refresh", needs_refresh)
    monkeypatch.setattr(pool, "_refresh_entry", refresh)
    return calls


def _rotate_unmatched(pool):
    return pool.mark_exhausted_and_rotate(
        status_code=401,
        error_context={"reason": "unauthorized", "message": "Provided authentication token is expired."},
        api_key_hint="oauth-token-the-session-sent-that-matches-no-entry",
    )


@pytest.mark.parametrize("provider", ["openai-codex", "anthropic"])  # deferred and inline refresh
def test_rotation_refreshes_expired_entries_instead_of_handing_them_out(tmp_path, monkeypatch, provider):
    pool = _seed_pool(
        tmp_path, monkeypatch, [_oauth(0, "expired-a", "manual"), _oauth(1, "expired-b", "manual")], provider=provider,
    )
    calls = _wire_refresh(pool, monkeypatch)

    nxt = _rotate_unmatched(pool)

    assert nxt is not None, "a refreshable pool must recover, not fall back"
    assert nxt.access_token.startswith("fresh-"), f"handed out an expired token: {nxt.access_token!r}"
    assert calls, "the expiring entries were never refreshed"


def test_every_rotation_in_the_retry_lap_carries_a_live_token(tmp_path, monkeypatch):
    pool = _seed_pool(tmp_path, monkeypatch, [_oauth(0, "expired-a"), _oauth(1, "expired-b")])
    _wire_refresh(pool, monkeypatch)

    handed_out = []
    for _ in range(5):
        nxt = _rotate_unmatched(pool)
        if nxt is None:
            break
        handed_out.append(nxt.access_token)

    assert handed_out, "no rotation succeeded"
    assert not [t for t in handed_out if t.startswith("expired")], handed_out


def test_failed_refresh_surfaces_the_error_instead_of_an_expired_token(tmp_path, monkeypatch):
    pool = _seed_pool(tmp_path, monkeypatch, [_oauth(0, "expired-a"), _oauth(1, "expired-b")])
    calls = _wire_refresh(pool, monkeypatch, succeeds=False)

    nxt = _rotate_unmatched(pool)

    assert calls, "a refresh should have been attempted"
    assert nxt is None or not nxt.access_token.startswith("expired"), nxt
    statuses = {e.id: e.last_status for e in pool._entries}
    assert all(status != "exhausted" for status in statuses.values()), statuses


def test_sole_expired_entry_recovers_through_its_refresh(tmp_path, monkeypatch):
    """The single-entry guard refuses to return an UNCHANGED credential; a refreshed one is a real recovery."""
    pool = _seed_pool(tmp_path, monkeypatch, [_oauth(0, "expired-a")])
    _wire_refresh(pool, monkeypatch)

    nxt = _rotate_unmatched(pool)

    assert nxt is not None
    assert nxt.access_token == "fresh-cred-0"


def test_sole_live_entry_still_refuses_to_report_a_fake_recovery(tmp_path, monkeypatch):
    pool = _seed_pool(tmp_path, monkeypatch, [_oauth(0, "live-a")])
    calls = _wire_refresh(pool, monkeypatch)

    assert _rotate_unmatched(pool) is None
    assert calls == []


@pytest.mark.parametrize("provider", ["openai-codex", "anthropic"])
def test_live_multi_entry_pool_rotation_is_unchanged(tmp_path, monkeypatch, provider):
    pool = _seed_pool(
        tmp_path, monkeypatch, [_oauth(0, "live-a", "manual"), _oauth(1, "live-b", "manual")], provider=provider,
    )
    calls = _wire_refresh(pool, monkeypatch)

    nxt = _rotate_unmatched(pool)

    assert nxt is not None and nxt.access_token in {"live-a", "live-b"}
    assert calls == []
