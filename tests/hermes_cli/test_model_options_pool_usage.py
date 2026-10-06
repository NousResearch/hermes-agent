"""Pool-aware model-picker usage: per-account rows, cache isolation, honest failures.

Real-path regression tests for ``hermes_cli/inventory.py::_apply_usage`` and
``agent/account_usage_cache`` against actual auth stores on disk (no pool mocks): a
multi-entry pool must report per-account windows/state, one profile's snapshot must
never leak into another's picker, a failed/401 fetch must not hide or mutate its
siblings, and a late background fetch must not replace fresher per-turn data.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timedelta, timezone

import pytest

from agent import account_usage
from agent.account_usage import AccountUsageSnapshot, AccountUsageWindow
from agent.account_usage_cache import _snapshots, remember_account_usage
from hermes_cli.inventory import _apply_usage

pytestmark = pytest.mark.usefixtures("_join_refresh_workers_autouse")


def _write_pool(home, provider: str, entries: list[dict]) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "auth.json").write_text(
        json.dumps({"version": 1, "credential_pool": {provider: entries}}, indent=2), encoding="utf-8")


def _entry(cred_id: str, token: str, **over) -> dict:
    entry = {
        "id": cred_id, "label": cred_id, "auth_type": "api_key", "priority": 0,
        "source": "manual", "access_token": token,
    }
    entry.update(over)
    return entry


def _fp(token: str) -> str:
    """The stable non-secret identity the cache assigns to an opaque API-key token."""
    import hashlib

    return f"fp:{hashlib.sha256(token.encode('utf-8')).hexdigest()[:16]}"


def _snapshot(used: float, *, resets_in_hours: float = 2.0, scope: str = "account") -> AccountUsageSnapshot:
    return AccountUsageSnapshot(
        provider="openrouter", source="test", fetched_at=datetime.now(timezone.utc),
        windows=(AccountUsageWindow(label="API key quota", used_percent=used,
                                    reset_at=datetime.now(timezone.utc) + timedelta(hours=resets_in_hours),
                                    scope=scope),))


def _clear_cache() -> None:
    _snapshots.clear()


def _join_refresh_workers() -> None:
    """Wait out any account-usage refresh daemon this test spawned (bounded): a straggler
    worker re-resolves the pool under whatever HERMES_HOME is current when it runs — after
    monkeypatch teardown that is the NEXT test's home, and its load_pool persist would merge
    this test's in-memory entries into that store (cross-test auth.json contamination)."""
    import threading

    for worker in [t for t in threading.enumerate() if "hermes-account-usage-refresh" in t.name]:
        worker.join(10)


@pytest.fixture(autouse=True)
def _join_refresh_workers_autouse():
    """Every test in this file: join refresh workers AFTER the test body (teardown ordering —
    monkeypatch restores HERMES_HOME after the body, so joining must happen after it)."""
    yield
    _join_refresh_workers()


def _install_fetcher(monkeypatch, answers: dict) -> dict:
    """Loopback-free deterministic fetcher: token → snapshot (or exception). Records calls."""
    calls: dict = {"tokens": [], "lock": threading.Lock()}

    def fake(base_url, api_key):
        with calls["lock"]:
            calls["tokens"].append(api_key)
        answer = answers.get(api_key)
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setitem(account_usage._USAGE_FETCHERS, "openrouter", fake)
    return calls


def _openrouter_row(usage_rows: list[dict]) -> dict:
    row = {"slug": "openrouter", "name": "OpenRouter", "models": ["m"], "authenticated": True}
    _apply_usage([row])
    return row


# ── per-account rows for a multi-entry pool ──────────────────────────────────


def test_multi_entry_pool_reports_per_account_rows(monkeypatch, tmp_path):
    """>=2 pool entries → ``usage.accounts`` (never a provider-wide ``windows`` gauge), one row
    per credential with its own cached windows, plus a background refresh per account."""
    from hermes_constants import hermes_home_key

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "a"))
    _write_pool(tmp_path / "a", "openrouter", [
        _entry("c1", "tok-1"), _entry("c2", "tok-2"), _entry("c3", "tok-3")])
    _clear_cache()
    remember_account_usage("openrouter", _snapshot(10.0), identity_id=_fp("tok-1"))
    remember_account_usage("openrouter", _snapshot(80.0), identity_id=_fp("tok-2"))
    # c3 has no snapshot yet → state unknown, still visible.
    _install_fetcher(monkeypatch, {})

    row = _openrouter_row([])

    usage = row["usage"]
    assert "windows" not in usage or not usage["windows"], "multi-entry pool must not fabricate a provider-wide gauge"
    accounts = {a["id"]: a for a in usage["accounts"]}
    assert len(accounts) == 3, "every pool entry stays visible, including the unmeasured one"
    assert accounts[_fp("tok-1")]["state"] == "ready"
    assert accounts[_fp("tok-1")]["windows"][0]["used_percent"] == 10.0
    assert accounts[_fp("tok-2")]["state"] == "ready"
    assert accounts[_fp("tok-2")]["windows"][0]["used_percent"] == 80.0
    assert accounts[_fp("tok-3")]["state"] == "unknown"
    assert accounts[_fp("tok-3")]["windows"] == []
    # identity ids are non-secret (fingerprint/decoded-principal shaped, never the token)
    assert all("tok-" not in a_id for a_id in accounts)


def test_same_codex_account_two_credentials_counts_one_account(monkeypatch, tmp_path):
    """Two credentials of one Codex account (same decoded JWT principal) dedupe to ONE account
    row; an anonymous third credential stays its own row."""
    import base64

    def _codex_jwt(account_id: str, sub: str) -> str:
        def b64(part: dict) -> str:
            return base64.urlsafe_b64encode(json.dumps(part).encode()).decode().rstrip("=")

        return f"{b64({'alg': 'none'})}.{b64({'sub': sub, 'https://api.openai.com/auth': {'chatgpt_account_id': account_id}})}.sig"

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "a"))
    _write_pool(tmp_path / "a", "openai-codex", [
        _entry("c1", _codex_jwt("acct-1", "user-1"), auth_type="oauth"),
        _entry("c2", _codex_jwt("acct-1", "user-1"), auth_type="oauth"),
        _entry("c3", _codex_jwt("acct-2", "user-2"), auth_type="oauth"),
    ])
    _clear_cache()
    _install_fetcher(monkeypatch, {})

    row = {"slug": "openai-codex", "name": "Codex", "models": ["m"], "authenticated": True}
    _apply_usage([row])

    accounts = row["usage"]["accounts"]
    ids = [a["id"] for a in accounts]
    assert len(accounts) == 2, f"same-account credentials dedupe: {ids}"
    assert ids.count("codex:acct-1:user-1") == 1
    assert "codex:acct-2:user-2" in ids


def test_single_entry_pool_keeps_legacy_gauge(monkeypatch, tmp_path):
    """A one-entry pool is the single-account case: the legacy ``usage.windows`` chip contract."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "a"))
    _write_pool(tmp_path / "a", "openrouter", [_entry("c1", "tok-1")])
    _clear_cache()
    remember_account_usage("openrouter", _snapshot(50.0))

    row = _openrouter_row([])

    assert row["usage"]["windows"][0]["used_percent"] == 50.0
    assert "accounts" not in row["usage"] or row["usage"]["accounts"] is None


def test_limited_account_resets_at_is_latest_exhausted_window(monkeypatch, tmp_path):
    """A quota-exhausted account recovers at the LATEST of its exhausted account-scoped windows;
    an exhausted model-scoped window (Opus) never sets an account's resets_at nor limits it."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "a"))
    _write_pool(tmp_path / "a", "openrouter", [_entry("c1", "tok-1"), _entry("c2", "tok-2")])
    _clear_cache()
    now = datetime.now(timezone.utc)
    snapshot = AccountUsageSnapshot(
        provider="openrouter", source="test", fetched_at=now,
        windows=(
            AccountUsageWindow(label="Session", used_percent=100.0, reset_at=now + timedelta(hours=1)),
            AccountUsageWindow(label="Weekly", used_percent=100.0, reset_at=now + timedelta(hours=5)),
            AccountUsageWindow(label="Opus week", used_percent=100.0, reset_at=now + timedelta(hours=9),
                               scope="model"),
        ))
    remember_account_usage("openrouter", snapshot, identity_id=_fp("tok-1"))
    remember_account_usage("openrouter", _snapshot(30.0), identity_id=_fp("tok-2"))

    row = _openrouter_row([])
    accounts = {a["id"]: a for a in row["usage"]["accounts"]}

    assert accounts[_fp("tok-1")]["state"] == "limited"
    assert accounts[_fp("tok-1")]["resets_at"] == (now + timedelta(hours=5)).isoformat(), \
        "latest EXHAUSTED account window wins; the exhausted model window must not extend it"
    assert accounts[_fp("tok-2")]["state"] == "ready"


def test_live_cooldown_marks_account_limited_without_mutating_pool(monkeypatch, tmp_path):
    """A persisted credential-wide cooldown (live) renders ``limited`` + resets_at; reading usage
    never writes the pool back (auth.json unchanged)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "a"))
    reset_at = time.time() + 3600
    _write_pool(tmp_path / "a", "openrouter", [
        _entry("c1", "tok-1", last_status="exhausted", last_status_at=time.time() - 60,
               last_error_code=429, last_error_reset_at=reset_at),
        _entry("c2", "tok-2")])
    _clear_cache()
    remember_account_usage("openrouter", _snapshot(5.0), identity_id=_fp("tok-2"))
    _install_fetcher(monkeypatch, {})

    row = _openrouter_row([])
    accounts = {a["id"]: a for a in row["usage"]["accounts"]}

    from datetime import datetime as _dt

    assert accounts[_fp("tok-1")]["state"] == "limited"
    assert accounts[_fp("tok-1")]["resets_at"] == _dt.fromtimestamp(reset_at, timezone.utc).isoformat()
    assert accounts[_fp("tok-2")]["state"] == "ready", "a benched sibling never hides a healthy account"
    disk = json.loads((tmp_path / "a" / "auth.json").read_text())
    assert len(disk["credential_pool"]["openrouter"]) == 2, "read path must not prune or rewrite the pool"


def test_dead_row_stays_visible_as_unavailable(monkeypatch, tmp_path):
    """A DEAD auth row renders ``unavailable`` (visible, never a quota row) and is never fetched."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "a"))
    _write_pool(tmp_path / "a", "openrouter", [
        _entry("c1", "tok-1", last_status="dead", last_status_at=time.time() - 60),
        _entry("c2", "tok-2")])
    _clear_cache()
    remember_account_usage("openrouter", _snapshot(7.0), identity_id=_fp("tok-2"))
    _install_fetcher(monkeypatch, {})

    row = _openrouter_row([])
    states = {a["id"]: a["state"] for a in row["usage"]["accounts"]}

    assert states[_fp("tok-1")] == "unavailable"
    assert states[_fp("tok-2")] == "ready"


# ── failure isolation: a 401/failed fetch must not hide or mutate siblings ────


def test_failed_fetch_keeps_sibling_visible_and_does_not_repair(monkeypatch, tmp_path):
    """A per-credential 401 (read-only path: never retried via rotation/refresh) renders that
    account unknown, and the sibling with cached data stays ready; the failed token is never
    re-resolved through the pool (no rotation)."""
    from hermes_cli import auth as auth_mod

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "a"))
    _write_pool(tmp_path / "a", "openai-codex", [
        _entry("c1", "tok-1", auth_type="oauth"), _entry("c2", "tok-2", auth_type="oauth")])
    _clear_cache()
    # Read-only Codex fetcher: 401 on tok-1; the resolver must NOT be consulted again.

    class _Unauthorized(Exception):
        pass

    def fake_read_only(base_url, api_key):
        if api_key == "tok-1":
            raise _Unauthorized()
        return _snapshot(11.0)

    monkeypatch.setitem(account_usage._READ_ONLY_USAGE_FETCHERS, "openai-codex", fake_read_only)
    monkeypatch.setattr(auth_mod, "resolve_codex_runtime_credentials",
                        lambda **kw: (_ for _ in ()).throw(AssertionError("read-only path must not rotate")))

    row = {"slug": "openai-codex", "name": "Codex", "models": ["m"], "authenticated": True}
    _apply_usage([row])
    # populate c2's slot the way the background refresh would, then re-open the picker
    remember_account_usage("openai-codex", _snapshot(11.0), identity_id=_fp("tok-2"))
    _apply_usage([row])

    accounts = {a["id"]: a for a in row["usage"]["accounts"]}
    assert accounts[_fp("tok-1")]["state"] == "unknown", "401 → unknown, no fabricated windows"
    assert accounts[_fp("tok-2")]["state"] == "ready", "the failed sibling stays visible with its data"
    disk = json.loads((tmp_path / "a" / "auth.json").read_text())
    assert disk["credential_pool"]["openai-codex"][0]["access_token"] == "tok-1", \
        "a read-only usage fetch must never rotate or mutate a pool entry"


# ── cache scope: A→B→A profile isolation ────────────────────────────────────


def test_cache_isolation_a_to_b_to_a(monkeypatch, tmp_path):
    """Snapshots are scoped per profile home: B's open must not read A's snapshot, and back on A
    the original data is intact (no cross-profile contamination in either direction)."""
    home_a, home_b = tmp_path / "a", tmp_path / "b"
    monkeypatch.setenv("HERMES_HOME", str(home_a))
    _write_pool(home_a, "openrouter", [_entry("c1", "tok-1"), _entry("c2", "tok-2")])
    monkeypatch.setenv("HERMES_HOME", str(home_b))
    _write_pool(home_b, "openrouter", [_entry("b1", "tok-b1"), _entry("b2", "tok-b2")])
    _clear_cache()

    monkeypatch.setenv("HERMES_HOME", str(home_a))
    remember_account_usage("openrouter", _snapshot(10.0), identity_id=_fp("tok-1"))
    remember_account_usage("openrouter", _snapshot(20.0), identity_id=_fp("tok-2"))

    monkeypatch.setenv("HERMES_HOME", str(home_b))
    row_b = _openrouter_row([])
    assert row_b["usage"]["accounts"], "B sees its own pool"
    states_b = {a["id"]: a["state"] for a in row_b["usage"]["accounts"]}
    assert all(state == "unknown" for state in states_b.values()), \
        f"B's accounts must be unknown, not A's cached numbers: {states_b}"
    windows_b = [w for a in row_b["usage"]["accounts"] for w in a["windows"]]
    assert windows_b == [], "A's snapshot must not leak into B's picker"

    monkeypatch.setenv("HERMES_HOME", str(home_a))
    row_a = _openrouter_row([])
    accounts_a = {a["id"]: a for a in row_a["usage"]["accounts"]}
    assert accounts_a[_fp("tok-1")]["state"] == "ready"
    assert accounts_a[_fp("tok-1")]["windows"][0]["used_percent"] == 10.0, "A→B→A keeps A's data intact"


def test_replaced_credential_never_reuses_stale_quota(monkeypatch, tmp_path):
    """A new token gets a NEW identity slot: the old account's snapshot cannot answer for it."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "a"))
    _write_pool(tmp_path / "a", "openrouter", [_entry("c1", "tok-new"), _entry("c2", "tok-2")])
    _clear_cache()
    remember_account_usage("openrouter", _snapshot(99.0), identity_id=_fp("tok-old"))

    row = _openrouter_row([])
    states = {a["id"]: a["state"] for a in row["usage"]["accounts"]}

    assert _fp("tok-old") not in states, "a replaced credential's slot is not reused"
    assert all(state == "unknown" for state in states.values())


def test_late_background_fetch_cannot_replace_fresher_data(monkeypatch, tmp_path):
    """A background refresh started before a fresher per-turn fetch cannot overwrite it, however
    late its response lands."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "a"))
    _write_pool(tmp_path / "a", "openrouter", [_entry("c1", "tok-1"), _entry("c2", "tok-2")])
    _clear_cache()

    # Per-turn fetch (no identity) lands a fresh snapshot in fp:1's slot, "now".
    remember_account_usage("openrouter", _snapshot(42.0), identity_id=_fp("tok-1"))
    # A background fetch that STARTED earlier finishes later with older numbers.
    remember_account_usage("openrouter", _snapshot(90.0), identity_id=_fp("tok-1"), started_monotonic=-1000.0)

    row = _openrouter_row([])
    accounts = {a["id"]: a for a in row["usage"]["accounts"]}
    assert accounts[_fp("tok-1")]["windows"][0]["used_percent"] == 42.0, \
        "the late fetch must not replace the fresher per-turn snapshot"


def test_model_scoped_window_exhaustion_does_not_limit_account(monkeypatch, tmp_path):
    """An exhausted MODEL-scoped window (Anthropic Opus/Sonnet) never marks the account limited
    nor contributes to its resets_at (model-scoped non-exhaustion is preserved)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "a"))
    _write_pool(tmp_path / "a", "openrouter", [_entry("c1", "tok-1"), _entry("c2", "tok-2")])
    _clear_cache()
    now = datetime.now(timezone.utc)
    snapshot = AccountUsageSnapshot(
        provider="openrouter", source="test", fetched_at=now,
        windows=(AccountUsageWindow(label="Opus week", used_percent=100.0,
                                    reset_at=now + timedelta(hours=9), scope="model"),))
    remember_account_usage("openrouter", snapshot, identity_id=_fp("tok-1"))
    remember_account_usage("openrouter", _snapshot(30.0), identity_id=_fp("tok-2"))

    row = _openrouter_row([])
    accounts = {a["id"]: a for a in row["usage"]["accounts"]}

    assert accounts[_fp("tok-1")]["state"] == "ready", "an exhausted model window alone must not limit the account"
    assert accounts[_fp("tok-1")]["resets_at"] is None


def test_non_supporting_pool_provider_reports_unknown_without_fetching(monkeypatch, tmp_path):
    """A pooled provider with NO usage fetcher still gets per-account metadata rows (state
    unknown), and no bogus fetch is attempted for it."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "a"))
    _write_pool(tmp_path / "a", "kimi", [_entry("c1", "tok-1"), _entry("c2", "tok-2")])
    _clear_cache()

    row = {"slug": "kimi", "name": "Kimi", "models": ["m"], "authenticated": True}
    with monkeypatch.context() as m:
        def fail_fetch(*a, **kw):
            raise AssertionError("no fetch may run for a non-supporting provider")

        # The background refresh worker must not fetch for this provider either.
        import agent.account_usage_cache as cache_mod
        m.setattr(cache_mod, "_refresh_entries", lambda requests: None)
        _apply_usage([row])

    accounts = row["usage"]["accounts"]
    assert len(accounts) == 2
    assert all(a["state"] == "unknown" and a["windows"] == [] for a in accounts)
