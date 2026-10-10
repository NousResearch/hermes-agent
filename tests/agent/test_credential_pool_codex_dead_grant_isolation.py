"""A terminally rejected Codex refresh token retires exactly one grant, never another account.

Regression for #120741.

The dead unit is a refresh-token chain. Quarantining by source dropped every ``device_code`` row, so
an independent ``hermes auth add`` account dying (``invalid_grant``) deleted the healthy singleton
account from the pool; a same-grant alias of a dead singleton stayed selectable; the rewrite after
the quarantine copied stale in-memory tokens over rows a peer had rotated meanwhile. The grant-scoped
retirement must also take the pool lock only after the auth-store flock is released (a pool-lock
holder that persisted otherwise timed out), retire a same-grant row a peer added after the pool
loaded, and never revert a peer-rotated alias row to a stamp-less legacy singleton's consumed pair.
Rows are driven through the real ``load_pool`` / ``select`` / ``_refresh_entry`` path against a temp
HERMES_HOME; only the token-endpoint POST is stubbed. All tokens are synthetic.
"""
from __future__ import annotations

import base64
import json
import threading
import time

import pytest

import hermes_cli.auth as auth_mod
from agent import credential_pool as cp
from agent.credential_pool import STATUS_DEAD, STATUS_EXHAUSTED, load_pool
from hermes_cli.auth import AuthError


def _jwt(account: str, sub: str, exp: float) -> str:
    def seg(obj: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(obj).encode()).rstrip(b"=").decode()

    payload = {"exp": int(exp), "sub": sub, "https://api.openai.com/auth": {"chatgpt_account_id": account}}
    return f"{seg({'alg': 'none'})}.{seg(payload)}.sig"


def _iso(ts: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))


NOW = time.time()
FRESH, EXPIRING = NOW + 8 * 3600, NOW + 60


def _row(row_id: str, source: str, priority: int, at: str, rt: str) -> dict:
    return {"id": row_id, "label": row_id, "auth_type": "oauth", "priority": priority, "source": source,
            "access_token": at, "refresh_token": rt, "last_refresh": _iso(NOW - 3600)}


def _write(home, singleton: dict, rows: list, *, singleton_last_refresh: str | None = _iso(NOW - 3600)) -> None:
    state = {"tokens": singleton, "auth_mode": "chatgpt"}
    if singleton_last_refresh:  # None = a legacy writer that stamped no rotation time
        state["last_refresh"] = singleton_last_refresh
    home.mkdir(parents=True, exist_ok=True)
    (home / "auth.json").write_text(json.dumps({
        "version": 1, "active_provider": "openai-codex",
        "providers": {"openai-codex": state},
        "credential_pool": {"openai-codex": rows},
    }), encoding="utf-8")


def _store(home) -> dict:
    return json.loads((home / "auth.json").read_text(encoding="utf-8"))


def _rows(home) -> dict:
    return {r["id"]: r for r in _store(home)["credential_pool"]["openai-codex"]}


def _peer_rewrites_row(home, row_id: str, at: str, rt: str) -> None:
    """Another process rotated / re-authed *row_id* after this pool loaded."""
    store = _store(home)
    for row in store["credential_pool"]["openai-codex"]:
        if row["id"] == row_id:
            row.update(access_token=at, refresh_token=rt, last_refresh=_iso(NOW))
    (home / "auth.json").write_text(json.dumps(store), encoding="utf-8")


def _peer_adds_row(home, row: dict) -> None:
    """Another process added *row* after this pool loaded."""
    store = _store(home)
    store["credential_pool"]["openai-codex"].append(row)
    (home / "auth.json").write_text(json.dumps(store), encoding="utf-8")


@pytest.fixture
def home(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


@pytest.fixture
def token_endpoint(monkeypatch):
    """Stubbed refresh POST: ``dead`` refresh tokens get a terminal invalid_grant, ``flaky`` ones a
    transient timeout, others rotate."""
    calls = {"posted": [], "dead": set(), "flaky": set()}

    def fake(access_token, refresh_token):
        calls["posted"].append(refresh_token)
        if refresh_token in calls["flaky"]:
            raise TimeoutError("token endpoint timed out")
        if refresh_token in calls["dead"]:
            raise AuthError("Codex token refresh failed: refresh token revoked",
                            provider="openai-codex", code="invalid_grant", relogin_required=True)
        return {"access_token": _jwt("rotated", refresh_token, FRESH), "refresh_token": f"{refresh_token}-next",
                "last_refresh": _iso(time.time())}

    monkeypatch.setattr(auth_mod, "refresh_codex_oauth_pure", fake)
    return calls


def test_independent_account_survives_another_accounts_dead_grant(home, token_endpoint):
    """Account B's grant dying must not delete account A: fill_first keeps serving A (same id, same
    tokens, in memory and on disk), B leaves rotation as DEAD, A's auth.json singleton is untouched."""
    a_at = _jwt("acct-A", "user-A", FRESH)
    _write(home, {"access_token": a_at, "refresh_token": "rt-A"}, [
        _row("default", "device_code", 0, a_at, "rt-A"),
        _row("acct-b", "manual:device_code", 1, _jwt("acct-B", "user-B", EXPIRING), "rt-B"),
    ])
    token_endpoint["dead"].add("rt-B")
    pool = load_pool("openai-codex")

    first = pool.select()  # picks A; B is expiring, so it is refreshed (and dies) behind the selection
    assert first is not None and first.id == "default"
    assert token_endpoint["posted"] == ["rt-B"]

    again = pool.select()
    assert again is not None and (again.id, again.access_token) == ("default", a_at)
    rows = _rows(home)
    assert (rows["default"]["access_token"], rows["default"]["refresh_token"]) == (a_at, "rt-A")
    assert rows["acct-b"]["last_status"] == STATUS_DEAD
    assert _store(home)["providers"]["openai-codex"]["tokens"] == {"access_token": a_at, "refresh_token": "rt-A"}
    reloaded = load_pool("openai-codex").select()
    assert reloaded is not None and reloaded.id == "default"


@pytest.mark.parametrize("failing_id", ["default", "alias"])
def test_same_grant_alias_is_not_selectable_after_the_grant_dies(home, token_endpoint, failing_id):
    """A legacy ``manual:device_code`` alias holding the dead refresh token is the same grant: whichever
    row's forced refresh hit invalid_grant, neither copy stays selectable, while an independent
    account (different grant) still is."""
    x_at = _jwt("acct-X", "user-X", FRESH)  # unexpired bearer the upstream just rejected with a 401
    _write(home, {"access_token": x_at, "refresh_token": "rt-X"}, [
        _row("default", "device_code", 0, x_at, "rt-X"),
        _row("alias", "manual:device_code", 1, x_at, "rt-X"),
        _row("other", "manual:device_code", 2, _jwt("acct-C", "user-C", FRESH), "rt-C"),
    ])
    token_endpoint["dead"].add("rt-X")
    pool = load_pool("openai-codex")

    assert pool._refresh_entry(next(e for e in pool.entries() if e.id == failing_id), force=True) is None

    chosen = pool.select()
    assert chosen is not None and chosen.id == "other"
    rows = _rows(home)
    assert "default" not in rows or rows["default"]["last_status"] == STATUS_DEAD
    assert rows["alias"]["last_status"] == STATUS_DEAD
    assert rows["other"]["last_status"] != STATUS_DEAD and rows["other"]["refresh_token"] == "rt-C"
    assert token_endpoint["posted"] == ["rt-X"]


def test_dead_grant_does_not_clobber_rows_a_peer_replaced(home, token_endpoint):
    """Rows another process rotated / re-authed after this pool loaded keep their NEW pair: neither
    retired as part of the dead grant nor overwritten with this process's stale snapshot."""
    x_at = _jwt("acct-X", "user-X", FRESH)
    c_at = _jwt("acct-C", "user-C", FRESH)
    _write(home, {"access_token": x_at, "refresh_token": "rt-X"}, [
        _row("default", "device_code", 0, x_at, "rt-X"),
        _row("alias", "manual:device_code", 1, x_at, "rt-X"),
        _row("other", "manual:device_code", 2, c_at, "rt-C"),
    ])
    token_endpoint["dead"].add("rt-X")
    pool = load_pool("openai-codex")
    z_at, c2_at = _jwt("acct-X", "user-X", FRESH + 60), _jwt("acct-C", "user-C", FRESH + 60)
    _peer_rewrites_row(home, "alias", z_at, "rt-Z")
    _peer_rewrites_row(home, "other", c2_at, "rt-C2")

    assert pool._refresh_entry(next(e for e in pool.entries() if e.id == "default"), force=True) is None

    rows = _rows(home)
    assert (rows["alias"]["access_token"], rows["alias"]["refresh_token"]) == (z_at, "rt-Z")
    assert rows["alias"].get("last_status") != STATUS_DEAD
    assert (rows["other"]["access_token"], rows["other"]["refresh_token"]) == (c2_at, "rt-C2")
    chosen = pool.select()
    assert chosen is not None and (chosen.id, chosen.access_token) == ("alias", z_at)


def test_dead_grant_retires_a_same_grant_row_a_peer_added(home, token_endpoint):
    """A row another process added after this pool loaded, still holding the dead refresh token, is
    the same grant: it goes DEAD on disk too, while a row the peer rotated keeps its new pair."""
    x_at = _jwt("acct-X", "user-X", FRESH)
    _write(home, {"access_token": x_at, "refresh_token": "rt-X"}, [
        _row("default", "device_code", 0, x_at, "rt-X"),
        _row("other", "manual:device_code", 2, _jwt("acct-C", "user-C", FRESH), "rt-C"),
    ])
    token_endpoint["dead"].add("rt-X")
    pool = load_pool("openai-codex")
    _peer_adds_row(home, _row("late-alias", "manual:device_code", 1, x_at, "rt-X"))
    c2_at = _jwt("acct-C", "user-C", FRESH + 60)
    _peer_rewrites_row(home, "other", c2_at, "rt-C2")

    assert pool._refresh_entry(next(e for e in pool.entries() if e.id == "default"), force=True) is None

    rows = _rows(home)
    assert rows["late-alias"]["last_status"] == STATUS_DEAD
    assert (rows["other"]["access_token"], rows["other"]["refresh_token"]) == (c2_at, "rt-C2")
    assert rows["other"].get("last_status") != STATUS_DEAD
    for live in (pool, load_pool("openai-codex")):
        chosen = live.select()
        assert chosen is not None and (chosen.id, chosen.access_token) == ("other", c2_at)
    assert token_endpoint["posted"] == ["rt-X"]


def test_dead_grant_retirement_does_not_block_a_pool_lock_holder_on_the_flock(home, token_endpoint, monkeypatch):
    """Thread 1's deferred refresh of B dies (invalid_grant) while it holds the auth-store flock; thread 2
    holds the pool lock and persists (a 429 on account C). The retirement must wait until the flock is
    released and take pool -> auth order: both finish without a lock timeout and A still serves."""
    a_at = _jwt("acct-A", "user-A", FRESH)
    _write(home, {"access_token": a_at, "refresh_token": "rt-A"}, [
        _row("default", "device_code", 0, a_at, "rt-A"),
        _row("acct-b", "manual:device_code", 1, _jwt("acct-B", "user-B", EXPIRING), "rt-B"),
        _row("acct-c", "manual:device_code", 2, _jwt("acct-C", "user-C", FRESH), "rt-C"),
    ])
    token_endpoint["dead"].add("rt-B")
    monkeypatch.setattr(auth_mod, "_probe_codex_quota_restored", lambda *a, **k: False)  # offline
    pool = load_pool("openai-codex")

    posting, writer_holds_pool_lock = threading.Event(), threading.Event()
    stub_post, real_persist = auth_mod.refresh_codex_oauth_pure, cp.persist_pool_entries

    def post_while_writer_waits(access_token, refresh_token):
        posting.set()  # thread 1 now holds the flock
        writer_holds_pool_lock.wait(10)
        return stub_post(access_token, refresh_token)

    def persist(*args, **kwargs):
        if threading.current_thread().name == "writer":
            writer_holds_pool_lock.set()  # inside mark_exhausted_and_rotate; the flock is next
        return real_persist(*args, **kwargs)

    monkeypatch.setattr(auth_mod, "refresh_codex_oauth_pure", post_while_writer_waits)
    monkeypatch.setattr(cp, "persist_pool_entries", persist)
    results, errors = {}, []

    def run(fn):
        try:
            results[threading.current_thread().name] = fn()
        except BaseException as exc:  # a lock timeout surfaces here
            errors.append(exc)

    refresher = threading.Thread(target=run, name="refresher", args=(pool.select,))
    writer = threading.Thread(target=run, name="writer", args=(
        lambda: pool.mark_exhausted_and_rotate(status_code=429, credential_id="acct-c"),))
    refresher.start()
    assert posting.wait(10)
    writer.start()
    refresher.join(60)
    writer.join(60)

    assert not refresher.is_alive() and not writer.is_alive()
    assert errors == [] and writer_holds_pool_lock.is_set()
    assert results["refresher"].id == "default" and results["writer"].id == "default"
    rows = _rows(home)
    assert rows["acct-b"]["last_status"] == STATUS_DEAD
    assert rows["acct-c"]["last_status"] == STATUS_EXHAUSTED
    assert (rows["default"]["access_token"], rows["default"]["refresh_token"]) == (a_at, "rt-A")
    chosen = pool.select()
    assert chosen is not None and (chosen.id, chosen.access_token) == ("default", a_at)


@pytest.mark.parametrize("case", ["peer_pair_fresh", "forced_refresh", "forced_refresh_fails_transiently"])
def test_peer_rotated_alias_is_not_reverted_to_a_legacy_singleton(home, token_endpoint, case):
    """A same-account ``manual:device_code`` alias adopts the pair a peer rotated into its persisted row.
    A legacy singleton (no ``last_refresh``) still holding the consumed pair cannot prove it is newer,
    so it must replace the adopted pair neither before the POST nor after a failed one."""
    old_at = _jwt("acct-A", "user-A", EXPIRING)
    _write(home, {"access_token": old_at, "refresh_token": "rt-A"},
           [_row("alias", "manual:device_code", 0, old_at, "rt-A")], singleton_last_refresh=None)
    token_endpoint["dead"].add("rt-A")  # consumed by the peer's rotation
    if case == "forced_refresh_fails_transiently":
        token_endpoint["flaky"].add("rt-P")
    pool = load_pool("openai-codex")
    new_at = _jwt("acct-A", "user-A", FRESH)
    _peer_rewrites_row(home, "alias", new_at, "rt-P")

    result = pool._refresh_entry(next(e for e in pool.entries() if e.id == "alias"), force=case != "peer_pair_fresh")

    assert "rt-A" not in token_endpoint["posted"]
    expected_rt = "rt-P-next" if case == "forced_refresh" else "rt-P"
    assert _rows(home)["alias"]["refresh_token"] == expected_rt
    assert next(e for e in pool.entries() if e.id == "alias").refresh_token == expected_rt
    if case == "peer_pair_fresh":
        assert result is not None and result.access_token == new_at and token_endpoint["posted"] == []
    # A manual row never writes back to the singleton.
    assert _store(home)["providers"]["openai-codex"]["tokens"] == {"access_token": old_at, "refresh_token": "rt-A"}


def test_failing_row_rotated_by_a_peer_is_adopted_not_replayed(home, token_endpoint):
    """Another process already rotated account B (its old refresh token is consumed): this process must
    adopt B's new pair instead of replaying the consumed token, and must neither retire B nor touch A."""
    a_at = _jwt("acct-A", "user-A", FRESH)
    _write(home, {"access_token": a_at, "refresh_token": "rt-A"}, [
        _row("default", "device_code", 0, a_at, "rt-A"),
        _row("acct-b", "manual:device_code", 1, _jwt("acct-B", "user-B", EXPIRING), "rt-B"),
    ])
    token_endpoint["dead"].add("rt-B")  # consumed by the peer's rotation
    pool = load_pool("openai-codex")
    b2_at = _jwt("acct-B", "user-B", FRESH)
    _peer_rewrites_row(home, "acct-b", b2_at, "rt-B2")

    assert pool.select().id == "default"

    assert "rt-B" not in token_endpoint["posted"]
    rows = _rows(home)
    assert (rows["acct-b"]["access_token"], rows["acct-b"]["refresh_token"]) == (b2_at, "rt-B2")
    assert rows["acct-b"].get("last_status") != STATUS_DEAD
    assert (rows["default"]["access_token"], rows["default"]["refresh_token"]) == (a_at, "rt-A")
    assert {e.id for e in pool.entries()} == {"default", "acct-b"}


def test_refresh_and_relogin_each_touch_only_their_own_grant(home, token_endpoint):
    """Control for the isolation above: a healthy refresh of account B rotates only B; a re-login of
    the singleton updates the singleton row and its legacy alias, never the independent account."""
    a_at = _jwt("acct-A", "user-A", FRESH)
    b_at = _jwt("acct-B", "user-B", EXPIRING)
    _write(home, {"access_token": a_at, "refresh_token": "rt-A"}, [
        _row("default", "device_code", 0, a_at, "rt-A"),
        _row("alias", "manual:device_code", 1, a_at, "rt-A"),
        _row("acct-b", "manual:device_code", 2, b_at, "rt-B"),
    ])
    pool = load_pool("openai-codex")
    assert pool.select().id == "default"
    assert token_endpoint["posted"] == ["rt-B"]
    rows = _rows(home)
    assert rows["acct-b"]["refresh_token"] == "rt-B-next"
    assert [rows[i]["refresh_token"] for i in ("default", "alias")] == ["rt-A", "rt-A"]
    assert _store(home)["providers"]["openai-codex"]["tokens"]["refresh_token"] == "rt-A"

    relogin_at = _jwt("acct-A", "user-A", FRESH + 120)
    auth_mod._save_codex_tokens({"access_token": relogin_at, "refresh_token": "rt-A-relogin"})
    rows = _rows(home)
    assert [rows[i]["refresh_token"] for i in ("default", "alias")] == ["rt-A-relogin", "rt-A-relogin"]
    assert rows["acct-b"]["refresh_token"] == "rt-B-next"


def test_dead_grant_does_not_clear_other_accounts_access_only_singleton(home, token_endpoint):
    """An access-only singleton may belong to A; B's dead grant must not clear A's bearer."""
    a_at = _jwt("acct-A", "user-A", FRESH)
    b_at = _jwt("acct-B", "user-B", EXPIRING)
    _write(home, {"access_token": a_at}, [
        _row("default", "device_code", 0, a_at, "rt-A"),
        _row("acct-b", "manual:device_code", 1, b_at, "rt-B"),
    ])
    token_endpoint["dead"].add("rt-B")
    pool = load_pool("openai-codex")
    failing = next(row for row in pool.entries() if row.id == "acct-b")
    assert pool._refresh_entry(failing, force=True) is None
    assert _store(home)["providers"]["openai-codex"]["tokens"] == {"access_token": a_at}
    assert _rows(home)["default"]["access_token"] == a_at


def test_dead_grant_clears_access_only_singleton_with_same_bearer(home, token_endpoint):
    """An access-only singleton matching the failed row is not another account."""
    b_at = _jwt("acct-B", "user-B", EXPIRING)
    _write(home, {"access_token": b_at}, [
        _row("acct-b", "manual:device_code", 0, b_at, "rt-B"),
    ])
    token_endpoint["dead"].add("rt-B")
    pool = load_pool("openai-codex")
    failing = next(row for row in pool.entries() if row.id == "acct-b")
    assert pool._refresh_entry(failing, force=True) is None
    state = _store(home)["providers"]["openai-codex"]
    assert state["tokens"] == {}
    assert state["last_auth_error"]["relogin_required"] is True
