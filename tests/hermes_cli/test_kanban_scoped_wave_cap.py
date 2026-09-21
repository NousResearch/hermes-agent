"""Phase2 P3 — scoped wave-cap + scoped circuit breaker (branch-only).

Ladder (each independently meaningful):
- key normalization (+ unknown bucket, still capped),
- wave-cap defers over-cap tasks / kill-switch 0 restores prior behaviour /
  running-with-same-key counts,
- scoped pause hits ONLY parked-key rows (healthy lane flows),
- circuit trips at >=3 signals (not 2), no status writes from the circuit,
- stale parks self-clear (expiry),
- exactly one probe spawn per park window (+ 1 reserved healthy slot),
- P1 cause contract (telemetry key when present, raw-text fallback),
- P2 union contract (either source suffices; raising/unparseable falls through),
- kill-switches off restore prior behaviour; success clears the park.

Resolution is stubbed per test (``resolve_task_upstream_key`` monkeypatched
by assignee); one integration test drives real crash -> signal -> park.
"""
from __future__ import annotations

import os
import sys
import tempfile
import time

import pytest


@pytest.fixture()
def isolated_kanban_home(monkeypatch):
    """Fresh HERMES_HOME with kanban DB + alpha/beta profiles; clean circuit."""
    test_home = tempfile.mkdtemp(prefix="kanban_p3_scoped_cap_test_")
    for prof in ("alpha", "beta", "default"):
        os.makedirs(os.path.join(test_home, "profiles", prof), exist_ok=True)
        with open(os.path.join(test_home, "profiles", prof, "config.yaml"), "w") as fh:
            fh.write("{}\n")  # identity marker: a bare dir is not a profile
    monkeypatch.setenv("HERMES_HOME", test_home)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    for mod in list(sys.modules.keys()):
        if mod.startswith("hermes_cli") or mod.startswith("hermes_state") or mod == "hermes_constants":
            del sys.modules[mod]
    from hermes_cli import kanban_db_dispatch as kbd
    kbd._UPSTREAM_CIRCUIT.clear()
    saved_providers = list(kbd._EXTRA_UPSTREAM_PARK_PROVIDERS)
    kbd._EXTRA_UPSTREAM_PARK_PROVIDERS[:] = []
    yield
    kbd._UPSTREAM_CIRCUIT.clear()
    kbd._EXTRA_UPSTREAM_PARK_PROVIDERS[:] = saved_providers


def _mods():
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    return kb, kbc, kbd


def _stub_keys(monkeypatch, kbd, mapping=None):
    """Stub upstream resolution: assignee -> pre-normalized lowercase key."""
    mapping = mapping or {}
    def fake(assignee, model_override=None, provider_override=None):
        if provider_override or model_override:
            return str(provider_override or model_override).lower()
        return mapping.get(assignee or "", "k")
    monkeypatch.setattr(kbd, "resolve_task_upstream_key", fake)
    return fake


def _fake_spawn(pid=999999999):
    def _spawn(*args, **kwargs):
        return pid
    return _spawn


def _park(kbd, board, key, n=3):
    for _ in range(n):
        kbd.note_upstream_signal(board, key)


# --- key normalization --------------------------------------------------------

def test_normalize_upstream_key(isolated_kanban_home):
    _, _, kbd = _mods()
    assert kbd.normalize_upstream_key("https://opencode.ai/zen/go/v1/") == "https://opencode.ai/zen/go/v1"
    assert kbd.normalize_upstream_key("  HTTPS://X.AI/ ") == "https://x.ai"
    assert kbd.normalize_upstream_key("") == "unknown"
    assert kbd.normalize_upstream_key(None) == "unknown"


def test_resolve_unknown_on_resolver_failure(isolated_kanban_home, monkeypatch):
    _, _, kbd = _mods()
    from hermes_cli import runtime_provider as rp
    def boom(**kwargs):
        raise RuntimeError("no creds")
    monkeypatch.setattr(rp, "resolve_runtime_provider", boom)
    assert kbd.resolve_task_upstream_key("alpha") == "unknown"


def test_normalize_wave_cap_setting(isolated_kanban_home):
    _, _, kbd = _mods()
    assert kbd._normalize_wave_cap_setting(None) == 3
    assert kbd._normalize_wave_cap_setting(0) == 0
    assert kbd._normalize_wave_cap_setting(1) == 1
    assert kbd._normalize_wave_cap_setting(4) == 4
    assert kbd._normalize_wave_cap_setting(5) == 3
    assert kbd._normalize_wave_cap_setting(-1) == 3
    assert kbd._normalize_wave_cap_setting("nope") == 3


def test_gateway_settings_defaults_and_kill_switches(isolated_kanban_home):
    from types import SimpleNamespace
    from gateway.kanban_watchers_dispatcher import _resolve_dispatcher_settings
    kb = SimpleNamespace(DEFAULT_FAILURE_LIMIT=2)
    s = _resolve_dispatcher_settings({}, kb)
    assert s.wave_cap_per_upstream == 3
    assert s.scoped_circuit_enabled is True
    s2 = _resolve_dispatcher_settings(
        {"wave_cap_per_upstream": 0, "scoped_circuit_enabled": False}, kb,
    )
    assert s2.wave_cap_per_upstream == 0
    assert s2.scoped_circuit_enabled is False
    s3 = _resolve_dispatcher_settings({"wave_cap_per_upstream": 9}, kb)
    assert s3.wave_cap_per_upstream == 3


# --- wave-cap -----------------------------------------------------------------

def test_wave_cap_defers_over_cap(isolated_kanban_home, monkeypatch):
    kb, kbc, kbd = _mods()
    _stub_keys(monkeypatch, kbd, {"alpha": "k"})
    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        for i in range(5):
            kb.create_task(conn, title=f"a{i}", assignee="alpha")
    with kbc.connect_closing() as conn:
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn(), dry_run=True, wave_cap_per_upstream=2,
        )
    assert len(res.spawned) == 2
    assert len(res.skipped_upstream_capped) == 3
    assert {t[1] for t in res.skipped_upstream_capped} == {"k"}
    assert all(t[2] == 2 for t in res.skipped_upstream_capped)


def test_wave_cap_zero_restores_prior_behaviour(isolated_kanban_home, monkeypatch):
    kb, kbc, kbd = _mods()
    _stub_keys(monkeypatch, kbd, {"alpha": "k"})
    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        for i in range(5):
            kb.create_task(conn, title=f"a{i}", assignee="alpha")
    with kbc.connect_closing() as conn:
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn(), dry_run=True, wave_cap_per_upstream=0,
        )
    assert len(res.spawned) == 5
    assert res.skipped_upstream_capped == []


def test_wave_cap_counts_running_with_same_key(isolated_kanban_home, monkeypatch):
    kb, kbc, kbd = _mods()
    _stub_keys(monkeypatch, kbd, {"alpha": "k"})
    # High crash grace: the tick-1 spawn stays running, so tick 2 measures
    # running-with-same-key (with grace=0 the reaper would crash it first).
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "99999")
    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        for i in range(3):
            kb.create_task(conn, title=f"a{i}", assignee="alpha")
    with kbc.connect_closing() as conn:
        res1 = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn(), dry_run=False, wave_cap_per_upstream=1,
        )
    assert len(res1.spawned) == 1
    assert len(res1.skipped_upstream_capped) == 2
    # Tick 2: the tick-1 spawn is still running -> running-with-same-key = 1
    # blocks everything new at cap 1.
    with kbc.connect_closing() as conn:
        res2 = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn(), dry_run=True, wave_cap_per_upstream=1,
        )
    assert res2.spawned == []
    assert len(res2.skipped_upstream_capped) == 2
    assert all(t[2] == 1 for t in res2.skipped_upstream_capped)


def test_wave_cap_scopes_per_key_not_global(isolated_kanban_home, monkeypatch):
    kb, kbc, kbd = _mods()
    _stub_keys(monkeypatch, kbd, {"alpha": "k", "beta": "j"})
    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        for i in range(2):
            kb.create_task(conn, title=f"a{i}", assignee="alpha")
        for i in range(2):
            kb.create_task(conn, title=f"b{i}", assignee="beta")
    with kbc.connect_closing() as conn:
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn(), dry_run=True, wave_cap_per_upstream=1,
        )
    by_key = {}
    for tid, key, current in res.skipped_upstream_capped:
        by_key.setdefault(key, []).append(tid)
    assert len(res.spawned) == 2  # one per key
    assert sorted(by_key) == ["j", "k"]


# --- scoped pause ---------------------------------------------------------------

def test_scoped_pause_hits_only_parked_key(isolated_kanban_home, monkeypatch):
    kb, kbc, kbd = _mods()
    _stub_keys(monkeypatch, kbd, {"alpha": "k", "beta": "j"})
    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        for i in range(3):
            kb.create_task(conn, title=f"a{i}", assignee="alpha")
        for i in range(2):
            kb.create_task(conn, title=f"b{i}", assignee="beta")
    _park(kbd, None, "k")
    with kbc.connect_closing() as conn:
        # dry_run: parked rows report scoped_paused, no probe consumes quota.
        res = kbd.dispatch_once(conn, spawn_fn=_fake_spawn(), dry_run=True)
    paused_ids = {t[0] for t in res.scoped_paused}
    assert len(res.scoped_paused) == 3
    assert all(t[1] == "k" for t in res.scoped_paused)
    spawned_ids = {t[0] for t in res.spawned}
    assert len(spawned_ids) == 2
    assert not (paused_ids & spawned_ids)
    # Parked rows were never claimed / status-written.
    with kbc.connect_closing() as conn:
        rows = conn.execute(
            "SELECT id, status, claim_lock FROM tasks WHERE status = 'ready'"
        ).fetchall()
        assert {r["id"] for r in rows} >= paused_ids
        assert all(r["claim_lock"] is None for r in rows if r["id"] in paused_ids)


def test_circuit_kill_switch_off_restores_prior_behaviour(isolated_kanban_home, monkeypatch):
    kb, kbc, kbd = _mods()
    _stub_keys(monkeypatch, kbd, {"alpha": "k"})
    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        kb.create_task(conn, title="a0", assignee="alpha")
    _park(kbd, None, "k")
    with kbc.connect_closing() as conn:
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn(), dry_run=True, scoped_circuit_enabled=False,
        )
    assert res.scoped_paused == []
    assert len(res.spawned) == 1


# --- circuit trip / expiry / probe ----------------------------------------------

def test_circuit_trips_at_3_not_2(isolated_kanban_home):
    _, _, kbd = _mods()
    assert kbd.note_upstream_signal(None, "k") is False
    assert kbd.get_parked_upstream_keys(None) == frozenset()
    assert kbd.note_upstream_signal(None, "k") is False
    assert kbd.get_parked_upstream_keys(None) == frozenset()
    assert kbd.note_upstream_signal(None, "k") is True
    assert kbd.get_parked_upstream_keys(None) == frozenset({"k"})


def test_stale_park_self_clears(isolated_kanban_home, monkeypatch):
    kb, kbc, kbd = _mods()
    _stub_keys(monkeypatch, kbd, {"alpha": "k"})
    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        kb.create_task(conn, title="a0", assignee="alpha")
    _park(kbd, None, "k")
    slot = ("", "k")
    kbd._UPSTREAM_CIRCUIT[slot]["parked_until"] = time.monotonic() - 1
    with kbc.connect_closing() as conn:
        res = kbd.dispatch_once(conn, spawn_fn=_fake_spawn(), dry_run=True)
    assert res.scoped_paused == []
    assert len(res.spawned) == 1
    assert slot not in kbd._UPSTREAM_CIRCUIT


def test_probe_is_single_spawn_per_window(isolated_kanban_home, monkeypatch):
    kb, kbc, kbd = _mods()
    _stub_keys(monkeypatch, kbd, {"alpha": "k", "beta": "j"})
    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        for i in range(3):
            kb.create_task(conn, title=f"a{i}", assignee="alpha")
        kb.create_task(conn, title="b0", assignee="beta")
    _park(kbd, None, "k")
    with kbc.connect_closing() as conn:
        res1 = kbd.dispatch_once(conn, spawn_fn=_fake_spawn(), dry_run=False)
    k_spawns_1 = [t for t in res1.spawned if t[1] == "alpha"]
    assert len(k_spawns_1) == 1  # exactly one probe
    assert len(res1.scoped_paused) == 2
    assert kbd._UPSTREAM_CIRCUIT[("", "k")]["probe_inflight"] is True
    with kbc.connect_closing() as conn:
        res2 = kbd.dispatch_once(conn, spawn_fn=_fake_spawn(), dry_run=False)
    k_spawns_2 = [t for t in res2.spawned if t[1] == "alpha"]
    assert k_spawns_2 == []  # probe consumed; park still holds


def test_probe_holds_one_slot_for_healthy_keys(isolated_kanban_home, monkeypatch):
    kb, kbc, kbd = _mods()
    _stub_keys(monkeypatch, kbd, {"alpha": "k", "beta": "j"})
    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        kb.create_task(conn, title="a0", assignee="alpha")  # parked row first
        kb.create_task(conn, title="b0", assignee="beta")
    _park(kbd, None, "k")
    with kbc.connect_closing() as conn:
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn(), dry_run=False, max_spawn=1,
        )
    # Budget 1: the healthy task wins; the probe must not steal the last slot.
    assert [t[1] for t in res.spawned] == ["beta"]
    assert len(res.scoped_paused) == 1
    assert kbd._UPSTREAM_CIRCUIT[("", "k")]["probe_inflight"] is not True


def test_success_clears_park(isolated_kanban_home, monkeypatch):
    kb, kbc, kbd = _mods()
    _stub_keys(monkeypatch, kbd, {"alpha": "k"})
    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        tid = kb.create_task(conn, title="a0", assignee="alpha")
    _park(kbd, None, "k")
    assert kbd.get_parked_upstream_keys(None) == frozenset({"k"})
    with kbc.connect_closing() as conn:
        kbd._clear_failure_counter(conn, tid)
    assert kbd.get_parked_upstream_keys(None) == frozenset()
    with kbc.connect_closing() as conn:
        events = kb.list_events(conn, tid)
    assert any(getattr(e, "kind", None) == "upstream_unparked" for e in events)


# --- P1 / P2 contracts ------------------------------------------------------------

def test_p1_cause_prefers_telemetry_then_text(isolated_kanban_home, monkeypatch):
    kb, kbc, kbd = _mods()
    _stub_keys(monkeypatch, kbd, {"alpha": "k"})
    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        tid = kb.create_task(conn, title="a0", assignee="alpha")
    with kbc.connect_closing() as conn:
        # No run row: falls back to raw-text classification.
        cause = kbd._read_failure_cause(conn, tid, "boom 429 too many requests")
        assert cause["source"] == "text" and cause["reason"] == "rate_limit"
        cause2 = kbd._read_failure_cause(conn, tid, "utterly mysterious")
        assert cause2["reason"] == "unclassified"
        # Additive telemetry key wins when present (never filters).
        with kb.write_txn(conn):
            run_id = kb._current_run_id(conn, tid)
            if run_id is None:
                run_id = kb._start_run(conn, tid, "worker") if hasattr(kb, "_start_run") else None
        if run_id is not None:
            with kb.write_txn(conn):
                conn.execute(
                    "UPDATE task_runs SET ended_at = ?, metadata = ? WHERE id = ?",
                    (1, '{"provider_error": {"reason": "billing", "code": 402}}', run_id),
                )
            cause3 = kbd._read_failure_cause(conn, tid, "utterly mysterious")
            assert cause3["source"] == "telemetry" and cause3["reason"] == "billing"


def test_p2_union_either_source_suffices_and_fallthrough(isolated_kanban_home, monkeypatch):
    kb, kbc, kbd = _mods()
    _stub_keys(monkeypatch, kbd, {"alpha": "k", "beta": "j"})
    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        kb.create_task(conn, title="a0", assignee="alpha")
        kb.create_task(conn, title="b0", assignee="beta")
    # P2-style provider alone suffices (no circuit signals at all).
    kbd._EXTRA_UPSTREAM_PARK_PROVIDERS.append(lambda board: ["j"])
    try:
        assert kbd.get_parked_upstream_keys(None) == frozenset({"j"})
        with kbc.connect_closing() as conn:
            res = kbd.dispatch_once(conn, spawn_fn=_fake_spawn(), dry_run=True)
        assert {t[1] for t in res.scoped_paused} == {"j"}
        assert [t[1] for t in res.spawned] == ["alpha"]
    finally:
        kbd._EXTRA_UPSTREAM_PARK_PROVIDERS[:] = []
    # Raising / unparseable providers fall through, never freeze.
    kbd._EXTRA_UPSTREAM_PARK_PROVIDERS.append(lambda board: 1 / 0)
    kbd._EXTRA_UPSTREAM_PARK_PROVIDERS.append(lambda board: [123, None, {"nope": 1}])
    try:
        with kbc.connect_closing() as conn:
            res2 = kbd.dispatch_once(conn, spawn_fn=_fake_spawn(), dry_run=True)
        assert res2.scoped_paused == []
        assert len(res2.spawned) == 2
    finally:
        kbd._EXTRA_UPSTREAM_PARK_PROVIDERS[:] = []


# --- real crash -> signal -> park integration --------------------------------------

def test_real_crashes_feed_circuit_then_park(isolated_kanban_home):
    kb, kbc, kbd = _mods()
    # Real resolver (no stub): whatever the test env resolves to, every cycle
    # shares one key — the circuit still learns and parks. Each tick both
    # reaps the previous dead worker (crash + signal) and respawns.
    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        tid = kb.create_task(conn, title="a0", assignee="alpha", max_retries=10)
    with kbc.connect_closing() as conn:
        res0 = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn(), dry_run=False, failure_limit=10,
        )
    assert [t[0] for t in res0.spawned] == [tid]
    for _ in range(3):
        with kbc.connect_closing() as conn:
            res = kbd.dispatch_once(
                conn, spawn_fn=_fake_spawn(), dry_run=False, failure_limit=10,
            )
        assert tid in res.crashed
    total_signals = sum(e.get("signals", 0) for e in kbd._UPSTREAM_CIRCUIT.values())
    assert total_signals == 3
    assert len(kbd.get_parked_upstream_keys(None)) == 1
    # The 3rd crash tick parked the key, then spent the single probe on the
    # same tick's respawn; the next tick holds the scoped pause (no new probe).
    with kbc.connect_closing() as conn:
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn(), dry_run=False, failure_limit=10,
        )
        row = conn.execute("SELECT status FROM tasks WHERE id = ?", (tid,)).fetchone()
    assert [t[0] for t in res.spawned] == []
    assert [t[0] for t in res.scoped_paused] == [tid]
    assert row["status"] == "ready"  # circuit never writes task status


def test_describe_suppression_surfaces_p3_buckets(isolated_kanban_home):
    _, _, kbd = _mods()
    res = kbd.DispatchResult(
        skipped_upstream_capped=[("t1", "k", 3), ("t2", "k", 3)],
        scoped_paused=[("t3", "j")],
    )
    held = kbd.describe_suppression([res])
    assert "upstream_capped=2" in held
    assert "scoped_paused=1" in held


# --- H3 fix: board-scoped, kill-switched success-clear --------------------------

def _upstream_unparked_events(kb, conn, tid):
    return [
        e for e in kb.list_events(conn, tid)
        if getattr(e, "kind", None) == "upstream_unparked"
    ]


def test_success_clear_kill_switch_off_restores_prior_behaviour(
    isolated_kanban_home, monkeypatch,
):
    # (1) Kill-switch restore: populated circuit + complete_task with the
    # circuit disabled -> dict untouched, no upstream_unparked event, only
    # the consecutive_failures UPDATE lands.
    kb, kbc, kbd = _mods()
    _stub_keys(monkeypatch, kbd, {"alpha": "k"})
    monkeypatch.setattr(kbd, "_resolve_scoped_circuit_enabled", lambda: False)
    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        tid = kb.create_task(conn, title="a0", assignee="alpha")
        conn.execute(
            "UPDATE tasks SET consecutive_failures = 3, last_failure_error = 'boom' WHERE id = ?",
            (tid,),
        )
    _park(kbd, "A", "k")
    before = dict(kbd._UPSTREAM_CIRCUIT)
    assert before != {}
    with kbc.connect_closing() as conn:
        assert kb.complete_task(conn, tid, board="A", result="verified ok") is True
    assert dict(kbd._UPSTREAM_CIRCUIT) == before
    with kbc.connect_closing() as conn:
        row = conn.execute(
            "SELECT status, consecutive_failures, last_failure_error FROM tasks WHERE id = ?",
            (tid,),
        ).fetchone()
        assert row["status"] == "done"
        assert row["consecutive_failures"] == 0
        assert row["last_failure_error"] is None
        assert _upstream_unparked_events(kb, conn, tid) == []


def test_success_clear_scopes_to_completing_board(isolated_kanban_home, monkeypatch):
    # (2) Cross-board isolation: parks on (A,K) + (B,K), complete on A ->
    # only A pops, B intact with its signal count. Repeat with the unknown
    # bucket: nothing pops anywhere (a blind-bucket success proves nothing).
    kb, kbc, kbd = _mods()
    _stub_keys(monkeypatch, kbd, {"alpha": "k"})
    monkeypatch.setattr(kbd, "_resolve_scoped_circuit_enabled", lambda: True)
    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        tid_a = kb.create_task(conn, title="a0", assignee="alpha")
        tid_u = kb.create_task(conn, title="u0", assignee="alpha")
    _park(kbd, "A", "k")
    _park(kbd, "B", "k")
    with kbc.connect_closing() as conn:
        assert kb.complete_task(conn, tid_a, board="A", result="verified ok") is True
    assert ("A", "k") not in kbd._UPSTREAM_CIRCUIT
    assert kbd._UPSTREAM_CIRCUIT[("B", "k")]["signals"] == 3
    with kbc.connect_closing() as conn:
        cleared = _upstream_unparked_events(kb, conn, tid_a)
        assert len(cleared) == 1
        assert (cleared[0].payload or {}).get("upstream_key") == "k"
    # Unknown bucket: parked on two boards, completing an unknown-keyed task
    # pops neither and emits no event.
    _park(kbd, "A", "unknown")
    _park(kbd, "B", "unknown")
    monkeypatch.setattr(kbd, "resolve_task_upstream_key", lambda *a, **k: "unknown")
    with kbc.connect_closing() as conn:
        assert kb.complete_task(conn, tid_u, board="A", result="verified ok") is True
    assert kbd._UPSTREAM_CIRCUIT[("A", "unknown")]["signals"] == 3
    assert kbd._UPSTREAM_CIRCUIT[("B", "unknown")]["signals"] == 3
    with kbc.connect_closing() as conn:
        assert _upstream_unparked_events(kb, conn, tid_u) == []


def test_success_clear_drops_stale_park_once(isolated_kanban_home, monkeypatch):
    # (3) Stale-park expiry: parked_until in the past -> the success-clear
    # drops it (same as a live park), emits a single event, and the probe
    # state resets (next signals start a fresh episode).
    kb, kbc, kbd = _mods()
    _stub_keys(monkeypatch, kbd, {"alpha": "k"})
    monkeypatch.setattr(kbd, "_resolve_scoped_circuit_enabled", lambda: True)
    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        tid = kb.create_task(conn, title="a0", assignee="alpha")
    _park(kbd, None, "k")
    kbd._UPSTREAM_CIRCUIT[("", "k")]["parked_until"] = time.monotonic() - 1
    with kbc.connect_closing() as conn:
        assert kb.complete_task(conn, tid, result="verified ok") is True
    assert ("", "k") not in kbd._UPSTREAM_CIRCUIT
    assert kbd.get_parked_upstream_keys(None) == frozenset()
    with kbc.connect_closing() as conn:
        assert len(_upstream_unparked_events(kb, conn, tid)) == 1
    # Probe reset: a fresh episode starts at signal 1, unparked, no probe.
    assert kbd.note_upstream_signal(None, "k") is False
    fresh = kbd._UPSTREAM_CIRCUIT[("", "k")]
    assert fresh["signals"] == 1 and fresh["parked_until"] is None
    assert fresh["probe_inflight"] is False
