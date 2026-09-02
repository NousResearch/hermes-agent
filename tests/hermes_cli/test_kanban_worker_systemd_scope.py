"""Kanban worker systemd-scope isolation — spawn, re-adoption, reaping.

Regression tests for the production incident (networkos-agent, 2026-09-01):
workers spawned as plain children of the gateway shared its cgroup, so
build workers (dev servers, browsers, DBs) OOM-throttled the gateway and
each gateway restart orphaned every in-flight run — the new gateway saw
claim_locks owned by the dead gateway pid and marked the runs crashed
("pid <n> not alive"), discarding ~18 runs / ~30h of build time.

Covered contracts:
  * ``_default_spawn`` wraps the worker argv in ``systemd-run --user
    --scope`` (own unit, description, MemoryMax/MemorySwapMax) when the
    probe passes and isolation allows it;
  * isolation ``none`` / unavailable systemd produce today's argv exactly;
  * the dispatcher tick re-adopts a running worker whose pid is verified
    alive (PID-reuse guard) with a fresh heartbeat after a gateway
    restart, and still marks dead-pid / stale-heartbeat / pid-reused rows
    crashed;
  * every dispatcher-side stop path reaps the worker's whole scope via
    ``systemctl --user stop``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    return home


@pytest.fixture
def conn(kanban_home):
    with kb.connect() as c:
        yield c


def _make_task(task_id="t_scope1", title="build the widget"):
    return kb.Task(
        id=task_id,
        title=title,
        body=None,
        assignee="elias",
        status="running",
        priority=0,
        created_by="test",
        created_at=1,
        started_at=None,
        completed_at=None,
        workspace_kind="dir",
        workspace_path=None,
        claim_lock="lock",
        claim_expires=None,
        tenant=None,
        current_run_id=7,
    )


def _patch_systemd_available(monkeypatch, available: bool):
    """Force the shared cached probe; kanban reads it at call time."""
    monkeypatch.setattr(
        "tools.process_registry._systemd_run_user_scope_available",
        lambda: available,
    )


def _patch_systemd_run_binary(monkeypatch):
    """Pretend ``systemd-run`` exists — the builder re-runs which() itself."""
    real_which = shutil.which

    def fake_which(name, *args, **kwargs):
        if name == "systemd-run":
            return "/usr/bin/systemd-run"
        return real_which(name, *args, **kwargs)

    monkeypatch.setattr(shutil, "which", fake_which)


def _fake_popen_capture(monkeypatch, captured, pid=4242):
    class FakeProc:
        pass

    FakeProc.pid = pid

    def fake_popen(cmd, *args, **kwargs):
        captured["cmd"] = list(cmd)
        captured["kwargs"] = dict(kwargs)
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)


def _write_kanban_config(home: Path, kanban_yaml: str):
    home.joinpath("config.yaml").write_text(
        f"kanban:\n{kanban_yaml}", encoding="utf-8"
    )


def _capture_worker_argv(
    monkeypatch, tmp_path, kanban_yaml: str, *, systemd_available: bool
) -> list[str]:
    """Spawn one worker with the given kanban config and return its argv.

    Writes the config exactly once per call site (load_config caches on
    mtime/size, so rewrites within a test would be unreliable); tests that
    need several captures share one config and flip only the probe.
    """
    home = tmp_path / ".hermes"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    _write_kanban_config(home, kanban_yaml)

    monkeypatch.setattr(kb, "_resolve_hermes_argv", lambda: ["hermes"])
    _patch_systemd_available(monkeypatch, systemd_available)
    if systemd_available:
        _patch_systemd_run_binary(monkeypatch)
    captured = {}
    _fake_popen_capture(monkeypatch, captured)

    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    kb._default_spawn(_make_task(), str(workspace))
    return captured["cmd"]


def _assert_plain_argv_shape(cmd: list[str]):
    """Today's (pre-isolation) worker argv shape, independent of which
    toolsets/model flags the config resolves: fixed hermes prefix, fixed
    chat suffix, and not one systemd token."""
    assert cmd[:5] == ["hermes", "-p", "elias", "--cli", "--accept-hooks"]
    assert cmd[-3:] == ["chat", "-q", "work kanban task t_scope1"]
    for token in ("systemd-run", "--user", "--scope", "--unit", "--collect",
                  "--description", "--property", "MemoryAccounting"):
        assert token not in cmd, f"systemd token {token!r} leaked into plain argv"


# ---------------------------------------------------------------------------
# Spawn wrapping
# ---------------------------------------------------------------------------

def test_default_spawn_wraps_argv_in_systemd_scope(monkeypatch, tmp_path):
    """Probe passes + isolation auto → systemd-run prefix with the unit
    name, description, per-worker memory properties, and the legacy argv
    intact after ``--``. The unwrapped baseline is captured from the same
    code path with the probe disabled, so the comparison is exact."""
    config = "  worker_isolation: auto\n  worker_memory_max_mb: 512\n"
    plain = _capture_worker_argv(
        monkeypatch, tmp_path, config, systemd_available=False
    )
    _assert_plain_argv_shape(plain)

    # Same config, probe now passes → same spawn, wrapped.
    wrapped = _capture_worker_argv(
        monkeypatch, tmp_path, config, systemd_available=True
    )

    assert wrapped[0] == "/usr/bin/systemd-run"
    # Flags before the command separator, in the builder's canonical order.
    head = wrapped[: wrapped.index("--")]
    assert head[1:5] == ["--user", "--scope", "--quiet", "--unit"]
    assert "hermes-kanban-t_scope1.scope" in head
    assert "--collect" in head
    assert head[head.index("--description") + 1] == (
        "Hermes kanban worker t_scope1: build the widget"
    )
    assert head[head.index("--property") + 1] == "MemoryAccounting=yes"
    assert head[head.index("--property") + 3] == f"MemoryMax={512 * 1024 * 1024}"
    assert head[head.index("--property") + 5] == f"MemorySwapMax={512 * 1024 * 1024}"
    assert head[head.index("--property") + 7] == "OOMPolicy=kill"
    # Legacy argv preserved verbatim after the separator.
    assert wrapped[wrapped.index("--") + 1:] == plain
    # The spawn published the unit for the dispatcher's bookkeeping.
    assert kb._default_spawn._last_scope_unit == "hermes-kanban-t_scope1.scope"


def test_default_spawn_none_keeps_legacy_argv_exactly(monkeypatch, tmp_path):
    """isolation 'none' must produce today's argv byte-for-byte, even when
    systemd is fully available — the rollback contract."""
    none_cmd = _capture_worker_argv(
        monkeypatch, tmp_path, "  worker_isolation: none\n",
        systemd_available=True,
    )
    _assert_plain_argv_shape(none_cmd)
    # 'none' ignores availability: a second capture with the probe down
    # (the classic macOS/container host) is byte-identical.
    fallback_cmd = _capture_worker_argv(
        monkeypatch, tmp_path, "  worker_isolation: none\n",
        systemd_available=False,
    )
    assert fallback_cmd == none_cmd
    assert kb._default_spawn._last_scope_unit == ""


def test_default_spawn_auto_without_systemd_keeps_legacy_argv(monkeypatch, tmp_path):
    """Unusable systemd (macOS / containers) with the default 'auto' mode
    silently falls back to the plain argv — no behavioural change."""
    cmd = _capture_worker_argv(
        monkeypatch, tmp_path, "  worker_isolation: auto\n",
        systemd_available=False,
    )
    _assert_plain_argv_shape(cmd)
    assert kb._default_spawn._last_scope_unit == ""


def test_forced_scope_without_systemd_still_spawns(monkeypatch, tmp_path):
    """'systemd-scope' + failed probe = loud warning, plain spawn. A board
    that cannot spawn at all would be worse than an unisolated worker."""
    cmd = _capture_worker_argv(
        monkeypatch, tmp_path, "  worker_isolation: systemd-scope\n",
        systemd_available=False,
    )
    _assert_plain_argv_shape(cmd)
    assert kb._default_spawn._last_scope_unit == ""


# ---------------------------------------------------------------------------
# Dispatcher bookkeeping
# ---------------------------------------------------------------------------

def test_set_worker_pid_records_scope_and_start_fingerprint(conn):
    """The pid, its start-time fingerprint (PID-reuse guard), and the scope
    unit land on both the task row and the active run; the ``spawned``
    event carries the scope for operators."""
    tid = kb.create_task(conn, title="record", assignee="w")
    kb.claim_task(conn, tid, claimer=kb._claimer_id())

    live = os.getpid()
    kb._set_worker_pid(
        conn, tid, live, scope_unit="hermes-kanban-%s.scope" % tid,
    )

    row = conn.execute(
        "SELECT worker_pid, worker_pid_started_at, worker_scope "
        "FROM tasks WHERE id = ?", (tid,)
    ).fetchone()
    assert row["worker_pid"] == live
    assert row["worker_pid_started_at"] == kb._worker_pid_start_time(live)
    assert row["worker_scope"] == "hermes-kanban-%s.scope" % tid

    run = conn.execute(
        "SELECT worker_pid, worker_scope FROM task_runs "
        "WHERE id = (SELECT current_run_id FROM tasks WHERE id = ?)",
        (tid,),
    ).fetchone()
    assert run["worker_pid"] == live
    assert run["worker_scope"] == "hermes-kanban-%s.scope" % tid

    event = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'spawned'",
        (tid,),
    ).fetchone()
    assert event and json.loads(event["payload"])["scope"] == (
        "hermes-kanban-%s.scope" % tid
    )


def test_dispatch_once_persists_scope_unit_for_scoped_spawn(
    monkeypatch, conn, kanban_home
):
    """End-to-end: the real ``_default_spawn`` (systemd available) → the
    dispatcher tick records the scope unit on the claimed task."""
    # The assignee must name a real Hermes profile or the dispatcher skips
    # the task as non-spawnable (control-plane lane).
    profile = Path(kanban_home) / "profiles" / "elias"
    profile.mkdir(parents=True, exist_ok=True)
    profile.joinpath("config.yaml").write_text("{}\n", encoding="utf-8")

    _patch_systemd_available(monkeypatch, True)
    _patch_systemd_run_binary(monkeypatch)
    captured = {}
    _fake_popen_capture(monkeypatch, captured, pid=5566)
    monkeypatch.setattr(kb, "_resolve_hermes_argv", lambda: ["hermes"])

    tid = kb.create_task(conn, title="scoped", assignee="elias")
    result = kb.dispatch_once(conn, dry_run=False)

    assert [s[0] for s in result.spawned] == [tid]
    row = conn.execute(
        "SELECT worker_pid, worker_scope FROM tasks WHERE id = ?", (tid,)
    ).fetchone()
    assert row["worker_pid"] == 5566
    assert row["worker_scope"] == kb._kanban_worker_scope_unit(tid)
    # The spawn itself really went through the scope wrapper.
    assert captured["cmd"][0] == "/usr/bin/systemd-run"


# ---------------------------------------------------------------------------
# Re-adoption after gateway restart
# ---------------------------------------------------------------------------

def _running_row(conn, tid, *, claimer, pid, pid_started, heartbeat):
    conn.execute(
        "UPDATE tasks SET status='running', claim_lock=?, claim_expires=?, "
        "worker_pid=?, worker_pid_started_at=?, last_heartbeat_at=? "
        "WHERE id=?",
        (claimer, int(time.time()) - 60, pid, pid_started, heartbeat, tid),
    )
    conn.execute(
        "UPDATE task_runs SET status='running', claim_lock=? "
        "WHERE id=(SELECT current_run_id FROM tasks WHERE id=?)",
        (claimer, tid),
    )
    conn.commit()


def test_adopt_surviving_worker_rewrites_claim_and_run_continues(conn):
    """A live, freshly-heartbeating worker owned by the previous gateway
    pid is re-adopted: claim moves to this claimer, run stays running, no
    failure counted, and crash detection leaves it alone."""
    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="survivor", assignee="w")
    kb.claim_task(conn, tid, claimer=f"{host}:4194304")
    live = os.getpid()
    _running_row(
        conn, tid,
        claimer=f"{host}:4194304",
        pid=live,
        pid_started=kb._worker_pid_start_time(live),
        heartbeat=int(time.time()),
    )

    adopted = kb.adopt_surviving_running_workers(conn)
    assert adopted == [tid]

    row = conn.execute(
        "SELECT status, claim_lock, claim_expires, consecutive_failures "
        "FROM tasks WHERE id = ?", (tid,)
    ).fetchone()
    assert row["status"] == "running"
    assert row["claim_lock"] == kb._claimer_id()
    assert row["claim_expires"] > int(time.time())
    assert row["consecutive_failures"] == 0

    event = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'adopted'",
        (tid,),
    ).fetchone()
    payload = json.loads(event["payload"])
    assert payload["previous_claimer"] == f"{host}:4194304"
    assert payload["claimer"] == kb._claimer_id()

    # The adopted run is not a crash: detection must skip it entirely.
    assert kb.detect_crashed_workers(conn) == []
    # Idempotent: a second pass finds nothing to adopt.
    assert kb.adopt_surviving_running_workers(conn) == []


def test_adoption_skips_stale_heartbeat(conn):
    """Alive pid but no observable progress for > 1h → NOT adopted; the
    existing stale paths own that case."""
    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="wedged", assignee="w")
    kb.claim_task(conn, tid, claimer=f"{host}:4194304")
    live = os.getpid()
    _running_row(
        conn, tid,
        claimer=f"{host}:4194304",
        pid=live,
        pid_started=kb._worker_pid_start_time(live),
        heartbeat=int(time.time()) - 7200,
    )

    assert kb.adopt_surviving_running_workers(conn) == []
    row = conn.execute(
        "SELECT claim_lock FROM tasks WHERE id = ?", (tid,)
    ).fetchone()
    assert row["claim_lock"] == f"{host}:4194304"


def test_dead_pid_still_crashes_and_counts_failure(conn):
    """Adoption must not rescue a genuinely dead worker: crash detection
    still fires, marks the run crashed, and counts the failure."""
    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="crashed", assignee="w")
    kb.claim_task(conn, tid, claimer=f"{host}:4194304")
    dead = subprocess.Popen(["true"])
    dead.wait()
    _running_row(
        conn, tid,
        claimer=f"{host}:4194304",
        pid=dead.pid,
        pid_started=None,
        heartbeat=int(time.time()),
    )
    conn.execute("UPDATE tasks SET started_at = started_at - 9999 WHERE id=?", (tid,))
    conn.commit()

    assert kb.adopt_surviving_running_workers(conn) == []
    crashed = kb.detect_crashed_workers(conn)
    assert tid in crashed
    row = conn.execute(
        "SELECT status, consecutive_failures FROM tasks WHERE id = ?", (tid,)
    ).fetchone()
    assert row["status"] != "running"
    assert row["consecutive_failures"] >= 1


def test_recycled_pid_is_not_mistaken_for_the_worker(conn):
    """PID-reuse guard: a live but different process at the recorded pid is
    a dead worker, not a survivor — crash detection fires with the reuse
    flag, and adoption never claims it."""
    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="reused", assignee="w")
    kb.claim_task(conn, tid, claimer=f"{host}:4194304")
    # A very much alive process (this test) whose start-time fingerprint
    # differs from the recorded one — exactly what pid reuse looks like.
    _running_row(
        conn, tid,
        claimer=f"{host}:4194304",
        pid=os.getpid(),
        pid_started=12345,
        heartbeat=int(time.time()),
    )
    conn.execute("UPDATE tasks SET started_at = started_at - 9999 WHERE id=?", (tid,))
    conn.commit()

    assert kb.adopt_surviving_running_workers(conn) == []
    crashed = kb.detect_crashed_workers(conn)
    assert tid in crashed

    event = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'crashed'",
        (tid,),
    ).fetchone()
    payload = json.loads(event["payload"])
    assert payload.get("pid_reused") is True


# ---------------------------------------------------------------------------
# Scope reaping on stop paths
# ---------------------------------------------------------------------------

def _patch_scope_stopper(monkeypatch, calls):
    def fake_stop(unit):
        calls.append(unit)
        return True

    monkeypatch.setattr("tools.process_registry._stop_systemd_unit", fake_stop)


def test_release_stale_claims_stops_worker_scope(monkeypatch, conn):
    """TTL-expired reclaim of a scoped worker stops the whole unit before
    the pid kill backstop, and clears the scope bookkeeping."""
    calls: list[str] = []
    _patch_scope_stopper(monkeypatch, calls)
    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="stale", assignee="w")
    kb.claim_task(conn, tid, claimer=f"{host}:4194304")
    dead = subprocess.Popen(["true"])
    dead.wait()
    unit = "hermes-kanban-%s.scope" % tid
    conn.execute(
        "UPDATE tasks SET worker_pid=?, worker_scope=?, "
        "claim_expires=?, last_heartbeat_at=? WHERE id=?",
        (dead.pid, unit, int(time.time()) - 60, int(time.time()) - 7200, tid),
    )
    conn.commit()

    reclaimed = kb.release_stale_claims(conn)
    assert reclaimed == 1
    assert calls == [unit]
    row = conn.execute(
        "SELECT status, worker_scope FROM tasks WHERE id = ?", (tid,)
    ).fetchone()
    assert row["status"] != "running"
    assert row["worker_scope"] is None


def test_enforce_max_runtime_stops_worker_scope(monkeypatch, conn):
    """Timeout of a scoped run stops the scope (SIGTERM→SIGKILL across the
    whole cgroup) — the dev servers a worker spawned die with it."""
    calls: list[str] = []
    _patch_scope_stopper(monkeypatch, calls)
    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="timeout", assignee="w")
    kb.claim_task(conn, tid, claimer=f"{host}:4194304")
    dead = subprocess.Popen(["true"])
    dead.wait()
    unit = "hermes-kanban-%s.scope" % tid
    conn.execute(
        "UPDATE tasks SET worker_pid=?, worker_scope=?, max_runtime_seconds=1, "
        "started_at=? WHERE id=?",
        (dead.pid, unit, int(time.time()) - 100, tid),
    )
    conn.execute(
        "UPDATE task_runs SET started_at = started_at - 9999 "
        "WHERE id=(SELECT current_run_id FROM tasks WHERE id=?)", (tid,),
    )
    conn.commit()

    def fake_kill(pid, sig):
        raise ProcessLookupError()

    timed_out = kb.enforce_max_runtime(conn, signal_fn=fake_kill)
    assert timed_out == [tid]
    assert calls == [unit]
    row = conn.execute(
        "SELECT status, worker_scope FROM tasks WHERE id = ?", (tid,)
    ).fetchone()
    assert row["worker_scope"] is None


def test_reclaim_task_stops_worker_scope(monkeypatch, conn):
    """Operator reclaim of a scoped worker stops its scope too."""
    calls: list[str] = []
    _patch_scope_stopper(monkeypatch, calls)
    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="manual", assignee="w")
    kb.claim_task(conn, tid, claimer=f"{host}:4194304")
    dead = subprocess.Popen(["true"])
    dead.wait()
    unit = "hermes-kanban-%s.scope" % tid
    conn.execute(
        "UPDATE tasks SET worker_pid=?, worker_scope=? WHERE id=?",
        (dead.pid, unit, tid),
    )
    conn.commit()

    assert kb.reclaim_task(conn, tid, reason="operator abort") is True
    assert calls == [unit]
    row = conn.execute(
        "SELECT status, worker_scope FROM tasks WHERE id = ?", (tid,)
    ).fetchone()
    assert row["worker_scope"] is None
