"""Tests for the kanban dispatch-reliability fixes.

Covers the three staged fixes:
  1. Assignee routing classification + create-time validation gate, and the
     dispatcher's split of unroutable (STUCK) from known pull lanes (idle).
  2. Dispatcher heartbeat write/read + hung/dead/healthy classification.
  3. Read-only stale-task auditor: dependency-deadlocked todos + stale ready.

All of it is exercised against a temp board — nothing here touches a live
board, and the auditor is asserted to never write.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_doctor as doc


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


# ---------------------------------------------------------------------------
# Fix 1 — assignee classification, create-time validation, dispatch split
# ---------------------------------------------------------------------------

def test_classify_assignee_categories(kanban_home, monkeypatch):
    # A real profile classifies as "profile"; force profile_exists so the
    # test doesn't depend on on-disk profiles.
    monkeypatch.setattr(
        "hermes_cli.profiles.profile_exists",
        lambda n: n in {"ghost", "patch"},
    )
    assert kb.classify_assignee("ghost") == "profile"
    assert kb.classify_assignee("Ghost") == "profile"  # normalized
    assert kb.classify_assignee("fable") == "pull_lane"
    assert kb.classify_assignee("orion-cc") == "pull_lane"
    assert kb.classify_assignee("nonesuch-typo") == "unroutable"
    for empty in (None, "", "   "):
        assert kb.classify_assignee(empty) == "unassigned"
    assert kb.assignee_is_routable("nonesuch-typo") is False
    assert kb.assignee_is_routable("fable") is True
    assert kb.assignee_is_routable(None) is True


def test_create_validation_gate(kanban_home):
    conn = kb.connect(board="default")
    # Default (flag off): unroutable assignee is accepted (legacy behavior).
    tid = kb.create_task(conn, title="legacy", assignee="nonesuch-typo",
                         created_by="t")
    assert kb.get_task(conn, tid).status == "ready"
    # Armed: unroutable assignee is refused.
    with pytest.raises(ValueError, match="not routable"):
        kb.create_task(conn, title="armed", assignee="nonesuch-typo",
                       created_by="t", validate_assignee=True)
    # Armed still accepts a pull lane and an empty assignee.
    kb.create_task(conn, title="lane", assignee="fable", created_by="t",
                   validate_assignee=True)
    kb.create_task(conn, title="none", assignee=None, created_by="t",
                   validate_assignee=True)
    conn.close()


def test_dispatch_splits_unroutable_from_pull_lane(kanban_home):
    conn = kb.connect(board="default")
    u = kb.create_task(conn, title="typo", assignee="nonesuch-typo", created_by="t")
    f = kb.create_task(conn, title="fable", assignee="fable", created_by="t")
    res = kb.dispatch_once(conn, dry_run=True)
    # The typo is surfaced as STUCK; the pull lane is correctly-idle.
    assert u in res.skipped_unroutable
    assert f in res.skipped_nonspawnable
    assert u not in res.skipped_nonspawnable
    assert kb.list_unroutable_ready(conn) == [u]
    conn.close()


# ---------------------------------------------------------------------------
# Fix 2 — dispatcher heartbeat / hang detection
# ---------------------------------------------------------------------------

def test_heartbeat_roundtrip_and_clear(kanban_home):
    assert kb.read_dispatcher_heartbeat() is None  # none yet
    kb.write_dispatcher_heartbeat(tick=3, interval_seconds=60)
    hb = kb.read_dispatcher_heartbeat()
    assert hb["tick"] == 3
    assert hb["age_seconds"] is not None and hb["age_seconds"] < 5
    assert hb["pid_alive"] is True
    kb.clear_dispatcher_heartbeat()
    assert kb.read_dispatcher_heartbeat() is None


def _write_hb(pid, ts):
    path = kb.dispatcher_heartbeat_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    boot = None
    try:
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except Exception:
        boot = None
    path.write_text(json.dumps({
        "pid": pid, "boot_id": boot, "host": "test",
        "ts": ts, "tick": 1, "interval_seconds": 60,
    }))


def test_heartbeat_classification(kanban_home):
    import os
    now = int(time.time())
    # alive + stale -> HUNG (the systemd-blind case)
    _write_hb(os.getpid(), now - 9999)
    r = doc.audit_heartbeat(stale_seconds=300)
    assert r["state"] == "hung" and r["ok"] is False
    # dead + stale -> dead (auto-recovers when flock frees)
    _write_hb(999999, now - 9999)
    r = doc.audit_heartbeat(stale_seconds=300)
    assert r["state"] == "dead" and r["ok"] is True
    # alive + fresh -> healthy
    _write_hb(os.getpid(), now)
    r = doc.audit_heartbeat(stale_seconds=300)
    assert r["state"] == "healthy" and r["ok"] is True


# ---------------------------------------------------------------------------
# Fix 3 — read-only stale-task auditor
# ---------------------------------------------------------------------------

def test_deadlocked_todo_surfaced(kanban_home):
    conn = kb.connect(board="default")
    # parent that will be blocked; child todo gated behind it
    parent = kb.create_task(conn, title="parent", assignee="ghost", created_by="t")
    child = kb.create_task(conn, title="child", assignee="patch", created_by="t",
                           parents=[parent])
    # child should be 'todo' (parent not done)
    assert kb.get_task(conn, child).status == "todo"
    kb.block_task(conn, parent, reason="dead")
    # backdate both so they exceed the default todo_days threshold
    old = int(time.time()) - 30 * 86400
    conn.execute("UPDATE tasks SET created_at=? WHERE id IN (?,?)",
                 (old, parent, child))
    conn.commit()
    conn.close()

    report = doc.audit_stale_tasks(ready_days=2, todo_days=7)
    ids = {t["id"] for t in report["deadlocked_todos"]}
    assert child in ids
    entry = next(t for t in report["deadlocked_todos"] if t["id"] == child)
    assert any(b["parent"] == parent and b["parent_status"] == "blocked"
               for b in entry["blocking_parents"])


def test_auditor_is_read_only(kanban_home):
    conn = kb.connect(board="default")
    kb.create_task(conn, title="typo", assignee="nonesuch-typo", created_by="t")
    conn.close()
    db_path = kb.kanban_db_path("default")
    before = db_path.stat().st_mtime_ns
    time.sleep(0.01)
    doc.run_doctor()  # full audit
    after = db_path.stat().st_mtime_ns
    assert before == after, "doctor must not write to the board"


# ---------------------------------------------------------------------------
# CLI exit-code propagation — cron/schedulers must see unhealthy boards
# ---------------------------------------------------------------------------


def test_main_propagates_kanban_doctor_findings_exit_code(kanban_home):
    """`hermes kanban doctor` must exit non-zero when findings are present.

    The doctor already returns 1 on findings; this catches the top-level CLI
    dispatch layer swallowing that return code and making cron health checks
    falsely green.
    """
    import os
    import subprocess
    import sys

    with kb.connect(board="default") as conn:
        kb.create_task(
            conn,
            title="stranded typo",
            assignee="nonesuch-typo",
            created_by="test",
        )

    env = os.environ.copy()
    result = subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", "kanban", "doctor", "--json"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["assignees"]["unroutable"][0]["assignee"] == "nonesuch-typo"


def test_main_propagates_armed_create_gate_refusal_exit_code(kanban_home):
    """The config-armed create gate must fail the process, not just stderr."""
    import os
    import subprocess
    import sys

    (kanban_home / "config.yaml").write_text(
        'kanban:\n  validate_assignee_on_create: true\n',
        encoding="utf-8",
    )
    env = os.environ.copy()
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "hermes_cli.main",
            "kanban",
            "create",
            "blocked typo",
            "--assignee",
            "nonesuch-typo",
            "--json",
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == 2
    assert "not routable" in result.stderr
