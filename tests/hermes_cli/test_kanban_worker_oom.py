"""A Kanban worker OOM-killed by its own scope is an INFRASTRUCTURE failure.

Kanban t_4d12c8eb (2026-10-24).

Two good implementation cards were killed mid-flight on hermes by their own
worker's systemd scope hitting its ``MemoryMax``:

    Oct 24 05:56:08 kernel: memory: usage 1048576kB, limit 1048576kB
    Oct 24 05:56:08 kernel: oom-kill:constraint=CONSTRAINT_MEMCG,...,
        oom_memcg=.../hermes-worker-kanban-t_000bf106-run-33301.scope,task=hermes,pid=759473

The dispatcher recorded ``"pid <n> not alive"`` and tripped the crash-loop
breaker with ``gave_up``. That reason string reads like the worker's CODE was
broken — the exact misdiagnosis that made two cards with finished
implementations look like failures.

These tests pin the corrected behaviour:
  * the kernel OOM record is parsed and attributed to the right task,
  * an OOM-killed worker is classified as ``oom_killed``, not a generic crash,
  * the failure text names the OOM and says the work was not at fault,
  * the OOM does NOT consume the crash budget (the card stays retryable),
  * a NON-OOM death is unaffected.
"""
from __future__ import annotations

import time

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_worker_oom as oom


# The kernel lines from the real incident (2026-10-24 05:56:08 on hermes).
REAL_KERNEL_OOM_LOG = """
Oct 24 05:56:08 hermes.loc.wallacearizona.us kernel: uv-compile invoked oom-killer: gfp_mask=0x100cca(GFP_HIGHUSER_MOVABLE), order=0, oom_score_adj=100
Oct 24 05:56:08 hermes.loc.wallacearizona.us kernel:  mem_cgroup_out_of_memory+0x145/0x170
Oct 24 05:56:08 hermes.loc.wallacearizona.us kernel: memory: usage 1048576kB, limit 1048576kB, failcnt 1240786
Oct 24 05:56:08 hermes.loc.wallacearizona.us kernel: Memory cgroup stats for /user.slice/user-1000.slice/user@1000.service/app.slice/hermes-worker-kanban-t_000bf106-run-33301.scope:
Oct 24 05:56:08 hermes.loc.wallacearizona.us kernel: oom-kill:constraint=CONSTRAINT_MEMCG,nodemask=(null),cpuset=user.slice,mems_allowed=0-1,oom_memcg=/user.slice/user-1000.slice/user@1000.service/app.slice/hermes-worker-kanban-t_000bf106-run-33301.scope,task_memcg=/user.slice/user-1000.slice/user@1000.service/app.slice/hermes-worker-kanban-t_000bf106-run-33301.scope,task=hermes,pid=759473,uid=1000
Oct 24 05:56:08 hermes.loc.wallacearizona.us kernel: Memory cgroup out of memory: Killed process 759473 (hermes) total-vm:2712396kB, anon-rss:21732kB, file-rss:29568kB, shmem-rss:0kB, UID:1000 pgtables:1076kB oom_score_adj:100
Oct 24 06:01:22 hermes.loc.wallacearizona.us kernel: memory: usage 1048576kB, limit 1048576kB, failcnt 661055
Oct 24 06:01:22 hermes.loc.wallacearizona.us kernel: oom-kill:constraint=CONSTRAINT_MEMCG,nodemask=(null),cpuset=user.slice,mems_allowed=0-1,oom_memcg=/user.slice/user-1000.slice/user@1000.service/app.slice/hermes-worker-kanban-t_38551eb5-run-33303.scope,task_memcg=/user.slice/user-1000.slice/user@1000.service/app.slice/hermes-worker-kanban-t_38551eb5-run-33303.scope,task=hermes,pid=791052,uid=1000
Oct 24 06:01:22 hermes.loc.wallacearizona.us kernel: Memory cgroup out of memory: Killed process 791052 (hermes) total-vm:3369008kB, anon-rss:30112kB, file-rss:31488kB, shmem-rss:0kB, UID:1000 pgtables:1092kB oom_score_adj:100
"""


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    # Reclaim the dead worker on the very next sweep: a just-claimed worker
    # otherwise sits inside the launch-window grace period.
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    kb.init_db()
    return home


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_parses_the_real_kernel_oom_log():
    """The parser reads the real kernel lines from the incident."""
    records = oom._parse_oom_records(REAL_KERNEL_OOM_LOG)

    assert len(records) == 2
    first = records[0]
    assert first["task_id"] == "t_000bf106"
    assert first["constraint"] == "CONSTRAINT_MEMCG"
    assert first["limit_kb"] == 1048576
    assert first["usage_kb"] == 1048576
    assert first["victim_pid"] == 759473
    assert first["victim_comm"] == "hermes"

    second = records[1]
    assert second["task_id"] == "t_38551eb5"
    assert second["victim_pid"] == 791052


def test_a_cgroup_local_kill_is_distinguishable_from_a_host_wide_one():
    """``CONSTRAINT_MEMCG`` means the scope hit ITS OWN limit — not a starved host.

    The distinction decides the fix: a cgroup-local kill means the LIMIT was
    wrong; a host-wide kill means the host was out of memory.
    """
    records = oom._parse_oom_records(REAL_KERNEL_OOM_LOG)
    assert all(r["constraint"] == "CONSTRAINT_MEMCG" for r in records)


def test_find_worker_oom_matches_by_task_id(monkeypatch):
    monkeypatch.setattr(
        oom, "read_oom_records",
        lambda *a, **k: oom._parse_oom_records(REAL_KERNEL_OOM_LOG),
    )
    record = oom.find_worker_oom(task_id="t_000bf106")
    assert record is not None
    assert record["task_id"] == "t_000bf106"
    assert record["limit_kb"] == 1048576


def test_find_worker_oom_returns_none_for_an_unrelated_task(monkeypatch):
    """No record for this card means the death was NOT an OOM."""
    monkeypatch.setattr(
        oom, "read_oom_records",
        lambda *a, **k: oom._parse_oom_records(REAL_KERNEL_OOM_LOG),
    )
    assert oom.find_worker_oom(task_id="t_something_else") is None


def test_find_worker_oom_matches_by_victim_pid(monkeypatch):
    monkeypatch.setattr(
        oom, "read_oom_records",
        lambda *a, **k: oom._parse_oom_records(REAL_KERNEL_OOM_LOG),
    )
    record = oom.find_worker_oom(worker_pid=791052)
    assert record is not None
    assert record["task_id"] == "t_38551eb5"


def test_describe_oom_names_the_limit_and_says_the_work_is_not_at_fault():
    record = oom._parse_oom_records(REAL_KERNEL_OOM_LOG)[0]
    text = oom.describe_oom(record)

    assert "OOM-killed" in text
    assert "CONSTRAINT_MEMCG" in text
    assert "1024MiB" in text
    assert "not a failure of the work" in text


# ---------------------------------------------------------------------------
# Dispatcher classification
# ---------------------------------------------------------------------------


def test_oom_killed_worker_is_classified_as_oom(monkeypatch):
    """The dead-worker classifier recognises an OOM instead of "not alive"."""
    monkeypatch.setattr(
        kbd, "_worker_oom_record",
        lambda pid=None, task_id=None: {
            "task_id": "t_000bf106",
            "constraint": "CONSTRAINT_MEMCG",
            "limit_kb": 1048576,
            "usage_kb": 1048576,
            "victim_comm": "hermes",
        },
    )
    # The reap registry is empty (a per-tick dispatcher never reaped the
    # worker) — the exact state that produced "pid <n> not alive".
    monkeypatch.setattr(kbd, "_classify_worker_exit", lambda pid: ("unknown", None))
    monkeypatch.setattr(kbd, "_worker_log_exit_code", lambda *a, **k: None)

    dead = kbd._classify_dead_worker_exit(
        759473, "host:1234", task_id="t_000bf106",
    )

    assert dead.oom_killed is True
    assert dead.kind == "oom_killed"
    assert "OOM-killed" in dead.error_text
    assert "not alive" not in dead.error_text
    assert dead.event_payload["oom_killed"] is True
    assert dead.event_payload["oom"]["constraint"] == "CONSTRAINT_MEMCG"


def test_a_non_oom_death_is_unaffected(monkeypatch):
    """A worker that died for another reason is still a plain crash."""
    monkeypatch.setattr(kbd, "_worker_oom_record", lambda pid=None, task_id=None: None)
    monkeypatch.setattr(kbd, "_classify_worker_exit", lambda pid: ("unknown", None))
    monkeypatch.setattr(kbd, "_worker_log_exit_code", lambda *a, **k: None)

    dead = kbd._classify_dead_worker_exit(1234, "host:1234", task_id="t_abc")

    assert dead.oom_killed is False
    assert dead.kind == "unknown"
    assert dead.error_text == "pid 1234 not alive"


def test_oom_lookup_never_raises(monkeypatch):
    """Forensics that break the next dispatch tick are worse than no forensics."""
    def _boom(*_args, **_kwargs):
        raise RuntimeError("journalctl exploded")

    monkeypatch.setattr(
        "hermes_cli.kanban_worker_oom.find_worker_oom", _boom,
    )
    assert kbd._worker_oom_record(pid=1, task_id="t_x") is None


# ---------------------------------------------------------------------------
# Accounting: an OOM must not consume the crash budget
# ---------------------------------------------------------------------------


def test_oom_kill_does_not_consume_the_crash_budget(kanban_home, monkeypatch):
    """An OOM-killed worker is requeued; the card is NOT parked as a failure.

    Before the fix, two OOMs in a row tripped the crash-loop breaker and the
    card landed in ``blocked`` with ``gave_up`` — reading as "the code was
    broken" when the implementation was already on disk.
    """
    monkeypatch.setattr(
        kbd, "_worker_oom_record",
        lambda pid=None, task_id=None: {
            "task_id": "t_oom",
            "constraint": "CONSTRAINT_MEMCG",
            "limit_kb": 1048576,
            "usage_kb": 1048576,
        },
    )
    monkeypatch.setattr(kbd, "_classify_worker_exit", lambda pid: ("unknown", None))
    monkeypatch.setattr(kbd, "_worker_log_exit_code", lambda *a, **k: None)
    monkeypatch.setattr(kbd, "_worker_alive", lambda *a, **k: False)

    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="oom card", assignee="a")
        kb.claim_task(conn, task_id)
        run_id = kb.get_task(conn, task_id).current_run_id
        conn.execute(
            "UPDATE tasks SET worker_pid=?, worker_started_at=NULL, started_at=? WHERE id=?",
            (4242, time.time() - 3600, task_id),
        )
        conn.commit()

        # Two OOM deaths in a row — the shape that used to trip the breaker.
        for _ in range(2):
            kbd.detect_crashed_workers(conn)
            conn.execute(
                "UPDATE tasks SET status='running', worker_pid=4242, "
                "worker_started_at=NULL, current_run_id=? WHERE id=?",
                (run_id, task_id),
            )
            conn.commit()

        row = conn.execute(
            "SELECT status, consecutive_failures, last_failure_error FROM tasks WHERE id=?",
            (task_id,),
        ).fetchone()

        assert row["status"] != "blocked", (
            "an OOM-killed worker must not park the card as a failure: "
            f"status={row['status']!r} error={row['last_failure_error']!r}"
        )
        assert int(row["consecutive_failures"] or 0) == 0
        assert "OOM-killed" in (row["last_failure_error"] or "")


def test_oom_failure_text_is_persisted_as_the_reason(kanban_home, monkeypatch):
    """The reason the next reader sees must name the OOM."""
    monkeypatch.setattr(
        kbd, "_worker_oom_record",
        lambda pid=None, task_id=None: {
            "task_id": "t_oom",
            "constraint": "CONSTRAINT_MEMCG",
            "limit_kb": 1048576,
        },
    )
    monkeypatch.setattr(kbd, "_classify_worker_exit", lambda pid: ("unknown", None))
    monkeypatch.setattr(kbd, "_worker_log_exit_code", lambda *a, **k: None)
    monkeypatch.setattr(kbd, "_worker_alive", lambda *a, **k: False)

    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="oom card", assignee="a")
        kb.claim_task(conn, task_id)
        conn.execute(
            "UPDATE tasks SET worker_pid=4242, worker_started_at=NULL, started_at=? WHERE id=?",
            (time.time() - 3600, task_id),
        )
        conn.commit()

        kbd.detect_crashed_workers(conn)

        row = conn.execute(
            "SELECT last_failure_error FROM tasks WHERE id=?", (task_id,),
        ).fetchone()
        error = row["last_failure_error"] or ""
        assert "OOM-killed" in error
        assert "not alive" not in error
