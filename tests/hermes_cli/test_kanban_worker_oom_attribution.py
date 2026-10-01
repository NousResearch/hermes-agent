"""A worker whose scope the kernel OOM-killed must not be booked as a protocol violation.

A worker scope runs with the default ``OOMPolicy=stop``: when the kernel kills a process
inside it, systemd stops the whole scope and the worker's signal path exits 0 with no
terminal kanban call made. The dispatcher used to read that as a worker skipping its
paperwork, book a protocol violation, and feed the violation streak - so a card could
``give_up`` on a death the kernel caused.

The scope is transient and ``--collect`` removes it on exit, so ``systemctl show`` cannot
report ``Result=oom-kill`` afterwards; the dispatcher reads the user manager's journal
instead. An OOM death arrives as either a clean exit or as a bare "pid not alive", so both
bookings must name the cause. These tests pin every branch.
"""

from unittest.mock import patch

import hermes_cli.kanban_db_dispatch as d


def _dead(oom: bool, kind: str = "clean_exit", pid: int = 4242):
    with patch.object(d, "_classify_worker_exit", return_value=(kind, 0 if kind == "clean_exit" else None)), \
         patch.object(d, "_worker_log_exit_code", return_value=None), \
         patch.object(d, "_worker_scope_oom_killed", return_value=oom):
        return d._classify_dead_worker_exit(pid, None, task_id="t_fixture")


def test_oom_killed_scope_is_a_crash_not_a_protocol_violation():
    dead = _dead(oom=True)
    assert dead.event_kind == "crashed"
    assert dead.protocol_violation is False
    assert dead.rate_limited is False
    assert dead.event_payload["oom_killed"] is True
    assert "OOM killer stopped its worker scope" in dead.error_text


def test_clean_exit_without_an_oom_stays_a_protocol_violation():
    dead = _dead(oom=False)
    assert dead.event_kind == "protocol_violation"
    assert dead.protocol_violation is True
    assert "oom_killed" not in dead.event_payload


def test_pid_not_alive_on_an_oom_killed_scope_names_the_cause():
    dead = _dead(oom=True, kind="unknown")
    assert dead.event_kind == "crashed"
    assert dead.event_payload["oom_killed"] is True
    assert "the kernel OOM killer stopped this run's worker scope" in dead.error_text


def test_pid_not_alive_without_an_oom_stays_bare():
    dead = _dead(oom=False, kind="unknown")
    assert dead.event_kind == "crashed"
    assert "oom_killed" not in dead.event_payload
    assert dead.error_text == "pid 4242 not alive"


def test_scope_oom_probe_reads_the_journal_line():
    hit = type("R", (), {"stdout": "hermes-worker-x.scope: Failed with result 'oom-kill'.\n"})()
    miss = type("R", (), {"stdout": "hermes-worker-x.scope: Deactivated successfully.\n"})()
    with patch.object(d.subprocess, "run", return_value=hit):
        assert d._worker_scope_oom_killed("t_fixture") is True
    with patch.object(d.subprocess, "run", return_value=miss):
        assert d._worker_scope_oom_killed("t_fixture") is False


def test_scope_oom_probe_survives_a_missing_journalctl():
    with patch.object(d.subprocess, "run", side_effect=OSError("no journalctl")):
        assert d._worker_scope_oom_killed("t_fixture") is False


def test_scope_oom_probe_is_bounded_to_the_one_task_and_a_short_window():
    """One `journalctl` call per booked death, scoped to that task's own scopes: the
    dispatcher must not scan the whole user journal (or every task's scopes) on a tick."""
    with patch.object(d.subprocess, "run", return_value=type("R", (), {"stdout": ""})()) as run:
        d._worker_scope_oom_killed("t_fixture")

    argv = run.call_args.args[0] if run.call_args.args else run.call_args.kwargs["args"]
    assert argv[0] == "journalctl"
    assert "--user" in argv
    assert "hermes-worker-kanban-t_fixture-run-*.scope" in argv
    assert "-o" in argv and "cat" in argv
    since = argv[argv.index("--since") + 1]
    assert since.endswith("h") and int(since.lstrip("-").rstrip("h")) <= 6
