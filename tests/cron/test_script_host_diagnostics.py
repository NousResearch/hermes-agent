"""Cron script execution-host diagnostics — #29849.

Script-backed cron jobs always execute on the host that ticks cron, never on
``terminal.backend``. That is deliberate (host-side watchdogs must run where
``HERMES_HOME`` is), but it was *silent*: a user whose agent wrote a script to
a remote backend got a bare ``Script not found: /home/.../id_check.sh``
pointing at a path they could ``ls`` on the backend seconds earlier, and an
operator who configured ``terminal.backend: docker`` had no signal that
script-backed cron runs outside that sandbox with the scheduler process's
environment.

These tests cover the diagnostics only — nothing here changes where a script
runs. Routing execution through the backend needs a decision on defaults and
stays open on #29849.
"""

import logging

import pytest

import cron.scheduler as sched


@pytest.fixture(autouse=True)
def _reset_notice_state():
    sched._SCRIPT_HOST_NOTICES.clear()
    yield
    sched._SCRIPT_HOST_NOTICES.clear()


# ---------------------------------------------------------------------------
# Backend resolution
# ---------------------------------------------------------------------------


def test_backend_defaults_to_local(monkeypatch):
    monkeypatch.setattr(sched, "load_config", lambda: {})
    assert sched._configured_terminal_backend() == "local"


def test_backend_is_read_and_normalized(monkeypatch):
    monkeypatch.setattr(sched, "load_config", lambda: {"terminal": {"backend": "  SSH "}})
    assert sched._configured_terminal_backend() == "ssh"


def test_backend_read_failure_degrades_to_local(monkeypatch):
    def _boom():
        raise RuntimeError("config exploded")

    monkeypatch.setattr(sched, "load_config", _boom)
    # Diagnostics must never raise into the job runner.
    assert sched._configured_terminal_backend() == "local"


# ---------------------------------------------------------------------------
# The hint text
# ---------------------------------------------------------------------------


def test_no_hint_for_local_backend():
    """Scheduler host and backend are the same machine — nothing to explain."""
    assert sched._script_runs_on_scheduler_host_hint("local") == ""


@pytest.mark.parametrize("backend", ["ssh", "docker", "modal", "daytona"])
def test_hint_names_the_backend_and_the_contract(backend):
    hint = sched._script_runs_on_scheduler_host_hint(backend)
    assert backend in hint
    assert "ticks cron" in hint
    assert "terminal.backend" in hint


# ---------------------------------------------------------------------------
# The not-found message (the reported symptom)
# ---------------------------------------------------------------------------


def test_missing_script_message_explains_the_host_under_remote_backend(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(sched, "_get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(sched, "load_config", lambda: {"terminal": {"backend": "ssh"}})

    ok, output = sched._run_job_script("id_check.sh")

    assert ok is False
    # Says WHERE it looked...
    assert "scheduler host" in output
    # ...and why the user's `ls` on the backend disagreed.
    assert "ssh" in output
    assert "terminal.backend" in output


def test_missing_script_message_stays_terse_for_local_backend(monkeypatch, tmp_path):
    """No remote backend configured → no confusing explanation appended."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(sched, "_get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(sched, "load_config", lambda: {"terminal": {"backend": "local"}})

    ok, output = sched._run_job_script("id_check.sh")

    assert ok is False
    assert "Script not found on the scheduler host" in output
    assert "terminal.backend" not in output


def test_path_traversal_block_is_unchanged(monkeypatch, tmp_path):
    """The containment guard still fires first and keeps its own message."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(sched, "_get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(sched, "load_config", lambda: {"terminal": {"backend": "ssh"}})

    ok, output = sched._run_job_script("../../etc/passwd")

    assert ok is False
    assert "Blocked" in output
    assert "outside the scripts directory" in output


# ---------------------------------------------------------------------------
# The isolation-visibility warning
# ---------------------------------------------------------------------------


def test_isolation_warning_fires_for_non_local_backend(caplog, tmp_path):
    script = tmp_path / "watchdog.sh"
    with caplog.at_level(logging.WARNING, logger="cron.scheduler"):
        sched._note_script_runs_outside_backend(script, "docker")

    messages = [r.getMessage() for r in caplog.records]
    assert len(messages) == 1
    assert "OUTSIDE the configured" in messages[0]
    assert "docker" in messages[0]
    assert "not sandboxed" in messages[0]


def test_isolation_warning_is_silent_for_local_backend(caplog, tmp_path):
    with caplog.at_level(logging.WARNING, logger="cron.scheduler"):
        sched._note_script_runs_outside_backend(tmp_path / "w.sh", "local")
    assert not caplog.records


def test_isolation_warning_is_once_per_script(caplog, tmp_path):
    """A per-minute job must not emit a warning per run."""
    script = tmp_path / "watchdog.sh"
    with caplog.at_level(logging.WARNING, logger="cron.scheduler"):
        for _ in range(5):
            sched._note_script_runs_outside_backend(script, "ssh")
    assert len([r for r in caplog.records if "OUTSIDE" in r.getMessage()]) == 1


def test_distinct_scripts_each_get_noted(caplog, tmp_path):
    with caplog.at_level(logging.WARNING, logger="cron.scheduler"):
        sched._note_script_runs_outside_backend(tmp_path / "a.sh", "ssh")
        sched._note_script_runs_outside_backend(tmp_path / "b.sh", "ssh")
    assert len([r for r in caplog.records if "OUTSIDE" in r.getMessage()]) == 2


def test_notice_registry_is_bounded(caplog, tmp_path):
    """The dedup set can't grow without bound on job churn."""
    cap = sched._SCRIPT_HOST_NOTICES_MAX
    with caplog.at_level(logging.WARNING, logger="cron.scheduler"):
        for i in range(cap + 10):
            sched._note_script_runs_outside_backend(tmp_path / f"s{i}.sh", "ssh")
    assert len(sched._SCRIPT_HOST_NOTICES) <= cap
