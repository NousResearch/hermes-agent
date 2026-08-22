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
    monkeypatch.delenv("TERMINAL_ENV", raising=False)
    _raw(monkeypatch, {})  # don't read this machine's real config.yaml
    monkeypatch.setattr(sched, "load_config", lambda: {})
    assert sched._effective_terminal_backend() == "local"


def test_backend_is_read_and_normalized(monkeypatch):
    """Merged-config fallback: no raw section, no env."""
    monkeypatch.delenv("TERMINAL_ENV", raising=False)
    _raw(monkeypatch, {})
    monkeypatch.setattr(sched, "load_config", lambda: {"terminal": {"backend": "  SSH "}})
    assert sched._effective_terminal_backend() == "ssh"


def test_backend_read_failure_degrades_to_local(monkeypatch):
    monkeypatch.delenv("TERMINAL_ENV", raising=False)
    _raw(monkeypatch, {})

    def _boom():
        raise RuntimeError("config exploded")

    monkeypatch.setattr(sched, "load_config", _boom)
    # Diagnostics must never raise into the job runner.
    assert sched._effective_terminal_backend() == "local"


# --- backend resolution order ----------------------------------------------
#
# These diagnostics must name the backend the terminal tool ACTUALLY uses; a
# wrong answer sends the operator to the wrong machine while they debug a
# missing script. terminal_tool's order (_ensure_terminal_env_bridged /
# _get_env_config) is:
#
#   1. explicit RAW config.yaml terminal.backend  (bridged with override=True,
#      so it beats even a deliberate TERMINAL_ENV, which may be stale from
#      `hermes setup`)
#   2. an existing TERMINAL_ENV selection
#   3. the merged config default, floor "local"
#
# Only keys present in the raw terminal section override env, so a raw section
# without `backend` leaves TERMINAL_ENV authoritative — hence raw vs merged.


def _raw(monkeypatch, value):
    """Stub hermes_cli.config.read_raw_config, which the helper imports."""
    import hermes_cli.config as hc

    monkeypatch.setattr(hc, "read_raw_config", lambda: value, raising=False)


def test_raw_config_backend_wins_over_env(monkeypatch):
    """terminal.backend in config.yaml beats TERMINAL_ENV (override=True)."""
    monkeypatch.setenv("TERMINAL_ENV", "local")
    _raw(monkeypatch, {"terminal": {"backend": "docker"}})
    assert sched._effective_terminal_backend() == "docker"
    # ...and the mismatch IS explained, because terminal really runs docker.
    assert "docker" in sched._script_runs_on_scheduler_host_hint()


def test_raw_config_backend_is_normalized(monkeypatch):
    monkeypatch.delenv("TERMINAL_ENV", raising=False)
    _raw(monkeypatch, {"terminal": {"backend": "  SSH "}})
    assert sched._effective_terminal_backend() == "ssh"


def test_env_used_when_raw_section_has_no_backend_key(monkeypatch):
    """A raw terminal section without `backend` doesn't override env."""
    monkeypatch.setenv("TERMINAL_ENV", "ssh")
    _raw(monkeypatch, {"terminal": {"docker_image": "x"}})
    assert sched._effective_terminal_backend() == "ssh"


def test_env_used_when_no_raw_terminal_section(monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "  DOCKER ")
    _raw(monkeypatch, {})
    assert sched._effective_terminal_backend() == "docker"
    assert "docker" in sched._script_runs_on_scheduler_host_hint()


def test_env_local_wins_when_config_is_silent(monkeypatch):
    """No raw backend + TERMINAL_ENV=local → local, so no hint is emitted."""
    monkeypatch.setenv("TERMINAL_ENV", "local")
    _raw(monkeypatch, {})
    monkeypatch.setattr(sched, "load_config", lambda: {"terminal": {"backend": "docker"}})
    assert sched._effective_terminal_backend() == "local"
    assert sched._script_runs_on_scheduler_host_hint() == ""


def test_blank_env_falls_back_to_merged_config(monkeypatch):
    """An empty/whitespace TERMINAL_ENV is not a deliberate choice."""
    monkeypatch.setenv("TERMINAL_ENV", "   ")
    _raw(monkeypatch, {})
    monkeypatch.setattr(sched, "load_config", lambda: {"terminal": {"backend": "docker"}})
    assert sched._effective_terminal_backend() == "docker"


def test_raw_config_read_failure_degrades_to_env(monkeypatch):
    """Diagnostics must never raise into the job runner."""
    import hermes_cli.config as hc

    def _boom():
        raise RuntimeError("raw config exploded")

    monkeypatch.setattr(hc, "read_raw_config", _boom, raising=False)
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    assert sched._effective_terminal_backend() == "docker"


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
    assert "OUTSIDE the terminal backend in effect" in messages[0]
    # Names both sources, so the operator knows where to look.
    assert "TERMINAL_ENV" in messages[0]
    assert "terminal.backend" in messages[0]
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
    """The dedup registry can't grow without bound on job churn."""
    cap = sched._SCRIPT_HOST_NOTICES_MAX
    with caplog.at_level(logging.WARNING, logger="cron.scheduler"):
        for i in range(cap + 10):
            sched._note_script_runs_outside_backend(tmp_path / f"s{i}.sh", "ssh")
    assert len(sched._SCRIPT_HOST_NOTICES) <= cap


# --- eviction policy at the cap --------------------------------------------
#
# Paths past the cap used to be logged without being recorded, so the dedupe
# check could never hit them and a job at that position re-warned on EVERY
# tick — the once-per-script guarantee silently stopped holding exactly where
# log volume mattered most. The registry now evicts the oldest entry instead.


def test_once_per_script_still_holds_at_the_cap(caplog, tmp_path):
    """The regression: a path past the cap must not warn on every run."""
    cap = sched._SCRIPT_HOST_NOTICES_MAX
    for i in range(cap):
        sched._note_script_runs_outside_backend(tmp_path / f"filler{i}.sh", "ssh")
    assert len(sched._SCRIPT_HOST_NOTICES) == cap  # registry is full

    overflow = tmp_path / "per_minute_job.sh"
    with caplog.at_level(logging.WARNING, logger="cron.scheduler"):
        for _ in range(10):
            sched._note_script_runs_outside_backend(overflow, "ssh")
    # Exactly once across 10 ticks — not once per tick.
    assert len([r for r in caplog.records if "per_minute_job.sh" in r.getMessage()]) == 1


def test_cap_evicts_the_oldest_entry(caplog, tmp_path):
    cap = sched._SCRIPT_HOST_NOTICES_MAX
    first = tmp_path / "oldest.sh"
    sched._note_script_runs_outside_backend(first, "ssh")
    for i in range(cap - 1):
        sched._note_script_runs_outside_backend(tmp_path / f"filler{i}.sh", "ssh")
    assert str(first) in sched._SCRIPT_HOST_NOTICES

    # One more distinct path evicts the oldest, keeping the registry at the cap.
    sched._note_script_runs_outside_backend(tmp_path / "newest.sh", "ssh")
    assert len(sched._SCRIPT_HOST_NOTICES) == cap
    assert str(first) not in sched._SCRIPT_HOST_NOTICES
    assert str(tmp_path / "newest.sh") in sched._SCRIPT_HOST_NOTICES


def test_evicted_script_can_warn_again_but_not_per_tick(caplog, tmp_path):
    """Eviction trades a bounded re-warn for never silently dropping a script."""
    cap = sched._SCRIPT_HOST_NOTICES_MAX
    script = tmp_path / "evicted.sh"
    sched._note_script_runs_outside_backend(script, "ssh")
    # Push it out with cap distinct paths.
    for i in range(cap):
        sched._note_script_runs_outside_backend(tmp_path / f"churn{i}.sh", "ssh")
    assert str(script) not in sched._SCRIPT_HOST_NOTICES

    caplog.clear()  # drop the setup warnings; count only the re-entry
    with caplog.at_level(logging.WARNING, logger="cron.scheduler"):
        for _ in range(5):
            sched._note_script_runs_outside_backend(script, "ssh")
    # Warns once on re-entry, then dedupes again — not five times.
    assert len([r for r in caplog.records if "evicted.sh" in r.getMessage()]) == 1


def test_recorded_and_emitted_never_disagree(caplog, tmp_path):
    """Every emitted notice has a matching record; no log-without-record."""
    cap = sched._SCRIPT_HOST_NOTICES_MAX
    with caplog.at_level(logging.WARNING, logger="cron.scheduler"):
        for i in range(cap + 25):
            path = tmp_path / f"s{i}.sh"
            before = str(path) in sched._SCRIPT_HOST_NOTICES
            sched._note_script_runs_outside_backend(path, "ssh")
            # A first-time path is always recorded when it is logged.
            if not before:
                assert str(path) in sched._SCRIPT_HOST_NOTICES
    # cap + 25 distinct paths, each logged exactly once as it was first seen.
    assert len([r for r in caplog.records if "OUTSIDE" in r.getMessage()]) == cap + 25
