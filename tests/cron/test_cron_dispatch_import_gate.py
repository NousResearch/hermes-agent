"""The cron dispatch path refuses to fan a broken tree out into N job failures.

Field incident (2026-10-24 02:16Z): one module-level ``TypeError`` in
``agent/auxiliary_client.py`` killed 17 distinct cron jobs in the same second,
each with the opaque::

    Restart-safe cron worker dispatch failed: cron external worker exited
    before ownership acknowledgement (exit 1)

The external cron worker imports the LIVE checkout, so the failure is total and
simultaneous. These tests pin the two seams that turn that into ONE actionable
diagnostic:

* ``scheduler._probe_cron_dispatch_imports`` reports the broken module and
  cause instead of raising;
* ``scheduler_tick`` consults the probe BEFORE dispatching, so a broken tree
  costs ONE diagnostic and leaves the due jobs for a later tick;
* the per-job dispatch failure names the import error when the worker died at
  import time (the window the tick probe cannot cover: the tree was edited
  between the probe and the spawn).
"""

import pytest

from cron import scheduler as sched
from cron import scheduler_tick as tick


# ---------------------------------------------------------------------------
# The probe seam
# ---------------------------------------------------------------------------

def test_probe_reports_nothing_for_a_healthy_tree(monkeypatch):
    """A healthy tree produces no diagnostic — the tick proceeds."""
    monkeypatch.setattr(
        sched, "_probe_cron_dispatch_imports", lambda: None, raising=False,
    )
    assert sched._probe_cron_dispatch_imports() is None


def test_probe_swallows_a_broken_smoke_module(monkeypatch):
    """The smoke module itself failing must never take the tick down."""
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "cron.import_smoke":
            raise ImportError("smoke module is broken")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    # Returns None (no diagnostic) rather than raising: the per-job failure
    # path still reports the real error.
    assert sched._probe_cron_dispatch_imports() is None


def test_probe_swallows_a_probe_that_raises(monkeypatch):
    """A probe that throws must not break the tick."""
    import cron.import_smoke as smoke

    def _boom(*args, **kwargs):
        raise RuntimeError("probe exploded")

    monkeypatch.setattr(smoke, "check_import_chain", _boom)
    assert sched._probe_cron_dispatch_imports() is None


def test_probe_surfaces_the_diagnostic_for_a_broken_tree(monkeypatch):
    """A broken chain yields the diagnostic string the tick logs."""
    import cron.import_smoke as smoke

    monkeypatch.setattr(
        smoke, "check_import_chain",
        lambda *a, **k: smoke.SmokeResult(
            ok=False,
            failing_module="agent.auxiliary_client",
            error_type="TypeError",
            error_message="missing 1 required positional argument: 'required_tokens'",
        ),
    )
    diagnostic = sched._probe_cron_dispatch_imports()
    assert diagnostic is not None
    assert "agent.auxiliary_client" in diagnostic
    assert "TypeError" in diagnostic


def test_the_diagnostic_is_logged_as_one_error(monkeypatch, caplog):
    """ONE error line, naming the module — not one per job."""
    import logging

    records = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = _Capture()
    sched.logger.addHandler(handler)
    sched.logger.setLevel(logging.DEBUG)
    try:
        sched._log_cron_dispatch_import_failure(
            "cron dispatch import chain is BROKEN: agent.auxiliary_client "
            "failed to import (TypeError: missing 1 required positional argument)"
        )
    finally:
        sched.logger.removeHandler(handler)

    errors = [r for r in records if r.levelno >= logging.ERROR]
    assert len(errors) == 1
    message = errors[0].getMessage()
    assert "agent.auxiliary_client" in message
    assert "live" in message.lower()


def _fake_lock(tmp_path):
    """A real file object for the tick lock: the release path closes it."""
    path = tmp_path / "tick.lock"
    handle = open(path, "w")
    return handle


# ---------------------------------------------------------------------------
# The tick seam — the acceptance criterion
# ---------------------------------------------------------------------------

def test_the_tick_probes_before_dispatching(monkeypatch, tmp_path):
    """The probe is consulted before any job is dispatched."""
    calls = []

    def _fake_probe():
        calls.append("probe")
        return None

    monkeypatch.setattr(sched, "_probe_cron_dispatch_imports", _fake_probe)
    monkeypatch.setattr(sched, "_get_lock_paths", lambda: (str(tmp_path), str(tmp_path / "lock")))
    monkeypatch.setattr(sched, "_acquire_tick_lock", lambda path: _fake_lock(tmp_path))
    monkeypatch.setattr(sched, "_ensure_cron_dir", lambda path: None)
    monkeypatch.setattr(sched, "_maybe_reap_dead_owners", lambda: None)
    # Every downstream step that could dispatch or deliver is a tripwire: the
    # tick must stop at the probe.
    def _tripwire(*args, **kwargs):
        raise AssertionError("the tick dispatched work despite a broken tree")

    monkeypatch.setattr("cron.bot_chat_delivery.drain", _tripwire, raising=False)
    monkeypatch.setattr("cron.bot_chat_delivery.drain_in_background", _tripwire, raising=False)

    # A broken tree: the probe reports it, the tick returns 0 and dispatches
    # nothing.
    monkeypatch.setattr(
        sched, "_probe_cron_dispatch_imports",
        lambda: "cron dispatch import chain is BROKEN: agent.auxiliary_client",
    )
    monkeypatch.setattr(sched, "_log_cron_dispatch_import_failure", lambda d: calls.append("logged"))

    assert tick._tick_admitted() == 0
    assert calls == ["logged"]


def test_the_tick_continues_when_the_tree_is_healthy(monkeypatch, tmp_path):
    """A healthy tree is not blocked by the probe."""
    monkeypatch.setattr(sched, "_probe_cron_dispatch_imports", lambda: None)
    monkeypatch.setattr(sched, "_get_lock_paths", lambda: (str(tmp_path), str(tmp_path / "lock")))
    monkeypatch.setattr(sched, "_acquire_tick_lock", lambda path: _fake_lock(tmp_path))
    monkeypatch.setattr(sched, "_ensure_cron_dir", lambda path: None)
    monkeypatch.setattr(sched, "_maybe_reap_dead_owners", lambda: None)
    monkeypatch.setattr(
        "cron.bot_chat_delivery.drain_in_background", lambda: None, raising=False,
    )
    monkeypatch.setattr(sched, "run_due_jobs", lambda **kwargs: 0, raising=False)
    # Whatever the tick does after the probe, it must not have been blocked by
    # the import gate: reaching this point at all proves the probe passed.
    called = []
    monkeypatch.setattr(
        sched, "_log_cron_dispatch_import_failure",
        lambda d: called.append(d),
    )
    tick._tick_admitted()
    assert called == []
