"""Regression tests for the croniter cache-poisoning guard (#105838).

CI runs the suite sharded per directory, so the real trigger — a
module-reload/import-mocking fixture in an earlier directory leaving
``sys.modules["croniter"]`` bound to a module whose ``croniter`` attribute
is the module itself — never occurs there. These tests drive the conftest
guard directly against the poisoned shape observed in the four-directory
batch run (gateway, agent, cli, cron in one pytest process).
"""

import sys
import types

from tests.cron.conftest import _heal_croniter_cache


def _install_poison():
    """Install the exact poisoned shape from the batch-run failures:
    ``sys.modules["croniter"]`` is a module object whose ``croniter``
    attribute points at the module itself, and ``cron.jobs`` has already
    lazily bound that object with ``HAS_CRONITER = True``."""
    fake = types.ModuleType("croniter")
    fake.croniter = fake
    sys.modules["croniter"] = fake
    import cron.jobs as jobs

    jobs.croniter = fake
    jobs.HAS_CRONITER = True
    return fake


def _run_guard():
    _heal_croniter_cache()


def test_guard_heals_poisoned_sys_modules_entry():
    fake = _install_poison()
    try:
        _run_guard()
        assert sys.modules["croniter"] is not fake
        assert callable(sys.modules["croniter"].croniter)
    finally:
        # Leave the genuine package (or nothing) behind, never the fake.
        sys.modules.pop("croniter", None)
        sys.modules.pop("croniter.croniter", None)


def test_guard_heals_cron_jobs_binding():
    _install_poison()
    try:
        _run_guard()
        import cron.jobs as jobs

        assert jobs.HAS_CRONITER is True
        assert callable(jobs.croniter)
        # A schedule parse goes through the healed binding end to end.
        assert jobs.parse_schedule("0 9 * * *")["kind"] == "cron"
    finally:
        sys.modules.pop("croniter", None)
        sys.modules.pop("croniter.croniter", None)
        import cron.jobs as jobs

        jobs.croniter = None
        jobs.HAS_CRONITER = None
        jobs._ensure_croniter()


def test_guard_leaves_healthy_cache_untouched():
    import cron.jobs as jobs

    jobs._ensure_croniter()  # genuine probe, healthy state
    healthy_mod = sys.modules.get("croniter")
    healthy_binding = jobs.croniter
    if healthy_mod is None:
        # croniter not installed in this environment; nothing to guard.
        return
    _run_guard()
    assert sys.modules["croniter"] is healthy_mod
    assert jobs.croniter is healthy_binding
