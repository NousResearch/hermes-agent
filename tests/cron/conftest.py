"""Cron-test fixtures.

Provides a default ``HERMES_MODEL`` for cron run_job tests so each one
doesn't have to spell out a model. The global conftest blanks
HERMES_MODEL hermetically; without this autouse fixture every cron test
that exercises ``run_job`` would hit the fail-fast guard added in
``cron/scheduler.py`` (see issue #23979) and have to be rewritten.

Tests that specifically need ``HERMES_MODEL`` unset — model-resolution
edge cases — call ``monkeypatch.delenv("HERMES_MODEL", raising=False)``
inside the test, which overrides this fixture's value for that scope.
"""

import sys

import pytest


@pytest.fixture()
def make_cron_provider():
    """Factory for minimal CronScheduler test doubles.

    ``make_cron_provider(register_job=...)`` returns a real ``CronScheduler``
    subclass instance whose ``register_job`` is the given callable — so tests
    exercising the creation-registration contract share one stub instead of
    redefining inline spy/failing classes, and an ABC rename breaks them
    loudly instead of silently passing a duck-type.
    """
    from cron.scheduler_provider import CronScheduler

    def _make(register_job=None, name="stub"):
        class _StubProvider(CronScheduler):
            @property
            def name(self):  # pragma: no cover - trivial
                return name

            def start(self, stop_event, **kw):  # pragma: no cover - unused
                pass

            def register_job(self, job):
                if register_job is not None:
                    return register_job(job)
                return None

        return _StubProvider()

    return _make


@pytest.fixture(autouse=True)
def _default_cron_test_model(monkeypatch):
    """Pin a default HERMES_MODEL so cron run_job tests have a resolvable model."""
    monkeypatch.setenv("HERMES_MODEL", "test-cron-default-model")
    yield


def _heal_croniter_cache() -> None:
    """Heal a croniter cache poisoned by an earlier test directory (#105838).

    A module-reload/import-mocking fixture in an earlier directory (gateway,
    agent, cli) can leave ``sys.modules["croniter"]`` pointing at a module
    whose ``croniter`` attribute is the *module* itself instead of the
    callable class. ``cron.jobs._ensure_croniter`` then binds that module
    object, and every later ``parse_schedule``/blueprint test fails with
    ``'module' object is not callable`` — the four-directory batch run fails
    ~55 cron tests that are green when the directory runs solo.

    Heal both caches: evict the poisoned module so the genuine package is
    re-imported, then re-probe the ``cron.jobs`` binding. A healthy
    ``sys.modules`` entry (callable ``croniter`` attribute) and legitimate
    monkeypatches (``HAS_CRONITER = False`` modeling a missing dependency,
    or a callable fake) are left untouched. Kept as a plain function so
    ``test_croniter_cache_healing.py`` can drive it directly — CI shards
    per directory, so the real cross-directory trigger never occurs there.
    """
    mod = sys.modules.get("croniter")
    if mod is not None and not callable(getattr(mod, "croniter", None)):
        for name in [
            k for k in sys.modules if k == "croniter" or k.startswith("croniter.")
        ]:
            del sys.modules[name]
        import croniter  # noqa: F401  # re-import the genuine package

    import cron.jobs as jobs

    if jobs.HAS_CRONITER and not callable(jobs.croniter):
        # The lazy probe cached the poisoned object; reset and re-probe so
        # ``_ensure_croniter`` picks up the healed ``sys.modules`` entry.
        jobs.croniter = None
        jobs.HAS_CRONITER = None
        jobs._ensure_croniter()


@pytest.fixture(autouse=True)
def _heal_poisoned_croniter_cache():
    """Run the croniter poisoning guard (see ``_heal_croniter_cache``) before
    each cron test so cross-directory batch runs match per-directory runs."""
    _heal_croniter_cache()
    yield


@pytest.fixture(autouse=True)
def _reset_session_context_vars():
    """Restore session ContextVars around cron tests that call run_job directly.

    Production confines each cron run to a copied context, but direct unit tests
    share the pytest context. ``run_job`` intentionally clears ordinary session
    variables to explicit empty values, which would otherwise shadow legacy env
    fallbacks used by later approval tests in the same process.
    """
    from gateway.session_context import _UNSET, _VAR_MAP

    def _reset_all():
        for var in _VAR_MAP.values():
            var.set(_UNSET)

    _reset_all()
    yield
    _reset_all()
