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

import pytest


@pytest.fixture(autouse=True)
def _default_cron_test_model(monkeypatch):
    """Pin a default HERMES_MODEL so cron run_job tests have a resolvable model."""
    monkeypatch.setenv("HERMES_MODEL", "test-cron-default-model")
    yield


@pytest.fixture(autouse=True)
def _reset_session_context_vars():
    """Reset every session-context ContextVar to its _UNSET default per test.

    Cron tests drive the real ``run_job`` directly in the pytest context, and
    its ``clear_session_vars`` finally intentionally pins every session var to
    an explicit ``""`` (the gateway relies on that to suppress the
    ``os.environ`` fallback). In production the ticker confines that to a
    per-job ``copy_context()``, but in a single-process test run it leaks into
    later tests that rely on the env fallback: the approval timeout tests
    resolve their session key through ``get_session_env`` and stop finding
    their registered gateway callback after any cron test has run ``run_job``.
    Restoring the defaults on both sides of each test keeps the cron suite
    order-independent.
    """
    from gateway.session_context import _VAR_MAP, _UNSET

    def _reset_all():
        for var in _VAR_MAP.values():
            var.set(_UNSET)

    _reset_all()
    yield
    _reset_all()
