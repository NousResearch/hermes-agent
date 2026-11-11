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

import asyncio
import json
from types import SimpleNamespace
from uuid import uuid4

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
def _no_managed_store(tmp_path, monkeypatch):
    """Point PM's store at an empty dir: script runs must not select the host install's
    dependency venv (POSIX cron scripts run on it when a store is committed)."""
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "no-pm-store"))


@pytest.fixture(autouse=True)
def _default_cron_test_model(monkeypatch):
    """Pin a default HERMES_MODEL so cron run_job tests have a resolvable model."""
    monkeypatch.setenv("HERMES_MODEL", "test-cron-default-model")
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


def owner_execution_runner():
    """The production owner execution seam as a ``run_canonical_job`` replacement.

    This is not admission/RPC coverage: the real scheduler runs under the exact owner-store
    identity and profile scope. A plain function so a subprocess tick can install it too.
    """
    from gateway.session_contract import SessionRef
    from gateway.session_cron import current_execution, execute
    from hermes_state_registry import acquire, release

    def run(job, *, extra_prompt=None, cancel_event=None, execution_id=None):
        from cron.scheduler import _get_hermes_home

        db = acquire(_get_hermes_home() / "state.db")
        identity = uuid4().hex
        ref = SessionRef("test-profile", "cron-" + identity)
        db.create_session(ref.session_id, source="cron")
        authority = SimpleNamespace(
            db=db, sessions={ref.session_id: SimpleNamespace(
                source=SimpleNamespace(user_id="cron-owner"))}, pending_results={})
        request_id = execution_id or identity
        row = {"admission_id": identity, "request_id": request_id,
               "principal_id": "cron-owner", "payload": {"text": extra_prompt or ""}}
        policy = SimpleNamespace(request_json=json.dumps({
            "cron_job": job, "extra_prompt": extra_prompt, "request_id": request_id}))

        async def execute_job():
            previous = current_execution()
            try:
                await execute(authority, ref, row, policy)
            except RuntimeError:
                if identity not in authority.pending_results:
                    raise
            assert current_execution() is previous
            assert not authority._cron_cancellations
            saved = authority.pending_results[identity]["result"]
            # The transport carries run-side verdicts back like run_canonical_job's status poll.
            job.update(saved.get("cron_job_flags") or {})
            return tuple(saved["cron_result"])

        try:
            return asyncio.run(execute_job())
        finally:
            release(db)

    return run


@pytest.fixture
def cron_owner(monkeypatch):
    """Replace only the client transport with the production owner execution seam."""
    run = owner_execution_runner()
    monkeypatch.setattr("cron.scheduler_authority.run_canonical_job", run)
    return run
