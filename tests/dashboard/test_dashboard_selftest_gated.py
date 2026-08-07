"""The dashboard self-test must stay armed on auth-gated binds.

``_dashboard_selftest_loop`` used to hit a bare ``continue`` whenever
``app.state.auth_required`` was set, because the in-process HTTP probe cannot
authenticate under the gate (``_require_token`` ignores ``_SESSION_TOKEN`` and
requires ``request.state.session``, which only ``gated_auth_middleware``
sets). The result: ``/api/status`` reported ``"selftest": "unknown"`` forever
on exactly the deployments that are remotely reachable — the health canary was
inert on the only binds that need one.

The gated path now probes the same sync helper ``/api/status`` builds its
``components`` rollup from, with no token minted and no hole punched through
the gate.
"""

from __future__ import annotations

import asyncio

import pytest

from hermes_cli import web_server


@pytest.fixture
def health(monkeypatch):
    """A fresh health recorder so counters can't leak between tests."""
    fresh = web_server.DashboardHealth()
    monkeypatch.setattr(web_server, "DASHBOARD_HEALTH", fresh)
    return fresh


@pytest.fixture
def storage(monkeypatch):
    """Stub ``gateway.readiness._probe_state_db``; returns the status setter."""
    import gateway.readiness as readiness

    state = {"status": "ok"}
    monkeypatch.setattr(
        readiness, "_probe_state_db", lambda home: {"status": state["status"]}
    )
    return state


class TestGatedSelfTestProbe:
    def test_records_ok_when_components_are_healthy(self, health, storage):
        asyncio.run(web_server._dashboard_selftest_gated_once())

        assert health.selftest_status == "ok"
        assert health.selftest_at is not None

    def test_records_failing_when_storage_is_degraded(self, health, storage):
        storage["status"] = "degraded"

        asyncio.run(web_server._dashboard_selftest_gated_once())

        assert health.selftest_status == "failing"

    def test_records_failing_on_recent_unhandled_errors(self, health, storage):
        health.record_error("OSError", "/api/sessions")

        asyncio.run(web_server._dashboard_selftest_gated_once())

        assert health.selftest_status == "failing"

    def test_probe_failure_records_failing_rather_than_raising(
        self, health, monkeypatch
    ):
        import gateway.readiness as readiness

        def boom(home):
            raise OSError(24, "Too many open files")

        monkeypatch.setattr(readiness, "_probe_state_db", boom)

        asyncio.run(web_server._dashboard_selftest_gated_once())

        assert health.selftest_status == "failing"

    def test_verdict_does_not_feed_on_its_own_status(self, health, storage):
        """A prior 'failing' must not pin the component to failing forever.

        ``DashboardHealth.snapshot()['status']`` folds in ``selftest_status``,
        so keying the verdict off it would latch.
        """
        health.record_selftest(False, None)

        asyncio.run(web_server._dashboard_selftest_gated_once())

        assert health.selftest_status == "ok"


class TestLoopDispatch:
    def _run_one_pass(self, monkeypatch, auth_required: bool) -> list:
        calls: list = []
        seen = asyncio.Event()

        async def record_gated():
            calls.append("gated")
            seen.set()

        async def record_http():
            calls.append("http")
            seen.set()

        monkeypatch.setattr(web_server, "_DASHBOARD_SELFTEST_INTERVAL_SECONDS", 0.0)
        monkeypatch.setattr(web_server, "_dashboard_selftest_gated_once", record_gated)
        monkeypatch.setattr(web_server, "_dashboard_selftest_once", record_http)
        # raising=False: app.state carries no auth_required until start_server
        # sets it, and monkeypatch removes it again on teardown.
        monkeypatch.setattr(
            web_server.app.state, "auth_required", auth_required, raising=False
        )

        async def drive():
            task = asyncio.create_task(web_server._dashboard_selftest_loop())
            try:
                await asyncio.wait_for(seen.wait(), timeout=5)
            finally:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task

        asyncio.run(drive())
        return calls

    def test_gated_bind_runs_the_component_probe(self, monkeypatch):
        assert "gated" in self._run_one_pass(monkeypatch, auth_required=True)

    def test_ungated_bind_still_runs_the_http_probe(self, monkeypatch):
        assert "http" in self._run_one_pass(monkeypatch, auth_required=False)

    def test_gated_bind_leaves_selftest_reported(self, health, storage, monkeypatch):
        """The regression proper: gated + one pass ⇒ no longer 'unknown'."""
        assert health.snapshot()["selftest"] == "unknown"

        asyncio.run(web_server._dashboard_selftest_gated_once())

        assert health.snapshot()["selftest"] != "unknown"
