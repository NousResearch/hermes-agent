"""Conformance suite for cron storage backends (``cron_store_port``).

Core runs it against the local backend (``tests/conformance/cron/test_local.py``). A backend
plugin runs the same suite by importing :class:`CronConformance` into its own test module and
supplying a ``backend`` fixture that returns a :class:`Backend` for its storage::

    from hermes_conformance.cron import Backend, CronConformance

    class TestMyBackend(CronConformance):
        @pytest.fixture
        def backend(self, tmp_path):
            yield Backend(...)

A test that needs a service it cannot reach fails, never skips. What the suite does not yet
cover is listed in :mod:`hermes_conformance.cron.suite`.
"""

from hermes_conformance.cron.suite import (
    Backend, CronConformance, ExecutionConformance, FireClaimConformance, PinningConformance,
    TickConformance,
)

__all__ = ["Backend", "CronConformance", "ExecutionConformance", "FireClaimConformance",
           "PinningConformance", "TickConformance"]
