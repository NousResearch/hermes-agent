"""Conformance suite for session storage backends (``hermes_state_port``).

Core runs it against the local backend (``tests/conformance/storage/test_local.py``). A backend
plugin runs the same suite by importing :class:`StorageConformance` into its own test module and
supplying a ``backend`` fixture that returns a :class:`Backend` for its storage::

    from hermes_conformance.storage import Backend, StorageConformance

    class TestMyBackend(StorageConformance):
        @pytest.fixture
        def backend(self, tmp_path):
            yield Backend(...)

A test that needs a service it cannot reach fails, never skips. What the suite does not yet
cover is listed in :mod:`hermes_conformance.storage.suite`.
"""

from hermes_conformance.storage.suite import (
    AdmissionConformance,
    Backend,
    FenceConformance,
    PinningConformance,
    StorageConformance,
)

__all__ = ["AdmissionConformance", "Backend", "FenceConformance", "PinningConformance", "StorageConformance"]
