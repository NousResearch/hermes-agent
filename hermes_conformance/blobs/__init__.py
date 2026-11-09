"""Conformance suite for blob storage backends (``hermes_blob_port``).

Core runs it against the local backend (``tests/conformance/blobs/test_local.py``); the
participant-level invariants need a backend that stores the participant level, which today's
local one does not (the documented local exception), so they run here against a two-level
reference backend (``tests/conformance/blobs/test_reference.py``). A backend plugin runs the
same suite by importing these classes into its own test module and supplying a ``backend``
fixture that returns a :class:`Backend` for its storage::

    from hermes_conformance.blobs import Backend, BlobConformance

    class TestMyBackend(BlobConformance):
        @pytest.fixture
        def backend(self, tmp_path):
            yield Backend(...)

A test that needs a service it cannot reach fails, never skips. What the suite does not yet
cover is listed in :mod:`hermes_conformance.blobs.suite`.
"""

from hermes_conformance.blobs.suite import (
    Backend, BlobConformance, BlobLevelRefusalConformance, ParticipantConformance,
    PinningConformance,
)

__all__ = [
    "Backend", "BlobConformance", "BlobLevelRefusalConformance", "ParticipantConformance",
    "PinningConformance",
]
