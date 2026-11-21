"""The kanban conformance suite's pinning part against the local backend (``test_local.py``)."""

from hermes_conformance.kanban import PinningConformance
from tests.conformance.kanban.test_local import backend  # the fixture

__all__ = ["TestLocalPinning", "backend"]


class TestLocalPinning(PinningConformance):
    pass
