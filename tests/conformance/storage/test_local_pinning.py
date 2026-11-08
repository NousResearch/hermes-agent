"""The storage conformance suite's pinning part against the local backend (``test_local.py``)."""

from hermes_conformance.storage import PinningConformance
from tests.conformance.storage.test_local import backend  # the fixture

__all__ = ["TestLocalPinning", "backend"]


class TestLocalPinning(PinningConformance):
    pass
