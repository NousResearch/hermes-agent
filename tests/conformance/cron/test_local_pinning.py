"""The cron conformance suite's pinning part against the local backend (``test_local.py``)."""

from hermes_conformance.cron import PinningConformance
from tests.conformance.cron.test_local import backend  # the fixture

__all__ = ["TestLocalPinning", "backend"]


class TestLocalPinning(PinningConformance):
    pass
