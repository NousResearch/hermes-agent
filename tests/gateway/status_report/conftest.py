"""Fixtures shared by the status report tests."""

import pytest

from tests.gateway.status_report.support import Clock


@pytest.fixture
def clock(monkeypatch):
    return Clock(monkeypatch)
