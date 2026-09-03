"""HTR test fixtures — loads Task 29 advisory inspection shared helpers."""

import pytest

pytest_plugins = ("tests.htr.conftest_advisory_inspection",)


def pytest_configure(config):
    config.addinivalue_line("markers", "provisional: Task 29 provisional T141-T168 (not locked)")
