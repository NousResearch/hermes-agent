"""The suite must not be able to allocate an unbounded estate by running its tests.

Measured 2026-10-28: `tmp/update-reliability-20261028` held 579 pytest basetemps,
1,016,073 files and 76.2 GB, because an explicit `--basetemp` inside the checkout was
accepted: `tmp/` is gitignored, so nothing reported it, and pytest never prunes a
basetemp it was handed. These pin the two admission rules that close it.
"""
from __future__ import annotations

import types
from pathlib import Path

import pytest

from tests import conftest as suite


def _config(basetemp=None):
    factory = types.SimpleNamespace(_given_basetemp=basetemp)
    return types.SimpleNamespace(_tmp_path_factory=factory)


def test_a_basetemp_inside_the_checkout_is_refused():
    inside = suite._CHECKOUT_ROOT / "tmp" / "update-reliability" / "attempt-01"
    with pytest.raises(pytest.UsageError) as exc:
        suite._refuse_basetemp_inside_the_checkout(_config(inside))
    assert "in-checkout pytest basetemp" in str(exc.value)


def test_the_checkout_root_itself_is_refused():
    with pytest.raises(pytest.UsageError):
        suite._refuse_basetemp_inside_the_checkout(_config(suite._CHECKOUT_ROOT))


def test_a_basetemp_outside_the_checkout_is_allowed(tmp_path):
    suite._refuse_basetemp_inside_the_checkout(_config(tmp_path / "b-abc"))


def test_no_basetemp_is_allowed():
    suite._refuse_basetemp_inside_the_checkout(_config(None))


def test_a_run_below_the_reserve_is_refused(monkeypatch):
    monkeypatch.setenv("HERMES_TEST_DISK_RESERVE_MB", "100000000")
    with pytest.raises(pytest.UsageError) as exc:
        suite._require_disk_reserve(_config())
    assert "below the" in str(exc.value)


def test_a_run_above_the_reserve_is_allowed(monkeypatch):
    monkeypatch.setenv("HERMES_TEST_DISK_RESERVE_MB", "1")
    suite._require_disk_reserve(_config())


def test_retention_keeps_failures_and_prunes_successes():
    """A successful run leaves nothing; a failing one leaves its sandbox."""
    assert suite.pytest_configure is not None
    option = types.SimpleNamespace()
    config = types.SimpleNamespace(option=option, addinivalue_line=lambda *a, **k: None,
                                   _tmp_path_factory=types.SimpleNamespace(_given_basetemp=None))
    suite.pytest_configure(config)
    assert option.tmp_path_retention_policy == "failed"
    assert option.tmp_path_retention_count == 3
