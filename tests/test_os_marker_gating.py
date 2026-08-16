"""The collection guard against a test carrying two host-OS markers.

Every marker in ``_OS_MARKS`` skips on all but one host, so two of them on one
item means it runs on no host at all while both the Linux suite and the
tests-os lanes report green. tests/conftest.py fails collection instead; this
pins that behaviour so the guard can't be dropped silently.
"""

from __future__ import annotations

import pytest

from tests.conftest import _OS_MARKS, _reject_multiple_os_marks


class _FakeItem:
    """Stands in for a collected item: the guard reads only these two."""

    def __init__(self, nodeid: str, *marks: str) -> None:
        self.nodeid = nodeid
        self._marks = [getattr(pytest.mark, name).mark for name in marks]

    def iter_markers(self):
        return iter(self._marks)


def test_single_os_marker_is_accepted():
    items = [_FakeItem(f"t.py::test_{name}", name) for name in _OS_MARKS]
    _reject_multiple_os_marks(items)  # must not raise


def test_unmarked_and_non_os_markers_are_accepted():
    _reject_multiple_os_marks([
        _FakeItem("t.py::test_plain"),
        _FakeItem("t.py::test_other", "slow", "integration"),
    ])


def test_two_os_markers_fail_collection():
    items = [
        _FakeItem("t.py::test_ok", "linux_only"),
        _FakeItem("t.py::test_bad", "linux_only", "windows_only"),
    ]
    with pytest.raises(pytest.UsageError) as excinfo:
        _reject_multiple_os_marks(items)

    message = str(excinfo.value)
    assert "t.py::test_bad" in message
    assert "linux_only, windows_only" in message
    # The passing item must not be named — the error is a list of offenders.
    assert "t.py::test_ok" not in message


def test_all_three_markers_are_reported_together():
    item = _FakeItem("t.py::test_worst", *_OS_MARKS)
    with pytest.raises(pytest.UsageError) as excinfo:
        _reject_multiple_os_marks([item])

    for name in _OS_MARKS:
        assert name in str(excinfo.value)
