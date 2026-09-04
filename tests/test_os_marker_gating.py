"""The collection guard against a test carrying two platforms() markers.

A module-level gate stacked on a per-test gate ran on no host at all while
both the full-suite and marked lanes reported green — the silent coverage
loss the guard exists for. tests/conftest.py fails collection instead; this
pins that behaviour so the guard can't be dropped silently.
"""

from __future__ import annotations

import pytest

from tests.conftest import _reject_contradictory_platform_marks


class _FakeItem:
    """Stands in for a collected item: the guard reads only these two."""

    def __init__(self, nodeid: str, *marks) -> None:
        self.nodeid = nodeid
        self._marks = list(marks)

    def iter_markers(self, name=None):
        if name is None:
            return iter(self._marks)
        return iter(m for m in self._marks if m.name == name)


def test_single_platforms_marker_is_accepted():
    items = [
        _FakeItem("t.py::test_linux", pytest.mark.platforms("linux")),
        _FakeItem("t.py::test_win", pytest.mark.platforms("windows", arch="arm64")),
        _FakeItem("t.py::test_not", pytest.mark.platforms("not macos")),
    ]
    _reject_contradictory_platform_marks(items)  # must not raise


def test_unmarked_and_non_platform_markers_are_accepted():
    _reject_contradictory_platform_marks(
        [
            _FakeItem("t.py::test_plain"),
            _FakeItem("t.py::test_slow", pytest.mark.slow),
        ]
    )


def test_two_platforms_markers_fail_collection():
    items = [
        _FakeItem("t.py::test_ok", pytest.mark.platforms("linux")),
        _FakeItem(
            "t.py::test_bad",
            pytest.mark.platforms("linux"),
            pytest.mark.platforms("windows"),
        ),
    ]
    with pytest.raises(pytest.UsageError) as excinfo:
        _reject_contradictory_platform_marks(items)

    message = str(excinfo.value)
    assert "t.py::test_bad" in message
    assert "at most one platforms()" in message
    # The passing item must not be named — the error is a list of offenders.
    assert "t.py::test_ok" not in message
