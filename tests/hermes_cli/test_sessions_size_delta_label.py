"""Tests for the session-store size-delta label (`hermes sessions optimize*`).

A negative before/after delta means the DB grew — printing
"reclaimed -163.0 MB" for that reads as data loss (issue #70146).
"""

from hermes_cli.main import _size_delta_label


def test_shrink_reports_reclaimed():
    assert _size_delta_label(15094.1) == "reclaimed 15094.1 MB"


def test_growth_reports_grew_by_not_negative_reclaimed():
    label = _size_delta_label(-163.0)
    assert label == "grew by 163.0 MB"
    assert "reclaimed" not in label
    assert "-" not in label


def test_zero_delta_reports_reclaimed_zero():
    assert _size_delta_label(0.0) == "reclaimed 0.0 MB"


def test_no_negative_number_ever_rendered():
    """Whatever the sign, the rendered number is never negative."""
    for value in (-9999.9, -0.1, 0.0, 0.1, 9999.9):
        assert "-" not in _size_delta_label(value), value
