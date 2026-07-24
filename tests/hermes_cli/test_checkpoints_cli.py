"""Tests for the `hermes checkpoints` CLI helpers."""

import argparse

import pytest

from hermes_cli import checkpoints


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan"), "bad", None])
def test_timestamp_formatters_hide_invalid_values(value):
    assert checkpoints._fmt_ts(value) == "—"
    assert checkpoints._fmt_age(value) == "—"


def test_age_formatter_keeps_future_finite_timestamps_as_now(monkeypatch):
    monkeypatch.setattr(checkpoints.time, "time", lambda: 1_000.0)

    assert checkpoints._fmt_age(1_001.0) == "now"


def test_status_tolerates_malformed_project_last_touch(monkeypatch, capsys):
    from tools import checkpoint_manager

    monkeypatch.setattr(
        checkpoint_manager,
        "store_status",
        lambda: {
            "base": "/tmp/checkpoints",
            "total_size_bytes": 0,
            "store_size_bytes": 0,
            "legacy_size_bytes": 0,
            "project_count": 3,
            "projects": [
                {"workdir": "/finite", "commits": 1, "last_touch": 900.0, "exists": True},
                {"workdir": "/nan", "commits": 1, "last_touch": float("nan"), "exists": True},
                {"workdir": "/string", "commits": 1, "last_touch": "bad", "exists": False},
            ],
            "legacy_archives": [],
        },
    )
    monkeypatch.setattr(checkpoints.time, "time", lambda: 1_000.0)

    assert checkpoints.cmd_status(argparse.Namespace(limit=20)) == 0
    output = capsys.readouterr().out

    assert "/finite" in output
    assert "/nan" in output
    assert "/string" in output
    assert "—" in output
