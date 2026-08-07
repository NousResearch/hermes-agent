"""Cron execution-ledger test isolation regressions."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Deliberately import during collection.  The cron fixture must repair modules
# that cached their store before per-test fixtures ran.
import cron.executions as executions


def test_execution_store_follows_the_per_test_hermes_home():
    expected = Path(os.environ["HERMES_HOME"]).resolve() / "cron" / "executions.db"

    assert executions.EXECUTIONS_FILE == expected
    record = executions.create_execution("isolation-probe", source="test")

    assert record["job_id"] == "isolation-probe"
    assert expected.exists()


def test_execution_store_guard_rejects_the_default_operator_home(monkeypatch):
    target = (Path.home() / ".hermes" / "cron" / "executions.db").resolve()
    monkeypatch.setattr(executions, "EXECUTIONS_FILE", target)

    with pytest.raises(RuntimeError, match="refusing to write outside"):
        executions.create_execution("guard-probe", source="test")


def test_execution_store_guard_rejects_any_path_outside_per_test_home(
    monkeypatch, tmp_path
):
    target = tmp_path.parent / "profile-home" / "cron" / "executions.db"
    monkeypatch.setattr(executions, "EXECUTIONS_FILE", target)

    with pytest.raises(RuntimeError, match="refusing to write outside"):
        executions.create_execution("profile-guard-probe", source="test")

    assert not target.exists()
