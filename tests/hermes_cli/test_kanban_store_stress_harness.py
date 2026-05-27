"""Stress-contract harness tests for Kanban store adapters."""

from __future__ import annotations

import os
import signal
from pathlib import Path

import pytest


def test_stress_harness_uses_isolated_sqlite_home_not_ai_team(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", "/tmp/should-not-be-used")
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "ai-team")
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)

    from hermes_cli.kanban_store_stress import StressHarnessConfig, run_claim_stress_contract

    result = run_claim_stress_contract(
        StressHarnessConfig(task_count=4, claim_workers=2, base_dir=tmp_path)
    )

    assert result.backend == "sqlite"
    assert result.board == "db-adapter-stress"
    assert result.board != "ai-team"
    assert Path(result.hermes_home).is_relative_to(tmp_path)
    assert Path(result.db_path).is_relative_to(tmp_path)
    assert os.environ["HERMES_KANBAN_BOARD"] == "ai-team"


def test_claim_stress_contract_has_unique_claims_and_no_leftover_running(tmp_path):
    from hermes_cli.kanban_store_stress import StressHarnessConfig, run_claim_stress_contract

    result = run_claim_stress_contract(
        StressHarnessConfig(task_count=12, claim_workers=4, base_dir=tmp_path)
    )

    assert result.task_count == 12
    assert len(result.claimed_task_ids) == 12
    assert sorted(result.claimed_task_ids) == sorted(set(result.claimed_task_ids))
    assert result.duplicate_claims == []
    assert result.status_counts == {"done": 12}
    assert result.open_run_count == 0


@pytest.mark.skipif(os.name == "nt", reason="POSIX signal crash harness only")
def test_dispatch_signal_stress_contract_covers_sigterm_and_sigkill(tmp_path):
    from hermes_cli.kanban_store_stress import StressHarnessConfig, run_dispatch_signal_contract

    result = run_dispatch_signal_contract(
        StressHarnessConfig(task_count=2, claim_workers=2, base_dir=tmp_path),
        signals=(signal.SIGTERM, signal.SIGKILL),
    )

    assert result.backend == "sqlite"
    assert result.signal_names == ["SIGTERM", "SIGKILL"]
    assert sorted(result.crashed_task_ids) == sorted(result.spawned_task_ids)
    assert result.status_counts == {"ready": 2}
    assert result.open_run_count == 0
    assert result.crash_event_count == 2
    assert result.unexpected_failures == []
