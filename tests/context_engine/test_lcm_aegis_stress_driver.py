from __future__ import annotations

from pathlib import Path

import pytest

from scripts import lcm_aegis_stress as stress


def test_dry_run_report_has_required_metrics_and_scenario_coverage(tmp_path: Path) -> None:
    out_path = tmp_path / "lcm-aegis-stress.md"

    run = stress.run_stress_gate(
        dry_run=True,
        turns=80,
        compaction_every=4,
        provider_down_compactions={5},
        restart_turns={33, 66},
        out_path=out_path,
    )

    assert run.status == "PASS", "\n".join(run.failures)
    assert run.compaction_count == 20
    assert run.degraded_count == 1
    assert run.degraded_rate == pytest.approx(0.05)
    assert run.max_consecutive_degraded == 1
    assert run.fail_closed_count == 0
    assert run.store_persistence is True
    assert run.estimated_spend_usd > 0
    assert run.observed_spend_usd > 0
    assert run.coverage_counts["normal_chat"] > 0
    assert run.coverage_counts["tool_call_like_row"] > 0
    assert run.coverage_counts["long_tool_output"] > 0
    assert run.coverage_counts["recall_query"] > 0
    assert run.coverage_counts["provider_down_window"] > 0
    assert run.coverage_counts["simulated_restart"] == 2

    report = out_path.read_text(encoding="utf-8")
    assert "# PRD-6 Aegis LCM Stress Gate" in report
    assert "Compaction count: 20" in report
    assert "Degraded count: 1" in report
    assert "Degraded rate: 5.000%" in report
    assert "Fail-closed count: 0" in report
    assert "Store persistence: PASS" in report
    assert "estimated spend" in report
    assert "observed spend" in report
    assert "provider-down window" in report
    assert "simulated restart" in report


def test_degraded_alert_contract_is_loud_for_three_consecutive_compactions(tmp_path: Path) -> None:
    run = stress.run_stress_gate(
        dry_run=True,
        turns=40,
        compaction_every=4,
        provider_down_compactions={2, 3, 4},
        restart_turns=set(),
        out_path=tmp_path / "consecutive.md",
    )

    assert run.status == "FAIL-LOUD"
    assert run.max_consecutive_degraded == 3
    assert any("3 consecutive" in failure for failure in run.failures)


def test_degraded_alert_contract_is_loud_above_five_percent_rate(tmp_path: Path) -> None:
    run = stress.run_stress_gate(
        dry_run=True,
        turns=40,
        compaction_every=4,
        provider_down_compactions={2},
        restart_turns=set(),
        out_path=tmp_path / "rate.md",
    )

    assert run.status == "FAIL-LOUD"
    assert run.degraded_rate == pytest.approx(0.10)
    assert any("> 5.000%" in failure for failure in run.failures)


def test_fail_closed_recall_is_counted_and_fails_loud(tmp_path: Path) -> None:
    run = stress.run_stress_gate(
        dry_run=True,
        turns=36,
        compaction_every=6,
        provider_down_compactions=set(),
        fail_closed_turns={20},
        restart_turns=set(),
        out_path=tmp_path / "fail-closed.md",
    )

    assert run.status == "FAIL-LOUD"
    assert run.fail_closed_count == 1
    assert any("fail-closed" in failure for failure in run.failures)


def test_runbook_names_activation_cutover_rollback_and_store_decisions() -> None:
    runbook = Path("docs/runbooks/lcm-aegis-apollo-cutover.md")

    text = runbook.read_text(encoding="utf-8")
    assert "Aegis activation" in text
    assert "Apollo cutover" in text
    assert "Rollback diff" in text
    assert "store preserve/purge" in text
    assert "Apollo owns the privileged" in text


def test_stress_script_contains_no_live_gateway_restart_command() -> None:
    source = Path(stress.__file__).read_text(encoding="utf-8")

    forbidden = [
        "launchctl kickstart",
        "systemctl --user restart",
        "hermes gateway restart",
        "gateway restart",
    ]
    assert not any(command in source for command in forbidden)
