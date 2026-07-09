from __future__ import annotations

from pathlib import Path

import pytest


def test_detects_required_output_paths_from_prompt_and_kanban_card(tmp_path):
    from hermes_cli.artifact_contracts import detect_required_artifacts

    report = tmp_path / "reports" / "final-synthesis.md"
    card_body = f"""
    Task: final synthesis
    Required output file: {report}
    Acceptance: do not finish with stdout-only prose; the file above is the artifact.
    """

    contracts = detect_required_artifacts(card_body)

    assert any(
        item["path"] == str(report) and item["required"] is True
        for item in contracts
    )


def test_max_turn_closeout_writes_required_report_from_useful_stdout(tmp_path, monkeypatch):
    from hermes_cli.closeout_state import write_closeout_state
    from hermes_cli.closure_artifacts import read_closure_artifact

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    report = tmp_path / "reports" / "final-synthesis.md"
    useful_stdout = "# Final synthesis\n\nUseful stdout that must survive max-turn closeout.\n"

    packet_path = write_closeout_state(
        session_id="required-artifact-session",
        task_id="task-9",
        task_contract=f"Produce the final report at {report}",
        final_response=useful_stdout,
        remaining_closeout_tasks=["verify required report artifact"],
    )

    assert report.exists()
    assert "Useful stdout" in report.read_text(encoding="utf-8")

    packet = read_closure_artifact(packet_path)
    assert packet["required_artifacts"][0]["path"] == str(report)
    assert packet["required_artifacts"][0]["status"] == "written"
    assert str(report) in packet["verified_artifacts"]


def test_required_artifact_write_preempts_exploration_near_budget(tmp_path):
    from hermes_cli.artifact_contracts import required_artifact_guard_action

    report = tmp_path / "reports" / "closeout.md"

    action = required_artifact_guard_action(
        prompt=f"Write the required report to {report}",
        stdout_draft="draft content from completed work",
        turns_remaining=1,
    )

    assert action["action"] == "write_required_artifact"
    assert action["path"] == str(report)
