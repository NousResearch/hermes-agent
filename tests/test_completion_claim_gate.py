from pathlib import Path

from agent.completion_claim_gate import (
    apply_completion_claim_gate,
    config_from_mapping,
    evaluate_completion_claim_gate,
)


def _write_report(path: Path, status_line: str = "Status: PASS") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "# Verification Report",
                status_line,
                "## Evidence",
                "- tests: unit tests — exit=0 — passed",
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_unverified_done_response_is_withheld():
    response = "Done. I created the files and everything is complete."

    result = apply_completion_claim_gate(
        response,
        enabled=True,
        require_report_for_done=True,
    )

    assert result.changed is True
    assert result.status == "completion_withheld"
    assert result.response.startswith("## Completion claim withheld pending verification")
    assert "could not verify a referenced report" in result.response
    assert "Done. I created" not in result.response


def test_done_response_with_passing_report_is_preserved(tmp_path):
    workspace = tmp_path / "workspace"
    report = workspace / "verification-report.md"
    _write_report(report, "Status: PASS")
    response = "Done. Verification report: ./verification-report.md"

    result = apply_completion_claim_gate(
        response,
        enabled=True,
        require_report_for_done=True,
        allowed_roots=[str(workspace)],
    )

    assert result.changed is False
    assert result.status == "completion_verified"
    assert result.response == response
    assert str(report.resolve()) in result.report_paths


def test_non_completion_response_is_not_changed():
    response = "I inspected the repository and found the relevant files."

    result = apply_completion_claim_gate(
        response,
        enabled=True,
        require_report_for_done=True,
    )

    assert result.changed is False
    assert result.status == "not_a_done_claim"
    assert result.response == response


def test_gate_disabled_is_noop():
    response = "Done. Complete."

    result = apply_completion_claim_gate(
        response,
        enabled=False,
        require_report_for_done=True,
    )

    assert result.changed is False
    assert result.status == "disabled"
    assert result.response == response


def test_passing_report_must_be_referenced_not_just_present(tmp_path):
    report = tmp_path / "verification-report.md"
    _write_report(report, "Status: PASS")
    response = "Done. The work is complete."

    result = evaluate_completion_claim_gate(
        response,
        enabled=True,
        require_report_for_done=True,
    )

    assert result.status == "completion_withheld"
    assert result.changed is True
    assert result.report_paths == []


def test_config_mapping_uses_defaults_and_overrides():
    cfg = config_from_mapping(
        {
            "agent": {
                "completion_claim_gate": {
                    "enabled": True,
                    "require_report_for_done": False,
                    "allowed_roots": [".", "./artifacts"],
                    "report_path_regex": r"(?P<path>.+\\.proof)",
                    "pass_regex": r"^OK$",
                    "remediation_command": "make verify",
                }
            }
        }
    )
    assert cfg == {
        "enabled": True,
        "require_report_for_done": False,
        "allowed_roots": [".", "./artifacts"],
        "report_path_regex": r"(?P<path>.+\\.proof)",
        "pass_regex": r"^OK$",
        "remediation_command": "make verify",
    }


def test_failed_report_still_withholds(tmp_path):
    report = tmp_path / "verification-report.md"
    _write_report(report, "Status: FAIL")
    response = f"Done. Verification report: {report}"

    result = evaluate_completion_claim_gate(
        response,
        enabled=True,
        require_report_for_done=True,
        allowed_roots=[str(tmp_path)],
    )

    assert result.status == "completion_withheld"
    assert result.changed is True
    assert str(report.resolve()) in result.report_paths


def test_default_pass_regex_accepts_json_reports(tmp_path):
    report = tmp_path / "verification-report.json"
    report.write_text('{"status": "PASS", "details": ["pytest ok"]}\n', encoding="utf-8")
    response = f"Done. Verification report: {report}"

    result = evaluate_completion_claim_gate(
        response,
        enabled=True,
        require_report_for_done=True,
        allowed_roots=[str(tmp_path)],
    )

    assert result.changed is False
    assert result.status == "completion_verified"
    assert str(report.resolve()) in result.report_paths


def test_chinese_analysis_text_with_wanchengdu_is_not_treated_as_done():
    response = "一、产品完成度雷达图\n\n这里是当前系统的成熟度与完成度分析。"

    result = evaluate_completion_claim_gate(
        response,
        enabled=True,
        require_report_for_done=True,
    )

    assert result.changed is False
    assert result.status == "not_a_done_claim"
    assert result.response == response


def test_explicit_chinese_completion_claim_still_triggers_gate():
    response = "已完成修复，请验收。"

    result = evaluate_completion_claim_gate(
        response,
        enabled=True,
        require_report_for_done=True,
    )

    assert result.changed is True
    assert result.status == "completion_withheld"
    assert result.response.startswith("## Completion claim withheld pending verification")


def test_custom_regexes_support_non_default_report_and_pass_markers(tmp_path):
    report = tmp_path / "proof.txt"
    _write_report(report, "Result: PASSED")
    response = f"Done. Evidence file: {report}"

    result = evaluate_completion_claim_gate(
        response,
        enabled=True,
        require_report_for_done=True,
        allowed_roots=[str(tmp_path)],
        report_path_regex=r"(?P<path>(?:~|/|\.)?[^\s`'\"]*proof\.txt)",
        pass_regex=r"^Result:\s*PASSED$",
        remediation_command="make verify",
    )

    assert result.changed is False
    assert result.status == "completion_verified"
    assert str(report.resolve()) in result.report_paths


def test_custom_remediation_command_is_rendered_in_withheld_message():
    response = "Done. Everything is complete."

    result = evaluate_completion_claim_gate(
        response,
        enabled=True,
        require_report_for_done=True,
        remediation_command="make verify",
    )

    assert result.changed is True
    assert "```bash\nmake verify\n```" in result.response


def test_unrelated_pass_file_outside_allowed_roots_does_not_bypass_gate(tmp_path):
    inside = tmp_path / "workspace"
    outside = tmp_path / "outside"
    outside_report = outside / "verification-report.md"
    _write_report(outside_report, "Status: PASS")
    response = f"Done. Verification report: {outside_report}"

    result = evaluate_completion_claim_gate(
        response,
        enabled=True,
        require_report_for_done=True,
        allowed_roots=[str(inside)],
    )

    assert result.changed is True
    assert result.status == "completion_withheld"
    assert result.report_paths == []


import pytest


@pytest.mark.parametrize(
    "response",
    [
        "I created a plan for the implementation.",
        "I fixed the typo in my explanation above.",
        "Work is not done yet.",
        "This report describes completed requests historically.",
        "The task is complete once tests pass.",
        "Complete these steps to reproduce.",
    ],
)
def test_false_positive_phrasings_are_not_treated_as_done_claims(response):
    result = evaluate_completion_claim_gate(
        response,
        enabled=True,
        require_report_for_done=True,
    )

    assert result.changed is False
    assert result.status == "not_a_done_claim"


@pytest.mark.parametrize(
    "response",
    [
        "Done. If you want, I can also summarize the diff.",
        "Completed. When ready, I can open a follow-up PR.",
        "Finished. After you review it, I can clean up the branch.",
        "Here is a summary of changes.\n\nDone. Verification report pending.",
        "Summary first line. Done.\nMore detail follows.",
    ],
)
def test_done_claims_with_follow_up_or_leading_summary_still_trigger_gate(response):
    result = evaluate_completion_claim_gate(
        response,
        enabled=True,
        require_report_for_done=True,
    )

    assert result.changed is True
    assert result.status == "completion_withheld"


def test_invalid_regex_config_fails_closed():
    result = evaluate_completion_claim_gate(
        "Done.",
        enabled=True,
        require_report_for_done=True,
        report_path_regex="(",
    )

    assert result.changed is True
    assert result.status == "completion_withheld"
    assert "invalid_gate_config" in result.reason
