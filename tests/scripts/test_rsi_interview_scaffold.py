from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[2] / "scripts" / "rsi_interview.py"
FULL_ROSTER = [
    "default",
    "buggy",
    "coder",
    "jade",
    "jade-ops",
    "product",
    "qa",
    "research",
    "reviewer",
    "rsi",
    "x",
    "yuki",
    "yuki-ops",
]


def _load_module():
    spec = importlib.util.spec_from_file_location("rsi_interview", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return _load_module()


def _empty_report(profile: str) -> dict:
    return {
        "profile": profile,
        "autonomous_failures": [],
        "incomplete_tasks": [],
        "incidents": [],
        "correction_feedback": [],
        "accounted_session_ids": [],
    }


def _audit(profile: str = "coder") -> dict:
    return {
        "profiles": {
            profile: {
                "sessions": [],
                "session_failures": [
                    {
                        "id": "session-failed",
                        "source": "kanban",
                        "title": "failed run",
                        "end_reason": "agent_close",
                        "fail_hits": ["tool:terminal:exit_code=1"],
                    },
                    {
                        "id": "session-needs-input",
                        "source": "kanban",
                        "title": "blocked run",
                        "end_reason": "cli_close",
                        "fail_hits": ["lifecycle:needs_input"],
                    },
                ],
                "cron_failures": [
                    {
                        "execution_id": "cron-failed",
                        "name": "eng-completion",
                        "status": "failed",
                        "error": "HTTP 429",
                    }
                ],
                "kanban_failures": [
                    {
                        "task_id": "t_failed",
                        "title": "failed task",
                        "status": "blocked",
                        "outcome": "needs_input",
                    }
                ],
            }
        }
    }


def _enrichment(profile: str = "coder") -> dict:
    report = _empty_report(profile)
    report["autonomous_failures"] = [
        {
            "id": item,
            "summary": f"Summary for {item}",
            "evidence": f"Evidence for {item}",
            "suggested_fix": f"Fix for {item}",
        }
        for item in ("session-failed", "session-needs-input", "cron-failed")
    ]
    report["incomplete_tasks"] = [
        {
            "id": item,
            "title": f"Title for {item}",
            "summary": f"Summary for {item}",
            "why_incomplete": f"Reason for {item}",
        }
        for item in ("session-needs-input", "t_failed")
    ]
    report["accounted_session_ids"] = [
        "session-failed",
        "session-needs-input",
        "cron-failed",
        "t_failed",
    ]
    return report


def test_scaffold_mechanically_covers_coder_and_reviewer_recurrent_omissions(mod):
    audit = {
        "profiles": {
            profile: {
                "session_failures": [
                    {
                        "id": f"{profile}-failed-id",
                        "fail_hits": ["tool:terminal:exit_code=1"],
                    }
                ],
                "cron_failures": [],
                "kanban_failures": [],
            }
            for profile in ("coder", "reviewer")
        }
    }

    for profile in ("coder", "reviewer"):
        scaffold = mod.build_scaffold(profile, audit)
        expected = f"{profile}-failed-id"
        assert [row["id"] for row in scaffold["autonomous_failures"]] == [expected]
        assert scaffold["accounted_session_ids"] == [expected]


def test_needs_input_session_is_required_in_both_categories(mod):
    scaffold = mod.build_scaffold("coder", _audit())

    assert [row["id"] for row in scaffold["autonomous_failures"]] == [
        "session-failed",
        "session-needs-input",
        "cron-failed",
    ]
    assert [row["id"] for row in scaffold["incomplete_tasks"]] == [
        "session-needs-input",
        "t_failed",
    ]


@pytest.mark.parametrize(
    "bad_record",
    [
        {
            "id": "session-failed-typo",
            "summary": "mistyped",
            "evidence": "mistyped",
            "suggested_fix": "retry",
        },
        {
            "id": "grouped-failures",
            "summary": "session-failed and session-needs-input",
            "evidence": "two grouped failures",
            "suggested_fix": "retry",
        },
    ],
)
def test_mistyped_or_grouped_model_records_cannot_replace_scaffold_rows(mod, bad_record):
    model_report = _empty_report("coder")
    model_report["autonomous_failures"] = [bad_record]

    result = mod.merge_interview("coder", model_report, _audit())

    assert [row["id"] for row in result.report["autonomous_failures"][:3]] == [
        "session-failed",
        "session-needs-input",
        "cron-failed",
    ]
    assert "session-failed" in result.missing_qualitative_ids


def test_audited_id_in_wrong_model_category_is_not_preserved_there(mod):
    model_report = _enrichment()
    model_report["incomplete_tasks"].append(
        {
            "id": "cron-failed",
            "title": "misclassified cron",
            "summary": "wrong category",
            "why_incomplete": "wrong category",
        }
    )

    result = mod.merge_interview("coder", model_report, _audit())

    assert "cron-failed" not in {row["id"] for row in result.report["incomplete_tasks"]}
    assert "cron-failed" in {row["id"] for row in result.report["autonomous_failures"]}


def test_model_deleted_entries_are_restored_from_scaffold(mod):
    model_report = _enrichment()
    model_report["autonomous_failures"] = []
    model_report["incomplete_tasks"] = []
    model_report["accounted_session_ids"] = []

    result = mod.merge_interview("coder", model_report, _audit())

    assert {row["id"] for row in result.report["autonomous_failures"]} >= {
        "session-failed",
        "session-needs-input",
        "cron-failed",
    }
    assert {row["id"] for row in result.report["incomplete_tasks"]} >= {
        "session-needs-input",
        "t_failed",
    }
    assert result.report["accounted_session_ids"] == [
        "session-failed",
        "session-needs-input",
        "cron-failed",
        "t_failed",
    ]


def test_duplicate_ids_merge_once_when_enrichment_is_compatible(mod):
    model_report = _enrichment()
    model_report["autonomous_failures"] = [
        {
            "id": "session-failed",
            "summary": "one summary",
            "evidence": "one evidence",
            "suggested_fix": "one fix",
        },
        {
            "id": "session-failed",
            "summary": "one summary",
            "evidence": "one evidence",
            "suggested_fix": "one fix",
        },
        *model_report["autonomous_failures"][1:],
    ]

    result = mod.merge_interview("coder", model_report, _audit())

    assert [row["id"] for row in result.report["autonomous_failures"]].count("session-failed") == 1
    assert result.conflicts == []


def test_conflicting_duplicate_enrichment_is_reported_for_grill(mod):
    model_report = _enrichment()
    model_report["autonomous_failures"].insert(
        1,
        {
            "id": "session-failed",
            "summary": "contradictory summary",
            "evidence": "Evidence for session-failed",
            "suggested_fix": "Fix for session-failed",
        },
    )

    result = mod.merge_interview("coder", model_report, _audit())

    assert result.conflicts == ["autonomous_failures id=session-failed has conflicting summary values"]


def test_valid_enrichment_preserves_model_qualitative_fields(mod):
    result = mod.merge_interview("coder", _enrichment(), _audit())

    assert result.conflicts == []
    assert result.missing_qualitative_ids == []
    row = next(row for row in result.report["autonomous_failures"] if row["id"] == "session-failed")
    assert row["summary"] == "Summary for session-failed"
    assert row["suggested_fix"] == "Fix for session-failed"


def test_empty_audit_slice_leaves_clean_empty_report_unchanged(mod):
    audit = {
        "profiles": {
            "qa": {
                "sessions": [],
                "session_failures": [],
                "cron_failures": [],
                "kanban_failures": [],
            }
        }
    }

    result = mod.merge_interview("qa", _empty_report("qa"), audit)

    assert result.report == _empty_report("qa")
    assert result.conflicts == []
    assert result.missing_qualitative_ids == []


@pytest.mark.parametrize("unavailable", [None, "HTTP 429: limit exhausted", {"error": "unavailable"}])
def test_unavailable_interview_still_emits_complete_accounting_scaffold(mod, unavailable):
    result = mod.merge_interview("coder", unavailable, _audit())

    assert result.report["accounted_session_ids"] == [
        "session-failed",
        "session-needs-input",
        "cron-failed",
        "t_failed",
    ]
    assert set(result.missing_qualitative_ids) == {
        "session-failed",
        "session-needs-input",
        "cron-failed",
        "t_failed",
    }


def test_builder_embeds_machine_readable_scaffold_before_interview(tmp_path, mod):
    import json
    import os
    import subprocess

    home = tmp_path / "home"
    store = home / ".hermes" / "rsi"
    (store / "audit").mkdir(parents=True)
    (store / "interview-prompt.txt").write_text("BASE\n", encoding="utf-8")
    (store / "audit" / "latest.json").write_text(json.dumps(_audit()), encoding="utf-8")
    builder = Path(__file__).parents[2] / "scripts" / "rsi-build-interview.py"

    run = subprocess.run(
        ["python3", str(builder), "coder"],
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, "HOME": str(home)},
    )

    assert run.returncode == 0, run.stderr
    assert '"autonomous_failures":[{"id":"<exact audited id>"' in run.stdout
    marker = "MANDATORY_REPORT_SCAFFOLD (runner-owned; enrich but do not alter IDs/categories):\n"
    assert marker in run.stdout
    scaffold = json.loads(run.stdout.split(marker, 1)[1])
    assert scaffold == mod.build_scaffold("coder", _audit())


def test_merge_cli_writes_scaffolded_report_and_grills_only_missing_quality(tmp_path):
    import json
    import subprocess

    runner = Path(__file__).parents[2] / "scripts" / "rsi-merge-interview.py"
    audit_path = tmp_path / "audit.json"
    raw_path = tmp_path / "raw.json"
    output_path = tmp_path / "merged.json"
    grill_path = tmp_path / "grill.txt"
    grill_base = tmp_path / "grill-base.txt"
    audit_path.write_text(json.dumps(_audit()), encoding="utf-8")
    raw_path.write_text(json.dumps(_empty_report("coder")), encoding="utf-8")
    grill_base.write_text("GRILL", encoding="utf-8")

    run = subprocess.run(
        [
            "python3",
            str(runner),
            "coder",
            str(raw_path),
            str(output_path),
            "--audit",
            str(audit_path),
            "--grill-prompt",
            str(grill_base),
            "--grill-output",
            str(grill_path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert run.returncode == 1
    merged = json.loads(output_path.read_text(encoding="utf-8"))
    assert merged["accounted_session_ids"] == [
        "session-failed",
        "session-needs-input",
        "cron-failed",
        "t_failed",
    ]
    payload = json.loads(run.stdout)
    assert not any("missing exact audited IDs" in error for error in payload["errors"])
    assert any("missing qualitative fields" in error for error in payload["errors"])
    assert grill_path.read_text(encoding="utf-8").startswith("GRILL")


def test_merge_cli_accepts_valid_enrichment_without_grill(tmp_path):
    import json
    import subprocess

    runner = Path(__file__).parents[2] / "scripts" / "rsi-merge-interview.py"
    audit_path = tmp_path / "audit.json"
    raw_path = tmp_path / "raw.json"
    output_path = tmp_path / "merged.json"
    grill_path = tmp_path / "grill.txt"
    audit_path.write_text(json.dumps(_audit()), encoding="utf-8")
    raw_path.write_text(json.dumps(_enrichment()), encoding="utf-8")

    run = subprocess.run(
        [
            "python3",
            str(runner),
            "coder",
            str(raw_path),
            str(output_path),
            "--audit",
            str(audit_path),
            "--grill-output",
            str(grill_path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert run.returncode == 0, run.stdout + run.stderr
    assert not grill_path.exists()
    assert json.loads(run.stdout)["valid"] is True


def test_thirteen_profile_fixture_matrix_is_complete_and_isolated(mod):
    audit = {
        "profiles": {
            profile: {
                "sessions": [],
                "session_failures": [
                    {"id": f"{profile}-session", "fail_hits": ["tool:terminal:exit_code=1"]}
                ],
                "cron_failures": [],
                "kanban_failures": [],
            }
            for profile in FULL_ROSTER
        }
    }

    for profile in FULL_ROSTER:
        result = mod.merge_interview(profile, _empty_report(profile), audit)
        assert result.report["accounted_session_ids"] == [f"{profile}-session"]
        assert [row["id"] for row in result.report["autonomous_failures"]] == [
            f"{profile}-session"
        ]
