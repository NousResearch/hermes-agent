"""Tests for non-destructive Hermes evolution reports."""

from __future__ import annotations


def test_build_report_summarizes_skill_evaluation_rows():
    from agent.evolution_report import build_report

    rows = [
        {
            "name": "alpha",
            "use_count": 5,
            "view_count": 2,
            "patch_count": 1,
            "success_count": 3,
            "failure_count": 0,
            "last_patched_at": "2026-06-20T00:00:00+00:00",
            "last_evaluated_at": "2026-06-21T00:00:00+00:00",
            "last_evaluation": "tests passed",
            "activity_count": 8,
            "provenance": "agent",
        },
        {
            "name": "beta",
            "use_count": 1,
            "view_count": 0,
            "patch_count": 2,
            "success_count": 0,
            "failure_count": 2,
            "last_patched_at": "2026-06-22T00:00:00+00:00",
            "last_evaluated_at": "2026-06-22T01:00:00+00:00",
            "last_evaluation": "user rejected output",
            "activity_count": 3,
            "provenance": "agent",
        },
        {
            "name": "gamma",
            "use_count": 2,
            "view_count": 0,
            "patch_count": 1,
            "success_count": 0,
            "failure_count": 0,
            "last_patched_at": "2026-06-19T00:00:00+00:00",
            "last_evaluated_at": None,
            "last_evaluation": None,
            "activity_count": 3,
            "provenance": "bundled",
        },
    ]

    report = build_report(rows)

    assert report["summary"] == {
        "total_skills": 3,
        "evaluated_skills": 2,
        "unevaluated_skills": 1,
        "total_successes": 3,
        "total_failures": 2,
    }
    assert [r["name"] for r in report["most_used"]] == ["alpha", "gamma", "beta"]
    assert [r["name"] for r in report["recently_patched"]] == ["beta", "alpha", "gamma"]
    assert [r["name"] for r in report["evaluation_gaps"]] == ["gamma"]
    assert [r["name"] for r in report["review_candidates"]] == ["beta", "gamma"]


def test_render_markdown_is_non_destructive_and_actionable():
    from agent.evolution_report import build_report, render_markdown

    report = build_report([
        {
            "name": "needs-measurement",
            "use_count": 4,
            "view_count": 0,
            "patch_count": 1,
            "success_count": 0,
            "failure_count": 0,
            "last_patched_at": "2026-06-19T00:00:00+00:00",
            "last_evaluated_at": None,
            "last_evaluation": None,
            "activity_count": 5,
            "provenance": "agent",
        }
    ])

    md = render_markdown(report)

    assert md.startswith("# Hermes Evolution Report")
    assert "report-only" in md
    assert "needs-measurement" in md
    assert "Evaluation gaps" in md
    assert "Recommended next loop" in md
    assert "skill_manage" not in md


def test_generate_report_uses_usage_report_and_writes_files(tmp_path, monkeypatch):
    from agent import evolution_report

    rows = [
        {
            "name": "alpha",
            "use_count": 1,
            "view_count": 0,
            "patch_count": 0,
            "success_count": 1,
            "failure_count": 0,
            "last_patched_at": None,
            "last_evaluated_at": "2026-06-21T00:00:00+00:00",
            "last_evaluation": "ok",
            "activity_count": 1,
            "provenance": "agent",
        }
    ]
    monkeypatch.setattr(evolution_report.skill_usage, "usage_report", lambda: rows)
    monkeypatch.setattr(evolution_report, "get_hermes_home", lambda: tmp_path)

    result = evolution_report.generate_report()

    assert result["report_dir"].startswith(str(tmp_path))
    assert (tmp_path / "logs" / "evolution").exists()
    assert result["markdown_path"].endswith("REPORT.md")
    assert result["json_path"].endswith("run.json")
    assert "# Hermes Evolution Report" in (tmp_path / "logs" / "evolution" / result["report_id"] / "REPORT.md").read_text(encoding="utf-8")

def test_build_report_includes_recent_run_trace_rows():
    from agent.evolution_report import build_report

    report = build_report(
        [],
        run_traces=[
            {
                "run_id": "20260622-abc123",
                "mission": "hermes-evolution",
                "final_status": "success",
                "summary": "Phase4 trace verified",
                "started_at": "2026-06-22T00:00:00+00:00",
                "completed_at": "2026-06-22T00:01:00+00:00",
                "success_signals": ["tests_passed"],
                "failure_signals": [],
            }
        ],
    )

    assert report["recent_runs"] == [
        {
            "run_id": "20260622-abc123",
            "mission": "hermes-evolution",
            "final_status": "success",
            "summary": "Phase4 trace verified",
            "started_at": "2026-06-22T00:00:00+00:00",
            "completed_at": "2026-06-22T00:01:00+00:00",
            "success_count": 1,
            "failure_count": 0,
            "evidence_count": 0,
            "gate_count": 0,
            "decision_count": 0,
            "context_pack_count": 0,
            "change_packet_count": 0,
        }
    ]


def test_render_markdown_includes_recent_evolution_runs():
    from agent.evolution_report import build_report, render_markdown

    report = build_report(
        [],
        run_traces=[
            {
                "run_id": "run-1",
                "mission": "hermes-evolution",
                "final_status": "success",
                "summary": "trace spine completed",
                "started_at": "2026-06-22T00:00:00+00:00",
                "completed_at": "2026-06-22T00:01:00+00:00",
                "success_signals": ["tests_passed"],
                "failure_signals": [],
            }
        ],
    )

    md = render_markdown(report)

    assert "## Recent evolution runs" in md
    assert "run-1" in md
    assert "trace spine completed" in md

def test_recent_runs_include_evidence_gate_and_decision_counts():
    from agent.evolution_report import build_report, render_markdown

    report = build_report(
        [],
        run_traces=[
            {
                "run_id": "genesis-run",
                "mission": "genesis",
                "final_status": "success",
                "summary": "Obsidian evidence checked",
                "started_at": "2026-06-22T00:00:00+00:00",
                "completed_at": "2026-06-22T00:01:00+00:00",
                "success_signals": ["obsidian_master_index_checked"],
                "failure_signals": [],
                "evidence_sources": [{"source_type": "obsidian_note", "path": "_master-index.md"}],
                "gate_results": [{"gate": "obsidian_master_index_checked", "status": "passed"}],
                "decisions": [{"decision": "HOLD", "status": "hold"}],
            }
        ],
    )

    run = report["recent_runs"][0]
    assert run["evidence_count"] == 1
    assert run["gate_count"] == 1
    assert run["decision_count"] == 1
    md = render_markdown(report)
    assert "Evidence" in md
    assert "| genesis-run | genesis | success | 1 | 0 | 1 | 1 | 1 | 0 | 0 | Obsidian evidence checked |" in md



def test_recent_runs_include_context_pack_counts():
    from agent.evolution_report import build_report, render_markdown

    report = build_report(
        [],
        run_traces=[
            {
                "run_id": "context-pack-run",
                "mission": "genesis",
                "final_status": "success",
                "summary": "Context pack attached",
                "started_at": "2026-06-22T00:00:00+00:00",
                "completed_at": "2026-06-22T00:01:00+00:00",
                "success_signals": [],
                "failure_signals": [],
                "context_packs": [{"path": "pack.md", "source_count": 3}],
            }
        ],
    )

    run = report["recent_runs"][0]
    assert run["context_pack_count"] == 1
    md = render_markdown(report)
    assert "Context Packs" in md
    assert "| context-pack-run | genesis | success | 0 | 0 | 0 | 0 | 0 | 1 | 0 | Context pack attached |" in md


def test_recent_runs_include_change_packet_counts():
    from agent.evolution_report import build_report, render_markdown

    report = build_report(
        [],
        run_traces=[
            {
                "run_id": "change-packet-run",
                "mission": "hermes-evolution",
                "final_status": "success",
                "summary": "change packet recorded",
                "started_at": "2026-06-22T00:00:00+00:00",
                "completed_at": "2026-06-22T00:01:00+00:00",
                "success_signals": [],
                "failure_signals": [],
                "change_packets": [{"path": "packet.md", "objective": "bound scope"}],
            }
        ],
    )

    run = report["recent_runs"][0]
    assert run["change_packet_count"] == 1
    md = render_markdown(report)
    assert "Change Packets" in md
    assert "| change-packet-run | hermes-evolution | success | 0 | 0 | 0 | 0 | 0 | 0 | 1 | change packet recorded |" in md
