from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pytest


@pytest.fixture
def report_env(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    logdir = hermes_home / "logs" / "memory-maintenance"
    logdir.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    def write_bundle(day: str, *, status: str, actions: int, best: str, doctor_exit: int = 0, maintain_exit: int = 0, proposal_counts: dict[str, int] | None = None) -> None:
        summary_path = logdir / f"summary-{day}.txt"
        doctor_path = logdir / f"doctor-{day}.json"
        maintain_path = logdir / f"maintain-{day}.json"
        proposal_counts = proposal_counts or {"NOOP": 2, "REVALIDATE": 1}

        doctor_path.write_text(
            json.dumps(
                {
                    "mode": "dry-run",
                    "overall_status": status,
                    "memory": {"status": "ok"},
                    "user": {"status": "ok"},
                    "fact_store": {"status": status},
                    "session_search": {"status": "ok"},
                    "skills": {"status": "ok"},
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        maintain_path.write_text(
            json.dumps(
                {
                    "mode": "dry-run",
                    "overall_status": status,
                    "summary": {"proposal_counts": proposal_counts},
                    "actions": [
                        {"scope": "fact_store", "target_id": "fact:7", "proposed_action": "REVALIDATE", "severity": "high"},
                        {"scope": "session_search", "target_id": "session_search", "proposed_action": "KEEP_IN_SESSION_SEARCH_ONLY", "severity": "low"},
                    ],
                    "warnings": ["No changes applied"],
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        summary_path.write_text(
            "\n".join(
                [
                    f"timestamp: 2026-04-26T04:30:00-0300",
                    f"doctor_exit_code: {doctor_exit}",
                    f"maintain_exit_code: {maintain_exit}",
                    f"doctor_report: {doctor_path}",
                    f"maintain_report: {maintain_path}",
                    f"doctor_overall_status: {status}",
                    f"maintain_overall_status: {status}",
                    f"maintain_actions_proposed: {actions}",
                    f"maintain_proposal_counts: {proposal_counts}",
                    "No changes applied",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    write_bundle("2026-04-25", status="warning", actions=3, best="Review fact_store proposal for fact:7 (REVALIDATE)", proposal_counts={"NOOP": 1, "REVALIDATE": 2})
    write_bundle("2026-04-26", status="critical", actions=17, best="Review fact_store proposal for fact:42 (REDACT_ON_OUTPUT_ONLY)", proposal_counts={"NOOP": 3, "REDACT_ON_OUTPUT_ONLY": 11, "REVALIDATE": 1, "KEEP_IN_SESSION_SEARCH_ONLY": 1})

    old_summary = logdir / "summary-2026-04-25.txt"
    new_summary = logdir / "summary-2026-04-26.txt"
    old_mtime = 1_700_000_000
    new_mtime = 1_700_100_000
    for path, mtime in ((old_summary, old_mtime), (new_summary, new_mtime)):
        os.utime(path, (mtime, mtime))

    return hermes_home


def test_memory_report_latest_reads_latest_summary_and_timer(report_env, monkeypatch):
    from hermes_cli import memory_report

    monkeypatch.setattr(
        memory_report,
        "_collect_timer_status",
        lambda: {
            "status": "active",
            "available": True,
            "active_state": "active",
            "unit_file_state": "enabled",
            "sub_state": "waiting",
            "next_run": "2026-04-26 04:30:00 -03",
            "last_trigger": "2026-04-26 00:16:24 -03",
        },
    )

    report = memory_report.build_memory_report_latest()

    assert report["mode"] == "latest"
    assert report["report_date"] == "2026-04-26"
    assert report["overall_status"] == "critical"
    assert report["doctor_exit_code"] == 0
    assert report["maintain_exit_code"] == 0
    assert report["maintain_actions_proposed"] == 17
    assert report["best_next_action"] == "Review fact_store proposal for fact:7 (REVALIDATE)"
    assert report["paths"]["summary"].endswith("summary-2026-04-26.txt")
    assert report["timer"]["status"] == "active"
    assert report["summary"]["proposal_counts"]["REDACT_ON_OUTPUT_ONLY"] == 11


def test_memory_report_latest_human_output_is_read_only(report_env, monkeypatch, capsys):
    from hermes_cli import memory_report

    monkeypatch.setattr(memory_report, "_collect_timer_status", lambda: {"status": "active", "unit_file_state": "enabled", "sub_state": "waiting", "next_run": "2026-04-26 04:30:00 -03"})
    report = memory_report.cmd_memory_report_latest(argparse.Namespace(json=False))
    out = capsys.readouterr().out

    assert report["overall_status"] == "critical"
    assert "Hermes memory report latest" in out
    assert "doctor_exit_code: 0" in out
    assert "maintain_exit_code: 0" in out
    assert "maintain_actions_proposed: 17" in out
    assert "Review fact_store proposal for fact:7 (REVALIDATE)" in out
    assert report["paths"]["summary"].endswith("summary-2026-04-26.txt")
    assert "No changes applied." in out
    assert "sk-" not in out


def test_memory_report_latest_json_shape(report_env, monkeypatch, capsys):
    from hermes_cli import memory_report

    monkeypatch.setattr(memory_report, "_collect_timer_status", lambda: {"status": "active", "unit_file_state": "enabled", "sub_state": "waiting"})
    memory_report.cmd_memory_report_latest(argparse.Namespace(json=True))
    payload = json.loads(capsys.readouterr().out)

    assert payload["mode"] == "latest"
    assert payload["paths"]["summary"].endswith("summary-2026-04-26.txt")
    assert payload["timer"]["status"] == "active"
    assert payload["summary"]["maintain_actions_proposed"] == 17
    assert payload["warnings"] == ["No changes applied"]


def test_memory_report_latest_main_dispatch(report_env, monkeypatch, capsys):
    from hermes_cli import memory_report
    import hermes_cli.main as main_mod

    monkeypatch.setattr(memory_report, "_collect_timer_status", lambda: {"status": "active", "unit_file_state": "enabled", "sub_state": "waiting"})
    monkeypatch.setattr(sys, "argv", ["hermes", "memory", "report", "latest", "--json"])
    main_mod.main()
    payload = json.loads(capsys.readouterr().out)

    assert payload["mode"] == "latest"
    assert payload["overall_status"] == "critical"
    assert payload["best_next_action"].startswith("Review fact_store proposal")
