from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from hermes_cli.status_lines import (
    StatusLineError,
    canonical_live_status,
    lineage_counts,
    lineage_headline,
    render_line,
    render_lineage_aggregate,
    render_status_pair,
    render_status_with_lineage_aggregate,
)


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "render_status_lines.py"


def test_render_line_uses_canonical_label_and_uppercase_status() -> None:
    assert render_line("self", "working", "editing status semantics") == (
        "Self status: WORKING — editing status semantics"
    )
    assert render_line("lineage", "dormant", "no descendants") == (
        "Lineage status: DORMANT — no descendants"
    )


def test_render_status_pair_outputs_two_canonical_lines() -> None:
    assert render_status_pair(
        "waiting",
        "CI is still running",
        "blocked",
        "one descendant needs Yunuen review",
    ) == (
        "Self status: WAITING — CI is still running\n"
        "Lineage status: BLOCKED — one descendant needs Yunuen review"
    )


@pytest.mark.parametrize(
    ("internal", "canonical"),
    [
        ("running", "working"),
        ("ready", "working"),
        ("queue", "working"),
        ("in_progress", "working"),
        ("review", "working"),
        ("todo", "waiting"),
        ("scheduled", "waiting"),
        ("blocked", "blocked"),
        ("triage", "dormant"),
        ("done", "dormant"),
        ("archived", "dormant"),
    ],
)
def test_canonical_live_status_aliases(internal: str, canonical: str) -> None:
    assert canonical_live_status(internal) == (canonical, None)


def test_lineage_aggregate_prefers_working_and_keeps_blocked_counts() -> None:
    output = render_status_with_lineage_aggregate(
        "working",
        "updating status contract",
        ["running", "blocked", "todo", "done"],
    )
    assert output == (
        "Self status: WORKING — updating status contract\n"
        "Lineage status: WORKING — working: 1; waiting: 1; blocked: 1; dormant: 1"
    )


def test_lineage_headline_rules() -> None:
    counts, unknown = lineage_counts(["blocked", "todo"])
    assert not unknown
    assert lineage_headline(counts) == "blocked"

    counts, unknown = lineage_counts(["scheduled"])
    assert not unknown
    assert lineage_headline(counts) == "waiting"

    counts, unknown = lineage_counts([])
    assert not unknown
    assert lineage_headline(counts) == "dormant"


def test_unknown_lineage_status_fails_closed_to_blocked() -> None:
    assert render_lineage_aggregate(["mystery"]) == (
        "Lineage status: BLOCKED — working: 0; waiting: 0; blocked: 1; dormant: 0; "
        "unknown→blocked: mystery"
    )


def test_invalid_scalar_status_rejected() -> None:
    with pytest.raises(StatusLineError):
        render_line("self", "ready", "ready is a storage alias, not scalar prompt status")


def test_script_renders_counts() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--self",
            "working",
            "running tests",
            "--lineage-count",
            "running",
            "2",
            "--lineage-count",
            "blocked",
            "1",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    assert proc.stdout.strip() == (
        "Self status: WORKING — running tests\n"
        "Lineage status: WORKING — working: 2; waiting: 0; blocked: 1; dormant: 0"
    )


def test_script_reads_json_statuses(tmp_path: Path) -> None:
    data = tmp_path / "lineage.json"
    data.write_text(json.dumps({"descendants": [{"kanban_status": "scheduled"}, {"status": "done"}]}))
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--self",
            "dormant",
            "no direct task",
            "--lineage-json",
            str(data),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    assert proc.stdout.strip() == (
        "Self status: DORMANT — no direct task\n"
        "Lineage status: WAITING — working: 0; waiting: 1; blocked: 0; dormant: 1"
    )
