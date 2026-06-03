"""Tests for the read-only ``hermes diagnostics index`` command."""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hermes_cli import diagnostics_index
from hermes_cli.main import build_top_level_parser


def _run_index(**kwargs) -> str:
    defaults = {
        "diagnostics_action": "index",
        "json": False,
        "area": None,
        "read_only": False,
        "approval_required": False,
    }
    defaults.update(kwargs)
    args = SimpleNamespace(**defaults)
    buf = io.StringIO()
    with redirect_stdout(buf):
        diagnostics_index.diagnostics_index_command(args)
    return buf.getvalue()


def test_parser_exposes_diagnostics_index_command():
    parser, _subparsers, _chat = build_top_level_parser()
    args = parser.parse_args(["diagnostics", "index", "--json"])

    assert args.command == "diagnostics"
    assert args.diagnostics_action == "index"
    assert args.json is True


def test_default_output_includes_baseline_entries():
    out = _run_index()

    assert "OR1 Start Gate" in out
    assert "python3 scripts/hermes_reliability_doctor.py" in out
    assert "hermes status" in out
    assert "hermes doctor" in out
    assert "hermes gateway status" in out
    assert "hermes kanban diagnostics" in out
    assert "hermes hooks doctor" in out


def test_json_output_is_valid_and_includes_expected_fields():
    payload = json.loads(_run_index(json=True))

    assert payload["count"] >= 7
    entry = next(item for item in payload["diagnostics"] if item["name"] == "OR1 Start Gate")
    assert entry["command"] == "OR1 Start Gate"
    assert entry["area"] == "repo"
    assert entry["read_only"] is True
    assert entry["approval_required"] is False
    assert entry["purpose"]
    assert entry["checks"]
    assert entry["notes"]


def test_filters_work():
    area_payload = json.loads(_run_index(json=True, area="gateway"))
    assert area_payload["count"] > 0
    assert all(item["area"] == "gateway" for item in area_payload["diagnostics"])
    assert {item["command"] for item in area_payload["diagnostics"]} >= {
        "hermes gateway status",
        "hermes gateway status --deep",
    }

    read_only_payload = json.loads(_run_index(json=True, read_only=True))
    assert read_only_payload["count"] > 0
    assert all(item["read_only"] is True for item in read_only_payload["diagnostics"])

    approval_payload = json.loads(_run_index(json=True, approval_required=True))
    assert approval_payload["count"] > 0
    assert all(item["approval_required"] is True for item in approval_payload["diagnostics"])


def test_index_does_not_execute_subprocesses():
    with patch("subprocess.run", side_effect=AssertionError("subprocess called")):
        out = _run_index()

    assert "Diagnostics Index" in out


def test_approval_required_markings_present():
    payload = json.loads(_run_index(json=True, approval_required=True))
    commands = {item["command"] for item in payload["diagnostics"]}

    assert "hermes doctor --fix" in commands
    assert "hermes gateway status --deep" in commands
    assert "hermes hooks doctor" in commands


def test_unknown_area_filter_rejected():
    with pytest.raises(SystemExit):
        diagnostics_index._filter_entries(area="not-real")
