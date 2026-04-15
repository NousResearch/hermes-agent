import argparse
import sys
from unittest.mock import MagicMock

import pytest

import hermes_cli.main as main_mod


class _FakeSessionDB:
    def list_sessions_rich(self, source=None, limit=20, **kwargs):
        return [{"id": "sess-1", "source": source or "cli"}]

    def get_messages(self, session_id):
        assert session_id == "sess-1"
        return [
            {"role": "tool", "tool_name": "web_extract", "content": "Error: Invalid URL '/v2/scrape'"},
            {"role": "user", "content": "Use requests instead of web_extract."},
        ]

    def close(self):
        return None


class _FakeEvolutionModule:
    @staticmethod
    def analyze_sessions(sessions, trajectory_entries=None, min_count=2):
        assert sessions[0]["id"] == "sess-1"
        assert min_count == 2
        return {
            "summary": {"sessions_analyzed": len(sessions), "trajectory_files": 0, "findings": 1},
            "findings": [],
            "recommendations": ["do something"],
            "prompt_deltas": ["tighten fallback"],
        }

    @staticmethod
    def render_markdown_report(report):
        return "# Hermes Self-Evolution Report\n\n- ok\n"

    @staticmethod
    def load_trajectory_entries(path):
        raise AssertionError("trajectory loading not expected in this test")


def test_evolve_report_help_includes_key_flags(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["hermes", "evolve", "report", "--help"])

    with pytest.raises(SystemExit) as excinfo:
        main_mod.main()

    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "hermes evolve report" in out
    assert "--source" in out
    assert "--trajectory" in out
    assert "--min-count" in out
    assert "--output" in out


def test_cmd_evolve_prints_report_to_stdout(monkeypatch, capsys):
    fake_hermes_state = MagicMock(SessionDB=lambda: _FakeSessionDB())
    monkeypatch.setitem(sys.modules, "hermes_state", fake_hermes_state)
    monkeypatch.setitem(sys.modules, "agent.evolution", _FakeEvolutionModule)

    args = argparse.Namespace(
        evolve_action="report",
        source="cli",
        limit=5,
        trajectory=[],
        min_count=2,
        output="-",
    )

    main_mod.cmd_evolve(args)

    out = capsys.readouterr().out
    assert "# Hermes Self-Evolution Report" in out


def test_cmd_evolve_writes_report_to_file(monkeypatch, tmp_path):
    fake_hermes_state = MagicMock(SessionDB=lambda: _FakeSessionDB())
    monkeypatch.setitem(sys.modules, "hermes_state", fake_hermes_state)
    monkeypatch.setitem(sys.modules, "agent.evolution", _FakeEvolutionModule)

    output_path = tmp_path / "report.md"
    args = argparse.Namespace(
        evolve_action="report",
        source=None,
        limit=5,
        trajectory=[],
        min_count=2,
        output=str(output_path),
    )

    main_mod.cmd_evolve(args)

    assert output_path.read_text(encoding="utf-8").startswith("# Hermes Self-Evolution Report")
