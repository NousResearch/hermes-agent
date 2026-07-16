"""Tests for ``hermes doctor --preflight`` (task 1.5).

Tests the ``preflight()`` function's behavior:
- Returns ``(bool, dict)`` and never raises.
- Report dict has keys for each check.
- A broken artifact root is reported, not raised.

Per AGENTS.md: test behavior via the function call, do NOT read source code.
"""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli.subcommands.doctor_preflight import preflight, run_preflight_cli


class TestPreflightReturnType:
    """preflight() returns (bool, dict) and never raises."""

    def test_returns_tuple_of_bool_and_dict(self):
        result = preflight()
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], dict)

    def test_never_raises_on_broken_venv(self, tmp_path):
        """Even with a broken python executable, preflight must not raise."""
        with patch(
            "hermes_cli.subcommands.doctor_preflight.sys"
        ) as mock_sys:
            mock_sys.executable = str(tmp_path / "nonexistent_python")
            # Should still not raise — the subprocess failure is captured.
            ok, report = preflight()
            assert isinstance(ok, bool)
            assert isinstance(report, dict)

    def test_never_raises_on_general_failure(self):
        """Even if something deep inside fails, preflight returns a result."""
        ok, report = preflight()
        assert isinstance(ok, bool)
        assert isinstance(report, dict)


class TestPreflightReportStructure:
    """The report dict has keys for each check."""

    def test_report_has_all_check_keys(self):
        ok, report = preflight()
        assert "ok" in report
        assert "core_imports" in report
        assert "config_parse" in report
        assert "config_version" in report
        assert "artifact_roots" in report

    def test_report_ok_matches_returned_bool(self):
        ok, report = preflight()
        assert report["ok"] == ok

    def test_core_imports_has_ok_key(self):
        _, report = preflight()
        assert "ok" in report["core_imports"]

    def test_config_parse_has_ok_key(self):
        _, report = preflight()
        assert "ok" in report["config_parse"]

    def test_config_version_has_ok_key(self):
        _, report = preflight()
        assert "ok" in report["config_version"]

    def test_artifact_roots_has_ok_key(self):
        _, report = preflight()
        assert "ok" in report["artifact_roots"]

    def test_artifact_roots_has_roots_dict(self):
        _, report = preflight()
        assert "roots" in report["artifact_roots"]
        assert isinstance(report["artifact_roots"]["roots"], dict)


class TestPreflightBrokenArtifactRoot:
    """A broken artifact root is reported, not raised."""

    def test_broken_artifact_root_reported_not_raised(self, tmp_path):
        """When get_artifact_root() returns a bogus path, the accessors
        should report failure in the report dict, not raise."""
        nonexistent = tmp_path / "does_not_exist"

        def fake_get_artifact_root():
            return nonexistent

        with patch(
            "hermes_constants.get_artifact_root", fake_get_artifact_root
        ), patch(
            "hermes_constants.bundled_skills_dir",
            lambda: nonexistent / "skills",
        ), patch(
            "hermes_constants.web_dist_dir",
            lambda: nonexistent / "web_dist",
        ), patch(
            "hermes_constants.tui_dist_dir",
            lambda: nonexistent / "tui_dist",
        ):
            ok, report = preflight()

        # Should not have raised; should report failure.
        assert ok is False
        assert report["artifact_roots"]["ok"] is False
        roots = report["artifact_roots"]["roots"]
        # At least one accessor should have reported a failure.
        assert any(
            not r.get("skipped") and r.get("ok") is False for r in roots.values()
        )

    def test_empty_artifact_dir_reported_not_raised(self, tmp_path):
        """When an accessor points at an existing but empty dir, it's reported."""

        def fake_get_artifact_root():
            return tmp_path

        empty_dir = tmp_path / "skills"
        empty_dir.mkdir()

        with patch(
            "hermes_constants.get_artifact_root", fake_get_artifact_root
        ), patch("hermes_constants.bundled_skills_dir", lambda: empty_dir), patch(
            "hermes_constants.web_dist_dir",
            lambda: tmp_path / "web_dist",
        ), patch(
            "hermes_constants.tui_dist_dir",
            lambda: tmp_path / "tui_dist",
        ):
            ok, report = preflight()

        assert ok is False
        roots = report["artifact_roots"]["roots"]
        skills_entry = roots.get("bundled_skills_dir", {})
        assert skills_entry.get("ok") is False
        assert "empty" in skills_entry.get("error", "").lower()


class TestRunPreflightCli:
    """The CLI entrypoint prints JSON and exits 0/1."""

    def test_returns_int_exit_code(self):
        code = run_preflight_cli()
        assert code in (0, 1)

    def test_exit_code_matches_preflight_result(self, capsys):
        ok, _ = preflight()
        code = run_preflight_cli()
        assert code == (0 if ok else 1)

    def test_prints_valid_json(self, capsys):
        run_preflight_cli()
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "ok" in data
        assert "core_imports" in data
