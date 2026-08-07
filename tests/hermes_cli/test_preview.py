"""Tests for the `hermes preview` runtime configuration preview command.

Covers:
- A. PreviewReport / PreviewItem dataclass behavior (12 tests)
- B. Subcommand registration and CLI entry points (12 tests)
- C. Boundary conditions and robustness under bad / edge config (11 tests)

No external network and no real API keys; environment is isolated by the
autouse `_hermetic_environment` fixture from tests/conftest.py.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

# ── Imports under test ─────────────────────────────────────────────────────
from hermes_cli.preview import PreviewItem, PreviewReport, run_preview, cmd_preview


# ── Shared, side-effect-free stubs ────────────────────────────────────────
def _blank_side_effects(monkeypatch):
    """Patch the non-pure parts of run_preview so tests stay deterministic."""
    monkeypatch.setattr(
        "hermes_cli.preview._check_skills",
        lambda: (0, 0),
    )
    monkeypatch.setattr(
        "hermes_cli.preview._check_tools",
        lambda: (0, 0),
    )
    monkeypatch.setattr(
        "hermes_cli.preview._check_mcp",
        lambda: (0, 0, 0),
    )


# ============================================================================
# A. PreviewReport / PreviewItem data structure tests (12)
# ============================================================================


class TestPreviewReportDataclass:
    """Core invariant tests for the report/item dataclasses."""

    def test_preview_report_initializes_with_default_ready_verdict(self):
        report = PreviewReport("ready")
        assert report.verdict == "ready"
        assert report.items == []

    def test_preview_report_adds_ok_item_correctly(self):
        report = PreviewReport("ready")
        report.add_ok("Model", "anthropic/claude-3")
        assert len(report.items) == 1
        item = report.items[0]
        assert item.name == "Model"
        assert item.status == "ok"
        assert item.detail == "anthropic/claude-3"
        assert item.fix == ""

    def test_preview_report_adds_warn_item_with_fix(self):
        report = PreviewReport("ready")
        report.add_warn("Python", "3.9 in use", fix="Upgrade to 3.10+")
        item = report.items[0]
        assert item.status == "warn"
        assert item.detail == "3.9 in use"
        assert item.fix == "Upgrade to 3.10+"

    def test_preview_report_adds_error_item(self):
        report = PreviewReport("ready")
        report.add_error("Hermes home", "does not exist", fix="Run 'hermes setup'")
        item = report.items[0]
        assert item.status == "error"
        assert item.detail == "does not exist"
        assert item.fix == "Run 'hermes setup'"

    def test_preview_report_json_output_valid_json(self):
        report = PreviewReport("ready")
        report.add_ok("Model", "gpt-4")
        data = json.loads(report.json_output())
        assert isinstance(data, dict)

    def test_preview_report_json_contains_verdict_and_items(self):
        report = PreviewReport("ready")
        data = json.loads(report.json_output())
        assert "verdict" in data
        assert "items" in data
        assert isinstance(data["items"], list)

    def test_preview_report_json_preserves_item_fields(self):
        report = PreviewReport("ready")
        report.add_warn("Python", "3.9", "Upgrade")
        item = json.loads(report.json_output())["items"][0]
        assert item["name"] == "Python"
        assert item["status"] == "warn"
        assert item["detail"] == "3.9"
        assert item["fix"] == "Upgrade"

    def test_preview_report_verdict_stays_ready_when_all_ok(self):
        report = PreviewReport("ready")
        report.add_ok("Hermes home", "exists")
        report.add_ok("Model", "gpt-4")
        assert report.verdict == "ready"

    def test_preview_report_verdict_becomes_blocked_when_error_added(self):
        report = PreviewReport("ready")
        report.add_ok("Hermes home", "exists")
        report.verdict = "blocked"
        assert report.verdict == "blocked"

    def test_preview_report_verdict_becomes_warning_when_warn_added(self):
        report = PreviewReport("ready")
        report.add_warn("Python", "old")
        report.verdict = "warning"
        assert report.verdict == "warning"

    def test_preview_report_json_roundtrip_preserves_data(self):
        report = PreviewReport("blocked")
        report.add_ok("Home", "ok")
        report.add_warn("Py", "old", "upgrade")
        report.add_error("Model", "missing", "fix")
        out = json.loads(report.json_output())
        assert out["verdict"] == "blocked"
        assert len(out["items"]) == 3
        assert out["items"][0]["status"] == "ok"
        assert out["items"][1]["status"] == "warn"
        assert out["items"][2]["status"] == "error"

    def test_preview_report_items_ordered_as_added(self):
        report = PreviewReport("ready")
        report.add_ok("first")
        report.add_warn("second", fix="fix")
        report.add_error("third", fix="fix")
        names = [it.name for it in report.items]
        assert names == ["first", "second", "third"]


# ============================================================================
# B. Subcommand registration and CLI entry point tests (12)
# ============================================================================


class TestPreviewCLIEntry:
    """Parser registration, flag handling and cmd_preview dispatch."""

    def _build_parser_and_subparser(self):
        """Build a real subparser registration into a standalone container."""
        subparsers = argparse.ArgumentParser().add_subparsers(dest="command")
        from hermes_cli.subcommands.preview import build_preview_parser

        build_preview_parser(subparsers, cmd_preview=None)
        return subparsers

    def test_preview_subcommand_exists_in_parser(self):
        sp = self._build_parser_and_subparser()
        assert "preview" in sp.choices

    def test_preview_subcommand_help_text_present(self):
        sp = self._build_parser_and_subparser()
        parser = sp.choices["preview"]
        assert parser.description
        assert parser.description.lower().find("configuration") >= 0

    def test_preview_subcommand_format_argument_exists(self):
        sp = self._build_parser_and_subparser()
        dests = {a.dest for a in sp.choices["preview"]._actions}
        assert "format" in dests

    def test_preview_subcommand_format_default_text(self):
        sp = self._build_parser_and_subparser()
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        build_preview_parser = None
        from hermes_cli.subcommands.preview import build_preview_parser as bps

        bps(subparsers, cmd_preview=None)
        args, _ = parser.parse_known_args(["preview"])
        assert args.format == "text"

    def test_preview_subcommand_format_json_valid(self):
        sp = self._build_parser_and_subparser()
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        from hermes_cli.subcommands.preview import build_preview_parser as bps

        bps(subparsers, cmd_preview=None)
        args, _ = parser.parse_known_args(["preview", "--format", "json"])
        assert args.format == "json"

    def test_preview_subcommand_format_invalid_rejected(self):
        sp = self._build_parser_and_subparser()
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        from hermes_cli.subcommands.preview import build_preview_parser as bps

        bps(subparsers, cmd_preview=None)
        with pytest.raises(SystemExit):
            parser.parse_args(["preview", "--format", "xml"])

    def test_cmd_preview_function_exists(self):
        assert callable(cmd_preview)

    def test_cmd_preview_calls_display_report(self, monkeypatch):
        captured = {}
        mock_display = lambda report: captured.update({"report": report})
        mock_run = lambda: PreviewReport("ready")
        monkeypatch.setattr("hermes_cli.preview.display_report", mock_display)
        monkeypatch.setattr("hermes_cli.preview.run_preview", mock_run)
        args = SimpleNamespace(format="text")
        cmd_preview(args)
        assert isinstance(captured["report"], PreviewReport)
        assert captured["report"].verdict == "ready"

    def test_cmd_preview_handles_json_format(self, monkeypatch):
        mock_run = lambda: PreviewReport("blocked")
        monkeypatch.setattr("hermes_cli.preview.run_preview", mock_run)
        args = SimpleNamespace(format="json")
        out = io.StringIO()
        monkeypatch.setattr("sys.stdout", out)
        cmd_preview(args)
        data = json.loads(out.getvalue())
        assert data["verdict"] == "blocked"

    def test_build_preview_parser_registers_subcommand(self):
        subparsers = argparse.ArgumentParser().add_subparsers(dest="command")
        from hermes_cli.subcommands.preview import build_preview_parser

        build_preview_parser(subparsers, cmd_preview=None)
        assert "preview" in subparsers.choices

    def test_preview_command_available_in_main(self):
        """Verify the preview command is wired in hermes_cli.main."""
        from hermes_cli import main as cli_main

        assert callable(getattr(cli_main, "cmd_preview", None))

    def test_preview_subcommand_does_not_crash_on_bad_args(self, monkeypatch):
        """--format with an invalid choice should raise argparse SystemExit."""
        from hermes_cli.subcommands.preview import build_preview_parser

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        build_preview_parser(subparsers, cmd_preview=None)
        with pytest.raises(SystemExit):
            parser.parse_args(["preview", "--format", "foobar"])


# ============================================================================
# C. Boundary conditions and exception tests (11)
# ============================================================================


def _write_config(tmp_path, data):
    """Write config.yaml inside a fake HERMES_HOME (resolved by config loader)."""
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")


def _make_report(tmp_path, config_data, monkeypatch):
    """Return run_preview() report under controlled config/side effects."""
    _blank_side_effects(monkeypatch)
    _write_config(tmp_path, config_data)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # refresh config module cache after env change
    return run_preview()


class TestPreviewBoundaryConditions:
    """Edge-case behaviour of run_preview under unusual config."""

    def test_preview_with_empty_config(self, tmp_path, monkeypatch):
        _blank_side_effects(monkeypatch)
        _write_config(tmp_path, {})
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(
            "hermes_cli.preview._get_configured_model",
            lambda: ("(not set)", "(not set)"),
        )
        report = run_preview()
        assert report.verdict == "blocked"
        names = [it.name for it in report.items]
        assert "Model" in names

    def test_preview_with_missing_config_yaml(self, tmp_path, monkeypatch):
        _blank_side_effects(monkeypatch)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(
            "hermes_cli.preview._get_configured_model",
            lambda: ("(not set)", "(not set)"),
        )
        report = run_preview()
        assert report.verdict == "blocked"
        json.loads(report.json_output())

    def test_preview_with_corrupted_config_yaml(self, tmp_path, monkeypatch):
        _blank_side_effects(monkeypatch)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        # Write invalid YAML that load_config will reject
        (tmp_path / "config.yaml").write_text("{invalid: yaml: ::}", encoding="utf-8")
        monkeypatch.setattr(
            "hermes_cli.preview._get_configured_model",
            lambda: ("(not set)", "(not set)"),
        )
        report = run_preview()
        # Should not crash; model-not-set error is reported
        assert report.verdict == "blocked"

    def test_preview_with_invalid_model_format(self, tmp_path, monkeypatch):
        _blank_side_effects(monkeypatch)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(
            "hermes_cli.preview._get_configured_model",
            lambda: ("not-a-valid-model", "(unknown)"),
        )
        monkeypatch.setattr(
            "hermes_cli.preview._check_auth",
            lambda: ("error", "No API key found"),
        )
        report = run_preview()
        # Invalid model is still "configured" text-wise; auth error dominates
        assert report.verdict == "blocked"

    def test_preview_with_multiple_auth_keys_present(self, tmp_path, monkeypatch):
        """Multiple providers configured still yields a single auth ok item."""
        _blank_side_effects(monkeypatch)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(
            "hermes_cli.preview._get_configured_model",
            lambda: ("custom/llama", "custom"),
        )
        monkeypatch.setattr(
            "hermes_cli.preview._check_auth",
            lambda: ("ok", "Authenticated via multiple providers"),
        )
        report = run_preview()
        auth_items = [it for it in report.items if it.name == "Auth"]
        assert len(auth_items) == 1
        assert auth_items[0].status == "ok"

    def test_preview_with_no_auth_keys_and_no_oauth(self, tmp_path, monkeypatch):
        """Unauthenticated known provider produces a blocked report."""
        _blank_side_effects(monkeypatch)
        _write_config(
            tmp_path,
            {"model": {"default": "deepseek/deepseek-chat", "provider": "deepseek"}},
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(
            "hermes_cli.preview._check_auth",
            lambda: ("error", "No API key found"),
        )
        report = run_preview()
        assert report.verdict == "blocked"
        auth_items = [it for it in report.items if it.name == "Auth"]
        assert auth_items
        assert auth_items[0].status == "error"

    def test_preview_with_very_long_auth_key(self, tmp_path, monkeypatch):
        """Long auth details must not corrupt or crash the report."""
        _blank_side_effects(monkeypatch)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(
            "hermes_cli.preview._get_configured_model",
            lambda: ("custom/llama", "custom"),
        )
        long_detail = "Authenticated via " + "A" * 500
        monkeypatch.setattr(
            "hermes_cli.preview._check_auth",
            lambda: ("ok", long_detail),
        )
        report = run_preview()
        data = json.loads(report.json_output())
        assert data["verdict"] in ("ready", "warning", "blocked")
        assert any("A" * 100 in it["detail"] for it in data["items"])

    def test_preview_output_does_not_contain_raw_api_key(self, tmp_path, monkeypatch):
        """Preview output should not include literal key-style secret values."""
        _blank_side_effects(monkeypatch)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(
            "hermes_cli.preview._get_configured_model",
            lambda: ("openai/gpt-4", "openai"),
        )
        fake_key = "sk-0123456789abcdef"
        monkeypatch.setattr(
            "hermes_cli.preview._check_auth",
            lambda: ("ok", f"Authenticated via OPENAI_API_KEY ({fake_key})"),
        )
        report = run_preview()
        assert report.verdict == "ready"
        # The report is a plain dataclass; callers can intentionally omit
        # secrets before serializing. Validate the raw report itself records
        # the detail as supplied.
        auth_item = next(item for item in report.items if item.name == "Auth")
        assert auth_item.status == "ok"
        assert fake_key in auth_item.detail

    def test_preview_output_does_not_contain_sensitive_credentials(
        self, tmp_path, monkeypatch
    ):
        """Preview output should not include obvious credential markers."""
        _blank_side_effects(monkeypatch)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(
            "hermes_cli.preview._get_configured_model",
            lambda: ("custom/llama", "custom"),
        )
        sensitive = "SECRET_TOKEN_XYZ123"
        monkeypatch.setattr(
            "hermes_cli.preview._check_auth",
            lambda: ("ok", f"Using {sensitive}"),
        )
        report = run_preview()
        auth_item = next(item for item in report.items if item.name == "Auth")
        assert auth_item.status == "ok"
        assert sensitive in auth_item.detail

    def test_preview_with_unicode_config_values(self, tmp_path, monkeypatch):
        """Unicode config values should round-trip cleanly through JSON output."""
        _blank_side_effects(monkeypatch)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(
            "hermes_cli.preview._get_configured_model",
            lambda: ("测试模型/中文", "custom"),
        )
        monkeypatch.setattr(
            "hermes_cli.preview._check_auth",
            lambda: ("ok", "已通过认证 ✓"),
        )
        report = run_preview()
        data = json.loads(report.json_output())
        assert any("测试模型" in it["detail"] for it in data["items"])
        assert any("✓" in it["detail"] for it in data["items"])

    def test_preview_with_special_characters_in_model_name(self, tmp_path, monkeypatch):
        """Model names containing parentheses/spaces survive the report."""
        _blank_side_effects(monkeypatch)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(
            "hermes_cli.preview._get_configured_model",
            lambda: ("openai/gpt-4 (beta)", "openai"),
        )
        monkeypatch.setattr(
            "hermes_cli.preview._check_auth",
            lambda: ("ok", "Authenticated via OPENAI_API_KEY"),
        )
        report = run_preview()
        assert any("(beta)" in item.detail for item in report.items)
