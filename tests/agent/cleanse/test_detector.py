"""Tests for agent.cleanse.detector — issue detection."""

import json
from pathlib import Path

import pytest

from agent.cleanse.detector import detect_issues


def test_detect_issues_empty_project(tmp_path):
    """detect_issues() returns empty result for empty project."""
    result = detect_issues(tmp_path)
    assert result.diagnostics == []
    assert result.recipe is None


def test_detect_issues_skips_missing_tools(tmp_path):
    """detect_issues() skips tools not on PATH without failing."""
    # Create a minimal TypeScript project
    (tmp_path / "tsconfig.json").write_text('{"compilerOptions": {}}')
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.ts").write_text("const x: string = 42;")

    result = detect_issues(tmp_path)
    # Should either collect diagnostics or skip tsc gracefully
    assert isinstance(result.diagnostics, list)
    assert isinstance(result.skipped_tools, list)


def test_detect_issues_python_project(tmp_path):
    """detect_issues() detects Python project structure."""
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'")
    (tmp_path / "main.py").write_text("import sys\n")

    result = detect_issues(tmp_path)
    assert result.recipe is not None
    assert result.recipe.kind == "python"
