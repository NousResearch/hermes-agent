"""Issue detection via VCS + manifests + linters.

Reuses/extends agent.verify.recipes + coding_context.detect_project_facts.
Runs actionlint when .github/workflows exist; skips missing binaries.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent.cleanse.parser import Diagnostic, normalize_diagnostics
from agent.coding_context import detect_project_facts
from agent.verify.recipes import Recipe, detect_recipe

logger = logging.getLogger("hermes.cleanse.detector")


@dataclass
class DetectionResult:
    """Result of running all project checkers."""

    diagnostics: list[Diagnostic] = field(default_factory=list)
    skipped_tools: list[str] = field(default_factory=list)
    recipe: Recipe | None = None
    project_facts: Any | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "diagnostics": [d.to_dict() for d in self.diagnostics],
            "skipped_tools": self.skipped_tools,
            "recipe": self.recipe.to_dict() if self.recipe else None,
        }


def _run_tool(
    tool: str,
    args: list[str],
    root: Path,
    timeout: float = 120.0,
) -> tuple[str, int | None]:
    """Run a checker tool and return (output, exit_code)."""
    if not shutil.which(tool):
        raise FileNotFoundError(f"{tool} not found on PATH")

    try:
        proc = subprocess.run(
            [tool, *args],
            cwd=str(root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            text=True,
            errors="replace",
        )
        return proc.stdout or "", proc.returncode
    except subprocess.TimeoutExpired as exc:
        raw = exc.output
        if isinstance(raw, bytes):
            output = raw.decode("utf-8", errors="replace")
        else:
            output = raw or ""
        return output, None


def _check_typescript(root: Path) -> tuple[list[Diagnostic], list[str]]:
    """Run tsc if tsconfig.json exists."""
    if not (root / "tsconfig.json").exists():
        return [], []

    try:
        output, _ = _run_tool("tsc", ["--noEmit"], root)
        return normalize_diagnostics("tsc", output, root), []
    except FileNotFoundError:
        return [], ["tsc"]


def _check_eslint(root: Path) -> tuple[list[Diagnostic], list[str]]:
    """Run eslint if .eslintrc* or eslint.config.* exists."""
    eslint_configs = [
        ".eslintrc.js",
        ".eslintrc.cjs",
        ".eslintrc.json",
        ".eslintrc.yml",
        ".eslintrc.yaml",
        "eslint.config.js",
        "eslint.config.mjs",
        "eslint.config.cjs",
    ]
    if not any((root / cfg).exists() for cfg in eslint_configs):
        return [], []

    try:
        output, _ = _run_tool("eslint", [".", "--format", "json"], root)
        return normalize_diagnostics("eslint", output, root), []
    except FileNotFoundError:
        return [], ["eslint"]


def _check_actionlint(root: Path) -> tuple[list[Diagnostic], list[str]]:
    """Run actionlint if .github/workflows exists."""
    workflows_dir = root / ".github" / "workflows"
    if not workflows_dir.exists() or not workflows_dir.is_dir():
        return [], []

    try:
        output, _ = _run_tool("actionlint", [], root)
        return normalize_diagnostics("actionlint", output, root), []
    except FileNotFoundError:
        logger.info("actionlint not on PATH; skipping workflow checks")
        return [], []


def _check_ruff(root: Path) -> tuple[list[Diagnostic], list[str]]:
    """Run ruff if pyproject.toml or ruff.toml exists."""
    if not ((root / "pyproject.toml").exists() or (root / "ruff.toml").exists()):
        return [], []

    try:
        output, _ = _run_tool("ruff", ["check", ".", "--output-format", "json"], root)
        return normalize_diagnostics("ruff", output, root), []
    except FileNotFoundError:
        return [], ["ruff"]


def _check_prettier(root: Path) -> tuple[list[Diagnostic], list[str]]:
    """Run prettier --check if .prettierrc* exists."""
    prettier_configs = [
        ".prettierrc",
        ".prettierrc.json",
        ".prettierrc.yml",
        ".prettierrc.yaml",
        ".prettierrc.js",
        ".prettierrc.cjs",
        "prettier.config.js",
        "prettier.config.cjs",
    ]
    if not any((root / cfg).exists() for cfg in prettier_configs):
        return [], []

    try:
        output, exit_code = _run_tool("prettier", ["--check", "."], root)
        # Prettier doesn't have structured output; just collect files that need formatting
        diagnostics = []
        if exit_code != 0:
            for line in output.splitlines():
                line = line.strip()
                if line and not line.startswith("["):
                    # Assume it's a file path
                    try:
                        rel_file = Path(line).relative_to(root)
                        diagnostics.append(
                            Diagnostic(
                                file=str(rel_file),
                                line=1,
                                col=None,
                                code="prettier",
                                severity="warning",
                                message="File needs formatting",
                            )
                        )
                    except (ValueError, OSError):
                        pass
        return diagnostics, []
    except FileNotFoundError:
        return [], ["prettier"]


def detect_issues(root: Path) -> DetectionResult:
    """Detect all project issues via available checkers.

    Runs:
    - TypeScript (tsc) if tsconfig.json exists
    - ESLint if eslint config exists
    - actionlint if .github/workflows exists
    - ruff if Python project
    - prettier if prettier config exists

    Skips tools not on PATH; never fails on missing binaries.
    """
    root = Path(root).resolve()
    result = DetectionResult()

    # Collect project context
    try:
        result.recipe = detect_recipe(root)
    except Exception as exc:
        logger.warning(f"Recipe detection failed: {exc}")

    try:
        result.project_facts = detect_project_facts(root)
    except Exception as exc:
        logger.warning(f"Project facts detection failed: {exc}")

    # Run all available checkers
    checkers = [
        _check_typescript,
        _check_eslint,
        _check_actionlint,
        _check_ruff,
        _check_prettier,
    ]

    for checker in checkers:
        try:
            diagnostics, skipped = checker(root)
            result.diagnostics.extend(diagnostics)
            result.skipped_tools.extend(skipped)
        except Exception as exc:
            logger.warning(f"Checker {checker.__name__} failed: {exc}")

    return result
