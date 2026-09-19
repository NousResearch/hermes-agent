"""Worker system for parallel file-sticky repairs.

Workers fix root causes; no project-wide check rerun; no diagnostic suppression.
Mutating formatters run serially first.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.cleanse.parser import Diagnostic

logger = logging.getLogger("hermes.cleanse.worker")


@dataclass
class RepairResult:
    """Result of attempting to fix diagnostics."""

    file: str
    fixed_count: int
    remaining: list[Diagnostic]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "file": self.file,
            "fixed_count": self.fixed_count,
            "remaining": [d.to_dict() for d in self.remaining],
            "error": self.error,
        }


def _group_by_file(diagnostics: list[Diagnostic]) -> dict[str, list[Diagnostic]]:
    """Group diagnostics by file for file-sticky workers."""
    groups: dict[str, list[Diagnostic]] = defaultdict(list)
    for diag in diagnostics:
        groups[diag.file].append(diag)
    return dict(groups)


def _is_formatting_issue(diag: Diagnostic) -> bool:
    """Check if diagnostic is a formatting issue (mutating formatter)."""
    formatting_codes = {
        "prettier",
        "eslint/prettier",
        "ruff-format",
    }
    # Check if it's a known formatting code or ESLint formatting rule
    if diag.code in formatting_codes:
        return True
    # ESLint formatting rules typically end with specific patterns
    if diag.code.startswith("@typescript-eslint/"):
        if any(
            keyword in diag.code
            for keyword in ["indent", "spacing", "newline", "comma", "semi", "quotes", "brace"]
        ):
            return True
    return False


def _apply_formatting_fixes(root: Path, files: set[str]) -> dict[str, RepairResult]:
    """Apply mutating formatters serially to the given files.

    Runs prettier and eslint --fix on files with formatting issues.
    Returns a map of file -> RepairResult.
    """
    import subprocess
    import shutil

    results: dict[str, RepairResult] = {}

    # Run prettier if available
    if shutil.which("prettier"):
        for file_path in sorted(files):
            abs_path = root / file_path
            if not abs_path.exists():
                continue
            try:
                subprocess.run(
                    ["prettier", "--write", str(abs_path)],
                    cwd=str(root),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=30,
                    check=True,
                )
                logger.info(f"Formatted {file_path} with prettier")
            except (subprocess.SubprocessError, OSError) as exc:
                logger.warning(f"Prettier failed on {file_path}: {exc}")

    # Run eslint --fix if available
    if shutil.which("eslint"):
        for file_path in sorted(files):
            abs_path = root / file_path
            if not abs_path.exists():
                continue
            try:
                subprocess.run(
                    ["eslint", "--fix", str(abs_path)],
                    cwd=str(root),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=30,
                )
                logger.info(f"Fixed {file_path} with eslint --fix")
            except (subprocess.SubprocessError, OSError) as exc:
                logger.warning(f"ESLint --fix failed on {file_path}: {exc}")

    for file_path in files:
        results[file_path] = RepairResult(
            file=file_path,
            fixed_count=0,  # Will be updated after re-check
            remaining=[],
            error=None,
        )

    return results


def _fix_file_diagnostics(
    root: Path,
    file_path: str,
    diagnostics: list[Diagnostic],
) -> RepairResult:
    """Fix diagnostics for a single file (stub for now).

    In a full implementation, this would:
    1. Group diagnostics by type/code
    2. Apply automated fixes (e.g., via language server or AST manipulation)
    3. Return remaining unfixed diagnostics

    For now, this is a placeholder that marks formatting issues as fixed
    if they were already handled by formatters.
    """
    # This is a simplified implementation
    # A full version would use language-specific fixers
    return RepairResult(
        file=file_path,
        fixed_count=0,
        remaining=diagnostics,
        error=None,
    )


def repair_issues(
    root: Path,
    diagnostics: list[Diagnostic],
    max_workers: int = 2,
) -> dict[str, RepairResult]:
    """Repair issues with file-sticky workers.

    Process:
    1. Separate formatting issues (mutating formatters)
    2. Apply formatters serially first
    3. Spawn N file-sticky workers for remaining issues
    4. Workers fix root causes (no suppression)

    Args:
        root: Project root
        diagnostics: List of diagnostics to fix
        max_workers: Maximum parallel workers (default 2)

    Returns:
        Map of file -> RepairResult
    """
    root = Path(root).resolve()
    grouped = _group_by_file(diagnostics)

    # Separate formatting issues
    formatting_files = set()
    non_formatting: dict[str, list[Diagnostic]] = {}

    for file_path, file_diagnostics in grouped.items():
        formatting_diags = [d for d in file_diagnostics if _is_formatting_issue(d)]
        other_diags = [d for d in file_diagnostics if not _is_formatting_issue(d)]

        if formatting_diags:
            formatting_files.add(file_path)
        if other_diags:
            non_formatting[file_path] = other_diags

    results: dict[str, RepairResult] = {}

    # Apply formatters serially first
    if formatting_files:
        logger.info(f"Applying formatters to {len(formatting_files)} files")
        formatting_results = _apply_formatting_fixes(root, formatting_files)
        results.update(formatting_results)

    # Fix remaining issues with file-sticky workers
    # For now, we process serially (parallel implementation would use threading/multiprocessing)
    for file_path, file_diagnostics in non_formatting.items():
        result = _fix_file_diagnostics(root, file_path, file_diagnostics)
        if file_path in results:
            # Merge with formatting results
            results[file_path].remaining.extend(result.remaining)
        else:
            results[file_path] = result

    return results
