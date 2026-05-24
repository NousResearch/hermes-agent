#!/usr/bin/env python3
"""Verify active prompt-loader policy invariants.

This checker intentionally validates the current ``agent.prompt_builder``
contract. It is not a replay of the legacy v0.8 priority-section checker.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.prompt_builder import (  # noqa: E402
    CONTEXT_FILE_MAX_CHARS,
    _truncate_content,
    build_context_files_prompt,
)

FORBIDDEN_POLICY_SNIPPETS = {
    "<claude-mem-context>": "generated claude memory context",
    "</claude-mem-context>": "generated claude memory context close tag",
    "<system-reminder>": "generated system reminder block",
}


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def check_active_policy_file(root: Path) -> list[str]:
    failures: list[str] = []
    agents_md = root / "AGENTS.md"
    if not agents_md.exists():
        return [f"missing active policy file: {agents_md}"]
    try:
        content = agents_md.read_text(encoding="utf-8").strip()
    except OSError as exc:
        return [f"could not read active policy file {agents_md}: {exc}"]
    if not content:
        failures.append(f"active policy file is empty: {agents_md}")
    for snippet, label in FORBIDDEN_POLICY_SNIPPETS.items():
        if snippet in content:
            failures.append(f"active policy file contains {label}: {snippet}")
    for shadow_name in (".hermes.md", "HERMES.md"):
        if (root / shadow_name).exists():
            failures.append(
                f"active AGENTS.md would be shadowed by higher-priority {shadow_name}"
            )
    return failures


def check_active_policy_uptake(root: Path) -> list[str]:
    failures: list[str] = []
    try:
        loaded = build_context_files_prompt(cwd=str(root), skip_soul=True)
    except Exception as exc:  # pragma: no cover - defensive operator output
        return [f"active prompt context failed to load: {type(exc).__name__}: {exc}"]

    if "# Project Context" not in loaded:
        failures.append("active prompt context did not include project context wrapper")
    if "## AGENTS.md" not in loaded:
        failures.append("active prompt context did not load AGENTS.md")
    if "<claude-mem-context>" in loaded:
        failures.append("active prompt context contains raw generated claude memory block")
    return failures


def check_loader_priority_contract() -> list[str]:
    failures: list[str] = []

    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / ".git").mkdir()
        child = root / "pkg" / "module"
        child.mkdir(parents=True)
        _write(root / ".hermes.md", "HERMES_SHOULD_LOAD")
        _write(child / "AGENTS.md", "AGENTS_SHOULD_NOT_LOAD")
        _write(child / "CLAUDE.md", "CLAUDE_SHOULD_NOT_LOAD")
        _write(child / ".cursorrules", "CURSOR_SHOULD_NOT_LOAD")
        _write(child / ".cursor" / "rules" / "rule.mdc", "MDC_SHOULD_NOT_LOAD")

        loaded = build_context_files_prompt(cwd=str(child), skip_soul=True)
        if "HERMES_SHOULD_LOAD" not in loaded:
            failures.append("dropped highest-priority .hermes.md project context")
        if "AGENTS_SHOULD_NOT_LOAD" in loaded:
            failures.append("loaded lower-priority AGENTS.md alongside .hermes.md")
        if "CLAUDE_SHOULD_NOT_LOAD" in loaded:
            failures.append("loaded lower-priority CLAUDE.md alongside .hermes.md")
        if "CURSOR_SHOULD_NOT_LOAD" in loaded or "MDC_SHOULD_NOT_LOAD" in loaded:
            failures.append("loaded lower-priority cursor rules alongside .hermes.md")

    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write(root / "AGENTS.md", "AGENTS_SHOULD_LOAD")
        _write(root / "CLAUDE.md", "CLAUDE_SHOULD_NOT_LOAD")
        _write(root / ".cursorrules", "CURSOR_SHOULD_NOT_LOAD")
        loaded = build_context_files_prompt(cwd=str(root), skip_soul=True)
        if "AGENTS_SHOULD_LOAD" not in loaded:
            failures.append("dropped AGENTS.md when no .hermes.md exists")
        if "CLAUDE_SHOULD_NOT_LOAD" in loaded:
            failures.append("loaded lower-priority CLAUDE.md alongside AGENTS.md")
        if "CURSOR_SHOULD_NOT_LOAD" in loaded:
            failures.append("loaded lower-priority cursor rules alongside AGENTS.md")

    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write(root / "CLAUDE.md", "CLAUDE_SHOULD_LOAD")
        _write(root / ".cursorrules", "CURSOR_SHOULD_NOT_LOAD")
        loaded = build_context_files_prompt(cwd=str(root), skip_soul=True)
        if "CLAUDE_SHOULD_LOAD" not in loaded:
            failures.append("dropped CLAUDE.md when no .hermes.md or AGENTS.md exists")
        if "CURSOR_SHOULD_NOT_LOAD" in loaded:
            failures.append("loaded lower-priority cursor rules alongside CLAUDE.md")

    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write(root / ".cursorrules", "CURSOR_SHOULD_LOAD")
        loaded = build_context_files_prompt(cwd=str(root), skip_soul=True)
        if "CURSOR_SHOULD_LOAD" not in loaded:
            failures.append("dropped cursor rules when no higher-priority file exists")

    return failures


def check_soul_independence() -> list[str]:
    failures: list[str] = []
    with TemporaryDirectory() as tmp:
        base = Path(tmp)
        project = base / "project"
        hermes_home = base / "hermes_home"
        project.mkdir()
        hermes_home.mkdir()
        _write(project / "AGENTS.md", "PROJECT_POLICY_MARKER")
        _write(project / "SOUL.md", "CWD_SOUL_SHOULD_NOT_LOAD")
        _write(hermes_home / "SOUL.md", "HERMES_HOME_SOUL_MARKER")

        with patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}):
            loaded = build_context_files_prompt(cwd=str(project), skip_soul=False)
            loaded_without_soul = build_context_files_prompt(cwd=str(project), skip_soul=True)

        if "PROJECT_POLICY_MARKER" not in loaded:
            failures.append("project policy dropped when HERMES_HOME SOUL.md exists")
        if "HERMES_HOME_SOUL_MARKER" not in loaded:
            failures.append("HERMES_HOME SOUL.md was not loaded independently")
        if "CWD_SOUL_SHOULD_NOT_LOAD" in loaded:
            failures.append("cwd SOUL.md was loaded instead of HERMES_HOME SOUL.md")
        if "HERMES_HOME_SOUL_MARKER" in loaded_without_soul:
            failures.append("skip_soul=True still loaded SOUL.md")
    return failures


def check_truncation_contract() -> list[str]:
    failures: list[str] = []
    content = (
        "HEAD_PROMPT_POLICY_MARKER\n"
        + ("A" * CONTEXT_FILE_MAX_CHARS)
        + "\nTAIL_PROMPT_POLICY_MARKER"
    )
    loaded = _truncate_content(content, "policy.md")
    if "HEAD_PROMPT_POLICY_MARKER" not in loaded:
        failures.append("truncation dropped head marker")
    if "TAIL_PROMPT_POLICY_MARKER" not in loaded:
        failures.append("truncation dropped tail marker")
    if "[...truncated policy.md:" not in loaded or "kept" not in loaded:
        failures.append("truncation marker missing or malformed")
    if len(loaded) >= len(content):
        failures.append("truncation did not reduce oversized content")
    return failures


def run_checks(root: Path = ROOT) -> list[str]:
    root = root.resolve()
    failures: list[str] = []
    failures.extend(check_active_policy_file(root))
    if not failures:
        failures.extend(check_active_policy_uptake(root))
    failures.extend(check_loader_priority_contract())
    failures.extend(check_soul_independence())
    failures.extend(check_truncation_contract())
    return failures


def main() -> int:
    failures = run_checks(ROOT)
    if failures:
        print("FAIL: prompt policy checks failed", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    print("PASS: active prompt policy loader checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
