"""Tests for the prompt-policy checker script."""

from __future__ import annotations

import pytest

from agent.prompt_builder import CONTEXT_FILE_MAX_CHARS


def test_checker_passes_for_active_repo() -> None:
    from scripts import check_prompt_policy

    assert check_prompt_policy.run_checks(check_prompt_policy.ROOT) == []


def test_checker_reports_missing_active_agents_md(tmp_path) -> None:
    from scripts import check_prompt_policy

    assert check_prompt_policy.run_checks(tmp_path) == [
        f"missing active policy file: {tmp_path / 'AGENTS.md'}"
    ]


def test_checker_reports_forbidden_generated_context_in_source(tmp_path) -> None:
    from scripts import check_prompt_policy

    agents_md = tmp_path / "AGENTS.md"
    agents_md.write_text(
        "# Policy\n\n" + ("A" * (CONTEXT_FILE_MAX_CHARS + 100)) + "\n<claude-mem-context>\nstale\n</claude-mem-context>\n",
        encoding="utf-8",
    )

    failures = check_prompt_policy.run_checks(tmp_path)

    assert "active policy file contains generated claude memory context: <claude-mem-context>" in failures
    assert "active policy file contains generated claude memory context close tag: </claude-mem-context>" in failures


def test_checker_reports_loader_priority_violation(monkeypatch) -> None:
    from scripts import check_prompt_policy

    def broken_prompt(*, cwd: str | None = None, skip_soul: bool = False) -> str:
        return "AGENTS_SHOULD_NOT_LOAD CLAUDE_SHOULD_NOT_LOAD CURSOR_SHOULD_NOT_LOAD"

    monkeypatch.setattr(check_prompt_policy, "build_context_files_prompt", broken_prompt)

    failures = check_prompt_policy.check_loader_priority_contract()

    assert "dropped highest-priority .hermes.md project context" in failures
    assert "loaded lower-priority AGENTS.md alongside .hermes.md" in failures
    assert "loaded lower-priority CLAUDE.md alongside .hermes.md" in failures
    assert "loaded lower-priority cursor rules alongside .hermes.md" in failures


def test_checker_proves_soul_md_independent(monkeypatch, tmp_path) -> None:
    from scripts import check_prompt_policy

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_home"))
    assert check_prompt_policy.check_soul_independence() == []

    def broken_prompt(*, cwd: str | None = None, skip_soul: bool = False) -> str:
        if skip_soul:
            return "PROJECT_POLICY_MARKER HERMES_HOME_SOUL_MARKER"
        return "PROJECT_POLICY_MARKER CWD_SOUL_SHOULD_NOT_LOAD"

    monkeypatch.setattr(check_prompt_policy, "build_context_files_prompt", broken_prompt)

    failures = check_prompt_policy.check_soul_independence()

    assert "HERMES_HOME SOUL.md was not loaded independently" in failures
    assert "cwd SOUL.md was loaded instead of HERMES_HOME SOUL.md" in failures
    assert "skip_soul=True still loaded SOUL.md" in failures


def test_checker_reports_truncation_regression(monkeypatch) -> None:
    from scripts import check_prompt_policy

    def broken_truncate(content: str, filename: str, max_chars: int = CONTEXT_FILE_MAX_CHARS) -> str:
        return content[:max_chars]

    monkeypatch.setattr(check_prompt_policy, "_truncate_content", broken_truncate)

    failures = check_prompt_policy.check_truncation_contract()

    assert "truncation dropped head marker" not in failures
    assert "truncation dropped tail marker" in failures
    assert "truncation marker missing or malformed" in failures


def test_main_returns_nonzero_for_policy_failure(monkeypatch, capsys) -> None:
    from scripts import check_prompt_policy

    monkeypatch.setattr(check_prompt_policy, "run_checks", lambda root: ["synthetic failure"])

    assert check_prompt_policy.main() == 1
    captured = capsys.readouterr()
    assert "FAIL: prompt policy checks failed" in captured.err
    assert "synthetic failure" in captured.err
