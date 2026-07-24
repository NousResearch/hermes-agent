"""Tests for the `/evolve-cc` slash-command parser and runner."""

from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli import evolve_cc as evolve_cc_module
from hermes_cli.evolve_cc import run_evolve_cc, run_evolve_cc_slash
from tools.claude_code_evolve import (
    AnalysisResult,
    AnalysisSummary,
    MemoryCandidate,
    PlannedWrite,
)


def test_run_evolve_cc_slash_parses_flags():
    seen = {}

    def fake_run(args: Namespace, confirm_fn=None):
        seen["args"] = args
        seen["confirm_fn"] = confirm_fn
        return 0

    with patch("hermes_cli.evolve_cc.run_evolve_cc", side_effect=fake_run):
        assert (
            run_evolve_cc_slash(
                '/evolve-cc --days 7 --scope repo --path "./demo path" --apply --limit-memory 4 --limit-skills 2'
            )
            == 0
        )

    args = seen["args"]
    assert args.days == 7
    assert args.scope == "repo"
    assert args.path == "./demo path"
    assert args.apply is True
    assert args.limit_memory == 4
    assert args.limit_skills == 2


def _empty_result(anchor: Path) -> AnalysisResult:
    summary = AnalysisSummary(
        scope="cwd",
        anchor_path=anchor,
        since=datetime(2026, 1, 1, tzinfo=timezone.utc),
        repo_root=None,
        active_worktree_paths=(),
        analysis_projects=(),
        memory_target_project_dirs=(anchor,),
        historical_worktree_project_dirs=(),
    )
    return AnalysisResult(summary=summary, memory_candidates=(), skill_candidates=())


def test_run_evolve_cc_dry_run_when_scope_cwd_has_no_project(tmp_path, capsys):
    args = Namespace(
        days=7,
        scope="cwd",
        path=str(tmp_path),
        apply=False,
        limit_memory=10,
        limit_skills=5,
    )

    with patch.object(
        evolve_cc_module,
        "analyze_claude_code_history",
        return_value=_empty_result(tmp_path),
    ):
        rc = run_evolve_cc(args)

    out = capsys.readouterr().out
    assert rc == 0
    assert "Dry run only" in out


def test_run_evolve_cc_apply_without_tty_errors(tmp_path, capsys, monkeypatch):
    candidate = MemoryCandidate(
        type="feedback",
        name="feedback-example",
        description="example",
        body="# Evolved Memory: example\n",
        target_project_dirs=(tmp_path,),
        evidence=("2026-01-01: test",),
        frequency=2,
    )
    result = AnalysisResult(
        summary=_empty_result(tmp_path).summary,
        memory_candidates=(candidate,),
        skill_candidates=(),
    )
    planned = (
        PlannedWrite(
            kind="memory",
            identifier="feedback-example",
            path=tmp_path / "memory" / "feedback-example.md",
            content="# Evolved Memory: example\n",
            already_exists=False,
        ),
    )

    args = Namespace(
        days=7,
        scope="cwd",
        path=str(tmp_path),
        apply=True,
        limit_memory=10,
        limit_skills=5,
    )

    fake_stdin = type("S", (), {"isatty": staticmethod(lambda: False)})()
    monkeypatch.setattr(evolve_cc_module.sys, "stdin", fake_stdin)

    with patch.object(
        evolve_cc_module, "analyze_claude_code_history", return_value=result
    ), patch.object(
        evolve_cc_module, "plan_candidate_writes", return_value=planned
    ), patch.object(
        evolve_cc_module, "apply_write_plan"
    ) as apply_mock:
        rc = run_evolve_cc(args)

    out = capsys.readouterr().out
    assert rc == 1
    assert "interactive TTY" in out
    apply_mock.assert_not_called()


def test_run_evolve_cc_apply_skips_when_all_plans_already_exist(tmp_path, capsys):
    candidate = MemoryCandidate(
        type="feedback",
        name="feedback-example",
        description="example",
        body="# body\n",
        target_project_dirs=(tmp_path,),
        evidence=("2026-01-01: test",),
        frequency=2,
    )
    result = AnalysisResult(
        summary=_empty_result(tmp_path).summary,
        memory_candidates=(candidate,),
        skill_candidates=(),
    )
    planned = (
        PlannedWrite(
            kind="memory",
            identifier="feedback-example",
            path=tmp_path / "memory" / "feedback-example.md",
            content="# body\n",
            already_exists=True,
        ),
    )

    args = Namespace(
        days=7,
        scope="cwd",
        path=str(tmp_path),
        apply=True,
        limit_memory=10,
        limit_skills=5,
    )

    with patch.object(
        evolve_cc_module, "analyze_claude_code_history", return_value=result
    ), patch.object(
        evolve_cc_module, "plan_candidate_writes", return_value=planned
    ), patch.object(
        evolve_cc_module, "apply_write_plan"
    ) as apply_mock:
        rc = run_evolve_cc(args)

    out = capsys.readouterr().out
    assert rc == 0
    assert "already exist" in out
    apply_mock.assert_not_called()


@pytest.mark.parametrize("bad", ["--days 0", "--days -1", "--limit-memory 0"])
def test_run_evolve_cc_slash_rejects_non_positive(bad, capsys):
    rc = run_evolve_cc_slash(f"/evolve-cc {bad}")
    assert rc == 2
    assert "Usage" in capsys.readouterr().out
