"""Regression coverage for Kanban CLI process exit status propagation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parents[2]


def _run_hermes(home: Path, *args: str, marker: bool = False) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["HERMES_HOME"] = str(home)
    env["HERMES_KANBAN_HOME"] = str(home)
    for name in (
        "HERMES_KANBAN_BOARD",
        "HERMES_KANBAN_DB",
        "HERMES_KANBAN_WORKSPACES_ROOT",
    ):
        env.pop(name, None)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    if marker:
        env["HERMES_DELEGATED_CHILD_CONTEXT"] = "1"
    else:
        env.pop("HERMES_DELEGATED_CHILD_CONTEXT", None)
    return subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", *args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


def test_delegated_child_kanban_cli_refusal_returns_nonzero_exit_status(tmp_path):
    """A printed Kanban mutation refusal must not look like CLI success."""
    home = tmp_path / "hermes"
    home.mkdir()

    created = _run_hermes(home, "kanban", "create", "exit status probe", "--json")
    assert created.returncode == 0, created.stderr
    task_id = json.loads(created.stdout)["id"]

    refused = _run_hermes(
        home,
        "kanban",
        "comment",
        task_id,
        "must be refused",
        marker=True,
    )

    assert refused.returncode == 1
    assert "delegate_task child contexts cannot mutate Kanban tasks via the CLI" in refused.stderr


def test_unanchored_worktree_create_returns_nonzero_exit_status(tmp_path):
    home = tmp_path / "hermes"
    home.mkdir()

    refused = _run_hermes(
        home,
        "kanban",
        "create",
        "unanchored",
        "--workspace",
        "worktree",
    )

    assert refused.returncode != 0
    assert "workspace_kind=worktree requires" in refused.stderr


def test_claim_refusal_after_skill_removal_reports_reason(tmp_path):
    """`kanban claim` on a card whose forced skill disappeared must exit
    non-zero AND name the missing skill (actionable, not just status=blocked)."""
    home = tmp_path / "hermes"
    home.mkdir()
    skill = home / "skills" / "claim-cli-skill" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text(
        "---\nname: claim-cli-skill\ndescription: Test fixture.\n---\n\n# t\n",
        encoding="utf-8",
    )

    created = _run_hermes(
        home,
        "kanban",
        "create",
        "claim guard cli",
        "--assignee",
        "default",
        "--skill",
        "claim-cli-skill",
        "--json",
    )
    assert created.returncode == 0, created.stderr
    task_id = json.loads(created.stdout)["id"]

    first_claim = _run_hermes(home, "kanban", "claim", task_id)
    assert first_claim.returncode == 0, first_claim.stderr

    reclaimed = _run_hermes(home, "kanban", "reclaim", task_id)
    assert reclaimed.returncode == 0, reclaimed.stderr

    skill.unlink()

    refused = _run_hermes(home, "kanban", "claim", task_id)

    assert refused.returncode != 0
    assert "claim-cli-skill" in refused.stderr
    assert "cannot claim" in refused.stderr
