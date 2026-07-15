"""Tests for flow_gate.py guard-write."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
FLOW_GATE = REPO_ROOT / "scripts/mw/flow_gate.py"
RULES = REPO_ROOT / "scripts/mw/flow-rules.yaml"


def _write(path: Path, content: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _project(root: Path) -> Path:
    _write(root / ".work/profile.yaml", "project: migrate-web\n")
    return root


def _run(path: Path, *, cwd: Path, env: Dict[str, str] | None = None) -> subprocess.CompletedProcess:
    process_env = os.environ.copy()
    if env:
        process_env.update(env)
    return subprocess.run(
        [sys.executable, str(FLOW_GATE), "guard-write", str(path)],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        env=process_env,
    )


def _load_rules() -> Dict[str, Any]:
    sys.path.insert(0, str(RULES.parent))
    try:
        from flow_eval import load_rules

        return load_rules(RULES)
    finally:
        sys.path.pop(0)


def _materialize(root: Path, menu: str, output: Dict[str, Any]) -> None:
    path = root / output["path"].replace("{menu}", menu)
    tokens: Iterable[str] = list(output["must_contain"]) + list(
        output["must_contain_any"][:1]
    )
    content = "\n".join(tokens) + "\n"
    needed = max(0, output["min_bytes"] - len(content.encode("utf-8")))
    _write(path, content + ("x" * needed))


def _seed_before(root: Path, menu: str, target_step: str) -> None:
    rules = _load_rules()
    for step in rules["steps"]:
        if step["id"] == target_step:
            return
        for output in step["outputs"]:
            _materialize(root, menu, output)
    raise AssertionError(f"unknown target step {target_step}")


def test_outside_mw_project_is_allowed(tmp_path: Path) -> None:
    proc = _run(tmp_path / "ordinary/file.py", cwd=tmp_path)
    assert proc.returncode == 0


def test_m2_is_blocked_and_names_m0_when_m0_is_missing(tmp_path: Path) -> None:
    root = _project(tmp_path / "mw")
    proc = _run(root / ".work/menus/home/design-brief.md", cwd=root)
    assert proc.returncode == 1
    assert "M0" in proc.stderr


def test_m0_first_output_is_allowed(tmp_path: Path) -> None:
    root = _project(tmp_path / "mw")
    proc = _run(root / ".project/menu-briefs/home.confirm.md", cwd=root)
    assert proc.returncode == 0


def test_m2_is_allowed_after_all_prior_steps(tmp_path: Path) -> None:
    root = _project(tmp_path / "mw")
    _seed_before(root, "home", "M2")
    proc = _run(root / ".work/menus/home/design-brief.md", cwd=root)
    assert proc.returncode == 0


def test_m8_fact_sheet_is_blocked_when_prior_steps_are_missing(tmp_path: Path) -> None:
    root = _project(tmp_path / "mw")
    proc = _run(root / ".work/deliverables/fact-sheets/home.md", cwd=root)
    assert proc.returncode == 1
    assert "M0" in proc.stderr


def test_other_project_file_is_allowed_with_current_user_lock(tmp_path: Path) -> None:
    root = _project(tmp_path / "mw")
    _write(root / ".work/menu-queue.md", "- menu: home locked_by: alice\n")
    proc = _run(root / "src/page.py", cwd=root, env={"MW_USER": "alice"})
    assert proc.returncode == 0


def test_other_project_file_is_blocked_without_current_user_lock(tmp_path: Path) -> None:
    root = _project(tmp_path / "mw")
    _write(root / ".work/menu-queue.md", "- menu: home locked_by: bob\n")
    proc = _run(root / "src/page.py", cwd=root, env={"MW_USER": "alice"})
    assert proc.returncode == 1
    assert "ต้องจองเมนูใน menu-queue" in proc.stderr


def test_other_project_file_warns_but_is_allowed_without_queue(tmp_path: Path) -> None:
    root = _project(tmp_path / "mw")
    proc = _run(root / "src/page.py", cwd=root, env={"MW_USER": "alice"})
    assert proc.returncode == 0
    assert "คำเตือน" in proc.stderr


def test_broken_project_rules_fail_closed(tmp_path: Path) -> None:
    root = _project(tmp_path / "mw")
    _write(root / ".work/flow-rules.yaml", "steps: [broken\n")
    proc = _run(root / ".project/menu-briefs/home.confirm.md", cwd=root)
    assert proc.returncode == 2
    assert "error:" in proc.stderr


def test_invalid_menu_slug_in_matching_output_fails_closed(tmp_path: Path) -> None:
    root = _project(tmp_path / "mw")
    proc = _run(root / ".work/menus/BAD/design-brief.md", cwd=root)
    assert proc.returncode == 2
    assert "invalid menu slug" in proc.stderr


def test_symlink_escape_from_mw_project_fails_closed(tmp_path: Path) -> None:
    root = _project(tmp_path / "mw")
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "linked").symlink_to(outside, target_is_directory=True)
    proc = _run(root / "linked/file.py", cwd=root)
    assert proc.returncode == 2
    assert "symlink" in proc.stderr
