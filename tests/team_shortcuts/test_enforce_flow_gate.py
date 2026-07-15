"""Tests for the Claude Code PreToolUse MW flow gate."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
HOOK = REPO_ROOT / "team-shortcuts/hooks/enforce-flow-gate.py"
INSTALLER = REPO_ROOT / "team-shortcuts/install-team-hooks.py"
MW_SCRIPTS = REPO_ROOT / "scripts/mw"


def _write(path: Path, content: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _project(root: Path, *, with_gate: bool = True) -> Path:
    _write(root / ".work/profile.yaml", "project: migrate-web\n")
    if with_gate:
        target = root / "scripts/mw"
        target.mkdir(parents=True, exist_ok=True)
        for name in ("flow_gate.py", "flow_eval.py", "flow-rules.yaml"):
            shutil.copy2(MW_SCRIPTS / name, target / name)
    return root


def _run_hook(
    payload: Dict[str, object], *, cwd: Path, hook: Path = HOOK, home: Path | None = None
) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    if home is not None:
        env["HOME"] = str(home)
    payload.setdefault("cwd", str(cwd))
    return subprocess.run(
        [sys.executable, str(hook)],
        input=json.dumps(payload),
        cwd=str(cwd),
        capture_output=True,
        text=True,
        env=env,
    )


def test_edit_m2_is_blocked_and_explains_m0_fix(tmp_path: Path) -> None:
    root = _project(tmp_path / "mw")
    proc = _run_hook(
        {
            "tool_name": "Edit",
            "tool_input": {"file_path": str(root / ".work/menus/home/design-brief.md")},
        },
        cwd=root,
    )
    assert proc.returncode == 2
    assert "M0" in proc.stderr
    assert "ทำ M0 ให้จบก่อน" in proc.stderr


def test_read_is_allowed_without_running_flow_gate(tmp_path: Path) -> None:
    isolated = tmp_path / "isolated/enforce-flow-gate.py"
    isolated.parent.mkdir(parents=True)
    shutil.copy2(HOOK, isolated)
    root = _project(tmp_path / "mw", with_gate=False)
    proc = _run_hook(
        {"tool_name": "Read", "tool_input": {"file_path": str(root / "README.md")}},
        cwd=root,
        hook=isolated,
        home=tmp_path / "home",
    )
    assert proc.returncode == 0


def test_bash_redirect_to_m4_is_checked_and_blocked(tmp_path: Path) -> None:
    root = _project(tmp_path / "mw")
    target = root / ".work/menus/home/m4-build.md"
    proc = _run_hook(
        {"tool_name": "Bash", "tool_input": {"command": f"echo x > {target}"}},
        cwd=root,
    )
    assert proc.returncode == 2
    assert "M0" in proc.stderr


@pytest.mark.parametrize(
    "command_template",
    [
        "printf x | tee {target}",
        "cp source.txt {target}",
        "mv source.txt {target}",
        "sed -i s/x/y/ {target}",
    ],
)
def test_bash_file_writer_patterns_are_checked(
    tmp_path: Path, command_template: str
) -> None:
    root = _project(tmp_path / "mw")
    target = root / ".work/menus/home/m4-build.md"
    proc = _run_hook(
        {
            "tool_name": "Bash",
            "tool_input": {"command": command_template.format(target=target)},
        },
        cwd=root,
    )
    assert proc.returncode == 2
    assert "M0" in proc.stderr


def test_complex_bash_write_in_mw_project_is_blocked(tmp_path: Path) -> None:
    root = _project(tmp_path / "mw")
    proc = _run_hook(
        {
            "tool_name": "Bash",
            "tool_input": {"command": "echo x > one.txt && echo y > two.txt"},
        },
        cwd=root,
    )
    assert proc.returncode == 2
    assert "ต้องแตกเป็นคำสั่งเดียวชัดๆ" in proc.stderr


def test_missing_flow_gate_in_mw_project_is_blocked(tmp_path: Path) -> None:
    isolated = tmp_path / "isolated/hooks/enforce-flow-gate.py"
    isolated.parent.mkdir(parents=True)
    shutil.copy2(HOOK, isolated)
    root = _project(tmp_path / "mw", with_gate=False)
    proc = _run_hook(
        {
            "tool_name": "Write",
            "tool_input": {"file_path": str(root / ".project/menu-briefs/home.confirm.md")},
        },
        cwd=root,
        hook=isolated,
        home=tmp_path / "empty-home",
    )
    assert proc.returncode == 2
    assert "ไม่พบ scripts/mw/flow_gate.py" in proc.stderr


def test_write_outside_mw_project_is_never_blocked(tmp_path: Path) -> None:
    proc = _run_hook(
        {"tool_name": "Write", "tool_input": {"file_path": str(tmp_path / "other/a.py")}},
        cwd=tmp_path,
        home=tmp_path / "home",
    )
    assert proc.returncode == 0


def test_installer_adds_one_pretooluse_entry_after_two_runs(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location("install_team_hooks", INSTALLER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "hooks": {
                    "PreToolUse": [
                        {
                            "matcher": "Read",
                            "hooks": [{"type": "command", "command": "keep-me"}],
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    runner = tmp_path / "hooks/enforce-flow-gate.py"
    module.install_pretooluse_entry(settings, runner)
    module.install_pretooluse_entry(settings, runner)

    data = json.loads(settings.read_text(encoding="utf-8"))
    entries = data["hooks"]["PreToolUse"]
    commands = [
        hook["command"]
        for entry in entries
        for hook in entry.get("hooks", [])
        if "enforce-flow-gate.py" in hook.get("command", "")
    ]
    assert commands == [str(runner)]
    assert any(hook.get("command") == "keep-me" for entry in entries for hook in entry["hooks"])
