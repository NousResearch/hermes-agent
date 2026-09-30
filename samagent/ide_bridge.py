"""IDE & Local Platform Bridge (samagent/ide_bridge.py).

Connects the SamAgent Platform to external IDEs (VS Code, Cursor, VSCodium, Zed) and the
host operating system (macOS / Linux / Windows) so users never have to use a CLI:
1. Generates `.vscode/{tasks.json,launch.json,settings.json,extensions.json}` in every project workspace.
2. Provides live workspace file tree, file read/write sync, and `git diff` inspection so edits made
   in VS Code are immediately visible and verifiable (L0–L4) in SamAgent before production deployment.
3. Launches VS Code (`code <path>` or `vscode://file/<abs_path>:<line>:1` URI).
4. Installs OS-level desktop launchers (macOS `.app` / `launchd`, Linux `.desktop` / `systemd`, Windows `.bat`)
   and the bundled VS Code extension (`integrations/vscode-samagent`).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Dict, List, Optional

from tools.subagent_worktree import _run_git

_SKIP_TREE_DIRS = frozenset({
    ".git",
    ".worktrees",
    "__pycache__",
    "node_modules",
    ".pytest_cache",
    ".venv",
})


def generate_vscode_workspace_config(project_dir: Path, *, platform_port: int = 8080) -> List[Path]:
    """Generate `.vscode/{tasks.json,launch.json,settings.json,extensions.json}` in *project_dir*."""
    root = Path(project_dir).resolve()
    vscode_dir = root / ".vscode"
    vscode_dir.mkdir(parents=True, exist_ok=True)

    tasks_json = {
        "version": "2.0.0",
        "tasks": [
            {
                "label": "SamAgent: Verify Workspace (L0-L4 + Security Probes)",
                "type": "shell",
                "command": "pytest .samagent/acceptance -v",
                "group": {"kind": "test", "isDefault": True},
                "presentation": {"reveal": "always", "panel": "shared"},
                "problemMatcher": [],
            },
            {
                "label": "SamAgent: Run Local Dev App Server",
                "type": "shell",
                "command": "python3 app/main.py --serve --port 3000",
                "isBackground": True,
                "presentation": {"reveal": "always", "panel": "dedicated"},
                "problemMatcher": [],
            },
            {
                "label": "SamAgent: Open Mission Control Platform",
                "type": "shell",
                "command": f"python3 -m webbrowser http://127.0.0.1:{platform_port}",
                "problemMatcher": [],
            },
        ],
    }

    launch_json = {
        "version": "0.2.0",
        "configurations": [
            {
                "name": "SamAgent: Debug Local App (Pre-Production)",
                "type": "debugpy",
                "request": "launch",
                "program": "${workspaceFolder}/app/main.py",
                "args": ["--serve", "--port", "3000"],
                "console": "integratedTerminal",
            },
            {
                "name": "SamAgent: Debug Acceptance & Security Suite",
                "type": "debugpy",
                "request": "launch",
                "module": "pytest",
                "args": ["${workspaceFolder}/.samagent/acceptance", "-v"],
                "console": "integratedTerminal",
            },
        ],
    }

    settings_json = {
        "python.testing.pytestEnabled": True,
        "python.testing.unittestEnabled": False,
        "python.testing.pytestArgs": [".samagent/acceptance"],
        "files.exclude": {
            "**/.git": True,
            "**/.worktrees": True,
            "**/__pycache__": True,
            "**/.pytest_cache": True,
        },
        "samagent.platformUrl": f"http://127.0.0.1:{platform_port}",
        "samagent.verifyOnSave": True,
    }

    extensions_json = {
        "recommendations": [
            "samagent.samagent-vscode",
            "ms-python.python",
            "redhat.vscode-yaml",
        ]
    }

    written: List[Path] = []
    for name, payload in (
        ("tasks.json", tasks_json),
        ("launch.json", launch_json),
        ("settings.json", settings_json),
        ("extensions.json", extensions_json),
    ):
        p = vscode_dir / name
        p.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        written.append(p)

    # Also write a root SamAgent.code-workspace file for one-click opening in VS Code
    ws_file = root / "project.code-workspace"
    ws_payload = {
        "folders": [{"path": "."}],
        "settings": settings_json,
        "extensions": extensions_json,
    }
    ws_file.write_text(json.dumps(ws_payload, indent=2) + "\n", encoding="utf-8")
    written.append(ws_file)
    return written


def list_workspace_files(project_dir: Path) -> List[Dict[str, Any]]:
    """Return all editable/inspectable files in *project_dir* with their VS Code deep links."""
    root = Path(project_dir).resolve()
    if not root.exists():
        return []
    items: List[Dict[str, Any]] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel_parts = p.relative_to(root).parts
        if any(part in _SKIP_TREE_DIRS for part in rel_parts):
            continue
        if p.name.endswith((".db", ".sqlite3", ".pyc")):
            continue
        rel = p.relative_to(root).as_posix()
        abs_posix = p.resolve().as_posix()
        category = (
            "vscode"
            if rel.startswith(".vscode/") or rel.endswith(".code-workspace")
            else "contract"
            if rel.startswith(".samagent/contract/")
            else "acceptance"
            if rel.startswith(".samagent/acceptance/")
            else "spec"
            if rel.startswith(".samagent/")
            else "app"
        )
        items.append(
            {
                "path": rel,
                "rel_path": rel,
                "abs_path": str(p.resolve()),
                "vscode_uri": f"vscode://file{abs_posix}:1:1",
                "cursor_uri": f"cursor://file{abs_posix}:1:1",
                "size_bytes": p.stat().st_size,
                "mtime": p.stat().st_mtime,
                "category": category,
            }
        )
    return items


def read_workspace_file(project_dir: Path, rel_path: str) -> Dict[str, Any]:
    """Safely read a file inside *project_dir* (blocks path traversal)."""
    root = Path(project_dir).resolve()
    target = (root / rel_path).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Path escapes workspace: {rel_path}") from exc
    if not target.exists() or not target.is_file():
        raise FileNotFoundError(f"File not found: {rel_path}")
    content = target.read_text(encoding="utf-8", errors="replace")
    abs_posix = target.as_posix()
    rel_str = target.relative_to(root).as_posix()
    return {
        "path": rel_str,
        "rel_path": rel_str,
        "abs_path": str(target),
        "vscode_uri": f"vscode://file{abs_posix}:1:1",
        "content": content,
    }


def write_workspace_file(project_dir: Path, rel_path: str, content: str) -> Dict[str, Any]:
    """Safely write an edited file inside *project_dir* (blocks path traversal)."""
    root = Path(project_dir).resolve()
    target = (root / rel_path).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Path escapes workspace: {rel_path}") from exc
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return read_workspace_file(root, rel_path)


def get_workspace_git_status_and_diff(project_dir: Path) -> Dict[str, Any]:
    """Return live `git status` and `git diff` so the user sees uncommitted IDE edits before deploy."""
    root = Path(project_dir).resolve()
    if not (root / ".git").exists():
        return {
            "is_git_repo": False,
            "dirty_files": [],
            "changed_files": [],
            "diff": "",
            "branch": "none",
            "head_commit": "none",
        }

    branch_res = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=str(root))
    head_res = _run_git(["rev-parse", "--short", "HEAD"], cwd=str(root))
    status_res = _run_git(["status", "--porcelain"], cwd=str(root))
    diff_res = _run_git(["diff", "HEAD"], cwd=str(root))
    log_res = _run_git(["log", "--oneline", "-n", "8"], cwd=str(root))

    dirty: List[str] = []
    changed_files: List[Dict[str, str]] = []
    for raw_line in (status_res.stdout or "").splitlines():
        if not raw_line.strip():
            continue
        dirty.append(raw_line.strip())
        st = raw_line[:2].strip() or "M"
        fpath = raw_line[3:].strip()
        changed_files.append({"status": st, "path": fpath})

    return {
        "is_git_repo": True,
        "branch": (branch_res.stdout or "main").strip(),
        "head_commit": (head_res.stdout or "HEAD").strip(),
        "dirty_files": dirty,
        "changed_files": changed_files,
        "diff": (diff_res.stdout or "")[:12000],
        "recent_commits": [line.strip() for line in (log_res.stdout or "").splitlines() if line.strip()],
    }


def open_in_vscode(
    project_dir: Path,
    rel_path: Optional[str] = None,
    *,
    line: int = 1,
) -> Dict[str, Any]:
    """Launch VS Code (`code`) on *project_dir* if installed on the host machine, and return deep-link URIs."""
    root = Path(project_dir).resolve()
    target = (root / rel_path).resolve() if rel_path else root
    if rel_path:
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Path escapes workspace: {rel_path}") from exc
    abs_posix = target.as_posix()
    suffix = f":{max(1, int(line))}:1" if rel_path else ""
    vscode_uri = f"vscode://file{abs_posix}{suffix}"
    cursor_uri = f"cursor://file{abs_posix}{suffix}"

    code_bin = shutil.which("code") or shutil.which("code-insiders") or shutil.which("codium")
    launched = False
    if code_bin and (os.environ.get("DISPLAY") or os.name in ("nt", "posix")):
        try:
            args = [code_bin, str(root)]
            if rel_path:
                args.extend(["-g", f"{target}:{max(1, int(line))}:1"])
            subprocess.Popen(
                args,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
            )
            launched = True
        except Exception:
            launched = False

    return {
        "launched_process": launched,
        "code_binary": code_bin,
        "project_dir": str(root),
        "target_path": str(target),
        "vscode_uri": vscode_uri,
        "cursor_uri": cursor_uri,
        "cli_fallback": f'code -g "{target}:{max(1, int(line))}"' if rel_path else f'code "{root}"',
    }


def evaluate_pre_production_gate(
    project_dir: Path,
    deliverable_or_verification: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Check whether the local workspace is safe and verified for production deployment."""
    from samagent.conductor.verify import scan_directory_security

    root = Path(project_dir).resolve()
    git_info = get_workspace_git_status_and_diff(root)
    payload = deliverable_or_verification or {}
    ver = payload.get("verification") if isinstance(payload.get("verification"), dict) else payload

    l0_l1_ok = bool(ver.get("l0_compile_passed", ver.get("passed", False))) and bool(
        ver.get("l1_unit_passed", ver.get("passed", False))
    )
    l2_ok = bool(ver.get("l2_contract_passed", ver.get("passed", False)))
    l3 = ver.get("l3_security") or {}
    l3_ok = bool(l3.get("passed")) if isinstance(l3, dict) and "passed" in l3 else any(
        l.get("level") == "L3" and l.get("passed") for l in (ver.get("layers") or [])
    )
    static_scan = scan_directory_security(root / "app") if (root / "app").exists() else {"passed": True, "findings": []}
    no_secrets = bool(static_scan.get("passed", True)) and len((l3.get("secret_leaks") if isinstance(l3, dict) else []) or []) == 0
    l4 = ver.get("l4_browser_smoke") or {}
    l4_ok = bool(l4.get("passed", ver.get("all_passed", ver.get("passed", False))))

    vscode_ready = (root / ".vscode" / "tasks.json").exists() and (root / ".vscode" / "launch.json").exists()
    app_ready = (root / "app" / "main.py").exists() and (root / "app" / "static" / "index.html").exists()

    checks_map = {
        "local_dev_app_ready": app_ready,
        "vscode_workspace_configured": vscode_ready,
        "l0_l1_syntax_contract": l0_l1_ok,
        "l2_ownership_and_tdd_red_green": l2_ok,
        "l3_security_owasp_idor_rbac": l3_ok,
        "l4_live_browser_dom_smoke": l4_ok,
        "no_secret_leaks": no_secrets,
    }
    blockers = [k for k, ok in checks_map.items() if not ok]
    return {
        "ready_for_production": len(blockers) == 0,
        "uncommitted_changes_count": len(git_info.get("dirty_files") or []),
        "checks": checks_map,
        "blockers": blockers,
    }
