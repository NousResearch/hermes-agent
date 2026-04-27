#!/usr/bin/env python3
"""
CodeWorkspaceService — detect and persist code workspace metadata.

Detects stack (Node/TS/React/Vite/Next, Go, Python, Docker),
package manager, available commands, and Git branch/remote.
No files in the workspace are modified.
"""

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stack detection
# ---------------------------------------------------------------------------

_NODE_FRAMEWORK_DEPS = {
    "react": "react",
    "next": "next",
    "vite": "vite",
    "zustand": "zustand",
    "tailwindcss": "tailwind",
    "fastapi": "fastapi",
    "flask": "flask",
    "django": "django",
}

_PYTHON_FRAMEWORK_DEPS = {
    "fastapi": "fastapi",
    "flask": "flask",
    "django": "django",
    "pytest": "pytest",
}


def _read_json_safe(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return {}


def _read_text_safe(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def detect_stack(workspace_path: str) -> list[str]:
    """Return list of detected stack identifiers for workspace_path."""
    root = Path(workspace_path)
    stack: list[str] = []

    # ---- Go ----
    if (root / "go.mod").exists():
        stack.append("go")
        if (root / "go.sum").exists():
            stack.append("go-module")

    # ---- Python ----
    py_indicators = [
        "pyproject.toml", "requirements.txt", "setup.py", "Pipfile",
        "uv.lock", "main.py", "app.py",
    ]
    if any((root / f).exists() for f in py_indicators):
        stack.append("python")
        # Look for known frameworks in pyproject.toml / requirements.txt
        for dep_file in ["pyproject.toml", "requirements.txt", "Pipfile"]:
            text = _read_text_safe(root / dep_file).lower()
            for dep_key, stack_name in _PYTHON_FRAMEWORK_DEPS.items():
                if stack_name not in stack and dep_key in text:
                    stack.append(stack_name)
        if (root / "pytest.ini").exists() or (root / "tests").is_dir():
            if "pytest" not in stack:
                stack.append("pytest")

    # ---- Node / Frontend ----
    if (root / "package.json").exists():
        stack.append("node")
        pkg = _read_json_safe(root / "package.json")
        all_deps: dict = {}
        all_deps.update(pkg.get("dependencies") or {})
        all_deps.update(pkg.get("devDependencies") or {})
        all_deps_lower = {k.lower(): v for k, v in all_deps.items()}

        for dep_key, stack_name in _NODE_FRAMEWORK_DEPS.items():
            if stack_name not in stack and dep_key in all_deps_lower:
                stack.append(stack_name)

    if (root / "tsconfig.json").exists() and "typescript" not in stack:
        stack.append("typescript")

    for vite_name in ("vite.config.ts", "vite.config.js", "vite.config.mts"):
        if (root / vite_name).exists() and "vite" not in stack:
            stack.append("vite")
            break

    for next_name in ("next.config.js", "next.config.ts", "next.config.mjs"):
        if (root / next_name).exists() and "next" not in stack:
            stack.append("next")
            break

    for tw_name in ("tailwind.config.js", "tailwind.config.ts"):
        if (root / tw_name).exists() and "tailwind" not in stack:
            stack.append("tailwind")
            break

    # ---- Docker / Infra ----
    if (root / "Dockerfile").exists():
        stack.append("docker")
    for compose_name in ("docker-compose.yml", "docker-compose.yaml", "compose.yml", "compose.yaml"):
        if (root / compose_name).exists() and "compose" not in stack:
            stack.append("compose")
            break
    if (root / "Makefile").exists() and "make" not in stack:
        stack.append("make")

    return stack


# ---------------------------------------------------------------------------
# Package manager detection
# ---------------------------------------------------------------------------

def detect_package_manager(workspace_path: str) -> Optional[str]:
    """Return the primary package manager for workspace_path."""
    root = Path(workspace_path)

    # Node lockfiles — priority order
    if (root / "pnpm-lock.yaml").exists():
        return "pnpm"
    if (root / "yarn.lock").exists():
        return "yarn"
    if (root / "bun.lock").exists() or (root / "bun.lockb").exists():
        return "bun"
    if (root / "package-lock.json").exists():
        return "npm"

    # Non-Node
    if (root / "go.mod").exists():
        return "go"
    if (root / "uv.lock").exists():
        return "uv"
    if (root / "Makefile").exists():
        return "make"

    return None


# ---------------------------------------------------------------------------
# Command detection
# ---------------------------------------------------------------------------

_SCRIPT_KIND_MAP = {
    "build": "build",
    "test": "test",
    "lint": "lint",
    "typecheck": "typecheck",
    "tsc": "typecheck",
    "dev": "dev",
    "start": "dev",
    "format": "format",
    "prettier": "format",
}


def _classify_script(name: str, command: str) -> str:
    needle = (name + " " + command).lower()
    for keyword, kind in _SCRIPT_KIND_MAP.items():
        if keyword in needle:
            return kind
    return "other"


def _detect_makefile_targets(root: Path) -> list[dict]:
    """Extract simple Makefile phony/rule targets."""
    text = _read_text_safe(root / "Makefile")
    if not text:
        return []
    targets = []
    interesting = {"dev", "build", "test", "lint", "typecheck", "clean", "format", "run", "install"}
    for line in text.splitlines():
        # Match lines like "target:" or "target: deps"
        if line and not line.startswith(("\t", " ", "#")):
            parts = line.split(":")
            name = parts[0].strip()
            if name and not name.startswith(".") and name in interesting:
                kind = _classify_script(name, "")
                targets.append({
                    "name": name,
                    "command": f"make {name}",
                    "source": "makefile",
                    "kind": kind,
                })
    return targets


def detect_commands(workspace_path: str, package_manager: Optional[str] = None) -> list[dict]:
    """Return list of detected commands for workspace_path."""
    root = Path(workspace_path)
    commands: list[dict] = []

    # --- package.json scripts ---
    if (root / "package.json").exists():
        pkg = _read_json_safe(root / "package.json")
        scripts: dict = pkg.get("scripts") or {}
        pm = package_manager or "npm"
        if pm not in ("npm", "pnpm", "yarn", "bun"):
            pm = "npm"
        prefix = f"{pm} run" if pm != "yarn" else "yarn"

        for name, cmd in scripts.items():
            kind = _classify_script(name, cmd or "")
            commands.append({
                "name": name,
                "command": f"{prefix} {name}",
                "source": "package.json",
                "kind": kind,
            })

    # --- Makefile targets ---
    if (root / "Makefile").exists():
        commands.extend(_detect_makefile_targets(root))

    # --- Go built-in commands ---
    if (root / "go.mod").exists():
        commands.extend([
            {"name": "test", "command": "go test ./...", "source": "go", "kind": "test"},
            {"name": "build", "command": "go build ./...", "source": "go", "kind": "build"},
            {"name": "lint", "command": "go vet ./...", "source": "go", "kind": "lint"},
        ])

    return commands


# ---------------------------------------------------------------------------
# Git detection
# ---------------------------------------------------------------------------

def detect_git_info(workspace_path: str) -> dict:
    """Return {is_git_repo, branch, repo_url} for workspace_path.

    Never raises — returns safe defaults on failure.
    """
    result = {"is_git_repo": False, "branch": None, "repo_url": None}
    root = Path(workspace_path)

    # Fast check: .git directory or file (worktrees)
    git_marker = root / ".git"
    if not git_marker.exists():
        # Could be a git worktree — probe subprocess
        try:
            check = subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                cwd=str(root),
                capture_output=True,
                text=True,
                timeout=3,
            )
            if check.returncode != 0 or check.stdout.strip() != "true":
                return result
        except Exception:
            return result

    result["is_git_repo"] = True

    # Branch
    try:
        r = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=3,
        )
        branch = r.stdout.strip()
        if not branch:
            # Detached HEAD
            r2 = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=str(root),
                capture_output=True,
                text=True,
                timeout=3,
            )
            branch = r2.stdout.strip() or None
        result["branch"] = branch or None
    except Exception:
        pass

    # Remote origin URL
    try:
        r = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=3,
        )
        if r.returncode == 0:
            result["repo_url"] = r.stdout.strip() or None
    except Exception:
        pass

    return result


# ---------------------------------------------------------------------------
# CodeWorkspaceService
# ---------------------------------------------------------------------------

class CodeWorkspaceService:
    """Inspect, register, and manage code workspaces.

    Uses WorkspaceDB (hermes_state) for persistence.
    """

    def inspect_path(self, path: str) -> dict:
        """Inspect a filesystem path and return workspace metadata without saving."""
        root = Path(path).resolve()
        if not root.exists():
            raise ValueError(f"Path does not exist: {path}")
        if not root.is_dir():
            raise ValueError(f"Path is not a directory: {path}")

        git_info = detect_git_info(str(root))
        stack = detect_stack(str(root))
        pm = detect_package_manager(str(root))
        commands = detect_commands(str(root), pm)

        return {
            "name": root.name,
            "path": str(root),
            "is_git_repo": git_info["is_git_repo"],
            "branch": git_info["branch"],
            "repo_url": git_info["repo_url"],
            "detected_stack": stack,
            "package_manager": pm,
            "commands": commands,
        }

    def open_workspace(self, path: str) -> dict:
        """Open/register a workspace. Upserts on path — no duplicates."""
        from hermes_state import WorkspaceDB

        meta = self.inspect_path(path)
        db = WorkspaceDB()
        try:
            return db.upsert_workspace(
                path=meta["path"],
                name=meta["name"],
                is_git_repo=meta["is_git_repo"],
                branch=meta["branch"],
                repo_url=meta["repo_url"],
                detected_stack=meta["detected_stack"],
                package_manager=meta["package_manager"],
                commands=meta["commands"],
            )
        finally:
            db.close()

    def list_workspaces(self) -> list[dict]:
        from hermes_state import WorkspaceDB

        db = WorkspaceDB()
        try:
            return db.list_workspaces()
        finally:
            db.close()

    def get_workspace(self, workspace_id: str) -> Optional[dict]:
        from hermes_state import WorkspaceDB

        db = WorkspaceDB()
        try:
            return db.get_workspace(workspace_id)
        finally:
            db.close()

    def refresh_workspace(self, workspace_id: str) -> dict:
        """Re-detect stack/branch/commands for an existing workspace."""
        from hermes_state import WorkspaceDB

        db = WorkspaceDB()
        try:
            ws = db.get_workspace(workspace_id)
            if not ws:
                raise ValueError(f"Workspace not found: {workspace_id}")
            meta = self.inspect_path(ws["path"])
            return db.upsert_workspace(
                path=meta["path"],
                name=meta["name"],
                is_git_repo=meta["is_git_repo"],
                branch=meta["branch"],
                repo_url=meta["repo_url"],
                detected_stack=meta["detected_stack"],
                package_manager=meta["package_manager"],
                commands=meta["commands"],
            )
        finally:
            db.close()
