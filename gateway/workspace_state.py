"""Workspace helpers shared by the gateway and status commands.

Gateway workspaces are intentionally lightweight: a session can store a current
absolute directory, and when unset we fall back to the WebUI's last/current
workspace when that state is available.  Helpers in this module never create or
modify project directories; they only validate existing directories and read the
WebUI state files.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable


def normalize_workspace_path(raw: str) -> tuple[str | None, str | None]:
    """Return ``(absolute_path, None)`` for an existing directory.

    ``raw`` may use ``~``.  Relative paths are resolved against the current
    process cwd so command handlers can accept short paths while still storing a
    stable absolute path.  The function is deliberately non-creating and returns
    a user-facing error string on failure.
    """
    text = (raw or "").strip()
    if not text:
        return None, "Workspace path is empty."
    try:
        path = Path(text).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        resolved = path.resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        return None, f"Could not resolve workspace path: {exc}"
    if not resolved.exists():
        return None, f"Workspace does not exist: {resolved}"
    if not resolved.is_dir():
        return None, f"Workspace is not a directory: {resolved}"
    return str(resolved), None


def _candidate_state_dirs() -> list[Path]:
    dirs: list[Path] = []
    env_dir = os.environ.get("HERMES_WEBUI_STATE_DIR")
    if env_dir:
        dirs.append(Path(env_dir).expanduser())

    try:
        from hermes_constants import get_hermes_home

        home = get_hermes_home()
        dirs.extend([
            home / "webui_state",
            home / "webui-mvp",
        ])
    except Exception:
        pass

    user_home = Path.home()
    dirs.extend([
        user_home / ".hermes" / "webui_state",
        user_home / ".hermes" / "webui-mvp",
        user_home / "hermes-webui" / "webui_state",
    ])

    seen: set[Path] = set()
    unique: list[Path] = []
    for d in dirs:
        try:
            r = d.expanduser().resolve()
        except Exception:
            r = d.expanduser()
        if r not in seen:
            unique.append(r)
            seen.add(r)
    return unique


def read_webui_last_workspace() -> str | None:
    """Read the WebUI's last workspace if present and still valid."""
    for state_dir in _candidate_state_dirs():
        path = state_dir / "last_workspace.txt"
        try:
            raw = path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            continue
        except Exception:
            continue
        normalized, _err = normalize_workspace_path(raw)
        if normalized:
            return normalized
    return None


def read_webui_workspaces() -> list[dict[str, str]]:
    """Return validated WebUI workspace entries from known state locations."""
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for state_dir in _candidate_state_dirs():
        path = state_dir / "workspaces.json"
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            continue
        except Exception:
            continue
        if not isinstance(raw, list):
            continue
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            normalized, _err = normalize_workspace_path(str(entry.get("path") or ""))
            if not normalized or normalized in seen:
                continue
            name = str(entry.get("name") or Path(normalized).name or normalized)
            result.append({"name": name, "path": normalized})
            seen.add(normalized)
    return result


def default_workspace() -> str:
    """Best-effort workspace fallback for gateway sessions."""
    last = read_webui_last_workspace()
    if last:
        return last
    workspaces = read_webui_workspaces()
    if workspaces:
        return workspaces[0]["path"]
    raw = os.environ.get("TERMINAL_CWD") or os.getcwd()
    normalized, _err = normalize_workspace_path(raw)
    return normalized or str(Path.home())


def format_workspace_list(workspaces: Iterable[dict[str, str]]) -> str:
    """Render a compact markdown list of workspace entries."""
    rows = []
    for i, entry in enumerate(workspaces, start=1):
        name = entry.get("name") or Path(entry.get("path", "")).name or "Workspace"
        path = entry.get("path") or ""
        rows.append(f"{i}. {name}: `{path}`")
    return "\n".join(rows)
