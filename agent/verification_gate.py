"""Postcondition verification helpers for delegated or multi-agent work.

The verifier is intentionally small and deterministic: it checks externally
observable claims (files exist, commands passed) and returns a structured verdict
that a parent agent can inspect before reporting success.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable


def _resolve_required_path(raw: str, root: str | Path | None) -> tuple[Path, bool]:
    path = Path(str(raw)).expanduser()
    if root is None:
        return path.resolve(), True
    root_path = Path(root).expanduser().resolve()
    if not path.is_absolute():
        path = root_path / path
    resolved = path.resolve()
    return resolved, (resolved == root_path or root_path in resolved.parents)


def verify_postconditions(
    *,
    summary: str,
    required_paths: Iterable[str] | None = None,
    command_results: Iterable[dict[str, Any]] | None = None,
    root: str | Path | None = None,
) -> dict[str, Any]:
    """Return a fail-closed verdict for claimed task completion."""

    failures: list[dict[str, Any]] = []
    paths_verified: list[str] = []
    commands_verified: list[dict[str, Any]] = []

    for raw in required_paths or []:
        path, in_root = _resolve_required_path(str(raw), root)
        if not in_root:
            failures.append({"kind": "path_outside_root", "path": str(path)})
        elif not path.exists():
            failures.append({"kind": "missing_path", "path": str(path)})
        else:
            paths_verified.append(str(path))

    for result in command_results or []:
        command = str(result.get("command") or result.get("canonical_command") or "")
        try:
            exit_code = int(result.get("exit_code", 1))
        except (TypeError, ValueError):
            exit_code = 1
        status = "passed" if exit_code == 0 else "failed"
        record = {
            "command": command,
            "exit_code": exit_code,
            "status": status,
            "output": str(result.get("output") or result.get("output_summary") or "")[:1000],
        }
        commands_verified.append(record)
        if exit_code != 0:
            failures.append({"kind": "command_failed", "command": command, "exit_code": exit_code})

    return {
        "passed": not failures,
        "summary": str(summary or ""),
        "failures": failures,
        "evidence": {
            "paths_verified": paths_verified,
            "commands_verified": commands_verified,
        },
    }
