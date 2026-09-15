"""Fail-closed policy for source trees owned by an external release manager."""

from __future__ import annotations

import json
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

SOURCE_UPDATE_POLICY_FILENAME = ".hermes-update-policy.json"
SOURCE_UPDATE_POLICY_SCHEMA = 1


@dataclass(frozen=True)
class SourceUpdatePolicy:
    marker_path: str
    update_command: str
    valid: bool = True
    error: Optional[str] = None


def _invalid(path: Path, reason: str) -> SourceUpdatePolicy:
    return SourceUpdatePolicy(str(path), "", valid=False, error=reason)


def read_source_update_policy(project_root: Path) -> Optional[SourceUpdatePolicy]:
    """Read the checkout-local policy; presence or read errors never permit updates."""
    path = Path(project_root) / SOURCE_UPDATE_POLICY_FILENAME
    try:
        marker_stat = path.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        return _invalid(path, f"marker_presence_unreadable:{type(exc).__name__}")
    if not stat.S_ISREG(marker_stat.st_mode):
        return _invalid(path, "marker_not_regular_file")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return _invalid(path, f"marker_unreadable:{type(exc).__name__}")
    if not isinstance(payload, dict) or type(payload.get("schema")) is not int or payload["schema"] != SOURCE_UPDATE_POLICY_SCHEMA:
        return _invalid(path, "unsupported_marker_schema")
    if payload.get("managed_externally") is not True:
        return _invalid(path, "invalid_managed_externally")
    command = payload.get("update_command", "")
    if not isinstance(command, str) or not command.strip():
        return _invalid(path, "missing_update_command")
    return SourceUpdatePolicy(str(path), command.strip())
