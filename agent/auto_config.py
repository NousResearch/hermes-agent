"""Shared base for agent-driven configuration tools.

The ``rules_configure`` tool needs CRUD + validation + path-safety
primitives. This module keeps that logic in one place.

Guarantees:

* Files are written under the active profile's ``rules/`` directory,
  never elsewhere.
* Frontmatter is validated before commit. A failed validation returns
  a structured error, not a stack trace.
* Existing files are never silently overwritten -- the agent must pass
  ``overwrite=True`` or use the ``update`` action.
* Writes go through :func:`safe_write`, which uses ``os.replace`` for
  atomicity on Windows and POSIX.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


# --- Errors ---------------------------------------------------------------


class AutoConfigError(Exception):
    """Base class for all auto-config failures."""

    code: str = "auto_config_error"


class NotFoundError(AutoConfigError):
    code = "not_found"


class AlreadyExistsError(AutoConfigError):
    code = "already_exists"


class ValidationError(AutoConfigError):
    code = "validation_error"


class PermissionDeniedError(AutoConfigError):
    code = "permission_denied"


# --- Result dataclass -----------------------------------------------------


@dataclass
class ConfigResult:
    """Returned to the agent after every CRUD action."""

    action: str
    path: str
    ok: bool = True
    message: str = ""

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "path": self.path,
            "ok": self.ok,
            "message": self.message,
        }


# --- Path safety ----------------------------------------------------------


_SAFE_NAME = re.compile(r"^[A-Za-z0-9_\-./]+$")


def safe_path(base_dir: Path, name: str, suffix: str) -> Path:
    """Resolve ``name`` under ``base_dir`` and reject traversal.

    Allows forward slashes for subdirectories (``ui/skills-router``)
    but rejects ``..``, absolute paths, and characters outside
    ``[A-Za-z0-9_-./]``.
    """
    if not name or not _SAFE_NAME.match(name):
        raise ValidationError(
            f"Invalid name {name!r}: only letters, digits, '_', '-', '.', '/' allowed"
        )
    if name.startswith("/") or name.startswith("~"):
        raise ValidationError(f"Invalid name {name!r}: must be relative")
    if ".." in name.split("/"):
        raise ValidationError(f"Invalid name {name!r}: '..' not allowed")
    candidate = (base_dir / name).resolve()
    base_resolved = base_dir.resolve()
    try:
        candidate.relative_to(base_resolved)
    except ValueError as exc:
        raise ValidationError(
            f"Path {candidate} escapes base {base_resolved}"
        ) from exc
    if not str(candidate).endswith(suffix):
        candidate = candidate.with_suffix(suffix)
    return candidate


def safe_write(path: Path, content: str) -> None:
    """Atomic write. Creates parent dirs. Refuses to leave a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(content, encoding="utf-8")
        os.replace(tmp, path)
    except Exception:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise


# --- Frontmatter helpers (rules) ------------------------------------------


def render_frontmatter(meta: dict, body: str) -> str:
    """Wrap a body in YAML frontmatter; returns the full file content."""
    if not meta:
        return body
    dumped = yaml.safe_dump(meta, sort_keys=False, allow_unicode=True).strip()
    return f"---\n{dumped}\n---\n\n{body.rstrip()}\n"


def validate_rule_frontmatter(meta: dict) -> None:
    """Raise ``ValidationError`` on malformed rule frontmatter."""
    if "description" in meta and not isinstance(meta["description"], str):
        raise ValidationError("'description' must be a string")
    if "alwaysApply" in meta and not isinstance(meta["alwaysApply"], bool):
        raise ValidationError("'alwaysApply' must be a boolean")
    if "globs" in meta:
        globs = meta["globs"]
        if not isinstance(globs, list) or not all(isinstance(g, str) for g in globs):
            raise ValidationError("'globs' must be a list of strings")


# --- Listing --------------------------------------------------------------


def list_entries(base_dir: Path, suffix: str) -> list[dict]:
    """Walk ``base_dir`` and return one entry per matching file."""
    if not base_dir.exists():
        return []
    entries: list[dict] = []
    for path in sorted(base_dir.rglob(f"*{suffix}")):
        try:
            rel = path.relative_to(base_dir)
            entries.append(
                {
                    "name": str(rel.with_suffix("")),
                    "path": str(path),
                    "size": path.stat().st_size,
                }
            )
        except OSError:
            continue
    return entries
