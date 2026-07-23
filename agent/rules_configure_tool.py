"""rules_configure -- agent-driven CRUD for the rules/ directory.

Tool schema::

    {
        "name": "rules_configure",
        "description": "Create, read, update, delete, or list rule files
                       in the profile's rules/ directory or a project's
                       .hermes/rules/ directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["create", "read", "update", "delete", "list"],
                },
                "name": {
                    "type": "string",
                    "description": "Rule path under rules/, e.g. "
                                   "'ui/skills-router'. Omit for 'list'.",
                },
                "body": {"type": "string", "description": "Markdown body."},
                "description": {
                    "type": "string",
                    "description": "Short label shown in the rule picker.",
                },
                "always_apply": {
                    "type": "boolean",
                    "description": "Inject into system prompt on every "
                                   "session. Default true when globs is empty.",
                },
                "globs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "File patterns that activate this rule.",
                },
                "overwrite": {
                    "type": "boolean",
                    "description": "Required true to overwrite an existing rule.",
                },
                "scope": {
                    "type": "string",
                    "enum": ["profile", "project"],
                    "description": "'profile' -> ~/.hermes/profiles/<profile>/rules/ "
                                   "'project' -> nearest ./.hermes/rules/ (cwd-first). "
                                   "Default: 'project' if a .hermes/rules/ exists, else 'profile'.",
                },
            },
            "required": ["action"],
        },
    }
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from agent.auto_config import (
    AlreadyExistsError,
    AutoConfigError,
    ConfigResult,
    NotFoundError,
    ValidationError,
    list_entries,
    render_frontmatter,
    safe_path,
    safe_write,
    validate_rule_frontmatter,
)
from agent.rules_loader import (
    discover_project_rules_dirs,
    load_rules,
    parse_frontmatter,
    resolve_rules_dir,
)

logger = logging.getLogger(__name__)


# --- Profile resolution ----------------------------------------------------


def get_active_profile_dir() -> Path:
    """Return the active Hermes profile directory."""
    from hermes_constants import get_hermes_home

    profile = os.environ.get("HERMES_PROFILE", "default").strip() or "default"
    return get_hermes_home() / "profiles" / profile


# --- Scope resolution ------------------------------------------------------


def resolve_rules_dir_for_scope(
    scope: Optional[str], cwd: Optional[Path] = None
) -> Path:
    """Return the rules directory for the requested scope.

    * ``profile`` -> ``~/.hermes/profiles/<profile>/rules/``
    * ``project`` -> nearest ``./.hermes/rules/`` from cwd
    * ``None``    -> ``project`` if one exists, else ``profile``
    """
    if scope == "profile":
        return resolve_rules_dir(get_active_profile_dir())
    if scope == "project":
        dirs = discover_project_rules_dirs(cwd)
        if dirs:
            return dirs[0]
        return (cwd or Path.cwd()) / ".hermes" / "rules"
    # Default: prefer project if one exists
    dirs = discover_project_rules_dirs(cwd)
    if dirs:
        return dirs[0]
    return resolve_rules_dir(get_active_profile_dir())


# --- Tool implementation ---------------------------------------------------


def _list(scope: Optional[str] = None) -> list[dict]:
    rules_dir = resolve_rules_dir_for_scope(scope)
    entries = list_entries(rules_dir, ".md") + list_entries(rules_dir, ".mdc")
    for entry in entries:
        try:
            text = Path(entry["path"]).read_text(encoding="utf-8")
            meta, _ = parse_frontmatter(text)
            entry["description"] = meta.get("description", "")
            entry["alwaysApply"] = bool(meta.get("alwaysApply", False))
            entry["globs"] = list(meta.get("globs", []) or [])
            entry["scope"] = (
                "project" if ".hermes" in Path(rules_dir).parts else "profile"
            )
        except OSError:
            continue
    return entries


def _create_or_update(
    action: str,
    name: str,
    body: str,
    description: Optional[str],
    always_apply: Optional[bool],
    globs: Optional[list[str]],
    overwrite: bool,
    scope: Optional[str] = None,
) -> ConfigResult:
    rules_dir = resolve_rules_dir_for_scope(scope)
    rules_dir.mkdir(parents=True, exist_ok=True)
    target = safe_path(rules_dir, name, ".md")

    meta: dict = {}
    if description is not None:
        meta["description"] = description
    if always_apply is not None:
        meta["alwaysApply"] = always_apply
    if globs is not None:
        meta["globs"] = globs

    validate_rule_frontmatter(meta)

    if target.exists() and action == "create" and not overwrite:
        raise AlreadyExistsError(
            f"Rule {name!r} already exists. Pass overwrite=true or use action=update."
        )

    content = render_frontmatter(meta, body or "")
    safe_write(target, content)
    return ConfigResult(
        action=action,
        path=str(target),
        message=f"Rule {name!r} {action}d ({len(content)} bytes)",
    )


def _read(name: str, scope: Optional[str] = None) -> dict:
    rules_dir = resolve_rules_dir_for_scope(scope)
    target = safe_path(rules_dir, name, ".md")
    if not target.exists():
        target = safe_path(rules_dir, name, ".mdc")
    if not target.exists():
        raise NotFoundError(f"Rule {name!r} not found")
    text = target.read_text(encoding="utf-8")
    meta, body = parse_frontmatter(text)
    return {
        "name": name,
        "path": str(target),
        "frontmatter": meta,
        "body": body,
        "size": len(text),
    }


def _delete(name: str, scope: Optional[str] = None) -> ConfigResult:
    rules_dir = resolve_rules_dir_for_scope(scope)
    target = safe_path(rules_dir, name, ".md")
    if not target.exists():
        target = safe_path(rules_dir, name, ".mdc")
    if not target.exists():
        raise NotFoundError(f"Rule {name!r} not found")
    target.unlink()
    return ConfigResult(
        action="delete",
        path=str(target),
        message=f"Rule {name!r} deleted",
    )


def run(
    action: str,
    name: Optional[str] = None,
    body: Optional[str] = None,
    description: Optional[str] = None,
    always_apply: Optional[bool] = None,
    globs: Optional[list[str]] = None,
    overwrite: bool = False,
    scope: Optional[str] = None,
) -> dict:
    """Public entry point called by the tool dispatcher."""
    try:
        if action == "list":
            return {"ok": True, "entries": _list(scope)}
        if action == "read":
            return {"ok": True, "rule": _read(name or "", scope)}
        if action in ("create", "update"):
            if not name:
                raise ValidationError("'name' is required for create/update")
            return _create_or_update(
                action, name, body or "", description,
                always_apply, globs, overwrite, scope
            ).to_dict()
        if action == "delete":
            if not name:
                raise ValidationError("'name' is required for delete")
            return _delete(name, scope).to_dict()
        raise ValidationError(f"Unknown action {action!r}")
    except AutoConfigError as exc:
        logger.debug("rules_configure failed: %s", exc)
        return {"ok": False, "error_code": exc.code, "error_message": str(exc)}
    except Exception as exc:
        logger.exception("rules_configure unexpected error")
        return {"ok": False, "error_code": "unexpected", "error_message": str(exc)}


def load_active_rules(cwd: Optional[Path] = None) -> list:
    """Convenience wrapper used by prompt_builder."""
    from agent.rules_loader import load_active_rules as _load

    return _load(get_active_profile_dir(), cwd)
