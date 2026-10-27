"""``hermes provider`` — inspect and remove configured inference providers.

Add is a guided flow that writes to several stores; remove had no equivalent. This
command reuses :mod:`hermes_cli.provider_removal` so the interactive
"Remove a saved custom provider" row and the CLI share one implementation.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Optional

from hermes_cli.provider_removal import (
    build_removal_plan,
    execute_removal,
    format_plan,
)


def _parse_assignments(values: list[str], flag: str) -> dict[str, str]:
    """Parse repeated ``PATH=VALUE`` arguments into a mapping."""
    parsed: dict[str, str] = {}
    for raw in values:
        path, sep, value = raw.partition("=")
        path, value = path.strip(), value.strip()
        if not sep or not path or not value:
            raise ValueError(f"{flag} expects PATH=PROVIDER, got {raw!r}")
        parsed[path] = value
    return parsed


def _emit(payload: dict[str, Any], as_json: bool, lines: list[str]) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for line in lines:
            print(line)


def provider_command(args) -> int:
    """Entry point for ``hermes provider``."""
    action = getattr(args, "provider_action", None)
    if action is None:
        # No sub-action: show this command's own help rather than the top-level one.
        print("usage: hermes provider remove <name> [options]", file=sys.stderr)
        print("       hermes provider remove --dry-run <name>", file=sys.stderr)
        return 2
    if action != "remove":
        print(f"hermes provider: unknown action '{action}'", file=sys.stderr)
        return 2
    return _remove_command(args)


def _remove_command(args) -> int:
    from hermes_cli.config import load_config, save_config

    name = str(getattr(args, "name", "") or "").strip()
    as_json = bool(getattr(args, "json", False))
    dry_run = bool(getattr(args, "dry_run", False))

    if not name:
        print("hermes provider remove: a provider name is required", file=sys.stderr)
        return 2

    try:
        reassign = _parse_assignments(list(getattr(args, "reassign", []) or []), "--reassign")
        clear = [entry.strip() for entry in (getattr(args, "clear", []) or []) if entry.strip()]
    except ValueError as exc:
        print(f"hermes provider remove: {exc}", file=sys.stderr)
        return 2

    config = load_config()
    plan = build_removal_plan(config, name)

    if not plan.found and not plan.references and not plan.credential_pool_keys:
        payload = {"provider": name, "found": False, "removed": False}
        _emit(payload, as_json, [f"No provider named '{name}' is configured — nothing to remove."])
        return 1

    lines = format_plan(plan, dry_run=dry_run)

    if dry_run:
        payload = {
            "provider": plan.provider,
            "found": plan.found,
            "dry_run": True,
            "stores": plan.stores,
            "credential_pool_keys": plan.credential_pool_keys,
            "env_vars": plan.exposed_env_vars,
            "references": [ref.describe() for ref in plan.references],
        }
        _emit(payload, as_json, lines + ["", "Dry run: nothing was written."])
        return 0

    active = _active_provider(config)
    if plan.found and _mentions(plan.provider, active) and "model.provider" not in reassign:
        lines.append("")
        lines.append(
            f"'{plan.provider}' is the active model provider. Pass "
            f"--reassign model.provider=<other> to switch before removing it.")
        _emit({"provider": plan.provider, "found": True, "removed": False,
               "error": "active_provider"}, as_json, lines)
        return 2

    unresolved = [ref for ref in plan.references if ref.path not in reassign and ref.path not in clear]
    if unresolved and not getattr(args, "force", False):
        lines.append("")
        lines.append("These references still point at the provider:")
        lines.extend(f"  {ref.path} -> {ref.provider}" for ref in unresolved)
        lines.append("Resolve each with --reassign PATH=PROVIDER or --clear PATH, or pass --force.")
        _emit({"provider": plan.provider, "found": True, "removed": False,
               "unresolved": [ref.path for ref in unresolved]}, as_json, lines)
        return 2

    if getattr(args, "force", False):
        clear = clear + [ref.path for ref in unresolved if ref.path not in clear]

    result = execute_removal(
        config, plan,
        reassign=reassign,
        clear=clear,
        keep_credentials=bool(getattr(args, "keep_credentials", False)),
    )

    if result["stores"]:
        _backup_config()
        save_config(config)

    result["removed"] = bool(result["stores"])
    lines = format_plan(plan, dry_run=False)
    lines.extend(_result_summary(result))
    _emit(result, as_json, lines)
    return 0 if result["removed"] else 1


def _active_provider(config: dict[str, Any]) -> str:
    model = config.get("model")
    if isinstance(model, dict):
        return str(model.get("provider") or "")
    return ""


def _mentions(provider: str, value: str) -> bool:
    left, right = provider.strip().lower(), str(value or "").strip().lower()
    if not left or not right:
        return False
    if left == right:
        return True
    strip = lambda v: v[len("custom:"):] if v.startswith("custom:") else v
    return strip(left) == strip(right)


def _backup_config() -> Optional[str]:
    try:
        from hermes_cli.config import get_config_path
        from hermes_cli.config_backups import backup_config
        path = backup_config(get_config_path(), "provider-remove")
        return str(path) if path else None
    except Exception:
        return None


def _result_summary(result: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    if result.get("stores"):
        lines.append(f"  config.yaml updated: {', '.join(result['stores'])}")
    removed = result.get("credentials_removed") or 0
    if removed:
        keys = ", ".join(result.get("credential_pools_removed", {}))
        lines.append(f"  credentials removed: {removed} ({keys})")
    if result.get("env_vars"):
        lines.append(f"  env vars left untouched: {', '.join(result['env_vars'])}")
    return lines
