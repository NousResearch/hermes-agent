#!/usr/bin/env python3
"""Merge operator stack defaults into ~/.hermes/config.yaml."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STACK = REPO_ROOT / "config" / "operator" / "hakua-stack.yaml"


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _resolve_env_placeholders(value: Any, hermes_home: Path) -> Any:
    if isinstance(value, str):
        return value.replace("$HERMES_HOME", str(hermes_home).replace("\\", "/"))
    if isinstance(value, list):
        return [_resolve_env_placeholders(item, hermes_home) for item in value]
    if isinstance(value, dict):
        return {k: _resolve_env_placeholders(v, hermes_home) for k, v in value.items()}
    return value


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8-sig") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping in {path}")
    return payload


def save_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def apply_stack(
    *,
    stack_file: Path,
    config_path: Path,
    hermes_home: Path,
    dry_run: bool,
) -> dict[str, Any]:
    current = load_yaml(config_path)
    overlay = _resolve_env_placeholders(load_yaml(stack_file), hermes_home)

    # Preserve explicit Bitwarden project_id if already configured.
    existing_project = (
        current.get("secrets", {})
        .get("bitwarden", {})
        .get("project_id", "")
    )
    if existing_project:
        overlay.setdefault("secrets", {}).setdefault("bitwarden", {})["project_id"] = existing_project

    # Allow .env to override memory vault path when present.
    env_file = hermes_home / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("MEMORY_VAULT_LOCAL_PATH="):
                local_path = line.split("=", 1)[1].strip().strip('"')
                if local_path:
                    overlay.setdefault("memory_vault", {})["local_path"] = local_path
            if line.startswith("MEMORY_VAULT_REMOTE="):
                remote = line.split("=", 1)[1].strip().strip('"')
                if remote:
                    overlay.setdefault("memory_vault", {})["remote"] = remote

    merged = _deep_merge(current, overlay)
    if not dry_run:
        save_yaml(config_path, merged)
    return merged


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply operator stack config overlay.")
    parser.add_argument("--stack-file", default=str(DEFAULT_STACK))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    try:
        from hermes_constants import get_hermes_home
    except Exception:
        get_hermes_home = lambda: Path.home() / ".hermes"  # type: ignore[assignment, misc]

    hermes_home = Path(get_hermes_home())
    config_path = hermes_home / "config.yaml"
    stack_file = Path(args.stack_file)

    if not stack_file.is_absolute():
        stack_file = (REPO_ROOT / stack_file).resolve()

    merged = apply_stack(
        stack_file=stack_file,
        config_path=config_path,
        hermes_home=hermes_home,
        dry_run=args.dry_run,
    )

    model = merged.get("model", {})
    delegation = merged.get("delegation", {})
    print("Applied operator stack:")
    print(f"  main: {model.get('provider')} / {model.get('default')}")
    print(f"  sub:  {delegation.get('provider')} / {delegation.get('model')}")
    print(f"  memory.provider: {merged.get('memory', {}).get('provider')}")
    print(f"  secrets.bitwarden.enabled: {merged.get('secrets', {}).get('bitwarden', {}).get('enabled')}")
    print(f"  memory_vault.enabled: {merged.get('memory_vault', {}).get('enabled')}")
    if args.dry_run:
        print("(dry-run: config not written)")
    else:
        print(f"  config: {config_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
