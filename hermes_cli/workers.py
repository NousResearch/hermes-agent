"""Manage worker profiles without launching agents or resolving credentials."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


def build_workers_parser(subparsers) -> None:
    parser = subparsers.add_parser("workers", help="Configure delegation worker profiles")
    commands = parser.add_subparsers(dest="workers_command")
    listing = commands.add_parser("list", aliases=["ls"], help="Discover configured workers")
    listing.add_argument("--json", action="store_true", help="Print the discovery catalog as JSON")
    inspect = commands.add_parser("inspect", help="Inspect one worker profile")
    inspect.add_argument("name")
    inspect.add_argument("--json", action="store_true")
    commands.add_parser("validate", help="Validate worker configuration without launching models")
    configure = commands.add_parser("set", help="Create or replace one profile from a YAML mapping")
    configure.add_argument("name")
    configure.add_argument("--file", required=True, type=Path)
    default = commands.add_parser("default", help="Select a default worker profile")
    default.add_argument("name", help="Configured profile name, or '-' to inherit legacy settings")
    parser.set_defaults(func=cmd_workers)


def _catalog(config: dict) -> dict:
    from agent.delegation_model_routing import discover_workers

    return discover_workers(config.get("delegation", {}))


def _validate(config: dict) -> None:
    from agent.delegation_model_routing import profile_config_errors

    errors = profile_config_errors(config.get("delegation", {}))
    if errors:
        raise ValueError("\n".join(errors))


def _write_profile(args: argparse.Namespace) -> None:
    import yaml
    from hermes_cli.config import _CONFIG_LOCK, is_managed, read_raw_config, save_config

    if is_managed():
        raise ValueError("Worker configuration is managed; change it through your administrator.")
    with _CONFIG_LOCK:
        config = copy.deepcopy(read_raw_config() or {})
        delegation = config.setdefault("delegation", {})
        if not isinstance(delegation, dict):
            raise ValueError("delegation must be a mapping before configuring workers")
        if args.workers_command == "set":
            try:
                profile = yaml.safe_load(args.file.read_text(encoding="utf-8"))
            except yaml.YAMLError as exc:
                raise ValueError("The profile file is not valid YAML") from exc
            if not isinstance(profile, dict):
                raise ValueError("The profile file must contain one YAML mapping")
            profiles = delegation.setdefault("profiles", {})
            if not isinstance(profiles, dict):
                raise ValueError("delegation.profiles must be a mapping")
            profiles[args.name] = profile
        else:
            delegation["default_profile"] = None if args.name == "-" else args.name
        _validate(config)
        save_config(config)
    print("Worker configuration saved. Existing worker conversations retain their original instructions.")


def cmd_workers(args: argparse.Namespace) -> None:
    from hermes_cli.config import load_config

    command = args.workers_command or "list"
    try:
        if command in {"set", "default"}:
            _write_profile(args)
            return
        config = load_config()
        _validate(config)
        if command == "validate":
            print("Worker configuration is valid. Provider availability has not been probed.")
            return
        catalog = _catalog(config)
        if command == "inspect":
            entries = catalog["profiles"]
            selected = next((entry for entry in entries if entry["name"] == args.name), None)
            if selected is None:
                raise ValueError(f"Unknown worker profile: {args.name}")
            catalog = selected
        # JSON keeps unknown availability and capabilities explicit, and is also
        # useful to scripts invoking Hermes without importing agent internals.
        print(json.dumps(catalog, indent=2, ensure_ascii=False))
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"Worker configuration error: {exc}")
        raise SystemExit(1) from exc
