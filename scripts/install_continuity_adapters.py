#!/usr/bin/env python3
"""Render project adapters and optionally install the sandbox Hermes adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
from pathlib import Path
from typing import Any

import yaml

from continuity_common import ContinuityError, atomic_write_text, load_json


HERMES_MANIFEST = "continuity-adapter-manifest.json"


CLAUDE_EVENT_NAMES = {
    "session_start": "SessionStart",
    "user_prompt": "UserPromptSubmit",
    "pre_tool": "PreToolUse",
    "post_tool": "PostToolUse",
    "stop": "Stop",
    "session_end": "SessionEnd",
}
CODEX_EVENT_NAMES = {
    "session_start": "SessionStart",
    "user_prompt": "UserPromptSubmit",
    "pre_tool": "PreToolUse",
    "post_tool": "PostToolUse",
    "stop": "Stop",
}


def _hook_block(
    command: str, timeout: int, *, matcher: str | None = None
) -> list[dict[str, Any]]:
    block: dict[str, Any] = {
        "hooks": [{"type": "command", "command": command, "timeout": timeout}]
    }
    if matcher:
        block["matcher"] = matcher
    return [block]


def render_project_adapters(repo_root: Path) -> dict[Path, str]:
    contract = load_json(repo_root / ".continuity/adapters.json")
    timeout = int(contract["timeout_seconds"])
    claude: dict[str, Any] = {"hooks": {}}
    for event in contract["surfaces"]["claude"]:
        command = (
            f'python3 "$CLAUDE_PROJECT_DIR/.specify/events.py" {event} --surface claude'
        )
        matcher = ".*" if event in {"pre_tool", "post_tool"} else None
        claude["hooks"][CLAUDE_EVENT_NAMES[event]] = _hook_block(
            command, timeout, matcher=matcher
        )
    codex: dict[str, Any] = {"hooks": {}}
    for event in contract["surfaces"]["codex"]:
        command = f"python3 .specify/events.py {event} --surface codex"
        matcher = ".*" if event in {"pre_tool", "post_tool"} else None
        codex["hooks"][CODEX_EVENT_NAMES[event]] = _hook_block(
            command, timeout, matcher=matcher
        )
    return {
        repo_root / ".claude/settings.json": json.dumps(
            claude, indent=2, sort_keys=True
        )
        + "\n",
        repo_root / ".codex/hooks.json": json.dumps(codex, indent=2, sort_keys=True)
        + "\n",
    }


def render_hermes_config(
    repo_root: Path, existing: dict[str, Any] | None = None
) -> str:
    contract = load_json(repo_root / ".continuity/adapters.json")
    timeout = int(contract["timeout_seconds"])
    config = dict(existing or {})
    hooks = dict(config.get("hooks") or {})
    entry = repo_root / ".specify/events.py"
    for event in contract["surfaces"]["hermes"]:
        hook = {
            "command": f"python3 {shlex.quote(str(entry))} {event} --surface hermes",
            "timeout": timeout,
        }
        if event == "pre_tool_call":
            hook["fail_closed"] = True
        hooks[event] = [hook]
    config["hooks"] = hooks
    return yaml.safe_dump(config, sort_keys=False, allow_unicode=True)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _load_yaml_mapping(path: Path) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        return {}, ""
    text = path.read_text(encoding="utf-8")
    loaded = yaml.safe_load(text) or {}
    if not isinstance(loaded, dict):
        raise ContinuityError(f"existing YAML is not a mapping: {path}")
    return loaded, text


def _managed_hermes_hooks(repo_root: Path) -> dict[str, Any]:
    rendered = yaml.safe_load(render_hermes_config(repo_root)) or {}
    return dict(rendered.get("hooks") or {})


def _before_hook_projection(manifest: dict[str, Any]) -> dict[str, Any]:
    installed = manifest.get("installed_hooks")
    before = manifest.get("before_hooks")
    absent = manifest.get("absent_before")
    if (
        not isinstance(installed, dict)
        or not isinstance(before, dict)
        or not isinstance(absent, list)
    ):
        raise ContinuityError("Hermes adapter manifest has an invalid before-image")
    return {
        event: before[event] if event in before else None
        for event in installed
        if event in before or event in absent
    }


def install_hermes_adapter(repo_root: Path, hermes_home: Path) -> dict[str, Any]:
    target = hermes_home.resolve() / "config.yaml"
    manifest_path = hermes_home.resolve() / HERMES_MANIFEST
    existing, before_text = _load_yaml_mapping(target)
    installed_hooks = _managed_hermes_hooks(repo_root)
    hooks = dict(existing.get("hooks") or {})

    if manifest_path.is_file():
        prior = load_json(manifest_path)
        if prior.get("status") in {"PREPARED", "INSTALLED"}:
            if prior.get("repo_root") != str(repo_root) or prior.get("target") != str(
                target
            ):
                raise ContinuityError("Hermes adapter is owned by another repository")
            installed_hooks = prior.get("installed_hooks")
            if not isinstance(installed_hooks, dict):
                raise ContinuityError("Hermes adapter manifest is malformed")
            current = {name: hooks.get(name) for name in installed_hooks}
            if current != prior.get("installed_hooks"):
                if prior.get(
                    "status"
                ) != "PREPARED" or current != _before_hook_projection(prior):
                    raise ContinuityError(
                        "managed Hermes hooks changed after installation"
                    )
            else:
                if prior.get("status") == "PREPARED":
                    prior["status"] = "INSTALLED"
                    atomic_write_text(
                        manifest_path,
                        json.dumps(prior, indent=2, sort_keys=True) + "\n",
                    )
                return {
                    "hermes_config": str(target),
                    "installed": True,
                    "changed": prior.get("status") == "PREPARED",
                }

    before_hooks = {name: hooks[name] for name in installed_hooks if name in hooks}
    absent_before = sorted(name for name in installed_hooks if name not in hooks)
    after_text = render_hermes_config(repo_root, existing)
    manifest = {
        "schema_version": 1,
        "status": "PREPARED",
        "repo_root": str(repo_root),
        "target": str(target),
        "before_sha256": _sha256_text(before_text),
        "after_sha256": _sha256_text(after_text),
        "before_hooks": before_hooks,
        "absent_before": absent_before,
        "installed_hooks": installed_hooks,
    }
    atomic_write_text(
        manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    atomic_write_text(target, after_text)
    manifest["status"] = "INSTALLED"
    atomic_write_text(
        manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return {"hermes_config": str(target), "installed": True, "changed": True}


def rollback_hermes_adapter(
    repo_root: Path, hermes_home: Path, *, apply: bool
) -> dict[str, Any]:
    manifest_path = hermes_home.resolve() / HERMES_MANIFEST
    manifest = load_json(manifest_path)
    if manifest.get("repo_root") != str(repo_root):
        raise ContinuityError("rollback manifest belongs to another repository")
    if manifest.get("status") not in {"PREPARED", "INSTALLED"}:
        raise ContinuityError("Hermes adapter is not in INSTALLED state")
    target = Path(str(manifest.get("target"))).resolve()
    if target != hermes_home.resolve() / "config.yaml":
        raise ContinuityError("rollback manifest target is outside Hermes home")
    config, _ = _load_yaml_mapping(target)
    hooks = dict(config.get("hooks") or {})
    installed = manifest.get("installed_hooks")
    if not isinstance(installed, dict):
        raise ContinuityError("rollback manifest has invalid installed hooks")
    current = {name: hooks.get(name) for name in installed}
    before_projection = _before_hook_projection(manifest)
    already_before = current == before_projection
    if current != installed and not already_before:
        raise ContinuityError("managed Hermes hooks changed after installation")
    if not apply:
        return {"rollback_valid": True, "applied": False, "target": str(target)}

    if not already_before:
        before_hooks = manifest["before_hooks"]
        absent_before = manifest["absent_before"]
        for event in installed:
            if event in before_hooks:
                hooks[event] = before_hooks[event]
            elif event in absent_before:
                hooks.pop(event, None)
        config["hooks"] = hooks
        restored = yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
        atomic_write_text(target, restored)
    else:
        restored = yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
    manifest["status"] = "ROLLED_BACK"
    manifest["restored_sha256"] = _sha256_text(restored)
    atomic_write_text(
        manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return {"rollback_valid": True, "applied": True, "target": str(target)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--hermes-home", type=Path)
    parser.add_argument("--apply-hermes", action="store_true")
    parser.add_argument("--rollback-dry-run", action="store_true")
    parser.add_argument("--rollback-apply", action="store_true")
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()
    try:
        mismatches: list[str] = []
        for path, expected in render_project_adapters(repo_root).items():
            if not path.is_file() or path.read_text(encoding="utf-8") != expected:
                mismatches.append(str(path.relative_to(repo_root)))
        if args.check and mismatches:
            raise ContinuityError(
                "generated adapters are stale: " + ", ".join(mismatches)
            )
        modes = sum((args.apply_hermes, args.rollback_dry_run, args.rollback_apply))
        if modes > 1:
            raise ContinuityError("choose only one Hermes mutation or rollback mode")
        if modes:
            if args.hermes_home is None:
                raise ContinuityError("--hermes-home is required for Hermes operations")
            if args.apply_hermes:
                result = install_hermes_adapter(repo_root, args.hermes_home)
            else:
                result = rollback_hermes_adapter(
                    repo_root, args.hermes_home, apply=args.rollback_apply
                )
            print(json.dumps(result, sort_keys=True))
        else:
            print(json.dumps({"valid": not mismatches, "mismatches": mismatches}))
        return 0
    except (
        ContinuityError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
        yaml.YAMLError,
    ) as exc:
        print(json.dumps({"valid": False, "errors": [str(exc)]}))
        return 2


if __name__ == "__main__":
    sys.exit(main())
