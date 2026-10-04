"""Deterministic Agent OS rule installer and readback helpers."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

MANAGED_BEGIN = "<!-- AGENT_OS_MANAGED_BLOCK_BEGIN -->"
MANAGED_END = "<!-- AGENT_OS_MANAGED_BLOCK_END -->"
VERSION_RE = re.compile(r"^AGENT_OS_RULE_VERSION:\s*(?P<value>\S+)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class TargetSpec:
    name: str
    command: str | None
    rule_path: str | None
    config_path: str | None = None
    supported: bool = True
    reason: str | None = None


def target_specs(home: Path) -> dict[str, TargetSpec]:
    return {
        "codex": TargetSpec(
            name="codex",
            command="codex",
            rule_path=str(home / ".codex" / "AGENTS.md"),
            config_path=str(home / ".codex" / "config.toml"),
        ),
        "claude": TargetSpec(
            name="claude",
            command="claude",
            rule_path=str(home / ".claude" / "CLAUDE.md"),
            config_path=str(home / ".claude"),
        ),
        "cursor": TargetSpec(
            name="cursor",
            command="cursor",
            rule_path=None,
            config_path=str(home / ".cursor" / "cli-config.json"),
            supported=False,
            reason="Only project-local .cursorrules or .cursor/rules/*.mdc are proven; no deterministic per-user rule file is used here.",
        ),
        "gemini": TargetSpec(
            name="gemini",
            command="gemini",
            rule_path=None,
            config_path=None,
            supported=False,
            reason="No stable local rule location was proven in this machine state.",
        ),
        "hermes": TargetSpec(
            name="hermes",
            command="hermes",
            rule_path=None,
            config_path=str(home / ".hermes" / "config.yaml"),
            supported=False,
            reason="Hermes loads repository SOUL.md + AGENTS.md and requires runtime activation, not a per-user rule file install.",
        ),
    }


def load_rule_text(repo_root: Path | str) -> str:
    repo_root = Path(repo_root).resolve()
    return (repo_root / "AGENT_OS_CORE.md").read_text(encoding="utf-8").strip() + "\n"


def rule_version(rule_text: str) -> str | None:
    match = VERSION_RE.search(rule_text)
    return match.group("value") if match else None


def render_managed_block(rule_text: str) -> str:
    return f"{MANAGED_BEGIN}\n{rule_text.strip()}\n{MANAGED_END}\n"


def merge_rule(existing: str, block: str) -> str:
    existing = existing or ""
    if MANAGED_BEGIN in existing and MANAGED_END in existing:
        pattern = re.compile(
            rf"{re.escape(MANAGED_BEGIN)}.*?{re.escape(MANAGED_END)}\n?",
            re.DOTALL,
        )
        merged = pattern.sub(block, existing, count=1)
    elif existing.strip():
        merged = existing.rstrip() + "\n\n" + block
    else:
        merged = block
    return merged


def readback_rule(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "exists": False,
            "managed_block_present": False,
            "rule_loaded": False,
            "rule_version": None,
            "content": "",
        }
    content = path.read_text(encoding="utf-8")
    block_present = MANAGED_BEGIN in content and MANAGED_END in content
    managed_content = _extract_managed_content(content)
    return {
        "exists": True,
        "managed_block_present": block_present,
        "rule_loaded": block_present,
        "rule_version": rule_version(managed_content or content),
        "content": content,
    }


def audit_targets(
    *,
    repo_root: Path | str = ".",
    home: Path | str | None = None,
    command_paths: Mapping[str, str | None] | None = None,
    process_snapshot: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    repo_root = Path(repo_root).resolve()
    home_path = Path(home).resolve() if home is not None else Path.home()
    specs = target_specs(home_path)
    resolved_commands = (
        dict(command_paths)
        if command_paths is not None
        else {name: shutil.which(spec.command) if spec.command else None for name, spec in specs.items()}
    )
    processes = list(process_snapshot) if process_snapshot is not None else collect_process_snapshot()

    results = []
    for name, spec in specs.items():
        command_path = resolved_commands.get(name)
        rule_path = Path(spec.rule_path).resolve() if spec.rule_path else None
        config_path = Path(spec.config_path).resolve() if spec.config_path else None
        if name == "cursor":
            results.append(
                {
                    "target": name,
                    "status": "UNSUPPORTED",
                    "command_path": command_path,
                    "rule_path": None,
                    "config_path": str(config_path) if config_path else None,
                    "actual_rule_locations": [".cursorrules", ".cursor/rules/*.mdc"],
                    "reason": spec.reason,
                }
            )
            continue
        if name == "gemini":
            status = "NOT_INSTALLED" if command_path is None else "UNSUPPORTED"
            results.append(
                {
                    "target": name,
                    "status": status,
                    "command_path": command_path,
                    "rule_path": None,
                    "config_path": None,
                    "actual_rule_locations": [],
                    "reason": spec.reason,
                }
            )
            continue
        if name == "hermes":
            active = any(
                "hermes" in str(item.get("name") or "").lower()
                or "hermes" in str(item.get("command_line") or "").lower()
                for item in processes
            )
            results.append(
                {
                    "target": name,
                    "status": "REQUIRES_RESTART" if active else "SUPPORTED",
                    "command_path": command_path,
                    "rule_path": str(repo_root / "AGENTS.md"),
                    "config_path": str(config_path) if config_path else None,
                    "actual_rule_locations": [str(repo_root / "SOUL.md"), str(repo_root / "AGENTS.md")],
                    "reason": (
                        "Runtime activation must be deferred until active Hermes work is quiescent."
                        if active
                        else spec.reason
                    ),
                }
            )
            continue
        if command_path is None:
            results.append(
                {
                    "target": name,
                    "status": "NOT_INSTALLED",
                    "command_path": None,
                    "rule_path": str(rule_path) if rule_path else None,
                    "config_path": str(config_path) if config_path else None,
                    "actual_rule_locations": [str(rule_path)] if rule_path else [],
                    "reason": "Executable not found on PATH.",
                }
            )
            continue
        rule_state = readback_rule(rule_path) if rule_path else None
        status = "ALREADY_CONFIGURED" if rule_state and rule_state["managed_block_present"] else "SAFE_TO_INSTALL"
        results.append(
            {
                "target": name,
                "status": status,
                "command_path": command_path,
                "rule_path": str(rule_path) if rule_path else None,
                "config_path": str(config_path) if config_path else None,
                "actual_rule_locations": [str(rule_path)] if rule_path else [],
                "reason": None,
            }
        )
    return results


def apply_rule(
    target: str,
    *,
    repo_root: Path | str = ".",
    home: Path | str | None = None,
) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    home_path = Path(home).resolve() if home is not None else Path.home()
    spec = target_specs(home_path).get(target)
    if spec is None:
        raise ValueError(f"Unknown target: {target}")
    if not spec.supported or not spec.rule_path:
        return {
            "target": target,
            "status": "UNSUPPORTED",
            "rule_path": spec.rule_path,
            "applied": False,
            "reason": spec.reason,
        }

    command_path = shutil.which(spec.command) if spec.command else None
    if command_path is None:
        return {
            "target": target,
            "status": "NOT_INSTALLED",
            "rule_path": spec.rule_path,
            "applied": False,
            "reason": "Executable not found on PATH.",
        }

    rule_text = load_rule_text(repo_root)
    block = render_managed_block(rule_text)
    path = Path(spec.rule_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    before = path.read_text(encoding="utf-8") if path.exists() else ""
    merged = merge_rule(before, block)
    path.write_text(merged, encoding="utf-8")
    first_readback = readback_rule(path)
    second_merged = merge_rule(path.read_text(encoding="utf-8"), block)
    idempotent_second_apply = second_merged == path.read_text(encoding="utf-8")
    if not idempotent_second_apply:
        path.write_text(second_merged, encoding="utf-8")
    preserved = before.strip() == "" or before.strip() in path.read_text(encoding="utf-8")
    return {
        "target": target,
        "status": "ALREADY_CONFIGURED" if first_readback["managed_block_present"] else "SAFE_TO_INSTALL",
        "rule_path": str(path),
        "rule_version": rule_version(rule_text),
        "rule_loaded": first_readback["rule_loaded"],
        "existing_user_content_preserved": preserved,
        "idempotent_second_apply": idempotent_second_apply,
        "applied": True,
    }


def collect_process_snapshot() -> list[dict[str, Any]]:
    if os.name != "nt":
        return []
    command = (
        "Get-CimInstance Win32_Process | "
        "Select-Object Name,ProcessId,ExecutablePath,CommandLine | ConvertTo-Json -Depth 4"
    )
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    payload = result.stdout.strip()
    if not payload:
        return []
    data = json.loads(payload)
    if isinstance(data, dict):
        data = [data]
    snapshot = []
    for item in data:
        snapshot.append(
            {
                "name": item.get("Name"),
                "process_id": item.get("ProcessId"),
                "path": item.get("ExecutablePath"),
                "command_line": item.get("CommandLine"),
            }
        )
    return snapshot


def _extract_managed_content(content: str) -> str:
    match = re.search(
        rf"{re.escape(MANAGED_BEGIN)}\n?(?P<body>.*?){re.escape(MANAGED_END)}",
        content,
        re.DOTALL,
    )
    return match.group("body").strip() if match else ""
