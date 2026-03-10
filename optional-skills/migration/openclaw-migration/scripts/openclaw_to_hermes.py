#!/usr/bin/env python3
"""OpenClaw -> Hermes migration helper.

This script migrates the parts of an OpenClaw user footprint that map cleanly
into Hermes Agent, archives selected unmapped docs for manual review, and
reports exactly what was skipped and why.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import yaml
except Exception:  # pragma: no cover - handled at runtime
    yaml = None


ENTRY_DELIMITER = "\n§\n"
DEFAULT_MEMORY_CHAR_LIMIT = 2200
DEFAULT_USER_CHAR_LIMIT = 1375
SKILL_CATEGORY_DIRNAME = "openclaw-imports"
SKILL_CATEGORY_DESCRIPTION = (
    "Skills migrated from an OpenClaw workspace."
)
SKILL_CONFLICT_MODES = {"skip", "overwrite", "rename"}
SUPPORTED_SECRET_TARGETS = {
    "TELEGRAM_BOT_TOKEN",
}
WORKSPACE_INSTRUCTIONS_FILENAME = "AGENTS" + ".md"
MIGRATION_OPTION_METADATA: Dict[str, Dict[str, str]] = {
    "soul": {
        "label": "SOUL.md",
        "description": "Import the OpenClaw persona file into Hermes.",
    },
    "workspace-agents": {
        "label": "Workspace instructions",
        "description": "Copy the OpenClaw workspace instructions file into a chosen workspace.",
    },
    "memory": {
        "label": "MEMORY.md",
        "description": "Import long-term memory entries into Hermes memories.",
    },
    "user-profile": {
        "label": "USER.md",
        "description": "Import user profile entries into Hermes memories.",
    },
    "messaging-settings": {
        "label": "Messaging settings",
        "description": "Import Hermes-compatible messaging settings such as allowlists and working directory.",
    },
    "secret-settings": {
        "label": "Allowlisted secrets",
        "description": "Import the small allowlist of Hermes-compatible secrets when explicitly enabled.",
    },
    "command-allowlist": {
        "label": "Command allowlist",
        "description": "Merge OpenClaw exec approval patterns into Hermes command_allowlist.",
    },
    "skills": {
        "label": "User skills",
        "description": "Copy OpenClaw skills into ~/.hermes/skills/openclaw-imports/.",
    },
    "tts-assets": {
        "label": "TTS assets",
        "description": "Copy compatible workspace TTS assets into ~/.hermes/tts/.",
    },
    "archive": {
        "label": "Archive unmapped docs",
        "description": "Archive compatible-but-unmapped docs for later manual review.",
    },
}
MIGRATION_PRESETS: Dict[str, set[str]] = {
    "user-data": {
        "soul",
        "workspace-agents",
        "memory",
        "user-profile",
        "messaging-settings",
        "command-allowlist",
        "skills",
        "tts-assets",
        "archive",
    },
    "full": set(MIGRATION_OPTION_METADATA),
}


@dataclass
class ItemResult:
    kind: str
    source: Optional[str]
    destination: Optional[str]
    status: str
    reason: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


def parse_selection_values(values: Optional[Sequence[str]]) -> List[str]:
    parsed: List[str] = []
    for value in values or ():
        for part in str(value).split(","):
            part = part.strip().lower()
            if part:
                parsed.append(part)
    return parsed


def resolve_selected_options(
    include: Optional[Sequence[str]] = None,
    exclude: Optional[Sequence[str]] = None,
    preset: Optional[str] = None,
) -> set[str]:
    include_values = parse_selection_values(include)
    exclude_values = parse_selection_values(exclude)
    valid = set(MIGRATION_OPTION_METADATA)
    preset_name = (preset or "").strip().lower()

    if preset_name and preset_name not in MIGRATION_PRESETS:
        raise ValueError(
            "Unknown migration preset: "
            + preset_name
            + ". Valid presets: "
            + ", ".join(sorted(MIGRATION_PRESETS))
        )

    unknown = (set(include_values) - {"all"} - valid) | (set(exclude_values) - {"all"} - valid)
    if unknown:
        raise ValueError(
            "Unknown migration option(s): "
            + ", ".join(sorted(unknown))
            + ". Valid options: "
            + ", ".join(sorted(valid))
        )

    if preset_name:
        selected = set(MIGRATION_PRESETS[preset_name])
    elif not include_values or "all" in include_values:
        selected = set(valid)
    else:
        selected = set(include_values)

    if "all" in exclude_values:
        selected.clear()
    selected -= (set(exclude_values) - {"all"})
    return selected


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_yaml_file(path: Path) -> Dict[str, Any]:
    if yaml is None or not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def dump_yaml_file(path: Path, data: Dict[str, Any]) -> None:
    if yaml is None:
        raise RuntimeError("PyYAML is required to update Hermes config.yaml")
    ensure_parent(path)
    path.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )


def parse_env_file(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    data: Dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        data[key.strip()] = value.strip()
    return data


def save_env_file(path: Path, data: Dict[str, str]) -> None:
    ensure_parent(path)
    lines = [f"{key}={value}" for key, value in data.items()]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def backup_existing(path: Path, backup_root: Path) -> Optional[Path]:
    if not path.exists():
        return None
    rel = Path(*path.parts[1:]) if path.is_absolute() and len(path.parts) > 1 else path
    dest = backup_root / rel
    ensure_parent(dest)
    if path.is_dir():
        shutil.copytree(path, dest, dirs_exist_ok=True)
    else:
        shutil.copy2(path, dest)
    return dest


def parse_existing_memory_entries(path: Path) -> List[str]:
    if not path.exists():
        return []
    raw = read_text(path)
    if not raw.strip():
        return []
    if ENTRY_DELIMITER in raw:
        return [e.strip() for e in raw.split(ENTRY_DELIMITER) if e.strip()]
    return extract_markdown_entries(raw)


def extract_markdown_entries(text: str) -> List[str]:
    entries: List[str] = []
    headings: List[str] = []
    paragraph_lines: List[str] = []

    def context_prefix() -> str:
        filtered = [h for h in headings if h and not re.search(r"\b(MEMORY|USER|SOUL|AGENTS|TOOLS|IDENTITY)\.md\b", h, re.I)]
        return " > ".join(filtered)

    def flush_paragraph() -> None:
        nonlocal paragraph_lines
        if not paragraph_lines:
            return
        text_block = " ".join(line.strip() for line in paragraph_lines).strip()
        paragraph_lines = []
        if not text_block:
            return
        prefix = context_prefix()
        if prefix:
            entries.append(f"{prefix}: {text_block}")
        else:
            entries.append(text_block)

    in_code_block = False
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()

        if stripped.startswith("```"):
            in_code_block = not in_code_block
            flush_paragraph()
            continue
        if in_code_block:
            continue

        heading_match = re.match(r"^(#{1,6})\s+(.*\S)\s*$", stripped)
        if heading_match:
            flush_paragraph()
            level = len(heading_match.group(1))
            text_value = heading_match.group(2).strip()
            while len(headings) >= level:
                headings.pop()
            headings.append(text_value)
            continue

        bullet_match = re.match(r"^\s*(?:[-*]|\d+\.)\s+(.*\S)\s*$", line)
        if bullet_match:
            flush_paragraph()
            content = bullet_match.group(1).strip()
            prefix = context_prefix()
            entries.append(f"{prefix}: {content}" if prefix else content)
            continue

        if not stripped:
            flush_paragraph()
            continue

        if stripped.startswith("|") and stripped.endswith("|"):
            flush_paragraph()
            continue

        paragraph_lines.append(stripped)

    flush_paragraph()

    deduped: List[str] = []
    seen = set()
    for entry in entries:
        normalized = normalize_text(entry)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(entry.strip())
    return deduped


def merge_entries(
    existing: Sequence[str],
    incoming: Sequence[str],
    limit: int,
) -> Tuple[List[str], Dict[str, int], List[str]]:
    merged = list(existing)
    seen = {normalize_text(entry) for entry in existing if entry.strip()}
    stats = {"existing": len(existing), "added": 0, "duplicates": 0, "overflowed": 0}
    overflowed: List[str] = []

    current_len = len(ENTRY_DELIMITER.join(merged)) if merged else 0

    for entry in incoming:
        normalized = normalize_text(entry)
        if not normalized:
            continue
        if normalized in seen:
            stats["duplicates"] += 1
            continue

        candidate_len = len(entry) if not merged else current_len + len(ENTRY_DELIMITER) + len(entry)
        if candidate_len > limit:
            stats["overflowed"] += 1
            overflowed.append(entry)
            continue

        merged.append(entry)
        seen.add(normalized)
        current_len = candidate_len
        stats["added"] += 1

    return merged, stats, overflowed


def relative_label(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def write_report(output_dir: Path, report: Dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for item in report["items"]:
        grouped.setdefault(item["status"], []).append(item)

    lines = [
        "# OpenClaw -> Hermes Migration Report",
        "",
        f"- Timestamp: {report['timestamp']}",
        f"- Mode: {report['mode']}",
        f"- Source: `{report['source_root']}`",
        f"- Target: `{report['target_root']}`",
        "",
        "## Summary",
        "",
    ]

    for key, value in report["summary"].items():
        lines.append(f"- {key}: {value}")

    lines.extend(["", "## What Was Not Fully Brought Over", ""])
    skipped = grouped.get("skipped", []) + grouped.get("conflict", []) + grouped.get("error", [])
    if not skipped:
        lines.append("- Nothing. All discovered items were either migrated or archived.")
    else:
        for item in skipped:
            source = item["source"] or "(n/a)"
            dest = item["destination"] or "(n/a)"
            reason = item["reason"] or item["status"]
            lines.append(f"- `{source}` -> `{dest}`: {reason}")

    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


class Migrator:
    def __init__(
        self,
        source_root: Path,
        target_root: Path,
        execute: bool,
        workspace_target: Optional[Path],
        overwrite: bool,
        migrate_secrets: bool,
        output_dir: Optional[Path],
        selected_options: Optional[set[str]] = None,
        preset_name: str = "",
        skill_conflict_mode: str = "skip",
    ):
        self.source_root = source_root
        self.target_root = target_root
        self.execute = execute
        self.workspace_target = workspace_target
        self.overwrite = overwrite
        self.migrate_secrets = migrate_secrets
        self.selected_options = set(selected_options or MIGRATION_OPTION_METADATA.keys())
        self.preset_name = preset_name.strip().lower()
        self.skill_conflict_mode = skill_conflict_mode.strip().lower() or "skip"
        self.timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        self.output_dir = output_dir or (
            target_root / "migration" / "openclaw" / self.timestamp if execute else None
        )
        self.archive_dir = self.output_dir / "archive" if self.output_dir else None
        self.backup_dir = self.output_dir / "backups" if self.output_dir else None
        self.overflow_dir = self.output_dir / "overflow" if self.output_dir else None
        self.items: List[ItemResult] = []

        config = load_yaml_file(self.target_root / "config.yaml")
        mem_cfg = config.get("memory", {}) if isinstance(config.get("memory"), dict) else {}
        self.memory_limit = int(mem_cfg.get("memory_char_limit", DEFAULT_MEMORY_CHAR_LIMIT))
        self.user_limit = int(mem_cfg.get("user_char_limit", DEFAULT_USER_CHAR_LIMIT))

        if self.skill_conflict_mode not in SKILL_CONFLICT_MODES:
            raise ValueError(
                "Unknown skill conflict mode: "
                + self.skill_conflict_mode
                + ". Valid modes: "
                + ", ".join(sorted(SKILL_CONFLICT_MODES))
            )

    def is_selected(self, option_id: str) -> bool:
        return option_id in self.selected_options

    def record(
        self,
        kind: str,
        source: Optional[Path],
        destination: Optional[Path],
        status: str,
        reason: str = "",
        **details: Any,
    ) -> None:
        self.items.append(
            ItemResult(
                kind=kind,
                source=str(source) if source else None,
                destination=str(destination) if destination else None,
                status=status,
                reason=reason,
                details=details,
            )
        )

    def source_candidate(self, *relative_paths: str) -> Optional[Path]:
        for rel in relative_paths:
            candidate = self.source_root / rel
            if candidate.exists():
                return candidate
        return None

    def resolve_skill_destination(self, destination: Path) -> Path:
        if self.skill_conflict_mode != "rename" or not destination.exists():
            return destination

        suffix = "-imported"
        candidate = destination.with_name(destination.name + suffix)
        counter = 2
        while candidate.exists():
            candidate = destination.with_name(f"{destination.name}{suffix}-{counter}")
            counter += 1
        return candidate

    def migrate(self) -> Dict[str, Any]:
        if not self.source_root.exists():
            self.record("source", self.source_root, None, "error", "OpenClaw directory does not exist")
            return self.build_report()

        config = self.load_openclaw_config()

        self.run_if_selected("soul", self.migrate_soul)
        self.run_if_selected("workspace-agents", self.migrate_workspace_agents)
        self.run_if_selected(
            "memory",
            lambda: self.migrate_memory(
                self.source_candidate("workspace/MEMORY.md", "workspace.default/MEMORY.md"),
                self.target_root / "memories" / "MEMORY.md",
                self.memory_limit,
                kind="memory",
            ),
        )
        self.run_if_selected(
            "user-profile",
            lambda: self.migrate_memory(
                self.source_candidate("workspace/USER.md", "workspace.default/USER.md"),
                self.target_root / "memories" / "USER.md",
                self.user_limit,
                kind="user-profile",
            ),
        )
        self.run_if_selected("messaging-settings", lambda: self.migrate_messaging_settings(config))
        self.run_if_selected("secret-settings", lambda: self.handle_secret_settings(config))
        self.run_if_selected("command-allowlist", self.migrate_command_allowlist)
        self.run_if_selected("skills", self.migrate_skills)
        self.run_if_selected(
            "tts-assets",
            lambda: self.copy_tree_non_destructive(
                self.source_candidate("workspace/tts"),
                self.target_root / "tts",
                kind="tts-assets",
                ignore_dir_names={".venv", "generated", "__pycache__"},
            ),
        )
        self.run_if_selected("archive", self.archive_docs)
        return self.build_report()

    def run_if_selected(self, option_id: str, func) -> None:
        if self.is_selected(option_id):
            func()
            return
        meta = MIGRATION_OPTION_METADATA[option_id]
        self.record(option_id, None, None, "skipped", "Not selected for this run", option_label=meta["label"])

    def build_report(self) -> Dict[str, Any]:
        summary: Dict[str, int] = {
            "migrated": 0,
            "archived": 0,
            "skipped": 0,
            "conflict": 0,
            "error": 0,
        }
        for item in self.items:
            summary[item.status] = summary.get(item.status, 0) + 1

        report = {
            "timestamp": self.timestamp,
            "mode": "execute" if self.execute else "dry-run",
            "source_root": str(self.source_root),
            "target_root": str(self.target_root),
            "workspace_target": str(self.workspace_target) if self.workspace_target else None,
            "output_dir": str(self.output_dir) if self.output_dir else None,
            "migrate_secrets": self.migrate_secrets,
            "preset": self.preset_name or None,
            "skill_conflict_mode": self.skill_conflict_mode,
            "selection": {
                "selected": sorted(self.selected_options),
                "preset": self.preset_name or None,
                "skill_conflict_mode": self.skill_conflict_mode,
                "available": [
                    {"id": option_id, **meta}
                    for option_id, meta in MIGRATION_OPTION_METADATA.items()
                ],
                "presets": [
                    {"id": preset_id, "selected": sorted(option_ids)}
                    for preset_id, option_ids in MIGRATION_PRESETS.items()
                ],
            },
            "summary": summary,
            "items": [asdict(item) for item in self.items],
        }

        if self.output_dir:
            write_report(self.output_dir, report)

        return report

    def maybe_backup(self, path: Path) -> Optional[Path]:
        if not self.execute or not self.backup_dir or not path.exists():
            return None
        return backup_existing(path, self.backup_dir)

    def write_overflow_entries(self, kind: str, entries: Sequence[str]) -> Optional[Path]:
        if not entries or not self.overflow_dir:
            return None
        self.overflow_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{kind.replace('-', '_')}_overflow.txt"
        path = self.overflow_dir / filename
        path.write_text("\n".join(entries) + "\n", encoding="utf-8")
        return path

    def copy_file(self, source: Path, destination: Path, kind: str) -> None:
        if not source or not source.exists():
            return

        if destination.exists():
            if sha256_file(source) == sha256_file(destination):
                self.record(kind, source, destination, "skipped", "Target already matches source")
                return
            if not self.overwrite:
                self.record(kind, source, destination, "conflict", "Target exists and overwrite is disabled")
                return

        if self.execute:
            backup_path = self.maybe_backup(destination)
            ensure_parent(destination)
            shutil.copy2(source, destination)
            self.record(kind, source, destination, "migrated", backup=str(backup_path) if backup_path else None)
        else:
            self.record(kind, source, destination, "migrated", "Would copy")

    def migrate_soul(self) -> None:
        source = self.source_candidate("workspace/SOUL.md", "workspace.default/SOUL.md")
        if not source:
            self.record("soul", None, self.target_root / "SOUL.md", "skipped", "No OpenClaw SOUL.md found")
            return
        self.copy_file(source, self.target_root / "SOUL.md", kind="soul")

    def migrate_workspace_agents(self) -> None:
        source = self.source_candidate(
            f"workspace/{WORKSPACE_INSTRUCTIONS_FILENAME}",
            f"workspace.default/{WORKSPACE_INSTRUCTIONS_FILENAME}",
        )
        if not source:
            return
        if not self.workspace_target:
            self.record("workspace-agents", source, None, "skipped", "No workspace target was provided")
            return
        destination = self.workspace_target / WORKSPACE_INSTRUCTIONS_FILENAME
        self.copy_file(source, destination, kind="workspace-agents")

    def migrate_memory(self, source: Optional[Path], destination: Path, limit: int, kind: str) -> None:
        if not source or not source.exists():
            self.record(kind, None, destination, "skipped", "Source file not found")
            return

        incoming = extract_markdown_entries(read_text(source))
        if not incoming:
            self.record(kind, source, destination, "skipped", "No importable entries found")
            return

        existing = parse_existing_memory_entries(destination)
        merged, stats, overflowed = merge_entries(existing, incoming, limit)
        details = {
            "existing_entries": stats["existing"],
            "added_entries": stats["added"],
            "duplicate_entries": stats["duplicates"],
            "overflowed_entries": stats["overflowed"],
            "char_limit": limit,
            "final_char_count": len(ENTRY_DELIMITER.join(merged)) if merged else 0,
        }
        overflow_file = self.write_overflow_entries(kind, overflowed)
        if overflow_file is not None:
            details["overflow_file"] = str(overflow_file)

        if self.execute:
            if stats["added"] == 0 and not overflowed:
                self.record(kind, source, destination, "skipped", "No new entries to import", **details)
                return
            backup_path = self.maybe_backup(destination)
            ensure_parent(destination)
            destination.write_text(ENTRY_DELIMITER.join(merged) + ("\n" if merged else ""), encoding="utf-8")
            self.record(
                kind,
                source,
                destination,
                "migrated",
                backup=str(backup_path) if backup_path else "",
                overflow_preview=overflowed[:5],
                **details,
            )
        else:
            self.record(kind, source, destination, "migrated", "Would merge entries", overflow_preview=overflowed[:5], **details)

    def migrate_command_allowlist(self) -> None:
        source = self.source_root / "exec-approvals.json"
        destination = self.target_root / "config.yaml"
        if not source.exists():
            self.record("command-allowlist", None, destination, "skipped", "No OpenClaw exec approvals file found")
            return
        if yaml is None:
            self.record("command-allowlist", source, destination, "error", "PyYAML is not available")
            return

        try:
            data = json.loads(source.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            self.record("command-allowlist", source, destination, "error", f"Invalid JSON: {exc}")
            return

        patterns: List[str] = []
        agents = data.get("agents", {})
        if isinstance(agents, dict):
            for agent_data in agents.values():
                allowlist = agent_data.get("allowlist", []) if isinstance(agent_data, dict) else []
                for entry in allowlist:
                    pattern = entry.get("pattern") if isinstance(entry, dict) else None
                    if pattern:
                        patterns.append(pattern)

        patterns = sorted(dict.fromkeys(patterns))
        if not patterns:
            self.record("command-allowlist", source, destination, "skipped", "No allowlist patterns found")
            return
        if not destination.exists():
            self.record("command-allowlist", source, destination, "skipped", "Hermes config.yaml does not exist yet")
            return

        config = load_yaml_file(destination)
        current = config.get("command_allowlist", [])
        if not isinstance(current, list):
            current = []
        merged = sorted(dict.fromkeys(list(current) + patterns))
        added = [pattern for pattern in merged if pattern not in current]
        if not added:
            self.record("command-allowlist", source, destination, "skipped", "All patterns already present")
            return

        if self.execute:
            backup_path = self.maybe_backup(destination)
            config["command_allowlist"] = merged
            dump_yaml_file(destination, config)
            self.record(
                "command-allowlist",
                source,
                destination,
                "migrated",
                backup=str(backup_path) if backup_path else "",
                added_patterns=added,
            )
        else:
            self.record("command-allowlist", source, destination, "migrated", "Would merge patterns", added_patterns=added)

    def load_openclaw_config(self) -> Dict[str, Any]:
        config_path = self.source_root / "openclaw.json"
        if not config_path.exists():
            return {}
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            return {}

    def merge_env_values(self, additions: Dict[str, str], kind: str, source: Path) -> None:
        destination = self.target_root / ".env"
        env_data = parse_env_file(destination)
        added: Dict[str, str] = {}
        conflicts: List[str] = []

        for key, value in additions.items():
            current = env_data.get(key)
            if current == value:
                continue
            if current and not self.overwrite:
                conflicts.append(key)
                continue
            env_data[key] = value
            added[key] = value

        if conflicts and not added:
            self.record(kind, source, destination, "conflict", "Destination .env already has different values", conflicting_keys=conflicts)
            return
        if not conflicts and not added:
            self.record(kind, source, destination, "skipped", "All env values already present")
            return

        if self.execute:
            backup_path = self.maybe_backup(destination)
            save_env_file(destination, env_data)
            self.record(
                kind,
                source,
                destination,
                "migrated",
                backup=str(backup_path) if backup_path else "",
                added_keys=sorted(added.keys()),
                conflicting_keys=conflicts,
            )
        else:
            self.record(
                kind,
                source,
                destination,
                "migrated",
                "Would merge env values",
                added_keys=sorted(added.keys()),
                conflicting_keys=conflicts,
            )

    def migrate_messaging_settings(self, config: Optional[Dict[str, Any]] = None) -> None:
        config = config or self.load_openclaw_config()
        additions: Dict[str, str] = {}

        workspace = (
            config.get("agents", {})
            .get("defaults", {})
            .get("workspace")
        )
        if isinstance(workspace, str) and workspace.strip():
            additions["MESSAGING_CWD"] = workspace.strip()

        allowlist_path = self.source_root / "credentials" / "telegram-default-allowFrom.json"
        if allowlist_path.exists():
            try:
                allow_data = json.loads(allowlist_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self.record("messaging-settings", allowlist_path, self.target_root / ".env", "error", "Invalid JSON in Telegram allowlist file")
            else:
                allow_from = allow_data.get("allowFrom", [])
                if isinstance(allow_from, list):
                    users = [str(user).strip() for user in allow_from if str(user).strip()]
                    if users:
                        additions["TELEGRAM_ALLOWED_USERS"] = ",".join(users)

        if additions:
            self.merge_env_values(additions, "messaging-settings", self.source_root / "openclaw.json")
        else:
            self.record("messaging-settings", self.source_root / "openclaw.json", self.target_root / ".env", "skipped", "No Hermes-compatible messaging settings found")

    def handle_secret_settings(self, config: Optional[Dict[str, Any]] = None) -> None:
        config = config or self.load_openclaw_config()
        if self.migrate_secrets:
            self.migrate_secret_settings(config)
            return

        config_path = self.source_root / "openclaw.json"
        if config_path.exists():
            self.record(
                "secret-settings",
                config_path,
                self.target_root / ".env",
                "skipped",
                "Secret migration disabled. Re-run with --migrate-secrets to import allowlisted secrets.",
                supported_targets=sorted(SUPPORTED_SECRET_TARGETS),
            )
        else:
            self.record(
                "secret-settings",
                config_path,
                self.target_root / ".env",
                "skipped",
                "OpenClaw config file not found",
                supported_targets=sorted(SUPPORTED_SECRET_TARGETS),
            )

    def migrate_secret_settings(self, config: Dict[str, Any]) -> None:
        secret_additions: Dict[str, str] = {}

        telegram_token = (
            config.get("channels", {})
            .get("telegram", {})
            .get("botToken")
        )
        if isinstance(telegram_token, str) and telegram_token.strip():
            secret_additions["TELEGRAM_BOT_TOKEN"] = telegram_token.strip()

        if secret_additions:
            self.merge_env_values(secret_additions, "secret-settings", self.source_root / "openclaw.json")
        else:
            self.record(
                "secret-settings",
                self.source_root / "openclaw.json",
                self.target_root / ".env",
                "skipped",
                "No allowlisted Hermes-compatible secrets found",
                supported_targets=sorted(SUPPORTED_SECRET_TARGETS),
            )

    def migrate_skills(self) -> None:
        source_root = self.source_candidate("workspace/skills")
        destination_root = self.target_root / "skills" / SKILL_CATEGORY_DIRNAME
        if not source_root or not source_root.exists():
            self.record("skills", None, destination_root, "skipped", "No OpenClaw skills directory found")
            return

        skill_dirs = [p for p in sorted(source_root.iterdir()) if p.is_dir() and (p / "SKILL.md").exists()]
        if not skill_dirs:
            self.record("skills", source_root, destination_root, "skipped", "No skills with SKILL.md found")
            return

        for skill_dir in skill_dirs:
            destination = destination_root / skill_dir.name
            final_destination = destination
            if destination.exists():
                if self.skill_conflict_mode == "skip":
                    self.record("skill", skill_dir, destination, "conflict", "Destination skill already exists")
                    continue
                if self.skill_conflict_mode == "rename":
                    final_destination = self.resolve_skill_destination(destination)
            if self.execute:
                backup_path = None
                if final_destination == destination and destination.exists():
                    backup_path = self.maybe_backup(destination)
                final_destination.parent.mkdir(parents=True, exist_ok=True)
                if final_destination == destination and destination.exists():
                    shutil.rmtree(destination)
                shutil.copytree(skill_dir, final_destination)
                details: Dict[str, Any] = {"backup": str(backup_path) if backup_path else ""}
                if final_destination != destination:
                    details["renamed_from"] = str(destination)
                self.record("skill", skill_dir, final_destination, "migrated", **details)
            else:
                if final_destination != destination:
                    self.record(
                        "skill",
                        skill_dir,
                        final_destination,
                        "migrated",
                        "Would copy skill directory under a renamed folder",
                        renamed_from=str(destination),
                    )
                else:
                    self.record("skill", skill_dir, final_destination, "migrated", "Would copy skill directory")

        desc_path = destination_root / "DESCRIPTION.md"
        if self.execute:
            desc_path.parent.mkdir(parents=True, exist_ok=True)
            if not desc_path.exists():
                desc_path.write_text(SKILL_CATEGORY_DESCRIPTION + "\n", encoding="utf-8")
        elif not desc_path.exists():
            self.record("skill-category", None, desc_path, "migrated", "Would create category description")

    def copy_tree_non_destructive(
        self,
        source_root: Optional[Path],
        destination_root: Path,
        kind: str,
        ignore_dir_names: Optional[set[str]] = None,
    ) -> None:
        if not source_root or not source_root.exists():
            self.record(kind, None, destination_root, "skipped", "Source directory not found")
            return

        ignore_dir_names = ignore_dir_names or set()
        files = [
            p
            for p in source_root.rglob("*")
            if p.is_file() and not any(part in ignore_dir_names for part in p.relative_to(source_root).parts[:-1])
        ]
        if not files:
            self.record(kind, source_root, destination_root, "skipped", "No files found")
            return

        copied = 0
        skipped = 0
        conflicts = 0

        for source in files:
            rel = source.relative_to(source_root)
            destination = destination_root / rel
            if destination.exists():
                if sha256_file(source) == sha256_file(destination):
                    skipped += 1
                    continue
                if not self.overwrite:
                    conflicts += 1
                    self.record(kind, source, destination, "conflict", "Destination file already exists")
                    continue

            if self.execute:
                self.maybe_backup(destination)
                ensure_parent(destination)
                shutil.copy2(source, destination)
            copied += 1

        status = "migrated" if copied else "skipped"
        reason = ""
        if not copied and conflicts:
            status = "conflict"
            reason = "All candidate files conflicted with existing destination files"
        elif not copied:
            reason = "No new files to copy"

        self.record(kind, source_root, destination_root, status, reason, copied_files=copied, unchanged_files=skipped, conflicts=conflicts)

    def archive_docs(self) -> None:
        candidates = [
            self.source_candidate("workspace/IDENTITY.md", "workspace.default/IDENTITY.md"),
            self.source_candidate("workspace/TOOLS.md", "workspace.default/TOOLS.md"),
            self.source_candidate("workspace/HEARTBEAT.md", "workspace.default/HEARTBEAT.md"),
        ]
        for candidate in candidates:
            if candidate:
                self.archive_path(candidate, reason="No direct Hermes destination; archived for manual review")

        for rel in ("workspace/.learnings", "workspace/memory"):
            candidate = self.source_root / rel
            if candidate.exists():
                self.archive_path(candidate, reason="No direct Hermes destination; archived for manual review")

        partially_extracted = [
            ("openclaw.json", "Selected Hermes-compatible values were extracted; raw OpenClaw config was not copied."),
            ("credentials/telegram-default-allowFrom.json", "Selected Hermes-compatible values were extracted; raw credentials file was not copied."),
        ]
        for rel, reason in partially_extracted:
            candidate = self.source_root / rel
            if candidate.exists():
                self.record("raw-config-skip", candidate, None, "skipped", reason)

        skipped_sensitive = [
            "memory/main.sqlite",
            "credentials",
            "devices",
            "identity",
            "workspace.zip",
        ]
        for rel in skipped_sensitive:
            candidate = self.source_root / rel
            if candidate.exists():
                self.record("sensitive-skip", candidate, None, "skipped", "Contains secrets, binary state, or product-specific runtime data")

    def archive_path(self, source: Path, reason: str) -> None:
        destination = self.archive_dir / relative_label(source, self.source_root) if self.archive_dir else None
        if self.execute and destination is not None:
            ensure_parent(destination)
            if source.is_dir():
                shutil.copytree(source, destination, dirs_exist_ok=True)
            else:
                shutil.copy2(source, destination)
            self.record("archive", source, destination, "archived", reason)
        else:
            self.record("archive", source, destination, "archived", reason)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migrate OpenClaw user state into Hermes Agent.")
    parser.add_argument("--source", default=str(Path.home() / ".openclaw"), help="OpenClaw home directory")
    parser.add_argument("--target", default=str(Path.home() / ".hermes"), help="Hermes home directory")
    parser.add_argument(
        "--workspace-target",
        help="Optional workspace root where the workspace instructions file should be copied",
    )
    parser.add_argument("--execute", action="store_true", help="Apply changes instead of reporting a dry run")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing Hermes targets after backing them up")
    parser.add_argument(
        "--migrate-secrets",
        action="store_true",
        help="Import a narrow allowlist of Hermes-compatible secrets into the target env file",
    )
    parser.add_argument(
        "--skill-conflict",
        choices=sorted(SKILL_CONFLICT_MODES),
        default="skip",
        help="How to handle imported skill directory conflicts: skip, overwrite, or rename the imported copy.",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(MIGRATION_PRESETS),
        help="Apply a named migration preset. 'user-data' excludes allowlisted secrets; 'full' includes all compatible groups.",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Comma-separated migration option ids to include (default: all). "
             f"Valid ids: {', '.join(sorted(MIGRATION_OPTION_METADATA))}",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Comma-separated migration option ids to skip. "
             f"Valid ids: {', '.join(sorted(MIGRATION_OPTION_METADATA))}",
    )
    parser.add_argument("--output-dir", help="Where to write report, backups, and archived docs")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        selected_options = resolve_selected_options(args.include, args.exclude, preset=args.preset)
    except ValueError as exc:
        print(json.dumps({"error": str(exc)}, indent=2, ensure_ascii=False))
        return 2
    migrator = Migrator(
        source_root=Path(os.path.expanduser(args.source)).resolve(),
        target_root=Path(os.path.expanduser(args.target)).resolve(),
        execute=bool(args.execute),
        workspace_target=Path(os.path.expanduser(args.workspace_target)).resolve() if args.workspace_target else None,
        overwrite=bool(args.overwrite),
        migrate_secrets=bool(args.migrate_secrets),
        output_dir=Path(os.path.expanduser(args.output_dir)).resolve() if args.output_dir else None,
        selected_options=selected_options,
        preset_name=args.preset or "",
        skill_conflict_mode=args.skill_conflict,
    )
    report = migrator.migrate()
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["summary"].get("error", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
