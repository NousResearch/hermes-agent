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
SUPPORTED_SECRET_TARGETS={
    "TELEGRAM_BOT_TOKEN",
    "OPENROUTER_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "ELEVENLABS_API_KEY",
    "VOICE_TOOLS_OPENAI_KEY",
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
    "discord-settings": {
        "label": "Discord settings",
        "description": "Import Discord bot token and allowlist into Hermes .env.",
    },
    "slack-settings": {
        "label": "Slack settings",
        "description": "Import Slack bot/app tokens and allowlist into Hermes .env.",
    },
    "whatsapp-settings": {
        "label": "WhatsApp settings",
        "description": "Import WhatsApp allowlist into Hermes .env.",
    },
    "signal-settings": {
        "label": "Signal settings",
        "description": "Import Signal account, HTTP URL, and allowlist into Hermes .env.",
    },
    "provider-keys": {
        "label": "Provider API keys",
        "description": "Import model provider API keys into Hermes .env (requires --migrate-secrets).",
    },
    "model-config": {
        "label": "Default model",
        "description": "Import the default model setting into Hermes config.yaml.",
    },
    "tts-config": {
        "label": "TTS configuration",
        "description": "Import TTS provider and voice settings into Hermes config.yaml.",
    },
    "shared-skills": {
        "label": "Shared skills",
        "description": "Copy shared OpenClaw skills from ~/.openclaw/skills/ into Hermes.",
    },
    "daily-memory": {
        "label": "Daily memory files",
        "description": "Merge daily memory entries from workspace/memory/ into Hermes MEMORY.md.",
    },
    "archive": {
        "label": "Archive unmapped docs",
        "description": "Archive compatible-but-unmapped docs for later manual review.",
    },
    "mcp-servers": {
        "label": "MCP servers",
        "description": "Import MCP server definitions from OpenClaw into Hermes config.yaml.",
    },
    "plugins-config": {
        "label": "Plugins configuration",
        "description": "Archive OpenClaw plugin configuration and installed extensions for manual review.",
    },
    "cron-jobs": {
        "label": "Cron / scheduled tasks",
        "description": "Import cron job definitions. Archive for manual recreation via 'hermes cron'.",
    },
    "hooks-config": {
        "label": "Hooks and webhooks",
        "description": "Archive OpenClaw hook configuration (internal hooks, webhooks, Gmail integration).",
    },
    "agent-config": {
        "label": "Agent defaults and multi-agent setup",
        "description": "Import agent defaults (compaction, context, thinking) into Hermes config. Archive multi-agent list.",
    },
    "gateway-config": {
        "label": "Gateway configuration",
        "description": "Import gateway port and auth settings. Archive full gateway config for manual setup.",
    },
    "session-config": {
        "label": "Session configuration",
        "description": "Import session reset policies (daily/idle) into Hermes session_reset config.",
    },
    "full-providers": {
        "label": "Full model provider definitions",
        "description": "Import custom model providers (baseUrl, apiType, headers) into Hermes custom_providers.",
    },
    "deep-channels": {
        "label": "Deep channel configuration",
        "description": "Import extended channel settings (Matrix, Mattermost, IRC, group configs). Archive complex settings.",
    },
    "browser-config": {
        "label": "Browser configuration",
        "description": "Import browser automation settings into Hermes config.yaml.",
    },
    "tools-config": {
        "label": "Tools configuration",
        "description": "Import tool settings (exec timeout, sandbox, web search) into Hermes config.yaml.",
    },
    "approvals-config": {
        "label": "Approval rules",
        "description": "Import approval mode and rules into Hermes config.yaml approvals section.",
    },
    "memory-backend": {
        "label": "Memory backend configuration",
        "description": "Archive OpenClaw memory backend settings (QMD, vector search, citations) for manual review.",
    },
    "skills-config": {
        "label": "Skills registry configuration",
        "description": "Archive per-skill enabled/config/env settings from OpenClaw skills.entries.",
    },
    "ui-identity": {
        "label": "UI and identity settings",
        "description": "Archive OpenClaw UI theme, assistant identity, and display preferences.",
    },
    "logging-config": {
        "label": "Logging and diagnostics",
        "description": "Archive OpenClaw logging and diagnostics configuration.",
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
        "discord-settings",
        "slack-settings",
        "whatsapp-settings",
        "signal-settings",
        "model-config",
        "tts-config",
        "shared-skills",
        "daily-memory",
        "archive",
        "mcp-servers",
        "agent-config",
        "session-config",
        "browser-config",
        "tools-config",
        "approvals-config",
        "deep-channels",
        "full-providers",
        "plugins-config",
        "cron-jobs",
        "hooks-config",
        "memory-backend",
        "skills-config",
        "ui-identity",
        "logging-config",
        "gateway-config",
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


def resolve_secret_input(value: Any, env: Optional[Dict[str, str]] = None) -> Optional[str]:
    """Resolve an OpenClaw SecretInput value to a plain string.

    SecretInput can be:
    - A plain string: "sk-..."
    - An env template: "${OPENROUTER_API_KEY}"
    - A SecretRef object: {"source": "env", "id": "OPENROUTER_API_KEY"}
    """
    if isinstance(value, str):
        # Check for env template: "${VAR_NAME}"
        m = re.match(r"^\$\{(\w+)\}$", value.strip())
        if m and env:
            return env.get(m.group(1), "").strip() or None
        return value.strip() or None
    if isinstance(value, dict):
        source = value.get("source", "")
        ref_id = value.get("id", "")
        if source == "env" and ref_id and env:
            return env.get(ref_id, "").strip() or None
        # File/exec sources can't be resolved here — return None
    return None


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


# ── Brand rewriting ─────────────────────────────────────────
# Replace OpenClaw brand names with Hermes in migrated text so that
# memory entries, user profiles, SOUL.md, and workspace instructions
# read as self-referential to the new agent identity.
#
# Case-preserving: ``OpenClaw`` → ``Hermes`` (prose), but lowercase matches
# like ``openclaw`` → ``hermes`` (so filesystem paths like ``~/.openclaw``
# become ``~/.hermes`` — the real Hermes home — not the broken ``~/.Hermes``).
_REBRAND_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r'\bOpen[\s-]?Claw\b', re.IGNORECASE), 'Hermes'),
    (re.compile(r'\bClawdBot\b', re.IGNORECASE), 'Hermes'),
    (re.compile(r'\bMoltBot\b', re.IGNORECASE), 'Hermes'),
]


def _case_preserving_replacement(replacement: str):
    """Return a re.sub replacement fn that lowercases the result when the
    matched text was all-lowercase.

    Keeps ``OpenClaw`` → ``Hermes`` but maps ``openclaw`` → ``hermes`` so a
    filesystem path like ``~/.openclaw/config.yaml`` rewrites to
    ``~/.hermes/config.yaml`` (the real Hermes home) instead of the broken
    ``~/.Hermes/config.yaml``.
    """
    def _sub(match: "re.Match[str]") -> str:
        matched = match.group(0)
        if matched and matched.islower():
            return replacement.lower()
        return replacement
    return _sub


def rebrand_text(text: str) -> str:
    """Replace OpenClaw / ClawdBot / MoltBot brand names with Hermes.

    Preserves case so filesystem-path matches (lowercase) don't become
    capitalized directory names that don't exist.
    """
    for pattern, replacement in _REBRAND_PATTERNS:
        text = pattern.sub(_case_preserving_replacement(replacement), text)
    return text


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
            # OpenClaw renamed workspace/ to workspace-main/ (and workspace-{agentId}
            # for multi-agent).  Try the new path as a fallback.
            if rel.startswith("workspace/"):
                suffix = rel[len("workspace/"):]
                for variant in ("workspace-main", "workspace-assistant"):
                    alt = self.source_root / variant / suffix
                    if alt.exists():
                        return alt
            elif rel.startswith("workspace.default/"):
                suffix = rel[len("workspace.default/"):]
                alt = self.source_root / "workspace-main" / suffix
                if alt.exists():
                    return alt
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
        self.run_if_selected("discord-settings", lambda: self.migrate_discord_settings(config))
        self.run_if_selected("slack-settings", lambda: self.migrate_slack_settings(config))
        self.run_if_selected("whatsapp-settings", lambda: self.migrate_whatsapp_settings(config))
        self.run_if_selected("signal-settings", lambda: self.migrate_signal_settings(config))
        self.run_if_selected("provider-keys", lambda: self.handle_provider_keys(config))
        self.run_if_selected("model-config", lambda: self.migrate_model_config(config))
        self.run_if_selected("tts-config", lambda: self.migrate_tts_config(config))
        self.run_if_selected("command-allowlist", self.migrate_command_allowlist)
        self.run_if_selected("skills", self.migrate_skills)
        self.run_if_selected("shared-skills", self.migrate_shared_skills)
        self.run_if_selected("daily-memory", self.migrate_daily_memory)
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

        # ── v2 migration modules ──────────────────────────────
        self.run_if_selected("mcp-servers", lambda: self.migrate_mcp_servers(config))
        self.run_if_selected("plugins-config", lambda: self.migrate_plugins_config(config))
        self.run_if_selected("cron-jobs", lambda: self.migrate_cron_jobs(config))
        self.run_if_selected("hooks-config", lambda: self.migrate_hooks_config(config))
        self.run_if_selected("agent-config", lambda: self.migrate_agent_config(config))
        self.run_if_selected("gateway-config", lambda: self.migrate_gateway_config(config))
        self.run_if_selected("session-config", lambda: self.migrate_session_config(config))
        self.run_if_selected("full-providers", lambda: self.migrate_full_providers(config))
        self.run_if_selected("deep-channels", lambda: self.migrate_deep_channels(config))
        self.run_if_selected("browser-config", lambda: self.migrate_browser_config(config))
        self.run_if_selected("tools-config", lambda: self.migrate_tools_config(config))
        self.run_if_selected("approvals-config", lambda: self.migrate_approvals_config(config))
        self.run_if_selected("memory-backend", lambda: self.migrate_memory_backend(config))
        self.run_if_selected("skills-config", lambda: self.migrate_skills_config(config))
        self.run_if_selected("ui-identity", lambda: self.migrate_ui_identity(config))
        self.run_if_selected("logging-config", lambda: self.migrate_logging_config(config))

        # Generate migration notes
        self.generate_migration_notes()

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

    def copy_file(self, source: Path, destination: Path, kind: str,
                  transform: Optional[Any] = None) -> None:
        if not source or not source.exists():
            return

        if destination.exists():
            if not transform and sha256_file(source) == sha256_file(destination):
                self.record(kind, source, destination, "skipped", "Target already matches source")
                return
            if not self.overwrite:
                self.record(kind, source, destination, "conflict", "Target exists and overwrite is disabled")
                return

        if self.execute:
            backup_path = self.maybe_backup(destination)
            ensure_parent(destination)
            if transform:
                content = read_text(source)
                content = transform(content)
                destination.write_text(content, encoding="utf-8")
                shutil.copystat(source, destination)
            else:
                shutil.copy2(source, destination)
            self.record(kind, source, destination, "migrated", backup=str(backup_path) if backup_path else None)
        else:
            self.record(kind, source, destination, "migrated", "Would copy")

    def migrate_soul(self) -> None:
        source = self.source_candidate("workspace/SOUL.md", "workspace.default/SOUL.md")
        if not source:
            self.record("soul", None, self.target_root / "SOUL.md", "skipped", "No OpenClaw SOUL.md found")
            return
        self.copy_file(source, self.target_root / "SOUL.md", kind="soul", transform=rebrand_text)

    def migrate_workspace_agents(self) -> None:
        source = self.source_candidate(
            f"workspace/{WORKSPACE_INSTRUCTIONS_FILENAME}",
            f"workspace.default/{WORKSPACE_INSTRUCTIONS_FILENAME}",
        )
        if source is None:
            self.record("workspace-agents", "workspace/AGENTS.md", "", "skipped", "Source file not found")
            return
        if not self.workspace_target:
            self.record("workspace-agents", source, None, "skipped", "No workspace target was provided")
            return
        destination = self.workspace_target / WORKSPACE_INSTRUCTIONS_FILENAME
        self.copy_file(source, destination, kind="workspace-agents", transform=rebrand_text)

    def migrate_memory(self, source: Optional[Path], destination: Path, limit: int, kind: str) -> None:
        if not source or not source.exists():
            self.record(kind, None, destination, "skipped", "Source file not found")
            return

        incoming = extract_markdown_entries(read_text(source))
        if not incoming:
            self.record(kind, source, destination, "skipped", "No importable entries found")
            return
        incoming = [rebrand_text(entry) for entry in incoming]

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
        # Check current name and legacy config filenames
        for name in ("openclaw.json", "clawdbot.json", "moltbot.json"):
            config_path = self.source_root / name
            if config_path.exists():
                try:
                    data = json.loads(config_path.read_text(encoding="utf-8"))
                    return data if isinstance(data, dict) else {}
                except json.JSONDecodeError:
                    continue
        return {}

    def load_openclaw_env(self) -> Dict[str, str]:
        """Load the OpenClaw .env file for secrets that live there instead of config."""
        return parse_env_file(self.source_root / ".env")

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
            ws_path = workspace.strip()
            # Skip if the workspace points inside the OpenClaw source directory —
            # that path will be stale after migration and would cause the Hermes
            # gateway to use the old OpenClaw workspace as its cwd, picking up
            # OpenClaw's AGENTS.md, MEMORY.md, etc.
            try:
                inside_source = Path(ws_path).resolve().is_relative_to(self.source_root.resolve())
            except (ValueError, OSError):
                inside_source = False
            if not inside_source:
                additions["MESSAGING_CWD"] = ws_path

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

        tg_cfg = config.get("channels", {}).get("telegram", {})
        telegram_token = self._get_channel_field(tg_cfg, "botToken") if isinstance(tg_cfg, dict) else None
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

    def _resolve_channel_secret(self, value: Any) -> Optional[str]:
        """Resolve a channel config value that may be a SecretRef."""
        return resolve_secret_input(value, self.load_openclaw_env())

    @staticmethod
    def _get_channel_field(ch_cfg: Dict[str, Any], field: str) -> Any:
        """Get a field from channel config, checking both flat and accounts.default layout."""
        val = ch_cfg.get(field)
        if val is not None:
            return val
        accounts = ch_cfg.get("accounts")
        if isinstance(accounts, dict):
            default = accounts.get("default")
            if isinstance(default, dict):
                return default.get(field)
        return None

    def migrate_discord_settings(self, config: Optional[Dict[str, Any]] = None) -> None:
        config = config or self.load_openclaw_config()
        additions: Dict[str, str] = {}
        discord = config.get("channels", {}).get("discord", {})
        if isinstance(discord, dict):
            token = self._get_channel_field(discord, "token")
            if isinstance(token, str) and token.strip():
                additions["DISCORD_BOT_TOKEN"] = token.strip()
            allow_from = self._get_channel_field(discord, "allowFrom") or []
            if isinstance(allow_from, list):
                users = [str(u).strip() for u in allow_from if str(u).strip()]
                if users:
                    additions["DISCORD_ALLOWED_USERS"] = ",".join(users)
        if additions:
            self.merge_env_values(additions, "discord-settings", self.source_root / "openclaw.json")
        else:
            self.record("discord-settings", self.source_root / "openclaw.json", self.target_root / ".env", "skipped", "No Discord settings found")

    def migrate_slack_settings(self, config: Optional[Dict[str, Any]] = None) -> None:
        config = config or self.load_openclaw_config()
        additions: Dict[str, str] = {}
        slack = config.get("channels", {}).get("slack", {})
        if isinstance(slack, dict):
            bot_token = self._get_channel_field(slack, "botToken")
            if isinstance(bot_token, str) and bot_token.strip():
                additions["SLACK_BOT_TOKEN"] = bot_token.strip()
            app_token = self._get_channel_field(slack, "appToken")
            if isinstance(app_token, str) and app_token.strip():
                additions["SLACK_APP_TOKEN"] = app_token.strip()
            allow_from = self._get_channel_field(slack, "allowFrom") or []
            if isinstance(allow_from, list):
                users = [str(u).strip() for u in allow_from if str(u).strip()]
                if users:
                    additions["SLACK_ALLOWED_USERS"] = ",".join(users)
        if additions:
            self.merge_env_values(additions, "slack-settings", self.source_root / "openclaw.json")
        else:
            self.record("slack-settings", self.source_root / "openclaw.json", self.target_root / ".env", "skipped", "No Slack settings found")

    def migrate_whatsapp_settings(self, config: Optional[Dict[str, Any]] = None) -> None:
        config = config or self.load_openclaw_config()
        additions: Dict[str, str] = {}
        whatsapp = config.get("channels", {}).get("whatsapp", {})
        if isinstance(whatsapp, dict):
            allow_from = self._get_channel_field(whatsapp, "allowFrom") or []
            if isinstance(allow_from, list):
                users = [str(u).strip() for u in allow_from if str(u).strip()]
                if users:
                    additions["WHATSAPP_ALLOWED_USERS"] = ",".join(users)
        if additions:
            self.merge_env_values(additions, "whatsapp-settings", self.source_root / "openclaw.json")
        else:
            self.record("whatsapp-settings", self.source_root / "openclaw.json", self.target_root / ".env", "skipped", "No WhatsApp settings found")

    def migrate_signal_settings(self, config: Optional[Dict[str, Any]] = None) -> None:
        config = config or self.load_openclaw_config()
        additions: Dict[str, str] = {}
        signal = config.get("channels", {}).get("signal", {})
        if isinstance(signal, dict):
            account = self._get_channel_field(signal, "account")
            if isinstance(account, str) and account.strip():
                additions["SIGNAL_ACCOUNT"] = account.strip()
            http_url = self._get_channel_field(signal, "httpUrl")
            if isinstance(http_url, str) and http_url.strip():
                additions["SIGNAL_HTTP_URL"] = http_url.strip()
            allow_from = self._get_channel_field(signal, "allowFrom") or []
            if isinstance(allow_from, list):
                users = [str(u).strip() for u in allow_from if str(u).strip()]
                if users:
                    additions["SIGNAL_ALLOWED_USERS"] = ",".join(users)
        if additions:
            self.merge_env_values(additions, "signal-settings", self.source_root / "openclaw.json")
        else:
            self.record("signal-settings", self.source_root / "openclaw.json", self.target_root / ".env", "skipped", "No Signal settings found")

    def handle_provider_keys(self, config: Optional[Dict[str, Any]] = None) -> None:
        config = config or self.load_openclaw_config()
        if not self.migrate_secrets:
            config_path = self.source_root / "openclaw.json"
            self.record(
                "provider-keys",
                config_path,
                self.target_root / ".env",
                "skipped",
                "Secret migration disabled. Re-run with --migrate-secrets to import provider API keys.",
                supported_targets=sorted(SUPPORTED_SECRET_TARGETS),
            )
            return
        self.migrate_provider_keys(config)

    def migrate_provider_keys(self, config: Dict[str, Any]) -> None:
        secret_additions: Dict[str, str] = {}

        # Extract provider API keys from models.providers
        # Note: apiKey values can be strings, env templates, or SecretRef objects
        openclaw_env = self.load_openclaw_env()
        providers = config.get("models", {}).get("providers", {})
        if isinstance(providers, dict):
            for provider_name, provider_cfg in providers.items():
                if not isinstance(provider_cfg, dict):
                    continue
                raw_key = provider_cfg.get("apiKey")
                api_key = resolve_secret_input(raw_key, openclaw_env)
                if not api_key:
                    # Warn if a SecretRef with file/exec source was silently unresolvable
                    if isinstance(raw_key, dict) and raw_key.get("source") in ("file", "exec"):
                        self.record(
                            "provider-keys",
                            self.source_root / "openclaw.json",
                            None,
                            "skipped",
                            f"Provider '{provider_name}' uses a {raw_key['source']}-backed SecretRef "
                            f"that cannot be auto-migrated. Add this key manually via: hermes config set",
                        )
                    continue

                base_url = provider_cfg.get("baseUrl", "")
                api_type = provider_cfg.get("api", "")
                env_var = None

                # Match by baseUrl first
                if isinstance(base_url, str):
                    if "openrouter" in base_url.lower():
                        env_var = "OPENROUTER_API_KEY"
                    elif "openai.com" in base_url.lower():
                        env_var = "OPENAI_API_KEY"
                    elif "anthropic" in base_url.lower():
                        env_var = "ANTHROPIC_API_KEY"

                # Match by api type
                if not env_var and isinstance(api_type, str) and api_type == "anthropic-messages":
                    env_var = "ANTHROPIC_API_KEY"

                # Match by provider name
                if not env_var:
                    name_lower = provider_name.lower()
                    if name_lower == "openrouter":
                        env_var = "OPENROUTER_API_KEY"
                    elif "openai" in name_lower:
                        env_var = "OPENAI_API_KEY"

                if env_var:
                    secret_additions[env_var] = api_key

        # Extract TTS API keys
        tts = config.get("messages", {}).get("tts", {})
        if isinstance(tts, dict):
            elevenlabs = tts.get("elevenlabs", {})
            if isinstance(elevenlabs, dict):
                el_key = elevenlabs.get("apiKey")
                if isinstance(el_key, str) and el_key.strip():
                    secret_additions["ELEVENLABS_API_KEY"] = el_key.strip()
            openai_tts = tts.get("openai", {})
            if isinstance(openai_tts, dict):
                oai_key = openai_tts.get("apiKey")
                if isinstance(oai_key, str) and oai_key.strip():
                    secret_additions["VOICE_TOOLS_OPENAI_KEY"] = oai_key.strip()

        # Also check the OpenClaw .env file — many users store keys there
        # instead of inline in openclaw.json
        openclaw_env = self.load_openclaw_env()
        env_key_mapping = {
            "OPENROUTER_API_KEY": "OPENROUTER_API_KEY",
            "OPENAI_API_KEY": "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY": "ANTHROPIC_API_KEY",
            "ELEVENLABS_API_KEY": "ELEVENLABS_API_KEY",
            "TELEGRAM_BOT_TOKEN": "TELEGRAM_BOT_TOKEN",
            "DEEPSEEK_API_KEY": "DEEPSEEK_API_KEY",
            "GEMINI_API_KEY": "GEMINI_API_KEY",
            "ZAI_API_KEY": "ZAI_API_KEY",
            "MINIMAX_API_KEY": "MINIMAX_API_KEY",
        }
        for oc_key, hermes_key in env_key_mapping.items():
            val = openclaw_env.get(oc_key, "").strip()
            if val and hermes_key not in secret_additions:
                secret_additions[hermes_key] = val

        # Check the openclaw.json "env" sub-object — some OpenClaw setups
        # store API keys here instead of in a separate .env file.
        # Keys can be at env.<KEY> or env.vars.<KEY>.
        json_env = config.get("env")
        if isinstance(json_env, dict):
            env_vars = json_env.get("vars")
            sources = [json_env]
            if isinstance(env_vars, dict):
                sources.append(env_vars)
            for src in sources:
                for oc_key, hermes_key in env_key_mapping.items():
                    val = src.get(oc_key)
                    if isinstance(val, str) and val.strip() and hermes_key not in secret_additions:
                        secret_additions[hermes_key] = val.strip()

        # Check per-agent auth-profiles.json for additional credentials
        auth_profiles_path = self.source_root / "agents" / "main" / "agent" / "auth-profiles.json"
        if auth_profiles_path.exists():
            try:
                profiles = json.loads(auth_profiles_path.read_text(encoding="utf-8"))
                if isinstance(profiles, dict):
                    # auth-profiles.json wraps profiles in a "profiles" key
                    profile_entries = profiles.get("profiles", profiles) if isinstance(profiles.get("profiles"), dict) else profiles
                    for profile_name, profile_data in profile_entries.items():
                        if not isinstance(profile_data, dict):
                            continue
                        # Canonical field is "key", "apiKey" is accepted as alias
                        api_key = profile_data.get("key", "") or profile_data.get("apiKey", "")
                        if not isinstance(api_key, str) or not api_key.strip():
                            continue
                        name_lower = profile_name.lower()
                        if "openrouter" in name_lower and "OPENROUTER_API_KEY" not in secret_additions:
                            secret_additions["OPENROUTER_API_KEY"] = api_key.strip()
                        elif "openai" in name_lower and "OPENAI_API_KEY" not in secret_additions:
                            secret_additions["OPENAI_API_KEY"] = api_key.strip()
                        elif "anthropic" in name_lower and "ANTHROPIC_API_KEY" not in secret_additions:
                            secret_additions["ANTHROPIC_API_KEY"] = api_key.strip()
            except (json.JSONDecodeError, OSError):
                pass

        if secret_additions:
            self.merge_env_values(secret_additions, "provider-keys", self.source_root / "openclaw.json")
        else:
            self.record(
                "provider-keys",
                self.source_root / "openclaw.json",
                self.target_root / ".env",
                "skipped",
                "No provider API keys found",
                supported_targets=sorted(SUPPORTED_SECRET_TARGETS),
            )

    def migrate_model_config(self, config: Optional[Dict[str, Any]] = None) -> None:
        config = config or self.load_openclaw_config()
        destination = self.target_root / "config.yaml"
        source_path = self.source_root / "openclaw.json"

        model_value = config.get("agents", {}).get("defaults", {}).get("model")
        if model_value is None:
            self.record("model-config", source_path, destination, "skipped", "No default model found in OpenClaw config")
            return

        if isinstance(model_value, dict):
            model_str = model_value.get("primary")
        else:
            model_str = model_value

        if not isinstance(model_str, str) or not model_str.strip():
            self.record("model-config", source_path, destination, "skipped", "Default model value is empty or invalid")
            return

        model_str = model_str.strip()

        if yaml is None:
            self.record("model-config", source_path, destination, "error", "PyYAML is not available")
            return

        hermes_config = load_yaml_file(destination)
        current_model = hermes_config.get("model")
        if current_model == model_str:
            self.record("model-config", source_path, destination, "skipped", "Model already set to the same value")
            return
        if current_model and not self.overwrite:
            self.record("model-config", source_path, destination, "conflict", "Model already set and overwrite is disabled", current=current_model, incoming=model_str)
            return

        if self.execute:
            backup_path = self.maybe_backup(destination)
            existing_model = hermes_config.get("model")
            if isinstance(existing_model, dict):
                existing_model["default"] = model_str
            else:
                hermes_config["model"] = {"default": model_str}
            dump_yaml_file(destination, hermes_config)
            self.record("model-config", source_path, destination, "migrated", backup=str(backup_path) if backup_path else "", model=model_str)
        else:
            self.record("model-config", source_path, destination, "migrated", "Would set model", model=model_str)

    def migrate_tts_config(self, config: Optional[Dict[str, Any]] = None) -> None:
        config = config or self.load_openclaw_config()
        destination = self.target_root / "config.yaml"
        source_path = self.source_root / "openclaw.json"

        tts = config.get("messages", {}).get("tts", {})
        if not isinstance(tts, dict) or not tts:
            self.record("tts-config", source_path, destination, "skipped", "No TTS configuration found in OpenClaw config")
            return

        if yaml is None:
            self.record("tts-config", source_path, destination, "error", "PyYAML is not available")
            return

        tts_data: Dict[str, Any] = {}

        provider = tts.get("provider")
        if isinstance(provider, str) and provider in ("elevenlabs", "openai", "edge", "microsoft"):
            # OpenClaw renamed "edge" to "microsoft"; Hermes still uses "edge"
            tts_data["provider"] = "edge" if provider == "microsoft" else provider

        # TTS provider settings live under messages.tts.providers.{provider}
        # in OpenClaw (not messages.tts.elevenlabs directly)
        providers = tts.get("providers") or {}

        # Also check the top-level "talk" config which has provider settings too
        talk_cfg = (config or self.load_openclaw_config()).get("talk") or {}
        talk_providers = talk_cfg.get("providers") or {}

        # Merge: messages.tts.providers takes priority, then talk.providers,
        # then legacy flat keys (messages.tts.elevenlabs, etc.)
        elevenlabs = (
            (providers.get("elevenlabs") or {})
            if isinstance(providers.get("elevenlabs"), dict) else
            (talk_providers.get("elevenlabs") or {})
            if isinstance(talk_providers.get("elevenlabs"), dict) else
            (tts.get("elevenlabs") or {})
        )
        if isinstance(elevenlabs, dict):
            el_settings: Dict[str, str] = {}
            voice_id = elevenlabs.get("voiceId") or talk_cfg.get("voiceId")
            if isinstance(voice_id, str) and voice_id.strip():
                el_settings["voice_id"] = voice_id.strip()
            model_id = elevenlabs.get("modelId") or talk_cfg.get("modelId")
            if isinstance(model_id, str) and model_id.strip():
                el_settings["model_id"] = model_id.strip()
            if el_settings:
                tts_data["elevenlabs"] = el_settings

        openai_tts = (
            (providers.get("openai") or {})
            if isinstance(providers.get("openai"), dict) else
            (talk_providers.get("openai") or {})
            if isinstance(talk_providers.get("openai"), dict) else
            (tts.get("openai") or {})
        )
        if isinstance(openai_tts, dict):
            oai_settings: Dict[str, str] = {}
            oai_model = openai_tts.get("model") or openai_tts.get("modelId")
            if isinstance(oai_model, str) and oai_model.strip():
                oai_settings["model"] = oai_model.strip()
            oai_voice = openai_tts.get("voice")
            if isinstance(oai_voice, str) and oai_voice.strip():
                oai_settings["voice"] = oai_voice.strip()
            if oai_settings:
                tts_data["openai"] = oai_settings

        edge_tts = (
            (providers.get("edge") or providers.get("microsoft") or {})
            if isinstance(providers.get("edge"), dict) or isinstance(providers.get("microsoft"), dict) else
            (tts.get("edge") or tts.get("microsoft") or {})
        )
        if isinstance(edge_tts, dict):
            edge_voice = edge_tts.get("voice")
            if isinstance(edge_voice, str) and edge_voice.strip():
                tts_data["edge"] = {"voice": edge_voice.strip()}

        if not tts_data:
            self.record("tts-config", source_path, destination, "skipped", "No compatible TTS settings found")
            return

        hermes_config = load_yaml_file(destination)
        existing_tts = hermes_config.get("tts", {})
        if not isinstance(existing_tts, dict):
            existing_tts = {}

        if self.execute:
            backup_path = self.maybe_backup(destination)
            merged_tts = dict(existing_tts)
            for key, value in tts_data.items():
                if isinstance(value, dict) and isinstance(merged_tts.get(key), dict):
                    merged_tts[key] = {**merged_tts[key], **value}
                else:
                    merged_tts[key] = value
            hermes_config["tts"] = merged_tts
            dump_yaml_file(destination, hermes_config)
            self.record("tts-config", source_path, destination, "migrated", backup=str(backup_path) if backup_path else "", settings=list(tts_data.keys()))
        else:
            self.record("tts-config", source_path, destination, "migrated", "Would set TTS config", settings=list(tts_data.keys()))

    def migrate_shared_skills(self) -> None:
        # Check all OpenClaw skill sources: managed, personal, project-level
        skill_sources = [
            (self.source_root / "skills", "shared-skills", "managed skills"),
            (Path.home() / ".agents" / "skills", "personal-skills", "personal cross-project skills"),
            (self.source_root / "workspace" / ".agents" / "skills", "project-skills", "project-level shared skills"),
            (self.source_root / "workspace.default" / ".agents" / "skills", "project-skills", "project-level shared skills"),
        ]
        found_any = False
        for source_root, kind_label, desc in skill_sources:
            if source_root.exists():
                found_any = True
                self._import_skill_directory(source_root, kind_label, desc)
        if not found_any:
            destination_root = self.target_root / "skills" / SKILL_CATEGORY_DIRNAME
            self.record("shared-skills", None, destination_root, "skipped", "No shared OpenClaw skills directories found")

    def _import_skill_directory(self, source_root: Path, kind_label: str, desc: str) -> None:
        """Import skills from a single source directory into openclaw-imports."""
        destination_root = self.target_root / "skills" / SKILL_CATEGORY_DIRNAME

        skill_dirs = [p for p in sorted(source_root.iterdir()) if p.is_dir() and (p / "SKILL.md").exists()]
        if not skill_dirs:
            self.record(kind_label, source_root, destination_root, "skipped", f"No skills with SKILL.md found in {desc}")
            return

        for skill_dir in skill_dirs:
            destination = destination_root / skill_dir.name
            final_destination = destination
            if destination.exists():
                if self.skill_conflict_mode == "skip":
                    self.record(kind_label, skill_dir, destination, "conflict", "Destination skill already exists")
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
                self.record(kind_label, skill_dir, final_destination, "migrated", **details)
            else:
                if final_destination != destination:
                    self.record(
                        kind_label,
                        skill_dir,
                        final_destination,
                        "migrated",
                        f"Would copy {desc} directory under a renamed folder",
                        renamed_from=str(destination),
                    )
                else:
                    self.record(kind_label, skill_dir, final_destination, "migrated", f"Would copy {desc} directory")

        desc_path = destination_root / "DESCRIPTION.md"
        if self.execute:
            desc_path.parent.mkdir(parents=True, exist_ok=True)
            if not desc_path.exists():
                desc_path.write_text(SKILL_CATEGORY_DESCRIPTION + "\n", encoding="utf-8")
        elif not desc_path.exists():
            self.record("shared-skill-category", None, desc_path, "migrated", "Would create category description")

    def migrate_daily_memory(self) -> None:
        source_dir = self.source_candidate("workspace/memory")
        destination = self.target_root / "memories" / "MEMORY.md"
        if not source_dir or not source_dir.is_dir():
            self.record("daily-memory", None, destination, "skipped", "No workspace/memory/ directory found")
            return

        md_files = sorted(p for p in source_dir.iterdir() if p.is_file() and p.suffix == ".md")
        if not md_files:
            self.record("daily-memory", source_dir, destination, "skipped", "No .md files found in workspace/memory/")
            return

        all_incoming: List[str] = []
        for md_file in md_files:
            entries = extract_markdown_entries(read_text(md_file))
            all_incoming.extend(entries)

        if not all_incoming:
            self.record("daily-memory", source_dir, destination, "skipped", "No importable entries found in daily memory files")
            return
        all_incoming = [rebrand_text(entry) for entry in all_incoming]

        existing = parse_existing_memory_entries(destination)
        merged, stats, overflowed = merge_entries(existing, all_incoming, self.memory_limit)
        details = {
            "source_files": len(md_files),
            "existing_entries": stats["existing"],
            "added_entries": stats["added"],
            "duplicate_entries": stats["duplicates"],
            "overflowed_entries": stats["overflowed"],
            "char_limit": self.memory_limit,
            "final_char_count": len(ENTRY_DELIMITER.join(merged)) if merged else 0,
        }
        overflow_file = self.write_overflow_entries("daily-memory", overflowed)
        if overflow_file is not None:
            details["overflow_file"] = str(overflow_file)

        if self.execute:
            if stats["added"] == 0 and not overflowed:
                self.record("daily-memory", source_dir, destination, "skipped", "No new entries to import", **details)
                return
            backup_path = self.maybe_backup(destination)
            ensure_parent(destination)
            destination.write_text(ENTRY_DELIMITER.join(merged) + ("\n" if merged else ""), encoding="utf-8")
            self.record(
                "daily-memory",
                source_dir,
                destination,
                "migrated",
                backup=str(backup_path) if backup_path else "",
                overflow_preview=overflowed[:5],
                **details,
            )
        else:
            self.record("daily-memory", source_dir, destination, "migrated", "Would merge daily memory entries", overflow_preview=overflowed[:5], **details)

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
            self.source_candidate("workspace/BOOTSTRAP.md", "workspace.default/BOOTSTRAP.md"),
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

    # ── MCP servers ─────────────────────────────────────────────
    def migrate_mcp_servers(self, config: Optional[Dict[str, Any]] = None) -> None:
        config = config or self.load_openclaw_config()
        mcp_raw = (config.get("mcp") or {}).get("servers") or {}
        if not mcp_raw:
            self.record("mcp-servers", None, None, "skipped", "No MCP servers found in OpenClaw config")
            return

        hermes_cfg_path = self.target_root / "config.yaml"
        hermes_cfg = load_yaml_file(hermes_cfg_path)
        existing_mcp = hermes_cfg.get("mcp_servers") or {}
        added = 0

        for name, srv in mcp_raw.items():
            if not isinstance(srv, dict):
                continue
            if name in existing_mcp and not self.overwrite:
                self.record("mcp-servers", f"mcp.servers.{name}", f"mcp_servers.{name}", "conflict",
                            "MCP server already exists in Hermes config")
                continue

            hermes_srv: Dict[str, Any] = {}
            # STDIO transport
            if srv.get("command"):
                hermes_srv["command"] = srv["command"]
                if srv.get("args"):
                    hermes_srv["args"] = srv["args"]
                if srv.get("env"):
                    hermes_srv["env"] = srv["env"]
                if srv.get("cwd"):
                    hermes_srv["cwd"] = srv["cwd"]
            # HTTP/SSE transport
            if srv.get("url"):
                hermes_srv["url"] = srv["url"]
                if srv.get("headers"):
                    hermes_srv["headers"] = srv["headers"]
                if srv.get("auth"):
                    hermes_srv["auth"] = srv["auth"]
            # Common fields
            if srv.get("enabled") is False:
                hermes_srv["enabled"] = False
            if srv.get("timeout"):
                hermes_srv["timeout"] = srv["timeout"]
            if srv.get("connectTimeout"):
                hermes_srv["connect_timeout"] = srv["connectTimeout"]
            # Tool filtering
            tools_cfg = srv.get("tools") or {}
            if tools_cfg.get("include") or tools_cfg.get("exclude"):
                hermes_srv["tools"] = {}
                if tools_cfg.get("include"):
                    hermes_srv["tools"]["include"] = tools_cfg["include"]
                if tools_cfg.get("exclude"):
                    hermes_srv["tools"]["exclude"] = tools_cfg["exclude"]
            # Sampling
            sampling = srv.get("sampling")
            if sampling and isinstance(sampling, dict):
                hermes_srv["sampling"] = {
                    k: v for k, v in {
                        "enabled": sampling.get("enabled"),
                        "model": sampling.get("model"),
                        "max_tokens_cap": sampling.get("maxTokensCap") or sampling.get("max_tokens_cap"),
                        "timeout": sampling.get("timeout"),
                        "max_rpm": sampling.get("maxRpm") or sampling.get("max_rpm"),
                    }.items() if v is not None
                }

            existing_mcp[name] = hermes_srv
            added += 1
            self.record("mcp-servers", f"mcp.servers.{name}", f"config.yaml mcp_servers.{name}",
                        "migrated", servers_added=added)

        if added > 0 and self.execute:
            self.maybe_backup(hermes_cfg_path)
            hermes_cfg["mcp_servers"] = existing_mcp
            dump_yaml_file(hermes_cfg_path, hermes_cfg)

    # ── Plugins ───────────────────────────────────────────────
    def migrate_plugins_config(self, config: Optional[Dict[str, Any]] = None) -> None:
        config = config or self.load_openclaw_config()
        plugins = config.get("plugins") or {}
        if not plugins:
            self.record("plugins-config", None, None, "skipped", "No plugins configuration found")
            return

        # Archive the full plugins config
        if self.archive_dir and self.execute:
            self.archive_dir.mkdir(parents=True, exist_ok=True)
            dest = self.archive_dir / "plugins-config.json"
            dest.write_text(json.dumps(plugins, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            self.record("plugins-config", "openclaw.json plugins.*", str(dest), "archived",
                        "Plugins config archived for manual review")
        else:
            self.record("plugins-config", "openclaw.json plugins.*", "archive/plugins-config.json",
                        "archived" if not self.execute else "migrated", "Would archive plugins config")

        # Copy extensions directory if it exists
        ext_dir = self.source_root / "extensions"
        if ext_dir.is_dir() and self.archive_dir:
            dest_ext = self.archive_dir / "extensions"
            if self.execute:
                shutil.copytree(ext_dir, dest_ext, dirs_exist_ok=True)
            self.record("plugins-config", str(ext_dir), str(dest_ext), "archived",
                        "Extensions directory archived")

        # Extract any plugin env vars
        entries = plugins.get("entries") or {}
        for plugin_name, plugin_cfg in entries.items():
            if isinstance(plugin_cfg, dict):
                env_vars = plugin_cfg.get("env") or {}
                api_key = plugin_cfg.get("apiKey")
                if api_key and self.migrate_secrets:
                    env_key = f"PLUGIN_{plugin_name.upper().replace('-', '_')}_API_KEY"
                    self._set_env_var(env_key, api_key, f"plugins.entries.{plugin_name}.apiKey")

    # ── Cron jobs ─────────────────────────────────────────────
    def migrate_cron_jobs(self, config: Optional[Dict[str, Any]] = None) -> None:
        config = config or self.load_openclaw_config()
        cron = config.get("cron") or {}
        cron_store = self.source_root / "cron"
        found_any = False

        # Archive the full cron config when present
        if cron:
            found_any = True
            if self.archive_dir and self.execute:
                self.archive_dir.mkdir(parents=True, exist_ok=True)
                dest = self.archive_dir / "cron-config.json"
                dest.write_text(json.dumps(cron, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                self.record("cron-jobs", "openclaw.json cron.*", str(dest), "archived",
                            "Cron config archived. Use 'hermes cron' to recreate jobs manually.")
            else:
                self.record("cron-jobs", "openclaw.json cron.*", "archive/cron-config.json",
                            "archived", "Would archive cron config")

        # Also check for cron store files even when config.cron is missing
        if cron_store.is_dir() and self.archive_dir:
            found_any = True
            dest_cron = self.archive_dir / "cron-store"
            if self.execute:
                shutil.copytree(cron_store, dest_cron, dirs_exist_ok=True)
            self.record("cron-jobs", str(cron_store), str(dest_cron), "archived",
                        "Cron job store archived")

        if not found_any:
            self.record("cron-jobs", None, None, "skipped", "No cron configuration found")

    # ── Hooks ─────────────────────────────────────────────────
    def migrate_hooks_config(self, config: Optional[Dict[str, Any]] = None) -> None:
        config = config or self.load_openclaw_config()
        hooks = config.get("hooks") or {}
        if not hooks:
            self.record("hooks-config", None, None, "skipped", "No hooks configuration found")
            return

        # Archive the full hooks config
        if self.archive_dir and self.execute:
            self.archive_dir.mkdir(parents=True, exist_ok=True)
            dest = self.archive_dir / "hooks-config.json"
            dest.write_text(json.dumps(hooks, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            self.record("hooks-config", "openclaw.json hooks.*", str(dest), "archived",
                        "Hooks config archived for manual review")
        else:
            self.record("hooks-config", "openclaw.json hooks.*", "archive/hooks-config.json",
                        "archived", "Would archive hooks config")

        # Copy workspace hooks directory
        for ws_name in ("workspace", "workspace.default"):
            hooks_dir = self.source_root / ws_name / "hooks"
            if hooks_dir.is_dir() and self.archive_dir:
                dest_hooks = self.archive_dir / "workspace-hooks"
                if self.execute:
                    shutil.copytree(hooks_dir, dest_hooks, dirs_exist_ok=True)
                self.record("hooks-config", str(hooks_dir), str(dest_hooks), "archived",
                            "Workspace hooks directory archived")
                break

    # ── Agent config ──────────────────────────────────────────
    def migrate_agent_config(self, config: Optional[Dict[str, Any]] = None) -> None:
        config = config or self.load_openclaw_config()
        agents = config.get("agents") or {}
        defaults = agents.get("defaults") or {}
        agent_list = agents.get("list") or []

        if not defaults and not agent_list:
            self.record("agent-config", None, None, "skipped", "No agent configuration found")
            return

        hermes_cfg_path = self.target_root / "config.yaml"
        hermes_cfg = load_yaml_file(hermes_cfg_path)
        changes = False

        # Map agent defaults
        agent_cfg = hermes_cfg.get("agent") or {}
        if defaults.get("contextTokens"):
            # No direct mapping but useful context
            pass
        if defaults.get("timeoutSeconds"):
            agent_cfg["max_turns"] = min(defaults["timeoutSeconds"] // 10, 200)
            changes = True
        if defaults.get("verboseDefault"):
            agent_cfg["verbose"] = defaults["verboseDefault"]
            changes = True
        if defaults.get("thinkingDefault"):
            # Map OpenClaw thinking -> Hermes reasoning_effort
            thinking = defaults["thinkingDefault"]
            if thinking in ("always", "high", "xhigh"):
                agent_cfg["reasoning_effort"] = "high"
            elif thinking in ("auto", "medium", "adaptive"):
                agent_cfg["reasoning_effort"] = "medium"
            elif thinking in ("off", "low", "none", "minimal"):
                agent_cfg["reasoning_effort"] = "low"
            changes = True

        # Map compaction -> compression
        compaction = defaults.get("compaction") or {}
        if compaction:
            compression = hermes_cfg.get("compression") or {}
            if compaction.get("mode") == "off":
                compression["enabled"] = False
            else:
                compression["enabled"] = True
            if compaction.get("timeout"):
                pass  # No direct mapping
            if compaction.get("model"):
                aux = hermes_cfg.setdefault("auxiliary", {})
                aux_comp = aux.setdefault("compression", {})
                aux_comp["model"] = compaction["model"]
            hermes_cfg["compression"] = compression
            changes = True

        # Map humanDelay
        human_delay = defaults.get("humanDelay") or {}
        if human_delay:
            hd = hermes_cfg.get("human_delay") or {}
            hd_mode = human_delay.get("mode") or ("natural" if human_delay.get("enabled") else None)
            if hd_mode and hd_mode != "off":
                hd["mode"] = hd_mode
            if human_delay.get("minMs"):
                hd["min_ms"] = human_delay["minMs"]
            if human_delay.get("maxMs"):
                hd["max_ms"] = human_delay["maxMs"]
            hermes_cfg["human_delay"] = hd
            changes = True

        # Map userTimezone
        if defaults.get("userTimezone"):
            hermes_cfg["timezone"] = defaults["userTimezone"]
            changes = True

        # Map terminal/exec settings
        exec_cfg = (config.get("tools") or {}).get("exec") or {}
        if exec_cfg:
            terminal_cfg = hermes_cfg.get("terminal") or {}
            if exec_cfg.get("timeoutSec") or exec_cfg.get("timeout"):
                terminal_cfg["timeout"] = exec_cfg.get("timeoutSec") or exec_cfg.get("timeout")
                changes = True
            hermes_cfg["terminal"] = terminal_cfg

        # Map sandbox -> terminal docker settings
        sandbox = defaults.get("sandbox") or {}
        if sandbox and sandbox.get("backend") == "docker":
            terminal_cfg = hermes_cfg.get("terminal") or {}
            terminal_cfg["backend"] = "docker"
            if sandbox.get("docker", {}).get("image"):
                terminal_cfg["docker_image"] = sandbox["docker"]["image"]
            hermes_cfg["terminal"] = terminal_cfg
            changes = True

        if changes:
            hermes_cfg["agent"] = agent_cfg
            if self.execute:
                self.maybe_backup(hermes_cfg_path)
                dump_yaml_file(hermes_cfg_path, hermes_cfg)
            self.record("agent-config", "openclaw.json agents.defaults", "config.yaml agent/compression/terminal",
                        "migrated", "Agent defaults mapped to Hermes config")

        # Archive multi-agent list
        if agent_list:
            if self.archive_dir and self.execute:
                self.archive_dir.mkdir(parents=True, exist_ok=True)
                dest = self.archive_dir / "agents-list.json"
                dest.write_text(json.dumps(agent_list, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            self.record("agent-config", "openclaw.json agents.list", "archive/agents-list.json",
                        "archived", f"Multi-agent setup ({len(agent_list)} agents) archived for manual recreation")

        # Archive bindings
        bindings = config.get("bindings") or []
        if bindings:
            if self.archive_dir and self.execute:
                self.archive_dir.mkdir(parents=True, exist_ok=True)
                dest = self.archive_dir / "bindings.json"
                dest.write_text(json.dumps(bindings, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            self.record("agent-config", "openclaw.json bindings", "archive/bindings.json",
                        "archived", f"Agent routing bindings ({len(bindings)} rules) archived")

    # ── Gateway config ────────────────────────────────────────
    def migrate_gateway_config(self, config: Optional[Dict[str, Any]] = None) -> None:
        config = config or self.load_openclaw_config()
        gateway = config.get("gateway") or {}
        if not gateway:
            self.record("gateway-config", None, None, "skipped", "No gateway configuration found")
            return

        # Archive the full gateway config (complex, many settings)
        if self.archive_dir and self.execute:
            self.archive_dir.mkdir(parents=True, exist_ok=True)
            dest = self.archive_dir / "gateway-config.json"
            dest.write_text(json.dumps(gateway, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        self.record("gateway-config", "openclaw.json gateway.*", "archive/gateway-config.json",
                    "archived", "Gateway config archived. Use 'hermes gateway' to configure.")

        # Extract gateway auth token to .env if present
        auth = gateway.get("auth") or {}
        if auth.get("token") and self.migrate_secrets:
            self._set_env_var("HERMES_GATEWAY_TOKEN", auth["token"], "gateway.auth.token")

    # ── Session config ────────────────────────────────────────
    def migrate_session_config(self, config: Optional[Dict[str, Any]] = None) -> None:
        config = config or self.load_openclaw_config()
        session = config.get("session") or {}
        if not session:
            self.record("session-config", None, None, "skipped", "No session configuration found")
            return

        hermes_cfg_path = self.target_root / "config.yaml"
        hermes_cfg = load_yaml_file(hermes_cfg_path)
        sr = hermes_cfg.get("session_reset") or {}
        changes = False

        # OpenClaw uses session.reset (structured) and session.resetTriggers (string array)
        reset = session.get("reset") or {}
        reset_triggers = session.get("resetTriggers") or session.get("reset_triggers") or []

        if reset:
            # Structured reset config: has mode, atHour, idleMinutes
            mode = reset.get("mode", "")
            if mode == "daily":
                sr["mode"] = "daily"
            elif mode == "idle":
                sr["mode"] = "idle"
            else:
                sr["mode"] = mode or "none"
            if reset.get("atHour") is not None:
                sr["at_hour"] = reset["atHour"]
            if reset.get("idleMinutes"):
                sr["idle_minutes"] = reset["idleMinutes"]
            changes = True
        elif isinstance(reset_triggers, list) and reset_triggers:
            # Simple string triggers: ["daily", "idle"]
            has_daily = "daily" in reset_triggers
            has_idle = "idle" in reset_triggers
            if has_daily and has_idle:
                sr["mode"] = "both"
            elif has_daily:
                sr["mode"] = "daily"
            elif has_idle:
                sr["mode"] = "idle"
            changes = True

        if changes:
            hermes_cfg["session_reset"] = sr
            if self.execute:
                self.maybe_backup(hermes_cfg_path)
                dump_yaml_file(hermes_cfg_path, hermes_cfg)
            self.record("session-config", "openclaw.json session.resetTriggers",
                        "config.yaml session_reset", "migrated")

        # Archive full session config (identity links, thread bindings, etc.)
        complex_keys = {"identityLinks", "threadBindings", "maintenance", "scope", "sendPolicy"}
        complex_session = {k: v for k, v in session.items() if k in complex_keys and v}
        if complex_session and self.archive_dir:
            if self.execute:
                self.archive_dir.mkdir(parents=True, exist_ok=True)
                dest = self.archive_dir / "session-config.json"
                dest.write_text(json.dumps(complex_session, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            self.record("session-config", "openclaw.json session (advanced)",
                        "archive/session-config.json", "archived",
                        "Advanced session settings archived (identity links, thread bindings, etc.)")

    # ── Full model providers ──────────────────────────────────
    def migrate_full_providers(self, config: Optional[Dict[str, Any]] = None) -> None:
        config = config or self.load_openclaw_config()
        models = config.get("models") or {}
        providers = models.get("providers") or {}
        if not providers:
            self.record("full-providers", None, None, "skipped", "No model providers found")
            return

        hermes_cfg_path = self.target_root / "config.yaml"
        hermes_cfg = load_yaml_file(hermes_cfg_path)
        custom_providers = hermes_cfg.get("custom_providers") or []
        added = 0

        # Well-known providers: just extract API keys
        WELL_KNOWN = {"openrouter", "openai", "anthropic", "deepseek", "google", "groq"}

        for prov_name, prov_cfg in providers.items():
            if not isinstance(prov_cfg, dict):
                continue

            # Extract API key to .env
            api_key = prov_cfg.get("apiKey") or prov_cfg.get("api_key")
            if api_key and self.migrate_secrets:
                env_key = f"{prov_name.upper().replace('-', '_')}_API_KEY"
                self._set_env_var(env_key, api_key, f"models.providers.{prov_name}.apiKey")

            # For non-well-known providers, create custom_providers entry
            if prov_name.lower() not in WELL_KNOWN and prov_cfg.get("baseUrl"):
                # Check if already exists
                existing_names = {p.get("name", "").lower() for p in custom_providers}
                if prov_name.lower() in existing_names and not self.overwrite:
                    self.record("full-providers", f"models.providers.{prov_name}",
                                "config.yaml custom_providers", "conflict",
                                f"Provider '{prov_name}' already exists")
                    continue

                api_type = prov_cfg.get("apiType") or prov_cfg.get("api") or prov_cfg.get("type") or "openai"
                api_mode_map = {
                    "openai": "chat_completions",
                    "openai-completions": "chat_completions",
                    "openai-responses": "chat_completions",
                    "anthropic": "anthropic_messages",
                    "anthropic-messages": "anthropic_messages",
                    "google-generative-ai": "chat_completions",
                    "cohere": "chat_completions",
                }
                entry = {
                    "name": prov_name,
                    "base_url": prov_cfg["baseUrl"],
                    "api_key": "",  # referenced from .env
                    "api_mode": api_mode_map.get(api_type, "chat_completions"),
                }
                custom_providers.append(entry)
                added += 1
                self.record("full-providers", f"models.providers.{prov_name}",
                            f"config.yaml custom_providers[{prov_name}]", "migrated")

        if added > 0 and self.execute:
            self.maybe_backup(hermes_cfg_path)
            hermes_cfg["custom_providers"] = custom_providers
            dump_yaml_file(hermes_cfg_path, hermes_cfg)

        # Archive model aliases/catalog
        agent_defaults = (config.get("agents") or {}).get("defaults") or {}
        model_aliases = agent_defaults.get("models") or {}
        if model_aliases:
            if self.archive_dir and self.execute:
                self.archive_dir.mkdir(parents=True, exist_ok=True)
                dest = self.archive_dir / "model-aliases.json"
                dest.write_text(json.dumps(model_aliases, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            self.record("full-providers", "agents.defaults.models", "archive/model-aliases.json",
                        "archived", f"Model aliases/catalog ({len(model_aliases)} entries) archived")

    # ── Deep channel config ───────────────────────────────────
    def migrate_deep_channels(self, config: Optional[Dict[str, Any]] = None) -> None:
        config = config or self.load_openclaw_config()
        channels = config.get("channels") or {}
        if not channels:
            self.record("deep-channels", None, None, "skipped", "No channel configuration found")
            return

        # Extended channel token/allowlist mapping
        CHANNEL_ENV_MAP = {
            "matrix": {"token": "MATRIX...OKEN", "tokenField": "accessToken", "allowFrom": "MATRIX_ALLOWED_USERS",
                        "extras": {"homeserverUrl": "MATRIX_HOMESERVER_URL", "userId": "MATRIX_USER_ID"}},
            "mattermost": {"token": "MATTERMOST_BOT_TOKEN", "allowFrom": "MATTERMOST_ALLOWED_USERS",
                           "extras": {"url": "MATTERMOST_URL", "teamId": "MATTERMOST_TEAM_ID"}},
            "irc": {"extras": {"server": "IRC_SERVER", "nick": "IRC_NICK", "channels": "IRC_CHANNELS"}},
            "googlechat": {"extras": {"serviceAccountKeyPath": "GOOGLE_CHAT_SA_KEY_PATH"}},
            "imessage": {},
            "bluebubbles": {"extras": {"server": "BLUEBUBBLES_SERVER", "password": "BLUEBUBBLES_PASSWORD"}},
            "msteams": {"token": "MSTEAMS_BOT_TOKEN", "allowFrom": "MSTEAMS_ALLOWED_USERS"},
            "nostr": {"extras": {"nsec": "NOSTR_NSEC", "relays": "NOSTR_RELAYS"}},
            "twitch": {"token": "TWITCH_BOT_TOKEN", "extras": {"channels": "TWITCH_CHANNELS"}},
        }

        for ch_name, ch_mapping in CHANNEL_ENV_MAP.items():
            ch_cfg = channels.get(ch_name) or {}
            if not ch_cfg:
                continue

            # Extract tokens (check flat path, then accounts.default)
            token_field = ch_mapping.get("tokenField", "botToken")
            bot_token = self._get_channel_field(ch_cfg, token_field)
            if ch_mapping.get("token") and bot_token and self.migrate_secrets:
                self._set_env_var(ch_mapping["token"], str(bot_token),
                                  f"channels.{ch_name}.{token_field}")
            allow_val = self._get_channel_field(ch_cfg, "allowFrom")
            if ch_mapping.get("allowFrom") and allow_val:
                if isinstance(allow_val, list):
                    allow_val = ",".join(str(x) for x in allow_val)
                self._set_env_var(ch_mapping["allowFrom"], str(allow_val),
                                  f"channels.{ch_name}.allowFrom")
            # Extra fields
            for oc_key, env_key in (ch_mapping.get("extras") or {}).items():
                val = self._get_channel_field(ch_cfg, oc_key)
                if val:
                    if isinstance(val, list):
                        val = ",".join(str(x) for x in val)
                    is_secret = "password" in oc_key.lower() or "token" in oc_key.lower() or "nsec" in oc_key.lower()
                    if is_secret and not self.migrate_secrets:
                        continue
                    self._set_env_var(env_key, str(val), f"channels.{ch_name}.{oc_key}")

        # Map Discord-specific settings to Hermes config
        discord_cfg = channels.get("discord") or {}
        if discord_cfg:
            hermes_cfg_path = self.target_root / "config.yaml"
            hermes_cfg = load_yaml_file(hermes_cfg_path)
            discord_hermes = hermes_cfg.get("discord") or {}
            changed = False
            if "requireMention" in discord_cfg:
                discord_hermes["require_mention"] = discord_cfg["requireMention"]
                changed = True
            if discord_cfg.get("autoThread") is not None:
                discord_hermes["auto_thread"] = discord_cfg["autoThread"]
                changed = True
            if changed and self.execute:
                hermes_cfg["discord"] = discord_hermes
                dump_yaml_file(hermes_cfg_path, hermes_cfg)

        # Archive complex channel configs (group settings, thread bindings, etc.)
        complex_archive = {}
        for ch_name, ch_cfg in channels.items():
            if not isinstance(ch_cfg, dict):
                continue
            complex_keys = {k: v for k, v in ch_cfg.items()
                          if k not in ("botToken", "appToken", "allowFrom", "enabled")
                          and v and k not in ("requireMention", "autoThread")}
            if complex_keys:
                complex_archive[ch_name] = complex_keys

        if complex_archive and self.archive_dir:
            if self.execute:
                self.archive_dir.mkdir(parents=True, exist_ok=True)
                dest = self.archive_dir / "channels-deep-config.json"
                dest.write_text(json.dumps(complex_archive, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            self.record("deep-channels", "openclaw.json channels (advanced settings)",
                        "archive/channels-deep-config.json", "archived",
                        f"Deep channel config for {len(complex_archive)} channels archived")

    # ── Browser config ────────────────────────────────────────
    def migrate_browser_config(self, config: Optional[Dict[str, Any]] = None) -> None:
        config = config or self.load_openclaw_config()
        browser = config.get("browser") or {}
        if not browser:
            self.record("browser-config", None, None, "skipped", "No browser configuration found")
            return

        hermes_cfg_path = self.target_root / "config.yaml"
        hermes_cfg = load_yaml_file(hermes_cfg_path)
        browser_hermes = hermes_cfg.get("browser") or {}
        changed = False

        # Map fields that have Hermes equivalents
        if browser.get("cdpUrl"):
            browser_hermes["cdp_url"] = browser["cdpUrl"]
            changed = True
        if browser.get("headless") is not None:
            browser_hermes["headless"] = browser["headless"]
            changed = True

        if changed:
            hermes_cfg["browser"] = browser_hermes
            if self.execute:
                self.maybe_backup(hermes_cfg_path)
                dump_yaml_file(hermes_cfg_path, hermes_cfg)
            self.record("browser-config", "openclaw.json browser.*", "config.yaml browser",
                        "migrated")

        # Archive remaining browser settings
        advanced = {k: v for k, v in browser.items()
                   if k not in ("cdpUrl", "headless") and v}
        if advanced and self.archive_dir:
            if self.execute:
                self.archive_dir.mkdir(parents=True, exist_ok=True)
                dest = self.archive_dir / "browser-config.json"
                dest.write_text(json.dumps(advanced, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            self.record("browser-config", "openclaw.json browser (advanced)",
                        "archive/browser-config.json", "archived")

    # ── Tools config ──────────────────────────────────────────
    def migrate_tools_config(self, config: Optional[Dict[str, Any]] = None) -> None:
        config = config or self.load_openclaw_config()
        tools = config.get("tools") or {}
        if not tools:
            self.record("tools-config", None, None, "skipped", "No tools configuration found")
            return

        hermes_cfg_path = self.target_root / "config.yaml"
        hermes_cfg = load_yaml_file(hermes_cfg_path)
        changed = False

        # Map exec timeout -> terminal timeout (field is timeoutSec in OpenClaw)
        exec_cfg = tools.get("exec") or {}
        timeout_val = exec_cfg.get("timeoutSec") or exec_cfg.get("timeout")
        if timeout_val:
            terminal_cfg = hermes_cfg.get("terminal") or {}
            terminal_cfg["timeout"] = timeout_val
            hermes_cfg["terminal"] = terminal_cfg
            changed = True

        # Map web search API key (path: tools.web.search.brave.apiKey in OpenClaw)
        web_cfg = tools.get("web") or tools.get("webSearch") or {}
        search_cfg = web_cfg.get("search") or web_cfg if not web_cfg.get("search") else web_cfg["search"]
        brave_cfg = search_cfg.get("brave") or {}
        brave_key = brave_cfg.get("apiKey") or search_cfg.get("braveApiKey") or web_cfg.get("braveApiKey")
        if brave_key and isinstance(brave_key, str) and self.migrate_secrets:
            self._set_env_var("BRAVE_API_KEY", brave_key, "tools.web.search.brave.apiKey")

        if changed and self.execute:
            self.maybe_backup(hermes_cfg_path)
            dump_yaml_file(hermes_cfg_path, hermes_cfg)
            self.record("tools-config", "openclaw.json tools.*", "config.yaml terminal",
                        "migrated")

        # Archive full tools config
        if self.archive_dir:
            if self.execute:
                self.archive_dir.mkdir(parents=True, exist_ok=True)
                dest = self.archive_dir / "tools-config.json"
                dest.write_text(json.dumps(tools, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            self.record("tools-config", "openclaw.json tools (full)", "archive/tools-config.json",
                        "archived", "Full tools config archived for reference")

    # ── Approvals config ──────────────────────────────────────
    def migrate_approvals_config(self, config: Optional[Dict[str, Any]] = None) -> None:
        config = config or self.load_openclaw_config()
        approvals = config.get("approvals") or {}
        if not approvals:
            self.record("approvals-config", None, None, "skipped", "No approvals configuration found")
            return

        hermes_cfg_path = self.target_root / "config.yaml"
        hermes_cfg = load_yaml_file(hermes_cfg_path)

        # Map approval mode (nested under approvals.exec.mode in OpenClaw)
        exec_approvals = approvals.get("exec") or {}
        mode = (exec_approvals.get("mode") if isinstance(exec_approvals, dict) else None) or approvals.get("mode") or approvals.get("defaultMode")
        if mode:
            mode_map = {"auto": "off", "always": "manual", "smart": "smart", "manual": "manual"}
            hermes_mode = mode_map.get(mode, "manual")
            hermes_cfg.setdefault("approvals", {})["mode"] = hermes_mode
            if self.execute:
                self.maybe_backup(hermes_cfg_path)
                dump_yaml_file(hermes_cfg_path, hermes_cfg)
            self.record("approvals-config", "openclaw.json approvals.mode",
                        "config.yaml approvals.mode", "migrated", f"Mapped '{mode}' -> '{hermes_mode}'")

        # Archive full approvals config
        if len(approvals) > 1 and self.archive_dir:
            if self.execute:
                self.archive_dir.mkdir(parents=True, exist_ok=True)
                dest = self.archive_dir / "approvals-config.json"
                dest.write_text(json.dumps(approvals, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            self.record("approvals-config", "openclaw.json approvals (rules)",
                        "archive/approvals-config.json", "archived")

    # ── Memory backend ────────────────────────────────────────
    def migrate_memory_backend(self, config: Optional[Dict[str, Any]] = None) -> None:
        config = config or self.load_openclaw_config()
        memory = config.get("memory") or {}
        if not memory:
            self.record("memory-backend", None, None, "skipped", "No memory backend configuration found")
            return

        if self.archive_dir and self.execute:
            self.archive_dir.mkdir(parents=True, exist_ok=True)
            dest = self.archive_dir / "memory-backend-config.json"
            dest.write_text(json.dumps(memory, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        self.record("memory-backend", "openclaw.json memory.*", "archive/memory-backend-config.json",
                    "archived", "Memory backend config (QMD, vector search, citations) archived for manual review")

    # ── Skills config ─────────────────────────────────────────
    def migrate_skills_config(self, config: Optional[Dict[str, Any]] = None) -> None:
        config = config or self.load_openclaw_config()
        skills = config.get("skills") or {}
        entries = skills.get("entries") or {}
        if not entries and not skills:
            self.record("skills-config", None, None, "skipped", "No skills registry configuration found")
            return

        if self.archive_dir and self.execute:
            self.archive_dir.mkdir(parents=True, exist_ok=True)
            dest = self.archive_dir / "skills-registry-config.json"
            dest.write_text(json.dumps(skills, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        self.record("skills-config", "openclaw.json skills.*", "archive/skills-registry-config.json",
                    "archived", f"Skills registry config ({len(entries)} entries) archived")

    # ── UI / Identity ─────────────────────────────────────────
    def migrate_ui_identity(self, config: Optional[Dict[str, Any]] = None) -> None:
        config = config or self.load_openclaw_config()
        ui = config.get("ui") or {}
        if not ui:
            self.record("ui-identity", None, None, "skipped", "No UI/identity configuration found")
            return

        if self.archive_dir and self.execute:
            self.archive_dir.mkdir(parents=True, exist_ok=True)
            dest = self.archive_dir / "ui-identity-config.json"
            dest.write_text(json.dumps(ui, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        self.record("ui-identity", "openclaw.json ui.*", "archive/ui-identity-config.json",
                    "archived", "UI theme and identity settings archived")

    # ── Logging / Diagnostics ─────────────────────────────────
    def migrate_logging_config(self, config: Optional[Dict[str, Any]] = None) -> None:
        config = config or self.load_openclaw_config()
        logging_cfg = config.get("logging") or {}
        diagnostics = config.get("diagnostics") or {}
        combined = {}
        if logging_cfg:
            combined["logging"] = logging_cfg
        if diagnostics:
            combined["diagnostics"] = diagnostics
        if not combined:
            self.record("logging-config", None, None, "skipped", "No logging/diagnostics configuration found")
            return

        if self.archive_dir and self.execute:
            self.archive_dir.mkdir(parents=True, exist_ok=True)
            dest = self.archive_dir / "logging-diagnostics-config.json"
            dest.write_text(json.dumps(combined, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        self.record("logging-config", "openclaw.json logging/diagnostics",
                    "archive/logging-diagnostics-config.json", "archived")

    # ── Helper: set env var ───────────────────────────────────
    def _set_env_var(self, key: str, value: str, source_label: str) -> None:
        env_path = self.target_root / ".env"
        if self.execute:
            env_data = parse_env_file(env_path)
            if key in env_data and not self.overwrite:
                self.record("env-var", source_label, f".env {key}", "conflict",
                            f"Env var {key} already set")
                return
            env_data[key] = value
            save_env_file(env_path, env_data)
        self.record("env-var", source_label, f".env {key}", "migrated")

    # ── Generate migration notes ──────────────────────────────
    def generate_migration_notes(self) -> None:
        if not self.output_dir:
            return
        notes = [
            "# OpenClaw -> Hermes Migration Notes",
            "",
            "This document lists items that require manual attention after migration.",
            "",
            "## PM2 / External Processes",
            "",
            "Your PM2 processes (Discord bots, Telegram bots, etc.) are NOT affected",
            "by this migration. They run independently and will continue working.",
            "No action needed for PM2-managed processes.",
            "",
        ]

        archived = [i for i in self.items if i.status == "archived"]
        if archived:
            notes.extend([
                "## Archived Items (Manual Review Needed)",
                "",
                "These OpenClaw configurations were archived because they don't have a",
                "direct 1:1 mapping in Hermes. Review each file and recreate manually:",
                "",
            ])
            for item in archived:
                notes.append(f"- **{item.kind}**: `{item.destination}` -- {item.reason}")
            notes.append("")

        conflicts = [i for i in self.items if i.status == "conflict"]
        if conflicts:
            notes.extend([
                "## Conflicts (Existing Hermes Config Not Overwritten)",
                "",
                "These items already existed in your Hermes config. Re-run with",
                "`--overwrite` to force, or merge manually:",
                "",
            ])
            for item in conflicts:
                notes.append(f"- **{item.kind}**: {item.reason}")
            notes.append("")

        has_cron_config_archive = any(
            i.kind == "cron-jobs" and i.status == "archived" and i.destination and i.destination.endswith("cron-config.json")
            for i in self.items
        )
        has_cron_store_archive = any(
            i.kind == "cron-jobs" and i.status == "archived" and i.destination and i.destination.endswith("cron-store")
            for i in self.items
        )

        notes.extend([
            "## IMPORTANT: Archive the OpenClaw Directory",
            "",
            "After migration, your OpenClaw directory still exists on disk with workspace",
            "state files (todo.json, sessions, logs). If the Hermes agent discovers these",
            "directories, it may read/write to them instead of the Hermes state, causing",
            "confusion (e.g., cron jobs reading a different todo list than interactive sessions).",
            "",
            "**Strongly recommended:** Run `hermes claw cleanup` to rename the OpenClaw",
            "directory to `.openclaw.pre-migration`. This prevents the agent from finding it.",
            "The directory is renamed, not deleted — you can undo this at any time.",
            "",
            "If you skip this step and notice the agent getting confused about workspaces",
            "or todo lists, run `hermes claw cleanup` to fix it.",
            "",
            "## Hermes-Specific Setup",
            "",
            "After migration, you may want to:",
            "- Run `hermes claw cleanup` to archive the OpenClaw directory (prevents state confusion)",
            "- Run `hermes setup` to configure any remaining settings",
            "- Run `hermes mcp list` to verify MCP servers were imported correctly",
        ])

        if has_cron_config_archive:
            notes.append("- Run `hermes cron` to recreate scheduled tasks (see archive/cron-config.json)")
        elif has_cron_store_archive:
            notes.append("- Run `hermes cron` to recreate scheduled tasks (see archived cron-store)")

        # Check if skills were imported
        has_skills = any(i.kind == "skills" and i.status == "migrated" for i in self.items)
        if has_skills:
            notes.extend([
                "",
                "## Imported Skills",
                "",
                "Imported skills require a new session to take effect. After migration,",
                "restart your agent or start a new chat session, then run `/skills`",
                "to verify they loaded correctly.",
                "",
            ])

        # Check if WhatsApp was detected
        has_whatsapp = any(i.kind == "whatsapp-settings" and i.status == "migrated" for i in self.items)
        if has_whatsapp:
            notes.extend([
                "",
                "## WhatsApp Requires Re-Pairing",
                "",
                "WhatsApp uses QR-code pairing, not token-based auth. Your allowlist",
                "was migrated, but you must re-pair the device by running:",
                "",
                "    hermes whatsapp",
                "",
            ])

        notes.extend([
            "- Run `hermes gateway install` if you need the gateway service",
            "- Review `~/.hermes/config.yaml` for any adjustments",
            "",
        ])

        if self.execute:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            (self.output_dir / "MIGRATION_NOTES.md").write_text(
                "\n".join(notes) + "\n", encoding="utf-8"
            )


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

    # ── Human-readable terminal recap ─────────────────────────
    s = report["summary"]
    items = report["items"]
    mode_label = "DRY RUN" if not args.execute else "EXECUTED"
    total = sum(s.values())

    print()
    print(f"  ╔══════════════════════════════════════════════════════╗")
    print(f"  ║   OpenClaw -> Hermes Migration   [{mode_label:>8s}]   ║")
    print(f"  ╠══════════════════════════════════════════════════════╣")
    print(f"  ║  Source:  {str(report['source_root'])[:42]:<42s}  ║")
    print(f"  ║  Target:  {str(report['target_root'])[:42]:<42s}  ║")
    print(f"  ╠══════════════════════════════════════════════════════╣")
    print(f"  ║  ✔ Migrated:  {s.get('migrated', 0):>3d}    ◆ Archived:  {s.get('archived', 0):>3d}        ║")
    print(f"  ║  ⊘ Skipped:   {s.get('skipped', 0):>3d}    ⚠ Conflicts: {s.get('conflict', 0):>3d}        ║")
    print(f"  ║  ✖ Errors:    {s.get('error', 0):>3d}    Total:       {total:>3d}        ║")
    print(f"  ╚══════════════════════════════════════════════════════╝")

    # Show what was migrated
    migrated = [i for i in items if i["status"] == "migrated"]
    if migrated:
        print()
        print("  Migrated:")
        seen_kinds = set()
        for item in migrated:
            label = item["kind"]
            if label in seen_kinds:
                continue
            seen_kinds.add(label)
            dest = item.get("destination") or ""
            if dest.startswith(str(report["target_root"])):
                dest = "~/.hermes/" + dest[len(str(report["target_root"])) + 1:]
            meta = MIGRATION_OPTION_METADATA.get(label, {})
            display = meta.get("label", label)
            print(f"    ✔ {display:<35s} -> {dest}")

    # Show what was archived
    archived = [i for i in items if i["status"] == "archived"]
    if archived:
        print()
        print("  Archived (manual review needed):")
        seen_kinds = set()
        for item in archived:
            label = item["kind"]
            if label in seen_kinds:
                continue
            seen_kinds.add(label)
            reason = item.get("reason", "")
            meta = MIGRATION_OPTION_METADATA.get(label, {})
            display = meta.get("label", label)
            short_reason = reason[:50] + "..." if len(reason) > 50 else reason
            print(f"    ◆ {display:<35s}  {short_reason}")

    # Show conflicts
    conflicts = [i for i in items if i["status"] == "conflict"]
    if conflicts:
        print()
        print("  Conflicts (use --overwrite to force):")
        for item in conflicts:
            print(f"    ⚠ {item['kind']}: {item.get('reason', '')}")

    # Show errors
    errors = [i for i in items if i["status"] == "error"]
    if errors:
        print()
        print("  Errors:")
        for item in errors:
            print(f"    ✖ {item['kind']}: {item.get('reason', '')}")

    # PM2 reassurance
    print()
    print("  ℹ PM2 processes (Discord/Telegram bots) are NOT affected.")

    # Next steps
    if args.execute:
        print()
        print("  Next steps:")
        print("    1. Review ~/.hermes/config.yaml")
        print("    2. Run: hermes mcp list")
        if any(i["kind"] == "cron-jobs" and i["status"] == "archived" for i in items):
            print("    3. Recreate cron jobs: hermes cron")
        if report.get("output_dir"):
            print(f"    → Full report: {report['output_dir']}/MIGRATION_NOTES.md")
    elif not args.execute:
        print()
        print("  This was a dry run. Add --execute to apply changes.")

    print()

    # Also dump JSON for programmatic use
    if os.environ.get("MIGRATION_JSON_OUTPUT"):
        print(json.dumps(report, indent=2, ensure_ascii=False))

    return 0 if s.get("error", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
