"""hermes import-agent — import Claude Code / Codex CLI setups into Hermes.

Usage:
    hermes import-agent                       # auto-detect ~/.claude or ~/.codex
    hermes import-agent claude-code           # import from ~/.claude
    hermes import-agent codex                 # import from ~/.codex
    hermes import-agent claude-code --dry-run # preview only, no changes
    hermes import-agent codex --source /path/to/.codex

Follows the OpenClaw migration pattern (``hermes claw migrate`` /
``optional-skills/migration/openclaw-migration/scripts/openclaw_to_hermes.py``):
detect → parse → map → apply, with a mandatory preview phase, per-item
imported/skipped/conflict/error records, and a ``--dry-run`` that writes
nothing.  The memory-entry merge and allowlist-merge primitives here are
self-contained ports of the openclaw script's equivalents so this command
works even when the optional migration skill is not installed.

Mappings
--------
claude-code (~/.claude):
    CLAUDE.md                       → memory entries in HERMES_HOME/memories/MEMORY.md
    settings.json permissions.allow → config.yaml command_allowlist (Bash(...) rules)
    settings.json permissions.deny  → config.yaml approvals.deny (Bash(...) rules)
    mcpServers (~/.claude.json or settings.json) → config.yaml mcp_servers
    skills/<name>/SKILL.md          → HERMES_HOME/skills/claude-code-imports/<name>/

codex (~/.codex):
    AGENTS.md                       → memory entries in HERMES_HOME/memories/MEMORY.md
    config.toml [mcp_servers.*]     → config.yaml mcp_servers
    memories/*.md                   → memory entries in HERMES_HOME/memories/MEMORY.md
    skills/<name>/SKILL.md          → HERMES_HOME/skills/codex-imports/<name>/

Secrets are NEVER imported: credential files (.credentials.json, auth.json)
are ignored, and MCP server env vars with secret-looking names (KEY, TOKEN,
SECRET, PASSWORD, ...) are stripped and reported so the user can re-add them
deliberately via ``hermes setup`` or config.yaml.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from utils import atomic_write_text, atomic_yaml_write

logger = logging.getLogger(__name__)

# Same entry delimiter as the Hermes memory store and the openclaw migration
# script — memories/MEMORY.md entries are separated by bare "§" lines.
ENTRY_DELIMITER = "\n§\n"

# Character budget for merged memory files (matches the openclaw script's
# default memory limit).
MEMORY_CHAR_LIMIT = 20_000

SUPPORTED_AGENTS = ("claude-code", "codex")

_AGENT_DEFAULT_DIRS = {
    "claude-code": ".claude",
    "codex": ".codex",
}

_SKILL_CATEGORY = {
    "claude-code": "claude-code-imports",
    "codex": "codex-imports",
}

# Env var names that look like credentials — never copied into config.yaml.
_SECRET_KEY_RE = re.compile(
    r"(?:^|_)(?:API[_-]?KEY|APIKEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIALS?|"
    r"AUTH|PRIVATE[_-]?KEY|ACCESS[_-]?KEY)(?:_|$)|KEY$",
    re.IGNORECASE,
)

# Files inside the source tree that hold credentials — never read.
_CREDENTIAL_FILENAMES = (".credentials.json", "auth.json", "credentials.json")


def is_secret_key(key: str) -> bool:
    """Return True when an env-var name looks like a credential."""
    return bool(_SECRET_KEY_RE.search(key or ""))


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).lower()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


class ConfigReadError(RuntimeError):
    """An existing config file is present but cannot be read or parsed.

    Signals that a read-modify-write round trip must be abandoned: the caller
    has no idea what the file holds, so writing a merged result back would
    replace real settings with only the keys it merged.
    """


def load_yaml_file(path: Path) -> Dict[str, Any]:
    """Load a YAML mapping, distinguishing "absent" from "unreadable".

    Callers read ``config.yaml``, merge a section in, and write the whole
    mapping back — so collapsing a present-but-unreadable file to ``{}``
    would replace every existing setting with just the merged keys.

    - Absent, or present but empty  -> ``{}``; first-time creation still works.
    - Present but unreadable, unparseable, or not a mapping -> raise
      :class:`ConfigReadError` so the caller refuses and leaves the file
      byte-identical.
    """
    import yaml

    if not path.exists():
        return {}
    try:
        raw = read_text(path)
    except OSError as exc:
        raise ConfigReadError(
            f"Refusing to overwrite {path}: the existing file cannot be read "
            f"({exc}). Fix the file permissions or move it aside first."
        ) from exc
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ConfigReadError(
            f"Refusing to overwrite {path}: the existing file is not valid YAML "
            f"({exc}). Fix it with `hermes config edit` (or move it aside), then "
            f"re-run the import."
        ) from exc
    # An empty file parses to None — a legitimate state with nothing to lose.
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigReadError(
            f"Refusing to overwrite {path}: expected the existing file to hold a "
            f"YAML mapping but found {type(data).__name__}. Fix it with "
            f"`hermes config edit` (or move it aside), then re-run the import."
        )
    return data


def dump_yaml_file(path: Path, data: Dict[str, Any]) -> None:
    """Write ``data`` as YAML atomically (temp file + fsync + rename).

    Only ever reached after :func:`load_yaml_file` has successfully read the
    same path, so the mapping being written is the real file's content plus
    the merged section — never a silently-empty stand-in.

    ``atomic_yaml_write`` keeps a symlinked config a symlink, creates the
    parent directory, and preserves the previous file's mode/owner (a
    ``0o600``-secured config stays ``0o600``).
    """
    atomic_yaml_write(path, data)


# ---------------------------------------------------------------------------
# Memory-entry primitives (ported from openclaw_to_hermes.py)
# ---------------------------------------------------------------------------

def extract_markdown_entries(text: str) -> List[str]:
    """Split a markdown document into individual memory entries.

    Headings become context prefixes, bullets and paragraphs become entries.
    Code blocks and tables are skipped.  Port of the openclaw migration
    script's extractor.
    """
    entries: List[str] = []
    headings: List[str] = []
    paragraph_lines: List[str] = []

    def context_prefix() -> str:
        filtered = [
            h for h in headings
            if h and not re.search(
                r"\b(MEMORY|USER|SOUL|AGENTS|TOOLS|IDENTITY|CLAUDE)\.md\b",
                h, re.I,
            )
        ]
        return " > ".join(filtered)

    def flush_paragraph() -> None:
        nonlocal paragraph_lines
        if not paragraph_lines:
            return
        block = " ".join(line.strip() for line in paragraph_lines).strip()
        paragraph_lines = []
        if not block:
            return
        prefix = context_prefix()
        entries.append(f"{prefix}: {block}" if prefix else block)

    in_code_block = False
    for raw_line in (text or "").splitlines():
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
            value = heading_match.group(2).strip()
            while len(headings) >= level:
                headings.pop()
            headings.append(value)
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


def parse_existing_memory_entries(path: Path) -> List[str]:
    """Parse the DESTINATION memory store into entries.

    ``memories/MEMORY.md`` is the entry-delimited store written by
    ``MemoryStore._write_file`` (tools/memory_tool.py), not a markdown
    document, so this splits on ``ENTRY_DELIMITER`` only — exactly what
    ``MemoryStore._parse_entries`` does.  A store with no delimiter (a single
    entry, or one that was hand-edited / shell-appended) is therefore ONE
    intact entry.

    Do NOT fall back to :func:`extract_markdown_entries` here.  That extractor
    is correct for CLAUDE.md / AGENTS.md *sources*, but it drops fenced code
    blocks and table rows and splits a block into one entry per bullet — and
    the merged result is written straight back over the user's store, so the
    loss is permanent.
    """
    if not path.exists():
        return []
    raw = read_text(path)
    if not raw.strip():
        return []
    return [e.strip() for e in raw.split(ENTRY_DELIMITER) if e.strip()]


def backup_memory_file(path: Path) -> Optional[Path]:
    """Snapshot ``path`` before a destructive rewrite; return the backup path.

    Restores parity with the openclaw migration script this module was ported
    from, which calls ``maybe_backup(destination)`` before rewriting a memory
    store.  Uses the same ``<name>.bak.<unix_ts>`` naming as
    ``MemoryStore._backup_drifted_file``.  Returns None when there is nothing
    to back up.
    """
    if not path.exists():
        return None
    backup = path.with_suffix(path.suffix + f".bak.{int(time.time())}")
    shutil.copy2(path, backup)
    return backup



def merge_entries(
    existing: Sequence[str],
    incoming: Sequence[str],
    limit: int,
) -> Tuple[List[str], Dict[str, int]]:
    merged = list(existing)
    seen = {normalize_text(e) for e in existing if e.strip()}
    stats = {"existing": len(existing), "added": 0, "duplicates": 0, "overflowed": 0}

    current_len = len(ENTRY_DELIMITER.join(merged)) if merged else 0
    for entry in incoming:
        normalized = normalize_text(entry)
        if not normalized:
            continue
        if normalized in seen:
            stats["duplicates"] += 1
            continue
        candidate_len = (
            len(entry) if not merged
            else current_len + len(ENTRY_DELIMITER) + len(entry)
        )
        if candidate_len > limit:
            stats["overflowed"] += 1
            continue
        merged.append(entry)
        seen.add(normalized)
        current_len = candidate_len
        stats["added"] += 1
    return merged, stats


# ---------------------------------------------------------------------------
# Claude Code permission rules → Hermes command patterns
# ---------------------------------------------------------------------------

_BASH_RULE_RE = re.compile(r"^Bash\((?P<inner>.*)\)$")


def claude_rule_to_command_pattern(rule: str) -> Optional[str]:
    """Convert a Claude Code ``Bash(...)`` permission rule into a Hermes glob.

    ``Bash(npm run build)``   → ``npm run build``
    ``Bash(npm run test:*)``  → ``npm run test*``  (Claude ':*' prefix match)
    ``Bash(git diff *)``      → ``git diff *``
    ``Bash``                  → None (blanket rule, too broad to import)
    Non-Bash rules (``Read(...)``, ``WebFetch(...)``, ...) → None: they gate
    Claude-specific tools with no command-allowlist equivalent.
    """
    rule = (rule or "").strip()
    m = _BASH_RULE_RE.match(rule)
    if not m:
        return None
    inner = m.group("inner").strip()
    if not inner:
        return None
    if inner.endswith(":*"):
        inner = inner[:-2] + "*"
    return inner


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def default_source_dir(agent: str) -> Path:
    return Path.home() / _AGENT_DEFAULT_DIRS[agent]


def detect_agents() -> List[str]:
    """Return the list of supported agents whose default dirs exist."""
    return [a for a in SUPPORTED_AGENTS if default_source_dir(a).is_dir()]


def sanitize_mcp_env(env: Any) -> Tuple[Dict[str, str], List[str]]:
    """Split an MCP server env dict into (kept, stripped-secret-names)."""
    kept: Dict[str, str] = {}
    stripped: List[str] = []
    if not isinstance(env, dict):
        return kept, stripped
    for key, value in env.items():
        if is_secret_key(str(key)):
            stripped.append(str(key))
        else:
            kept[str(key)] = value
    return kept, stripped


# ---------------------------------------------------------------------------
# Importer
# ---------------------------------------------------------------------------

class AgentImporter:
    """Detect/parse/map/apply importer for a single agent source tree.

    ``execute=False`` runs the full plan without touching disk (dry run);
    ``execute=True`` applies it.  Every item is recorded as
    imported/skipped/conflict/error with a human-readable reason so the CLI
    can print a per-item report.
    """

    def __init__(
        self,
        agent: str,
        source_root: Path,
        target_root: Path,
        execute: bool = False,
        overwrite: bool = False,
    ) -> None:
        if agent not in SUPPORTED_AGENTS:
            raise ValueError(f"Unsupported agent: {agent!r}")
        self.agent = agent
        self.source_root = Path(source_root)
        self.target_root = Path(target_root)
        self.execute = execute
        self.overwrite = overwrite
        self.items: List[Dict[str, Any]] = []
        self.stripped_secrets: List[str] = []

    # -- reporting ---------------------------------------------------------

    def record(self, kind: str, source, destination, status: str,
               reason: str = "", **details) -> None:
        item: Dict[str, Any] = {
            "kind": kind,
            "source": str(source) if source else None,
            "destination": str(destination) if destination else None,
            "status": status,
            "reason": reason,
        }
        if details:
            item.update(details)
        self.items.append(item)

    def load_target_config(self, kind: str, source, destination: Path
                           ) -> Optional[Dict[str, Any]]:
        """Read the destination config.yaml, or record a refusal and return None.

        The single chokepoint for the three importers that read config.yaml,
        merge a section into it and write it back.  When the existing file is
        present but unreadable there is nothing safe to merge into, so the
        item is recorded as an ``error`` and the file is left untouched —
        rather than being replaced by the merged section alone.

        Deliberately runs in dry-run too: ``--dry-run`` must report the
        refusal, not preview an ``imported`` that would destroy the config.
        """
        try:
            return load_yaml_file(destination)
        except ConfigReadError as exc:
            self.record(kind, source, destination, "error", str(exc))
            return None

    def build_report(self) -> Dict[str, Any]:
        summary = {"imported": 0, "skipped": 0, "conflict": 0, "error": 0}
        for item in self.items:
            status = item.get("status", "skipped")
            summary[status] = summary.get(status, 0) + 1
        report: Dict[str, Any] = {
            "agent": self.agent,
            "source": str(self.source_root),
            "target": str(self.target_root),
            "dry_run": not self.execute,
            "items": self.items,
            "summary": summary,
        }
        if self.stripped_secrets:
            report["stripped_secrets"] = sorted(set(self.stripped_secrets))
        return report

    # -- orchestration -----------------------------------------------------

    def run(self) -> Dict[str, Any]:
        if not self.source_root.is_dir():
            self.record("source", self.source_root, None, "error",
                        "Source directory does not exist")
            return self.build_report()
        if self.agent == "claude-code":
            self._run_claude_code()
        else:
            self._run_codex()
        return self.build_report()

    def _run_claude_code(self) -> None:
        settings = self._load_claude_settings()
        self.import_context_file(self.source_root / "CLAUDE.md", kind="claude-md")
        self.import_permission_allowlist(settings)
        self.import_permission_denylist(settings)
        self.import_mcp_servers(self._claude_mcp_servers(settings), kind="mcp-servers")
        self.import_skills(self.source_root / "skills")
        commands_dir = self.source_root / "commands"
        if commands_dir.is_dir() and any(commands_dir.glob("*.md")):
            self.record(
                "slash-commands", commands_dir, None, "skipped",
                "Claude slash commands have no direct Hermes equivalent — "
                "consider converting them into skills",
            )

    def _run_codex(self) -> None:
        config = self._load_codex_config()
        self.import_context_file(self.source_root / "AGENTS.md", kind="agents-md")
        mcp = config.get("mcp_servers") if isinstance(config, dict) else None
        self.import_mcp_servers(mcp if isinstance(mcp, dict) else {},
                                kind="mcp-servers")
        self.import_memories_dir(self.source_root / "memories")
        self.import_skills(self.source_root / "skills")

    # -- parsers (fail soft: bad files become per-item error records) -------

    def _load_claude_settings(self) -> Dict[str, Any]:
        path = self.source_root / "settings.json"
        if not path.exists():
            self.record("settings", None, None, "skipped",
                        "No settings.json found")
            return {}
        try:
            data = json.loads(read_text(path))
        except (json.JSONDecodeError, OSError) as exc:
            self.record("settings", path, None, "error",
                        f"Could not parse settings.json: {exc}")
            return {}
        if not isinstance(data, dict):
            self.record("settings", path, None, "error",
                        "settings.json is not a JSON object")
            return {}
        return data

    def _claude_mcp_servers(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Collect mcpServers from ~/.claude.json (preferred) and settings.json."""
        servers: Dict[str, Any] = {}
        # ~/.claude.json lives NEXT TO ~/.claude/, not inside it
        claude_json = self.source_root.parent / ".claude.json"
        if claude_json.exists():
            try:
                data = json.loads(read_text(claude_json))
                if isinstance(data, dict) and isinstance(data.get("mcpServers"), dict):
                    servers.update(data["mcpServers"])
            except (json.JSONDecodeError, OSError) as exc:
                self.record("mcp-servers", claude_json, None, "error",
                            f"Could not parse .claude.json: {exc}")
        from_settings = settings.get("mcpServers")
        if isinstance(from_settings, dict):
            for name, srv in from_settings.items():
                servers.setdefault(name, srv)
        return servers

    def _load_codex_config(self) -> Dict[str, Any]:
        path = self.source_root / "config.toml"
        if not path.exists():
            self.record("config", None, None, "skipped",
                        "No config.toml found")
            return {}
        try:
            import tomllib
            data = tomllib.loads(read_text(path))
        except Exception as exc:
            self.record("config", path, None, "error",
                        f"Could not parse config.toml: {exc}")
            return {}
        return data if isinstance(data, dict) else {}

    # -- mappers -------------------------------------------------------------

    def import_context_file(self, source: Path, kind: str) -> None:
        """CLAUDE.md / AGENTS.md → memory entries in memories/MEMORY.md."""
        destination = self.target_root / "memories" / "MEMORY.md"
        if not source.exists():
            self.record(kind, None, destination, "skipped",
                        f"No {source.name} found")
            return
        try:
            incoming = extract_markdown_entries(read_text(source))
        except OSError as exc:
            self.record(kind, source, destination, "error",
                        f"Could not read file: {exc}")
            return
        if not incoming:
            self.record(kind, source, destination, "skipped",
                        "No importable entries found")
            return
        self._merge_memory_entries(kind, source, destination, incoming)

    def import_memories_dir(self, memories_dir: Path) -> None:
        """codex memories/*.md → memory entries in memories/MEMORY.md."""
        destination = self.target_root / "memories" / "MEMORY.md"
        if not memories_dir.is_dir():
            self.record("memories", None, destination, "skipped",
                        "No memories directory found")
            return
        incoming: List[str] = []
        for md_file in sorted(memories_dir.glob("*.md")):
            try:
                incoming.extend(extract_markdown_entries(read_text(md_file)))
            except OSError as exc:
                self.record("memories", md_file, destination, "error",
                            f"Could not read file: {exc}")
        if not incoming:
            self.record("memories", memories_dir, destination, "skipped",
                        "No importable entries found")
            return
        self._merge_memory_entries("memories", memories_dir, destination, incoming)

    def _merge_memory_entries(self, kind: str, source: Path,
                              destination: Path, incoming: List[str]) -> None:
        existing = parse_existing_memory_entries(destination)
        merged, stats = merge_entries(existing, incoming, MEMORY_CHAR_LIMIT)
        details = {
            "existing_entries": stats["existing"],
            "added_entries": stats["added"],
            "duplicate_entries": stats["duplicates"],
            "overflowed_entries": stats["overflowed"],
        }
        if stats["added"] == 0:
            self.record(kind, source, destination, "skipped",
                        "No new entries to import", **details)
            return
        if self.execute:
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                backup = backup_memory_file(destination)
            except OSError as exc:
                # Never rewrite the store when the safety net failed.
                self.record(kind, source, destination, "error",
                            f"Could not back up existing memory file: {exc}",
                            **details)
                return
            if backup is not None:
                details["backup"] = str(backup)
            try:
                atomic_write_text(
                    destination,
                    ENTRY_DELIMITER.join(merged) + ("\n" if merged else ""),
                )
            except OSError as exc:
                self.record(kind, source, destination, "error",
                            f"Could not write merged memory file: {exc}",
                            **details)
                return
            self.record(kind, source, destination, "imported", **details)
        else:
            self.record(kind, source, destination, "imported",
                        "Would merge entries", **details)

    def import_permission_allowlist(self, settings: Dict[str, Any]) -> None:
        """settings.json permissions.allow → config.yaml command_allowlist."""
        destination = self.target_root / "config.yaml"
        permissions = settings.get("permissions")
        allow = permissions.get("allow") if isinstance(permissions, dict) else None
        if not isinstance(allow, list) or not allow:
            self.record("command-allowlist", None, destination, "skipped",
                        "No permissions.allow rules found")
            return

        patterns: List[str] = []
        skipped_rules: List[str] = []
        for rule in allow:
            if not isinstance(rule, str):
                continue
            pattern = claude_rule_to_command_pattern(rule)
            if pattern:
                patterns.append(pattern)
            else:
                skipped_rules.append(rule)
        patterns = sorted(dict.fromkeys(patterns))
        if not patterns:
            self.record("command-allowlist", None, destination, "skipped",
                        "No Bash(...) allow rules to import",
                        unmapped_rules=skipped_rules)
            return

        config = self.load_target_config(
            "command-allowlist", "settings.json permissions.allow", destination)
        if config is None:
            return
        current = config.get("command_allowlist", [])
        if not isinstance(current, list):
            current = []
        merged = sorted(dict.fromkeys(list(current) + patterns))
        added = [p for p in merged if p not in current]
        if not added:
            self.record("command-allowlist", "settings.json permissions.allow",
                        destination, "skipped", "All patterns already present")
            return
        details: Dict[str, Any] = {"added_patterns": added}
        if skipped_rules:
            details["unmapped_rules"] = skipped_rules
        if self.execute:
            config["command_allowlist"] = merged
            dump_yaml_file(destination, config)
            self.record("command-allowlist", "settings.json permissions.allow",
                        destination, "imported", **details)
        else:
            self.record("command-allowlist", "settings.json permissions.allow",
                        destination, "imported", "Would merge patterns", **details)

    def import_permission_denylist(self, settings: Dict[str, Any]) -> None:
        """settings.json permissions.deny → config.yaml approvals.deny."""
        destination = self.target_root / "config.yaml"
        permissions = settings.get("permissions")
        deny = permissions.get("deny") if isinstance(permissions, dict) else None
        if not isinstance(deny, list) or not deny:
            self.record("command-denylist", None, destination, "skipped",
                        "No permissions.deny rules found")
            return

        patterns: List[str] = []
        for rule in deny:
            if not isinstance(rule, str):
                continue
            pattern = claude_rule_to_command_pattern(rule)
            if pattern:
                patterns.append(pattern)
        patterns = sorted(dict.fromkeys(patterns))
        if not patterns:
            self.record("command-denylist", None, destination, "skipped",
                        "No Bash(...) deny rules to import")
            return

        config = self.load_target_config(
            "command-denylist", "settings.json permissions.deny", destination)
        if config is None:
            return
        approvals = config.get("approvals")
        if not isinstance(approvals, dict):
            approvals = {}
        current = approvals.get("deny", [])
        if not isinstance(current, list):
            current = []
        merged = sorted(dict.fromkeys(list(current) + patterns))
        added = [p for p in merged if p not in current]
        if not added:
            self.record("command-denylist", "settings.json permissions.deny",
                        destination, "skipped", "All patterns already present")
            return
        if self.execute:
            approvals["deny"] = merged
            config["approvals"] = approvals
            dump_yaml_file(destination, config)
            self.record("command-denylist", "settings.json permissions.deny",
                        destination, "imported", added_patterns=added)
        else:
            self.record("command-denylist", "settings.json permissions.deny",
                        destination, "imported", "Would merge patterns",
                        added_patterns=added)

    def import_mcp_servers(self, servers: Dict[str, Any], kind: str) -> None:
        """mcpServers / [mcp_servers.*] → config.yaml mcp_servers."""
        destination = self.target_root / "config.yaml"
        if not servers:
            self.record(kind, None, destination, "skipped",
                        "No MCP servers found")
            return

        config = self.load_target_config(kind, None, destination)
        if config is None:
            return
        existing = config.get("mcp_servers")
        if not isinstance(existing, dict):
            existing = {}
        added = 0
        for name, srv in servers.items():
            if not isinstance(srv, dict):
                self.record(kind, name, None, "skipped",
                            "Server entry is not a mapping")
                continue
            if name in existing and not self.overwrite:
                self.record(kind, name, f"mcp_servers.{name}", "conflict",
                            "MCP server already exists in Hermes config")
                continue

            hermes_srv: Dict[str, Any] = {}
            if srv.get("command"):
                hermes_srv["command"] = srv["command"]
                if srv.get("args"):
                    hermes_srv["args"] = srv["args"]
                env_kept, env_stripped = sanitize_mcp_env(srv.get("env"))
                if env_kept:
                    hermes_srv["env"] = env_kept
                if env_stripped:
                    self.stripped_secrets.extend(
                        f"mcp_servers.{name}.env.{k}" for k in env_stripped
                    )
                if srv.get("cwd"):
                    hermes_srv["cwd"] = srv["cwd"]
            if srv.get("url"):
                hermes_srv["url"] = srv["url"]
                headers = srv.get("headers")
                if isinstance(headers, dict):
                    kept_headers = {
                        k: v for k, v in headers.items()
                        if not is_secret_key(str(k))
                        and "authorization" not in str(k).lower()
                    }
                    if kept_headers:
                        hermes_srv["headers"] = kept_headers
                    for k in headers:
                        if k not in kept_headers:
                            self.stripped_secrets.append(
                                f"mcp_servers.{name}.headers.{k}"
                            )
            if not hermes_srv:
                self.record(kind, name, None, "skipped",
                            "Server has neither a command nor a url")
                continue

            existing[name] = hermes_srv
            added += 1
            self.record(kind, name, f"config.yaml mcp_servers.{name}",
                        "imported")

        if added > 0 and self.execute:
            config["mcp_servers"] = existing
            dump_yaml_file(destination, config)

    def import_skills(self, source_root: Path) -> None:
        """skills/<name>/SKILL.md dirs → HERMES_HOME/skills/<category>/<name>."""
        category = _SKILL_CATEGORY[self.agent]
        destination_root = self.target_root / "skills" / category
        if not source_root.is_dir():
            self.record("skills", None, destination_root, "skipped",
                        "No skills directory found")
            return
        skill_dirs = [
            p for p in sorted(source_root.iterdir())
            if p.is_dir() and (p / "SKILL.md").exists()
        ]
        if not skill_dirs:
            self.record("skills", source_root, destination_root, "skipped",
                        "No skills with SKILL.md found")
            return
        for skill_dir in skill_dirs:
            destination = destination_root / skill_dir.name
            if destination.exists() and not self.overwrite:
                self.record("skill", skill_dir, destination, "conflict",
                            "Destination skill already exists")
                continue
            if self.execute:
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    shutil.rmtree(destination)
                shutil.copytree(skill_dir, destination)
                self.record("skill", skill_dir, destination, "imported")
            else:
                self.record("skill", skill_dir, destination, "imported",
                            "Would copy skill directory")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def import_agent_command(args) -> None:
    """Handle ``hermes import-agent`` (invoked from hermes_cli.main)."""
    from hermes_cli.config import get_config_path, load_config, save_config
    from hermes_constants import get_hermes_home
    from hermes_cli.setup import (
        Colors,
        color,
        print_header,
        print_info,
        print_success,
        print_error,
        prompt_yes_no,
    )

    agent = getattr(args, "agent", None)
    explicit_source = getattr(args, "source", None)
    dry_run = getattr(args, "dry_run", False)
    overwrite = getattr(args, "overwrite", False)
    auto_yes = getattr(args, "yes", False)

    # -- detect --------------------------------------------------------------
    if agent is None:
        detected = detect_agents()
        if not detected:
            print()
            print_error("No supported agent setup found (~/.claude or ~/.codex).")
            print_info("Specify one explicitly: hermes import-agent claude-code --source /path")
            return
        if len(detected) > 1 and explicit_source is None:
            print()
            print_info("Multiple agent setups detected: " + ", ".join(detected))
            print_info("Pick one: hermes import-agent claude-code   or   hermes import-agent codex")
            return
        agent = detected[0]

    source_dir = Path(explicit_source) if explicit_source else default_source_dir(agent)

    print()
    print(color("┌─────────────────────────────────────────────────────────┐", Colors.MAGENTA))
    print(color("│          ⚕ Hermes — Import From Another Agent          │", Colors.MAGENTA))
    print(color("└─────────────────────────────────────────────────────────┘", Colors.MAGENTA))

    if not source_dir.is_dir():
        print()
        print_error(f"Agent directory not found: {source_dir}")
        print_info("Specify a custom path: hermes import-agent "
                    f"{agent} --source /path/to/{_AGENT_DEFAULT_DIRS[agent]}")
        return

    hermes_home = get_hermes_home()
    print()
    print_header("Import Settings")
    print_info(f"Agent:       {agent}")
    print_info(f"Source:      {source_dir}")
    print_info(f"Target:      {hermes_home}")
    print_info(f"Overwrite:   {'yes' if overwrite else 'no (skip conflicts)'}")
    print_info("Secrets:     never imported — run 'hermes setup' for credentials")

    # Ensure config.yaml exists before the import tries to merge into it
    config_path = get_config_path()
    if not config_path.exists():
        save_config(load_config())

    # -- Phase 1: preview (always) --------------------------------------------
    try:
        preview = AgentImporter(
            agent=agent,
            source_root=source_dir.resolve(),
            target_root=hermes_home.resolve(),
            execute=False,
            overwrite=overwrite,
        ).run()
    except Exception as e:
        print()
        print_error(f"Import preview failed: {e}")
        logger.debug("import-agent preview error", exc_info=True)
        return

    summary = preview.get("summary", {})
    if summary.get("imported", 0) == 0 and summary.get("conflict", 0) == 0:
        print()
        print_info(f"Nothing to import from {agent}.")
        print_import_report(preview, dry_run=True)
        return

    print()
    print_header(f"Import Preview — {summary.get('imported', 0)} item(s) would be imported")
    print_info("No changes have been made yet. Review the list below:")
    print_import_report(preview, dry_run=True)

    if dry_run:
        return

    # -- Phase 2: confirm and execute -----------------------------------------
    print()
    if not auto_yes:
        if not sys.stdin.isatty():
            print_info("Non-interactive session — preview only.")
            print_info(f"To execute, re-run with: hermes import-agent {agent} --yes")
            return
        if not prompt_yes_no("Proceed with import?", default=True):
            print_info("Import cancelled.")
            return

    try:
        report = AgentImporter(
            agent=agent,
            source_root=source_dir.resolve(),
            target_root=hermes_home.resolve(),
            execute=True,
            overwrite=overwrite,
        ).run()
    except Exception as e:
        print()
        print_error(f"Import failed: {e}")
        logger.debug("import-agent error", exc_info=True)
        return

    print_import_report(report, dry_run=False)
    print()
    print_success("Import complete.")
    print_info("API keys and credentials were NOT imported — run 'hermes setup' "
               "to configure providers, or add them to ~/.hermes/.env.")


def print_import_report(report: Dict[str, Any], dry_run: bool) -> None:
    """Print a formatted per-item import report (claw-migrate style)."""
    from hermes_cli.setup import Colors, color, print_header, print_info

    summary = report.get("summary", {})
    print()
    if dry_run:
        print_header("Dry Run Results")
        print_info("No files were modified. This is a preview of what would happen.")
    else:
        print_header("Import Results")
    print()

    items = report.get("items", [])
    groups = (
        ("imported", Colors.GREEN, "✓ Would import" if dry_run else "✓ Imported"),
        ("conflict", Colors.YELLOW, "⚠ Conflicts (skipped — use --overwrite to force)"),
        ("skipped", Colors.DIM, "─ Skipped"),
        ("error", Colors.RED, "✗ Errors"),
    )
    for status, col, label in groups:
        group_items = [i for i in items if i.get("status") == status]
        if not group_items:
            continue
        print(color(f"  {label}:", col))
        for item in group_items:
            kind = item.get("kind", "unknown")
            if status == "imported":
                dest = item.get("destination") or ""
                dest_short = str(dest).replace(str(Path.home()), "~")
                print(f"      {kind:<22s} → {dest_short}")
            else:
                reason = item.get("reason", "")
                print(f"      {kind:<22s}  {reason}")
        print()

    stripped = report.get("stripped_secrets") or []
    if stripped:
        print(color("  ⚷ Secrets stripped (never imported):", Colors.YELLOW))
        for name in stripped:
            print(f"      {name}")
        print_info("Re-add credentials deliberately via 'hermes setup' or ~/.hermes/.env.")
        print()

    parts = []
    if summary.get("imported"):
        action = "would import" if dry_run else "imported"
        parts.append(f"{summary['imported']} {action}")
    if summary.get("conflict"):
        parts.append(f"{summary['conflict']} conflict(s)")
    if summary.get("skipped"):
        parts.append(f"{summary['skipped']} skipped")
    if summary.get("error"):
        parts.append(f"{summary['error']} error(s)")
    if parts:
        print_info(f"Summary: {', '.join(parts)}")
