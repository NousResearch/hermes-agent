"""hermes import-agent — import Claude Code / Codex CLI setups into Hermes.

Secrets are NEVER imported: credential files are never read, and MCP env vars with secret-looking
names (KEY, TOKEN, SECRET, PASSWORD, ...) are stripped and reported so the user re-adds them via
``hermes setup`` or config.yaml.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import sys
import time
import tomllib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

from utils import atomic_write_text, atomic_yaml_write

logger = logging.getLogger(__name__)

# Entry delimiter of the Hermes memory store (memories/MEMORY.md) and the openclaw script.
ENTRY_DELIMITER = "\n§\n"
# Character budget for merged memory files (openclaw script default).
MEMORY_CHAR_LIMIT = 20_000
SUPPORTED_AGENTS = ("claude-code", "codex")
_AGENT_DEFAULT_DIRS = {"claude-code": ".claude", "codex": ".codex"}
_SKILL_CATEGORY = {"claude-code": "claude-code-imports", "codex": "codex-imports"}

# Env var names that look like credentials — never copied into config.yaml.
_SECRET_KEY_RE = re.compile(
    r"(?:^|_)(?:API[_-]?KEY|APIKEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIALS?|"
    r"AUTH|PRIVATE[_-]?KEY|ACCESS[_-]?KEY)(?:_|$)|KEY$", re.IGNORECASE)


def is_secret_key(key: str) -> bool:
    """Return True when an env-var name looks like a credential."""
    return bool(_SECRET_KEY_RE.search(key or ""))


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).lower()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


class ConfigReadError(RuntimeError):
    """Existing config file present but unreadable/unparseable: the read-modify-write round trip
    must be abandoned, else the merged result would replace real settings with only merged keys."""


def load_yaml_file(path: Path) -> Dict[str, Any]:
    """Load a YAML mapping: absent/empty -> ``{}``; unreadable/unparseable/non-mapping ->
    :class:`ConfigReadError` so the caller refuses and leaves the file byte-identical."""
    if not path.exists():
        return {}
    fix_hint = "Fix it with `hermes config edit` (or move it aside), then re-run the import."

    def refusal(detail: str) -> ConfigReadError:
        return ConfigReadError(f"Refusing to overwrite {path}: {detail}")

    try:
        raw = read_text(path)
    except OSError as exc:
        raise refusal(f"the existing file cannot be read ({exc}). "
                      "Fix the file permissions or move it aside first.") from exc
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise refusal(f"the existing file is not valid YAML ({exc}). {fix_hint}") from exc
    if data is None:  # empty file — a legitimate state with nothing to lose
        return {}
    if not isinstance(data, dict):
        raise refusal(f"expected the existing file to hold a YAML mapping but found "
                      f"{type(data).__name__}. {fix_hint}")
    return data


def dump_yaml_file(path: Path, data: Dict[str, Any]) -> None:
    """Atomic YAML write; only reached after :func:`load_yaml_file` succeeded on the same path."""
    atomic_yaml_write(path, data)


def extract_markdown_entries(text: str) -> List[str]:
    """Split markdown into memory entries: headings become context prefixes; bullets and
    paragraphs become entries; code blocks and tables are skipped."""
    entries: List[str] = []
    headings: List[str] = []
    paragraph_lines: List[str] = []

    def add_entry(content: str) -> None:
        prefix = " > ".join(
            h for h in headings
            if h and not re.search(r"\b(MEMORY|USER|SOUL|AGENTS|TOOLS|IDENTITY|CLAUDE)\.md\b",
                                   h, re.I))
        entries.append(f"{prefix}: {content}" if prefix else content)

    def flush_paragraph() -> None:
        block = " ".join(line.strip() for line in paragraph_lines).strip()
        paragraph_lines.clear()
        if block:
            add_entry(block)

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
            # Drop deeper (and same-level) headings, then push this one.
            headings[len(heading_match.group(1)) - 1:] = [heading_match.group(2).strip()]
            continue
        bullet_match = re.match(r"^\s*(?:[-*]|\d+\.)\s+(.*\S)\s*$", line)
        if bullet_match:
            flush_paragraph()
            add_entry(bullet_match.group(1).strip())
            continue
        if not stripped or (stripped.startswith("|") and stripped.endswith("|")):
            flush_paragraph()  # blank line or table row ends the paragraph
            continue
        paragraph_lines.append(stripped)

    flush_paragraph()
    deduped: List[str] = []
    seen = set()
    for entry in entries:
        normalized = normalize_text(entry)
        if normalized and normalized not in seen:
            seen.add(normalized)
            deduped.append(entry.strip())
    return deduped


def parse_existing_memory_entries(path: Path) -> List[str]:
    """Parse the DESTINATION memory store (``ENTRY_DELIMITER``-split, as ``MemoryStore`` does).
    Do NOT fall back to :func:`extract_markdown_entries`: it drops code blocks/table rows and
    splits bullets, and the merged result overwrites the user's store — the loss is permanent."""
    if not path.exists():
        return []
    return [e.strip() for e in read_text(path).split(ENTRY_DELIMITER) if e.strip()]


def merge_entries(existing: Sequence[str], incoming: Sequence[str],
                  limit: int) -> Tuple[List[str], Dict[str, int]]:
    merged = list(existing)
    seen = {normalize_text(e) for e in existing if e.strip()}
    stats = {"existing": len(existing), "added": 0, "duplicates": 0, "overflowed": 0}
    current_len = len(ENTRY_DELIMITER.join(merged))
    for entry in incoming:
        normalized = normalize_text(entry)
        if not normalized:
            continue
        if normalized in seen:
            stats["duplicates"] += 1
            continue
        candidate_len = (len(entry) if not merged
                         else current_len + len(ENTRY_DELIMITER) + len(entry))
        if candidate_len > limit:
            stats["overflowed"] += 1
            continue
        merged.append(entry)
        seen.add(normalized)
        current_len = candidate_len
        stats["added"] += 1
    return merged, stats


_BASH_RULE_RE = re.compile(r"^Bash\((?P<inner>.*)\)$")


def claude_rule_to_command_pattern(rule: str) -> Optional[str]:
    """``Bash(npm run test:*)`` -> ``npm run test*`` (Claude ':*' is a prefix match). Bare ``Bash``
    (too broad) and non-Bash rules (Claude-only tools, no allowlist equivalent) return None."""
    m = _BASH_RULE_RE.match((rule or "").strip())
    inner = m.group("inner").strip() if m else ""
    if not inner:
        return None
    return inner[:-2] + "*" if inner.endswith(":*") else inner


def detect_agents() -> List[str]:
    """Return the list of supported agents whose default dirs exist."""
    return [a for a in SUPPORTED_AGENTS if (Path.home() / _AGENT_DEFAULT_DIRS[a]).is_dir()]


def sanitize_mcp_env(env: Any) -> Tuple[Dict[str, str], List[str]]:
    """Split an MCP server env dict into (kept, stripped-secret-names)."""
    if not isinstance(env, dict):
        return {}, []
    kept = {str(k): v for k, v in env.items() if not is_secret_key(str(k))}
    return kept, [str(k) for k in env if str(k) not in kept]


def _translate_mcp_server(name: str, srv: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Map one Claude/Codex MCP server entry to Hermes shape; returns (server, stripped secret paths)."""
    hermes_srv: Dict[str, Any] = {}
    stripped: List[str] = []
    if srv.get("command"):
        hermes_srv["command"] = srv["command"]
        if srv.get("args"):
            hermes_srv["args"] = srv["args"]
        env_kept, env_stripped = sanitize_mcp_env(srv.get("env"))
        if env_kept:
            hermes_srv["env"] = env_kept
        stripped.extend(f"mcp_servers.{name}.env.{k}" for k in env_stripped)
        if srv.get("cwd"):
            hermes_srv["cwd"] = srv["cwd"]
    if srv.get("url"):
        hermes_srv["url"] = srv["url"]
        headers = srv.get("headers")
        if isinstance(headers, dict):
            kept_headers = {k: v for k, v in headers.items()
                            if not is_secret_key(str(k)) and "authorization" not in str(k).lower()}
            if kept_headers:
                hermes_srv["headers"] = kept_headers
            stripped.extend(f"mcp_servers.{name}.headers.{k}" for k in headers if k not in kept_headers)
    return hermes_srv, stripped


class AgentImporter:
    """Detect/parse/map/apply importer for one agent source tree. ``execute=False`` runs the full
    plan without touching disk; every item is recorded as imported/skipped/conflict/error."""

    def __init__(self, agent: str, source_root: Path, target_root: Path,
                 execute: bool = False, overwrite: bool = False) -> None:
        if agent not in SUPPORTED_AGENTS:
            raise ValueError(f"Unsupported agent: {agent!r}")
        self.agent = agent
        self.source_root = Path(source_root)
        self.target_root = Path(target_root)
        self.execute = execute
        self.overwrite = overwrite
        self.items: List[Dict[str, Any]] = []
        self.stripped_secrets: List[str] = []

    def record(self, kind: str, source, destination, status: str,
               reason: str = "", **details) -> None:
        self.items.append({"kind": kind, "source": str(source) if source else None,
                           "destination": str(destination) if destination else None,
                           "status": status, "reason": reason, **details})

    def load_target_config(self, kind: str, source,
                           destination: Path) -> Optional[Dict[str, Any]]:
        """Read the destination config.yaml, or record a refusal and return None. Runs in dry-run
        too: ``--dry-run`` must report the refusal, not preview an ``imported`` that destroys it."""
        try:
            return load_yaml_file(destination)
        except ConfigReadError as exc:
            self.record(kind, source, destination, "error", str(exc))
            return None

    def apply(self, kind: str, source, destination, would: str, action,
              details: Optional[Dict[str, Any]] = None) -> None:
        """Record ``imported`` (reason ``would`` in dry-run); in execute mode run ``action`` first.
        ``action`` may add keys to ``details`` (shared by reference) and returns an error string
        (recorded as ``error``) or None."""
        details = details or {}
        error = action() if self.execute else None
        self.record(kind, source, destination, "error" if error else "imported",
                    error or ("" if self.execute else would), **details)

    def build_report(self) -> Dict[str, Any]:
        summary = {"imported": 0, "skipped": 0, "conflict": 0, "error": 0}
        for item in self.items:
            summary[item["status"]] = summary.get(item["status"], 0) + 1
        report: Dict[str, Any] = {"agent": self.agent, "source": str(self.source_root),
                                  "target": str(self.target_root), "dry_run": not self.execute,
                                  "items": self.items, "summary": summary}
        if self.stripped_secrets:
            report["stripped_secrets"] = sorted(set(self.stripped_secrets))
        return report

    def run(self) -> Dict[str, Any]:
        if not self.source_root.is_dir():
            self.record("source", self.source_root, None, "error",
                        "Source directory does not exist")
        else:
            {"claude-code": self._run_claude_code, "codex": self._run_codex}[self.agent]()
        return self.build_report()

    def _run_claude_code(self) -> None:
        settings = self._load_source_mapping(
            "settings", self.source_root / "settings.json", json.loads, self._JSON_ERRORS,
            record_missing=True, non_mapping_error="settings.json is not a JSON object")
        self.import_context_file(self.source_root / "CLAUDE.md", "claude-md")
        self._import_permission_rules(settings, "allow")
        self._import_permission_rules(settings, "deny")
        # mcpServers: ~/.claude.json (preferred; lives NEXT TO ~/.claude/) then settings.json
        claude_json = self._load_source_mapping(
            "mcp-servers", self.source_root.parent / ".claude.json", json.loads, self._JSON_ERRORS)
        servers: Dict[str, Any] = {}
        for source in (claude_json.get("mcpServers"), settings.get("mcpServers")):
            if isinstance(source, dict):
                for name, srv in source.items():
                    servers.setdefault(name, srv)
        self.import_mcp_servers(servers, kind="mcp-servers")
        self.import_skills(self.source_root / "skills")
        commands_dir = self.source_root / "commands"
        if commands_dir.is_dir() and any(commands_dir.glob("*.md")):
            self.record("slash-commands", commands_dir, None, "skipped",
                        "Claude slash commands have no direct Hermes equivalent — "
                        "consider converting them into skills")

    def _run_codex(self) -> None:
        config = self._load_source_mapping("config", self.source_root / "config.toml",
                                           tomllib.loads, Exception, record_missing=True)
        self.import_context_file(self.source_root / "AGENTS.md", "agents-md")
        mcp = config.get("mcp_servers")
        self.import_mcp_servers(mcp if isinstance(mcp, dict) else {}, kind="mcp-servers")
        memories_dir = self.source_root / "memories"
        self._import_markdown_files(
            "memories", memories_dir,
            sorted(memories_dir.glob("*.md")) if memories_dir.is_dir() else None,
            "No memories directory found")
        self.import_skills(self.source_root / "skills")

    _JSON_ERRORS = (json.JSONDecodeError, OSError)

    def _load_source_mapping(self, kind: str, path: Path, parse, errors, *,
                             record_missing: bool = False,
                             non_mapping_error: str = "") -> Dict[str, Any]:
        """Parse ``path`` into a mapping; problems become per-item error records and ``{}``."""
        if not path.exists():
            if record_missing:
                self.record(kind, None, None, "skipped", f"No {path.name} found")
            return {}
        try:
            data = parse(read_text(path))
        except errors as exc:
            self.record(kind, path, None, "error", f"Could not parse {path.name}: {exc}")
            return {}
        if isinstance(data, dict):
            return data
        if non_mapping_error:
            self.record(kind, path, None, "error", non_mapping_error)
        return {}

    def import_context_file(self, source: Path, kind: str) -> None:
        """CLAUDE.md / AGENTS.md → memory entries in memories/MEMORY.md."""
        self._import_markdown_files(kind, source, [source] if source.exists() else None,
                                    f"No {source.name} found", single_file=True)

    def _import_markdown_files(self, kind: str, source: Path, files: Optional[List[Path]],
                               missing_reason: str, single_file: bool = False) -> None:
        """Extract entries from ``files`` (None = source missing) and merge into memories/MEMORY.md.
        An unreadable file records an error; a directory import then still reports "no entries"
        when nothing was extracted, while a single-file import stops at the error."""
        destination = self.target_root / "memories" / "MEMORY.md"
        if files is None:
            self.record(kind, None, destination, "skipped", missing_reason)
            return
        incoming: List[str] = []
        failed = False
        for md_file in files:
            try:
                incoming.extend(extract_markdown_entries(read_text(md_file)))
            except OSError as exc:
                failed = True
                self.record(kind, md_file, destination, "error", f"Could not read file: {exc}")
        if not incoming:
            if not (failed and single_file):
                self.record(kind, source, destination, "skipped", "No importable entries found")
            return
        existing = parse_existing_memory_entries(destination)
        merged, stats = merge_entries(existing, incoming, MEMORY_CHAR_LIMIT)
        details = {"existing_entries": stats["existing"], "added_entries": stats["added"],
                   "duplicate_entries": stats["duplicates"],
                   "overflowed_entries": stats["overflowed"]}
        if stats["added"] == 0:
            self.record(kind, source, destination, "skipped", "No new entries to import", **details)
            return

        def write() -> Optional[str]:
            destination.parent.mkdir(parents=True, exist_ok=True)
            # Snapshot first (``<name>.bak.<unix_ts>``, as MemoryStore does); never rewrite the
            # store when the safety net failed.
            step = "back up existing"
            try:
                if destination.exists():
                    backup = destination.with_suffix(f"{destination.suffix}.bak.{int(time.time())}")
                    shutil.copy2(destination, backup)
                    details["backup"] = str(backup)
                step = "write merged"
                atomic_write_text(destination, ENTRY_DELIMITER.join(merged) + ("\n" if merged else ""))
            except OSError as exc:
                return f"Could not {step} memory file: {exc}"
            return None

        self.apply(kind, source, destination, "Would merge entries", write, details)

    # (settings key, item kind, config path, dry-run tracks unmapped rules)
    _PERMISSION_RULES = {
        "allow": ("command-allowlist", ("command_allowlist",), True),
        "deny": ("command-denylist", ("approvals", "deny"), False)}

    def _import_permission_rules(self, settings: Dict[str, Any], key: str) -> None:
        """settings.json permissions.allow/deny → config.yaml command_allowlist / approvals.deny."""
        kind, config_path, track_unmapped = self._PERMISSION_RULES[key]
        label = f"settings.json permissions.{key}"
        destination = self.target_root / "config.yaml"
        permissions = settings.get("permissions")
        rules = permissions.get(key) if isinstance(permissions, dict) else None
        if not isinstance(rules, list) or not rules:
            self.record(kind, None, destination, "skipped", f"No permissions.{key} rules found")
            return
        mapped = [(r, claude_rule_to_command_pattern(r)) for r in rules if isinstance(r, str)]
        patterns = sorted(dict.fromkeys(p for _, p in mapped if p))
        skipped_rules = [r for r, p in mapped if not p]
        unmapped: Dict[str, Any] = {"unmapped_rules": skipped_rules} if track_unmapped else {}
        if not patterns:
            self.record(kind, None, destination, "skipped",
                        f"No Bash(...) {key} rules to import", **unmapped)
            return
        if not skipped_rules:
            unmapped = {}
        config = self.load_target_config(kind, label, destination)
        if config is None:
            return
        # Walk to the list's parent mapping, materializing missing/invalid levels.
        parent: Dict[str, Any] = config
        for part in config_path[:-1]:
            child = parent.get(part)
            parent[part] = parent = child if isinstance(child, dict) else {}
        current = parent.get(config_path[-1], [])
        current = current if isinstance(current, list) else []
        merged = sorted(dict.fromkeys(list(current) + patterns))
        added = [p for p in merged if p not in current]
        if not added:
            self.record(kind, label, destination, "skipped", "All patterns already present")
            return

        def write() -> None:
            parent[config_path[-1]] = merged
            dump_yaml_file(destination, config)

        self.apply(kind, label, destination, "Would merge patterns", write,
                   {"added_patterns": added, **unmapped})

    def import_mcp_servers(self, servers: Dict[str, Any], kind: str) -> None:
        """mcpServers / [mcp_servers.*] → config.yaml mcp_servers."""
        destination = self.target_root / "config.yaml"
        if not servers:
            self.record(kind, None, destination, "skipped", "No MCP servers found")
            return
        config = self.load_target_config(kind, None, destination)
        if config is None:
            return
        existing = config.get("mcp_servers")
        existing = existing if isinstance(existing, dict) else {}
        added = 0
        for name, srv in servers.items():
            if not isinstance(srv, dict):
                self.record(kind, name, None, "skipped", "Server entry is not a mapping")
                continue
            if name in existing and not self.overwrite:
                self.record(kind, name, f"mcp_servers.{name}", "conflict",
                            "MCP server already exists in Hermes config")
                continue
            hermes_srv, stripped = _translate_mcp_server(name, srv)
            self.stripped_secrets.extend(stripped)
            if not hermes_srv:
                self.record(kind, name, None, "skipped", "Server has neither a command nor a url")
                continue
            existing[name] = hermes_srv
            added += 1
            self.record(kind, name, f"config.yaml mcp_servers.{name}", "imported")
        if added > 0 and self.execute:
            config["mcp_servers"] = existing
            dump_yaml_file(destination, config)

    def import_skills(self, source_root: Path) -> None:
        """skills/<name>/SKILL.md dirs → HERMES_HOME/skills/<category>/<name>."""
        destination_root = self.target_root / "skills" / _SKILL_CATEGORY[self.agent]
        if not source_root.is_dir():
            self.record("skills", None, destination_root, "skipped", "No skills directory found")
            return
        skill_dirs = [p for p in sorted(source_root.iterdir())
                      if p.is_dir() and (p / "SKILL.md").exists()]
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

            def copy(skill_dir=skill_dir, destination=destination) -> None:
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    shutil.rmtree(destination)
                shutil.copytree(skill_dir, destination)

            self.apply("skill", skill_dir, destination, "Would copy skill directory", copy)


def import_agent_command(args) -> None:
    """Handle ``hermes import-agent`` (invoked from hermes_cli.main)."""
    from hermes_cli.config import get_config_path, load_config, save_config
    from hermes_constants import get_hermes_home
    from hermes_cli.setup import (Colors, color, print_header, print_info, print_success,
                                  print_error, prompt_yes_no)

    agent, explicit_source, overwrite = args.agent, args.source, args.overwrite

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
    source_dir = Path(explicit_source or Path.home() / _AGENT_DEFAULT_DIRS[agent])

    print()
    print(color("┌─────────────────────────────────────────────────────────┐", Colors.MAGENTA))
    print(color("│          ⚕ Hermes — Import From Another Agent          │", Colors.MAGENTA))
    print(color("└─────────────────────────────────────────────────────────┘", Colors.MAGENTA))
    if not source_dir.is_dir():
        print()
        print_error(f"Agent directory not found: {source_dir}")
        print_info(f"Specify a custom path: hermes import-agent {agent} --source /path/to/{_AGENT_DEFAULT_DIRS[agent]}")
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
    if not get_config_path().exists():
        save_config(load_config())

    def run_import(execute: bool, phase: str) -> Optional[Dict[str, Any]]:
        """Run the importer; on failure print the error and return None."""
        try:
            return AgentImporter(agent, source_dir.resolve(), hermes_home.resolve(),
                                 execute=execute, overwrite=overwrite).run()
        except Exception as e:
            print()
            print_error(f"Import{phase} failed: {e}")
            logger.debug(f"import-agent{phase} error", exc_info=True)
            return None

    # Phase 1: preview (always)
    preview = run_import(False, " preview")
    if preview is None:
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
    if args.dry_run:
        return

    # Phase 2: confirm and execute
    print()
    if not args.yes:
        if not sys.stdin.isatty():
            print_info("Non-interactive session — preview only.")
            print_info(f"To execute, re-run with: hermes import-agent {agent} --yes")
            return
        if not prompt_yes_no("Proceed with import?", default=True):
            print_info("Import cancelled.")
            return
    report = run_import(True, "")
    if report is None:
        return
    print_import_report(report, dry_run=False)
    print()
    print_success("Import complete.")
    print_info("API keys and credentials were NOT imported — run 'hermes setup' "
               "to configure providers, or add them to ~/.hermes/.env.")


def print_import_report(report: Dict[str, Any], dry_run: bool) -> None:
    """Print a formatted per-item import report (claw-migrate style)."""
    from hermes_cli.setup import Colors, color, print_header, print_info

    print()
    print_header("Dry Run Results" if dry_run else "Import Results")
    if dry_run:
        print_info("No files were modified. This is a preview of what would happen.")
    print()
    # (status, colour, group heading, summary label)
    groups = (
        ("imported", Colors.GREEN, "✓ Would import" if dry_run else "✓ Imported",
         "would import" if dry_run else "imported"),
        ("conflict", Colors.YELLOW, "⚠ Conflicts (skipped — use --overwrite to force)",
         "conflict(s)"),
        ("skipped", Colors.DIM, "─ Skipped", "skipped"),
        ("error", Colors.RED, "✗ Errors", "error(s)"))
    for status, col, label, _ in groups:
        group_items = [i for i in report.get("items", []) if i.get("status") == status]
        if not group_items:
            continue
        print(color(f"  {label}:", col))
        for item in group_items:
            tail = ("→ " + str(item.get("destination") or "").replace(str(Path.home()), "~")
                    if status == "imported" else f" {item.get('reason', '')}")
            print(f"      {item.get('kind', 'unknown'):<22s} {tail}")
        print()
    if stripped := report.get("stripped_secrets"):
        print(color("  ⚷ Secrets stripped (never imported):", Colors.YELLOW))
        for name in stripped:
            print(f"      {name}")
        print_info("Re-add credentials deliberately via 'hermes setup' or ~/.hermes/.env.")
        print()
    summary = report.get("summary", {})
    parts = [f"{summary[k]} {label}" for k, _, _, label in groups if summary.get(k)]
    if parts:
        print_info(f"Summary: {', '.join(parts)}")


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.

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

def default_source_dir(agent: str) -> Path:
    return Path.home() / _AGENT_DEFAULT_DIRS[agent]
# ---- END PLUGIN-COMPAT ----
