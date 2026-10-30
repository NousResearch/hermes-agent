"""WATCHDOG.yml discovery, validation and serialization for the native adviser.

Port of oh-my-pi's ``coding-agent/src/advisor/config.ts`` + ``watchdog.ts`` with the
search path adapted to Hermes: the user level is ``$HERMES_HOME`` and the project
level probes ``<dir>/.hermes/WATCHDOG.yml`` and ``<dir>/WATCHDOG.yml`` from the
session cwd up to the git root (or ``$HOME``). An entry from a more specific file
replaces the whole earlier entry with the same slug; top-level ``instructions``
concatenate user-first so the leaf file reads last.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import yaml

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

CONFIG_FILENAMES = ("WATCHDOG.yml", "WATCHDOG.yaml")
WATCHDOG_MD_FILENAME = "WATCHDOG.md"
PROJECT_CONFIG_DIR = ".hermes"

REVIEW_MODES = ("turn", "agent-end")
SYNC_BACKLOG_MODES = ("off", "1", "3", "5", "strict")
ADVISER_SCOPES = ("main", "subagent", "all")
_SYNC_BACKLOG_NUMERIC = {1: "1", 3: "3", 5: "5"}

#: oh-my-pi release this adviser port is pinned to (advisor subsystem, session roster and
#: prompts). Bump all three together with ``website/docs/user-guide/features/adviser.md``
#: when re-porting; ``UPSTREAM_ADVISER_COMMIT`` is the ``v18.14.0`` tag commit.
UPSTREAM_ADVISER_REPO = "can1357/oh-my-pi"
UPSTREAM_ADVISER_RELEASE = "18.14.0"
UPSTREAM_ADVISER_COMMIT = "31147f9d1a76608549821c3759f8487b4d66c72d"
UPSTREAM_ADVISER_PATHS = (
    "packages/coding-agent/src/advisor/",
    "packages/coding-agent/src/session/session-advisors.ts",
    "packages/coding-agent/src/prompts/advisor/",
    "docs/advisor-watchdog.md",
)

#: The read-only investigative set an adviser gets when its entry omits ``tools``.
DEFAULT_TOOL_NAMES = ("read_file", "search_files")

#: Legacy oh-my-pi tool names and the Hermes tool that serves the same purpose.
TOOL_ALIASES = {
    "read": "read_file",
    "cat": "read_file",
    "grep": "search_files",
    "glob": "search_files",
    "find": "search_files",
    "search": "search_files",
}

_SLUG_STRIP = re.compile(r"[^a-z0-9]+")
_AT_IMPORT = re.compile(r"(?<![\w`])@([A-Za-z0-9_./\\~-]+)")
_FENCE = re.compile(r"```")

MAX_NOTES_CAP = 32


@dataclass
class Candidate:
    path: str
    content: str
    level: str  # "user" | "project"
    depth: int


@dataclass
class AdviserConfig:
    name: str
    slug: str = ""
    model: Optional[str] = None
    provider: Optional[str] = None
    tools: Optional[list[str]] = None
    review_mode: Optional[str] = None
    review_interval: Optional[int] = None
    sync_backlog: Optional[str] = None
    scope: Optional[str] = None
    include_reasoning: Optional[bool] = None
    instructions: Optional[str] = None
    enabled: Optional[bool] = None
    max_notes_per_update: Optional[int] = None
    source: str = ""


@dataclass
class DiscoveredAdvisers:
    advisers: list[AdviserConfig] = field(default_factory=list)
    shared_instructions: Optional[str] = None
    shared_max_notes_per_update: Optional[int] = None
    warnings: list[str] = field(default_factory=list)


def slugify_adviser_name(name: str) -> str:
    slug = _SLUG_STRIP.sub("-", str(name or "").strip().lower()).strip("-")
    return slug or "adviser"


def _expand(path: str) -> str:
    return os.path.expanduser(os.path.expandvars(str(path)))


def repo_root(cwd: str, home: str) -> Optional[str]:
    """Nearest ancestor (inclusive) holding a ``.git`` entry, else None (walk stops at home)."""
    try:
        start = Path(cwd).resolve()
        home_path = Path(home).resolve()
    except OSError:
        return None
    current = start
    while True:
        if (current / ".git").exists():
            return str(current)
        if current == home_path:
            return None
        parent = current.parent
        if parent == current:
            return None
        current = parent


def _depth(cwd: str, directory: str) -> int:
    try:
        rel = os.path.relpath(directory, cwd)
    except ValueError:
        return 0
    return 0 if rel in ("", ".") else len([part for part in rel.split(os.sep) if part])


def _candidate_paths(cwd: str, home: str, filenames: Iterable[str]) -> list[tuple[str, bool, int]]:
    """``(path, is_user, depth)`` in insertion order — user first, project root → leaf."""
    home = _expand(home or str(get_hermes_home()))
    items: list[tuple[str, bool, int]] = []
    for filename in filenames:
        items.append((os.path.join(home, filename), True, -1))
    stop = repo_root(cwd, home) or home
    current = os.path.abspath(_expand(cwd or os.getcwd()))
    user_home = str(Path.home())
    while True:
        depth = _depth(cwd, current)
        for filename in filenames:
            # ``~/.hermes/<f>`` is the user-level candidate, never a *project* one.
            if current != user_home:
                items.append((os.path.join(current, PROJECT_CONFIG_DIR, filename), False, depth))
            items.append((os.path.join(current, filename), False, depth))
        if current == stop:
            break
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return items


def _visible(owner_dir: str, is_user: bool, parent_name: str) -> bool:
    if is_user:
        return True
    if not os.path.basename(owner_dir).startswith("."):
        return True
    return parent_name == PROJECT_CONFIG_DIR


def collect_config_candidates(cwd: str, home: str, filenames: Iterable[str]) -> list[Candidate]:
    """Read every candidate on the search path, deduped, user-first then depth-descending."""
    seen: set[str] = set()
    kept: list[tuple[Candidate, int]] = []
    for path, is_user, depth in _candidate_paths(cwd, home, filenames):
        absolute = os.path.abspath(path)
        if absolute in seen:
            continue
        seen.add(absolute)
        parent = os.path.dirname(absolute)
        parent_name = os.path.basename(parent)
        owner_dir = os.path.dirname(parent) if parent_name == PROJECT_CONFIG_DIR else parent
        if not _visible(owner_dir, is_user, parent_name):
            continue
        try:
            content = Path(absolute).read_text(encoding="utf-8-sig", errors="replace")
        except FileNotFoundError:
            continue
        except OSError as exc:
            logger.warning("adviser: cannot read %s: %s", absolute, exc)
            continue
        kept.append((Candidate(absolute, content, "user" if is_user else "project", depth), depth))
    kept.sort(key=lambda item: (0 if item[0].level == "user" else 1, -item[1]))
    return [candidate for candidate, _ in kept]


def expand_at_imports(text: str, base_path: str, depth: int = 0) -> str:
    """Expand ``@path`` includes relative to *base_path* (max depth 5).

    Fenced code is left alone, a missing or unreadable target stays verbatim, and
    cycles terminate at the depth cap.
    """
    if not text or depth >= 5:
        return text
    base_dir = os.path.dirname(os.path.abspath(base_path))

    def _replace(match: re.Match[str]) -> str:
        raw = match.group(1)
        target = _expand(raw if os.path.isabs(raw) else os.path.join(base_dir, raw))
        try:
            content = Path(target).read_text(encoding="utf-8-sig", errors="replace")
        except OSError:
            return match.group(0)
        return expand_at_imports(content, target, depth + 1)

    out: list[str] = []
    in_fence = False
    for line in text.splitlines(keepends=True):
        if _FENCE.search(line):
            in_fence = not in_fence
            out.append(line)
            continue
        out.append(line if in_fence else _AT_IMPORT.sub(_replace, line))
    return "".join(out)


def _normalize_sync_backlog(value: Any) -> Optional[str]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return _SYNC_BACKLOG_NUMERIC.get(value)
    if isinstance(value, str) and value in SYNC_BACKLOG_MODES:
        return value
    return None


def _positive_int(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value != value or value < 1:
        return None
    return int(value)


def _normalize_tools(raw: Any, source: str) -> Optional[list[str]]:
    if raw is None:
        return None
    if not isinstance(raw, list):
        logger.warning("adviser: %s: tools must be a list — ignored", source)
        return None
    names: list[str] = []
    for item in raw:
        name = str(item or "").strip()
        if not name:
            continue
        name = TOOL_ALIASES.get(name.lower(), name)
        if name not in names:
            names.append(name)
    return names


def parse_watchdog_doc(
    doc: Any, path: str
) -> tuple[Optional[str], list[dict[str, Any]], Optional[int], list[str]]:
    """Validate one YAML document per entry, so one bad adviser never drops the file."""
    warnings: list[str] = []
    if not isinstance(doc, dict):
        raise ValueError("expected a YAML mapping")
    raw_instructions = doc.get("instructions")
    instructions = raw_instructions if isinstance(raw_instructions, str) else None
    if raw_instructions is not None and instructions is None:
        warnings.append(f"{path}: instructions must be a string — ignored")
    shared_max = _positive_int(doc.get("maxNotesPerUpdate"))
    if shared_max is not None:
        shared_max = min(MAX_NOTES_CAP, shared_max)
    raw_advisers = doc.get("advisors")
    if raw_advisers is not None and not isinstance(raw_advisers, list):
        warnings.append(f"{path}: advisors must be a list — ignored")
    entries = [
        entry for index, raw in enumerate(raw_advisers if isinstance(raw_advisers, list) else [])
        if (entry := _parse_adviser_entry(raw, index, path, warnings)) is not None
    ]
    return instructions, entries, shared_max, warnings


def _parse_adviser_entry(raw: Any, index: int, path: str,
                         warnings: list[str]) -> Optional[dict[str, Any]]:
    """Validate one roster entry; ``None`` drops it with a warning naming the entry."""
    if not isinstance(raw, dict) or not str(raw.get("name") or "").strip():
        warnings.append(f"{path}: advisor #{index + 1} dropped — name is required")
        return None
    entry: dict[str, Any] = {"name": str(raw["name"]).strip()}
    label = f"{path}: advisor \"{entry['name']}\""
    if raw.get("model") is not None:
        entry["model"] = str(raw.get("model") or "").strip() or None
    if raw.get("provider") is not None:
        entry["provider"] = str(raw.get("provider") or "").strip() or None
    if "tools" in raw:
        entry["tools"] = _normalize_tools(raw.get("tools"), path)
    if raw.get("reviewMode") is not None:
        mode = str(raw.get("reviewMode") or "").strip()
        if mode in REVIEW_MODES:
            entry["reviewMode"] = mode
        else:
            warnings.append(f"{label} reviewMode must be one of {REVIEW_MODES}")
    if raw.get("reviewInterval") is not None:
        interval = _positive_int(raw.get("reviewInterval"))
        if interval is None:
            warnings.append(f"{label} reviewInterval must be a positive integer")
        else:
            entry["reviewInterval"] = interval
    if raw.get("syncBacklog") is not None:
        backlog = _normalize_sync_backlog(raw.get("syncBacklog"))
        if backlog is None:
            warnings.append(f"{label} syncBacklog must be one of {SYNC_BACKLOG_MODES}")
        else:
            entry["syncBacklog"] = backlog
    if raw.get("scope") is not None:
        scope = str(raw.get("scope") or "").strip()
        if scope not in ADVISER_SCOPES:
            warnings.append(f"{label} dropped — scope must be one of {ADVISER_SCOPES}")
            return None
        entry["scope"] = scope
    if raw.get("includeReasoning") is not None:
        entry["includeReasoning"] = bool(raw.get("includeReasoning"))
    if raw.get("instructions") is not None:
        text = str(raw.get("instructions") or "").strip()
        if text:
            entry["instructions"] = expand_at_imports(text, path).strip()
    if raw.get("enabled") is not None:
        entry["enabled"] = bool(raw.get("enabled"))
    if raw.get("maxNotesPerUpdate") is not None:
        budget = _positive_int(raw.get("maxNotesPerUpdate"))
        if budget is None:
            warnings.append(f"{label} maxNotesPerUpdate must be >= 1")
        else:
            entry["maxNotesPerUpdate"] = min(MAX_NOTES_CAP, budget)
    return entry


def discover_advisers(cwd: str, home: str = "") -> DiscoveredAdvisers:
    """Walk every ``WATCHDOG.yml``/``.yaml`` on the search path and merge the roster."""
    home = home or str(get_hermes_home())
    items = collect_config_candidates(cwd, home, CONFIG_FILENAMES)
    by_slug: dict[str, AdviserConfig] = {}
    shared_parts: list[str] = []
    shared_max: Optional[int] = None
    warnings: list[str] = []
    for item in items:
        try:
            parsed = yaml.safe_load(item.content)
        except yaml.YAMLError as exc:
            warnings.append(f"{item.path}: failed to parse YAML ({exc}) — file skipped")
            logger.warning("adviser config: %s: %s", item.path, exc)
            continue
        if parsed is None:
            continue
        if not isinstance(parsed, dict):
            warnings.append(f"{item.path}: expected a YAML mapping — file skipped")
            continue
        try:
            instructions, entries, doc_max, doc_warnings = parse_watchdog_doc(parsed, item.path)
        except ValueError as exc:
            warnings.append(f"{item.path}: {exc} — file skipped")
            continue
        warnings.extend(doc_warnings)
        if instructions and instructions.strip():
            shared_parts.append(expand_at_imports(instructions.strip(), item.path).strip())
        if doc_max is not None:
            shared_max = doc_max
        for entry in entries:
            slug = slugify_adviser_name(entry["name"])
            by_slug[slug] = AdviserConfig(
                name=entry["name"],
                slug=slug,
                model=entry.get("model"),
                provider=entry.get("provider"),
                tools=entry.get("tools"),
                review_mode=entry.get("reviewMode"),
                review_interval=entry.get("reviewInterval"),
                sync_backlog=entry.get("syncBacklog"),
                scope=entry.get("scope"),
                include_reasoning=entry.get("includeReasoning"),
                instructions=entry.get("instructions"),
                enabled=entry.get("enabled"),
                max_notes_per_update=entry.get("maxNotesPerUpdate"),
                source=item.path,
            )
    advisers = list(by_slug.values())
    used: dict[str, int] = {}
    for adviser in advisers:
        base = adviser.slug or "adviser"
        count = used.get(base, 0)
        used[base] = count + 1
        if count:
            adviser.slug = f"{base}-{count + 1}"
    return DiscoveredAdvisers(
        advisers=advisers,
        shared_instructions="\n\n".join(shared_parts) if shared_parts else None,
        shared_max_notes_per_update=shared_max,
        warnings=warnings,
    )


def discover_watchdog_files(cwd: str, home: str = "") -> str:
    """Concatenated ``WATCHDOG.md`` attention blocks (user first, leaf last)."""
    home = home or str(get_hermes_home())
    blocks: list[str] = []
    for item in collect_config_candidates(cwd, home, (WATCHDOG_MD_FILENAME,)):
        content = expand_at_imports(item.content, item.path).strip()
        if content:
            blocks.append(f"Especially pay attention to:\n<attention>\n{content}\n</attention>")
    return "\n\n".join(blocks)


def adviser_config_path(scope: str, project_dir: str, home: str = "") -> str:
    """Canonical ``WATCHDOG.yml`` a scope edits (user → HERMES_HOME, project → project dir)."""
    directory = (home or str(get_hermes_home())) if scope == "user" else project_dir
    return os.path.join(directory, CONFIG_FILENAMES[0])


def resolve_edit_path(scope: str, project_dir: str, home: str = "") -> str:
    """Prefer an existing ``WATCHDOG.yaml`` when no canonical ``WATCHDOG.yml`` exists."""
    canonical = adviser_config_path(scope, project_dir, home)
    if os.path.exists(canonical):
        return canonical
    alternate = canonical[:-4] + ".yaml"
    return alternate if os.path.exists(alternate) else canonical


def load_watchdog_config_file(path: str) -> dict[str, Any]:
    """Raw, un-merged, un-expanded doc for editing — never raises."""
    doc: dict[str, Any] = {"advisors": []}
    try:
        text = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    except OSError as exc:
        if not isinstance(exc, FileNotFoundError):
            logger.warning("adviser config: cannot read %s: %s", path, exc)
        return doc
    try:
        parsed = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        doc["warnings"] = [f"{path}: failed to parse YAML ({exc})"]
        return doc
    if parsed is None:
        return doc
    if not isinstance(parsed, dict):
        doc["warnings"] = [f"{path}: expected a YAML mapping — file skipped"]
        return doc
    if isinstance(parsed.get("instructions"), str):
        doc["instructions"] = parsed["instructions"]
    shared_max = _positive_int(parsed.get("maxNotesPerUpdate"))
    if shared_max is not None:
        doc["maxNotesPerUpdate"] = min(MAX_NOTES_CAP, shared_max)
    advisers = parsed.get("advisors")
    if isinstance(advisers, list):
        doc["advisors"] = [
            entry for raw in advisers
            if (entry := _clean_entry_for_edit(raw, path)) is not None
        ]
    return doc


def _clean_entry_for_edit(raw: Any, path: str) -> Optional[dict[str, Any]]:
    """One entry as the editor round-trips it: unknown keys dropped, known keys typed."""
    if not isinstance(raw, dict) or not str(raw.get("name") or "").strip():
        return None
    entry: dict[str, Any] = {"name": str(raw["name"]).strip()}
    for key in ("model", "provider", "instructions"):
        if raw.get(key) is not None:
            entry[key] = str(raw.get(key))
    if "tools" in raw:
        entry["tools"] = _normalize_tools(raw.get("tools"), path) or []
    if raw.get("reviewMode") in REVIEW_MODES:
        entry["reviewMode"] = raw["reviewMode"]
    interval = _positive_int(raw.get("reviewInterval"))
    if interval is not None:
        entry["reviewInterval"] = interval
    backlog = _normalize_sync_backlog(raw.get("syncBacklog"))
    if backlog is not None:
        entry["syncBacklog"] = backlog
    if raw.get("scope") in ADVISER_SCOPES:
        entry["scope"] = raw["scope"]
    if raw.get("includeReasoning") is not None:
        entry["includeReasoning"] = bool(raw.get("includeReasoning"))
    if raw.get("enabled") is not None:
        entry["enabled"] = bool(raw.get("enabled"))
    budget = _positive_int(raw.get("maxNotesPerUpdate"))
    if budget is not None:
        entry["maxNotesPerUpdate"] = min(MAX_NOTES_CAP, budget)
    return entry


_YAML_AMBIGUOUS = re.compile(r"^(?:true|false|null|yes|no|on|off|~|[-+]?\d+(?:\.\d+)?)$", re.IGNORECASE)


def _yaml_scalar(value: Any) -> str:
    """Render one scalar safely: JSON quoting is valid YAML, so only bare words stay bare."""
    text = str(value)
    if text and text == text.strip() and not any(char in text for char in ":#{}[],&*!|>'\"%@`\n\t"):
        if not _YAML_AMBIGUOUS.match(text):
            return text
    return json.dumps(text, ensure_ascii=False)


def _yaml_block(value: str, indent: str) -> str:
    if any(line[:1] in (" ", "\t") for line in value.splitlines()):
        return _yaml_scalar(value)  # leading whitespace cannot round-trip through ``|``
    if "\n" not in value:
        return _yaml_scalar(value)
    body = value.rstrip("\n")
    trailing = len(value) - len(body)
    chomp = "|2-" if trailing == 0 else ("|2" if trailing == 1 else "|2+")
    lines = [f"{indent}{chomp}"]
    for line in body.split("\n"):
        lines.append(f"{indent}{line}" if line else "")
    if trailing > 1:
        lines.extend("" for _ in range(trailing - 1))
    return "\n".join(lines)


def _serialize_entry_fields(entry: dict[str, Any]) -> list[str]:
    """Everything after ``- name:`` for one roster entry, in canonical order."""
    field_indent = "    "
    lines: list[str] = []
    if entry.get("model"):
        lines.append(f"{field_indent}model: {_yaml_scalar(entry['model'])}")
    if entry.get("provider"):
        lines.append(f"{field_indent}provider: {_yaml_scalar(entry['provider'])}")
    if "tools" in entry:
        tools = entry.get("tools") or []
        if not tools:
            lines.append(f"{field_indent}tools: []")
        else:
            lines.append(f"{field_indent}tools:")
            for tool in tools:
                lines.append(f"{field_indent}  - {_yaml_scalar(tool)}")
    if entry.get("reviewMode") in REVIEW_MODES:
        lines.append(f"{field_indent}reviewMode: {_yaml_scalar(entry['reviewMode'])}")
    if entry.get("syncBacklog") in SYNC_BACKLOG_MODES:
        lines.append(f"{field_indent}syncBacklog: {_yaml_scalar(entry['syncBacklog'])}")
    if entry.get("scope") in ADVISER_SCOPES:
        lines.append(f"{field_indent}scope: {_yaml_scalar(entry['scope'])}")
    if entry.get("includeReasoning") is not None:
        lines.append(f"{field_indent}includeReasoning: {'true' if entry['includeReasoning'] else 'false'}")
    text = entry.get("instructions")
    if isinstance(text, str) and text.strip():
        if "\n" in text:
            lines.append(f"{field_indent}instructions:\n" + _yaml_block(text, field_indent + "  "))
        else:
            lines.append(f"{field_indent}instructions: {_yaml_scalar(text)}")
    if entry.get("enabled") is not None:
        lines.append(f"{field_indent}enabled: {'true' if entry['enabled'] else 'false'}")
    budget = _positive_int(entry.get("maxNotesPerUpdate"))
    if budget is not None:
        lines.append(f"{field_indent}maxNotesPerUpdate: {min(MAX_NOTES_CAP, budget)}")
    interval = _positive_int(entry.get("reviewInterval"))
    if interval is not None:
        lines.append(f"{field_indent}reviewInterval: {interval}")
    return lines


def serialize_watchdog_config(doc: dict[str, Any]) -> str:
    """Canonical, hand-editable ``WATCHDOG.yml`` text (fixed key order)."""
    lines: list[str] = []
    instructions = doc.get("instructions")
    if isinstance(instructions, str) and instructions.strip():
        if "\n" not in instructions:
            lines.append(f"instructions: {_yaml_scalar(instructions)}")
        else:
            lines.append("instructions:\n" + _yaml_block(instructions, "  "))
    max_notes = _positive_int(doc.get("maxNotesPerUpdate"))
    if max_notes is not None:
        lines.append(f"maxNotesPerUpdate: {max_notes}")
    lines.append("advisors:")
    for entry in doc.get("advisors") or []:
        if not isinstance(entry, dict):
            continue
        lines.append(f"  - name: {_yaml_scalar(entry.get('name') or '')}")
        lines.extend(_serialize_entry_fields(entry))
    if not any(isinstance(entry, dict) for entry in doc.get("advisors") or []) and \
            not (isinstance(instructions, str) and instructions.strip()) and max_notes is None:
        return ""
    text = "\n".join(lines)
    return text + "\n" if text.strip() else ""


def save_watchdog_config_file(path: str, doc: dict[str, Any]) -> None:
    """Write *doc*; an empty document deletes the file so discovery falls back to the default."""
    text = serialize_watchdog_config(doc)
    if not text.strip():
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")


__all__ = [
    "ADVISER_SCOPES",
    "CONFIG_FILENAMES",
    "DEFAULT_TOOL_NAMES",
    "MAX_NOTES_CAP",
    "PROJECT_CONFIG_DIR",
    "REVIEW_MODES",
    "SYNC_BACKLOG_MODES",
    "TOOL_ALIASES",
    "UPSTREAM_ADVISER_COMMIT",
    "UPSTREAM_ADVISER_PATHS",
    "UPSTREAM_ADVISER_RELEASE",
    "UPSTREAM_ADVISER_REPO",
    "WATCHDOG_MD_FILENAME",
    "AdviserConfig",
    "Candidate",
    "DiscoveredAdvisers",
    "adviser_config_path",
    "collect_config_candidates",
    "discover_advisers",
    "discover_watchdog_files",
    "expand_at_imports",
    "load_watchdog_config_file",
    "parse_watchdog_doc",
    "repo_root",
    "resolve_edit_path",
    "save_watchdog_config_file",
    "serialize_watchdog_config",
    "slugify_adviser_name",
]
