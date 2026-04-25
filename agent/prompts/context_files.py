"""Context file loading and scanning for the system prompt.

Handles SOUL.md, .hermes.md, AGENTS.md, CLAUDE.md, .cursorrules — including
prompt injection scanning.
"""

import logging
import os
import re
from pathlib import Path
from typing import Optional

from hermes_constants import get_hermes_home

from agent.prompts.constants import (
    CONTEXT_FILE_MAX_CHARS,
    CONTEXT_TRUNCATE_HEAD_RATIO,
    CONTEXT_TRUNCATE_TAIL_RATIO,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Context file scanning — detect prompt injection in AGENTS.md, .cursorrules,
# SOUL.md before they get injected into the system prompt.
# ---------------------------------------------------------------------------

_CONTEXT_THREAT_PATTERNS = [
    (r'ignore\s+(previous|all|above|prior)\s+instructions', "prompt_injection"),
    (r'do\s+not\s+tell\s+the\s+user', "deception_hide"),
    (r'system\s+prompt\s+override', "sys_prompt_override"),
    (r'disregard\s+(your|all|any)\s+(instructions|rules|guidelines)', "disregard_rules"),
    (r'act\s+as\s+(if|though)\s+you\s+(have\s+no|don\'t\s+have)\s+(restrictions|limits|rules)', "bypass_restrictions"),
    (r'<!--[^>]*(?:ignore|override|system|secret|hidden)[^>]*-->', "html_comment_injection"),
    (r'<\s*div\s+style\s*=\s*["\'][\s\S]*?display\s*:\s*none', "hidden_div"),
    (r'translate\s+.*\s+into\s+.*\s+and\s+(execute|run|eval)', "translate_execute"),
    (r'curl\s+[^\n]*\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)', "exfil_curl"),
    (r'cat\s+[^\n]*(\.env|credentials|\.netrc|\.pgpass)', "read_secrets"),
]

_CONTEXT_INVISIBLE_CHARS = {
    '\u200b', '\u200c', '\u200d', '\u2060', '\ufeff',
    '\u202a', '\u202b', '\u202c', '\u202d', '\u202e',
}


def _scan_context_content(content: str, filename: str) -> str:
    """Scan context file content for injection. Returns sanitized content."""
    findings = []

    # Check invisible unicode
    for char in _CONTEXT_INVISIBLE_CHARS:
        if char in content:
            findings.append(f"invisible unicode U+{ord(char):04X}")

    # Check threat patterns
    for pattern, pid in _CONTEXT_THREAT_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE):
            findings.append(pid)

    if findings:
        logger.warning("Context file %s blocked: %s", filename, ", ".join(findings))
        return f"[BLOCKED: {filename} contained potential prompt injection ({', '.join(findings)}). Content not loaded.]"

    return content


def _find_git_root(start: Path) -> Optional[Path]:
    """Walk *start* and its parents looking for a ``.git`` directory.

    Returns the directory containing ``.git``, or ``None`` if we hit the
    filesystem root without finding one.
    """
    current = start.resolve()
    for parent in [current, *current.parents]:
        if (parent / ".git").exists():
            return parent
    return None


_HERMES_MD_NAMES = (".hermes.md", "HERMES.md")


def _find_hermes_md(cwd: Path) -> Optional[Path]:
    """Discover the nearest ``.hermes.md`` or ``HERMES.md``.

    Search order: *cwd* first, then each parent directory up to (and
    including) the git repository root.  Returns the first match, or
    ``None`` if nothing is found.
    """
    stop_at = _find_git_root(cwd)
    current = cwd.resolve()

    for directory in [current, *current.parents]:
        for name in _HERMES_MD_NAMES:
            candidate = directory / name
            if candidate.is_file():
                return candidate
        # Stop walking at the git root (or filesystem root).
        if stop_at and directory == stop_at:
            break
    return None


def _strip_yaml_frontmatter(content: str) -> str:
    """Remove optional YAML frontmatter (``---`` delimited) from *content*.

    The frontmatter may contain structured config (model overrides, tool
    settings) that will be handled separately in a future PR.  For now we
    strip it so only the human-readable markdown body is injected into the
    system prompt.
    """
    if content.startswith("---"):
        end = content.find("\n---", 3)
        if end != -1:
            # Skip past the closing --- and any trailing newline
            body = content[end + 4:].lstrip("\n")
            return body if body else content
    return content


def _truncate_content(content: str, filename: str, max_chars: int = CONTEXT_FILE_MAX_CHARS) -> str:
    """Head/tail truncation with a marker in the middle."""
    if len(content) <= max_chars:
        return content
    head_chars = int(max_chars * CONTEXT_TRUNCATE_HEAD_RATIO)
    tail_chars = int(max_chars * CONTEXT_TRUNCATE_TAIL_RATIO)
    head = content[:head_chars]
    tail = content[-tail_chars:]
    marker = f"\n\n[...truncated {filename}: kept {head_chars}+{tail_chars} of {len(content)} chars. Use file tools to read the full file.]\n\n"
    return head + marker + tail


def load_soul_md() -> Optional[str]:
    """Load SOUL.md from HERMES_HOME and return its content, or None.

    Used as the agent identity (slot #1 in the system prompt).  When this
    returns content, ``build_context_files_prompt`` should be called with
    ``skip_soul=True`` so SOUL.md isn't injected twice.
    """
    try:
        from hermes_cli.config import ensure_hermes_home
        ensure_hermes_home()
    except Exception as e:
        logger.debug("Could not ensure HERMES_HOME before loading SOUL.md: %s", e)

    soul_path = get_hermes_home() / "knowledge" / "facts" / "SOUL.md"
    if not soul_path.exists():
        # Fallback to old location for backward compatibility during transition
        old_path = get_hermes_home() / "SOUL.md"
        if old_path.exists():
            soul_path = old_path
        else:
            return None
    try:
        content = soul_path.read_text(encoding="utf-8").strip()
        if not content:
            return None
        content = _scan_context_content(content, "SOUL.md")
        content = _truncate_content(content, "SOUL.md")
        return content
    except Exception as e:
        logger.debug("Could not read SOUL.md from %s: %s", soul_path, e)
        return None


def _load_hermes_md(cwd_path: Path) -> str:
    """.hermes.md / HERMES.md — walk to git root."""
    hermes_md_path = _find_hermes_md(cwd_path)
    if not hermes_md_path:
        return ""
    try:
        content = hermes_md_path.read_text(encoding="utf-8").strip()
        if not content:
            return ""
        content = _strip_yaml_frontmatter(content)
        rel = hermes_md_path.name
        try:
            rel = str(hermes_md_path.relative_to(cwd_path))
        except ValueError:
            pass
        content = _scan_context_content(content, rel)
        result = f"## {rel}\n\n{content}"
        return _truncate_content(result, ".hermes.md")
    except Exception as e:
        logger.debug("Could not read %s: %s", hermes_md_path, e)
        return ""


def _load_agents_md(cwd_path: Path) -> str:
    """AGENTS.md — top-level only (no recursive walk)."""
    for name in ["AGENTS.md", "agents.md"]:
        candidate = cwd_path / name
        if candidate.exists():
            try:
                content = candidate.read_text(encoding="utf-8").strip()
                if content:
                    content = _scan_context_content(content, name)
                    result = f"## {name}\n\n{content}"
                    return _truncate_content(result, "AGENTS.md")
            except Exception as e:
                logger.debug("Could not read %s: %s", candidate, e)
    return ""


def _load_claude_md(cwd_path: Path) -> str:
    """CLAUDE.md / claude.md — cwd only."""
    for name in ["CLAUDE.md", "claude.md"]:
        candidate = cwd_path / name
        if candidate.exists():
            try:
                content = candidate.read_text(encoding="utf-8").strip()
                if content:
                    content = _scan_context_content(content, name)
                    result = f"## {name}\n\n{content}"
                    return _truncate_content(result, "CLAUDE.md")
            except Exception as e:
                logger.debug("Could not read %s: %s", candidate, e)
    return ""


def _load_cursorrules(cwd_path: Path) -> str:
    """.cursorrules + .cursor/rules/*.mdc — cwd only."""
    cursorrules_content = ""
    cursorrules_file = cwd_path / ".cursorrules"
    if cursorrules_file.exists():
        try:
            content = cursorrules_file.read_text(encoding="utf-8").strip()
            if content:
                content = _scan_context_content(content, ".cursorrules")
                cursorrules_content += f"## .cursorrules\n\n{content}\n\n"
        except Exception as e:
            logger.debug("Could not read .cursorrules: %s", e)

    cursor_rules_dir = cwd_path / ".cursor" / "rules"
    if cursor_rules_dir.exists() and cursor_rules_dir.is_dir():
        mdc_files = sorted(cursor_rules_dir.glob("*.mdc"))
        for mdc_file in mdc_files:
            try:
                content = mdc_file.read_text(encoding="utf-8").strip()
                if content:
                    content = _scan_context_content(content, f".cursor/rules/{mdc_file.name}")
                    cursorrules_content += f"## .cursor/rules/{mdc_file.name}\n\n{content}\n\n"
            except Exception as e:
                logger.debug("Could not read %s: %s", mdc_file, e)

    if not cursorrules_content:
        return ""
    return _truncate_content(cursorrules_content, ".cursorrules")


def build_context_files_prompt(cwd: Optional[str] = None, skip_soul: bool = False) -> str:
    """Discover and load context files for the system prompt.

    Priority (first found wins — only ONE project context type is loaded):
      1. .hermes.md / HERMES.md  (walk to git root)
      2. AGENTS.md / agents.md   (cwd only)
      3. CLAUDE.md / claude.md   (cwd only)
      4. .cursorrules / .cursor/rules/*.mdc  (cwd only)

    SOUL.md from HERMES_HOME is independent and always included when present.
    Each context source is capped at 20,000 chars.

    When *skip_soul* is True, SOUL.md is not included here (it was already
    loaded via ``load_soul_md()`` for the identity slot).
    """
    if cwd is None:
        cwd = os.getcwd()

    cwd_path = Path(cwd).resolve()
    sections = []

    # Priority-based project context: first match wins
    project_context = (
        _load_hermes_md(cwd_path)
        or _load_agents_md(cwd_path)
        or _load_claude_md(cwd_path)
        or _load_cursorrules(cwd_path)
    )
    if project_context:
        sections.append(project_context)

    # SOUL.md from HERMES_HOME only — skip when already loaded as identity
    if not skip_soul:
        soul_content = load_soul_md()
        if soul_content:
            sections.append(soul_content)

    if not sections:
        return ""
    return "# Project Context\n\nThe following project context files have been loaded and should be followed:\n\n" + "\n".join(sections)
