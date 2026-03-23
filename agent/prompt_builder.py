"""System prompt assembly -- identity, platform hints, skills index, context files.

All functions are stateless. AIAgent._build_system_prompt() calls these to
assemble pieces, then combines them with memory and ephemeral prompts.
"""

import logging
import os
import re
import time
from pathlib import Path
from typing import Optional

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
    (r'<\s*div\s+style\s*=\s*["\'].*display\s*:\s*none', "hidden_div"),
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


# =========================================================================
# Constants
# =========================================================================

DEFAULT_AGENT_IDENTITY = (
    "You are Hermes Agent, an intelligent AI assistant created by Nous Research. "
    "You are helpful, knowledgeable, and direct. You assist users with a wide "
    "range of tasks including answering questions, writing and editing code, "
    "analyzing information, creative work, and executing actions via your tools. "
    "You communicate clearly, admit uncertainty when appropriate, and prioritize "
    "being genuinely useful over being verbose unless otherwise directed below. "
    "Be targeted and efficient in your exploration and investigations."
)

MEMORY_GUIDANCE = (
    "You have persistent memory across sessions. Save durable facts using the memory "
    "tool: user preferences, environment details, tool quirks, and stable conventions. "
    "Memory is injected into every turn, so keep it compact and focused on facts that "
    "will still matter later.\n"
    "Prioritize what reduces future user steering — the most valuable memory is one "
    "that prevents the user from having to correct or remind you again. "
    "User preferences and recurring corrections matter more than procedural task details.\n"
    "Do NOT save task progress, session outcomes, completed-work logs, or temporary TODO "
    "state to memory; use session_search to recall those from past transcripts. "
    "If you've discovered a new way to do something, solved a problem that could be "
    "necessary later, save it as a skill with the skill tool."
)

SESSION_SEARCH_GUIDANCE = (
    "When the user references something from a past conversation or you suspect "
    "relevant cross-session context exists, use session_search to recall it before "
    "asking them to repeat themselves."
)

SKILLS_GUIDANCE = (
    "After completing a complex task (5+ tool calls), fixing a tricky error, "
    "or discovering a non-trivial workflow, save the approach as a "
    "skill with skill_manage so you can reuse it next time.\n"
    "When using a skill and finding it outdated, incomplete, or wrong, "
    "patch it immediately with skill_manage(action='patch') — don't wait to be asked. "
    "Skills that aren't maintained become liabilities."
)

DEBUGGING_GUIDANCE = (
    "When debugging or fixing errors: "
    "1) Read the full error trace before acting. "
    "2) Search for related code before editing. "
    "3) Make the smallest possible fix first. "
    "4) Run tests or verify after making changes. "
    "5) If stuck after 3 failed attempts, step back, reconsider the approach, "
    "and explain your reasoning before trying again."
)

SELF_REVIEW_GUIDANCE = (
    "After making code changes, briefly verify: "
    "1) All imports are present. "
    "2) Function signatures match their call sites. "
    "3) No syntax errors in the logic. "
    "4) Edge cases are handled. "
    "If you spot an issue, fix it immediately rather than waiting."
)

THINK_BEFORE_ACT_GUIDANCE = (
    "For complex code changes: Before editing, briefly state WHAT you'll change "
    "and WHY. This catches errors early and helps you plan multi-file changes "
    "coherently. For simple fixes, act directly."
)

PLATFORM_HINTS = {
    "whatsapp": (
        "You are on a text messaging communication platform, WhatsApp. "
        "Please do not use markdown as it does not render. "
        "You can send media files natively: to deliver a file to the user, "
        "include MEDIA:/absolute/path/to/file in your response. The file "
        "will be sent as a native WhatsApp attachment — images (.jpg, .png, "
        ".webp) appear as photos, videos (.mp4, .mov) play inline, and other "
        "files arrive as downloadable documents. You can also include image "
        "URLs in markdown format ![alt](url) and they will be sent as photos."
    ),
    "telegram": (
        "You are on a text messaging communication platform, Telegram. "
        "Please do not use markdown as it does not render. "
        "You can send media files natively: to deliver a file to the user, "
        "include MEDIA:/absolute/path/to/file in your response. Images "
        "(.png, .jpg, .webp) appear as photos, audio (.ogg) sends as voice "
        "bubbles, and videos (.mp4) play inline. You can also include image "
        "URLs in markdown format ![alt](url) and they will be sent as native photos."
    ),
    "discord": (
        "You are in a Discord server or group chat communicating with your user. "
        "You can send media files natively: include MEDIA:/absolute/path/to/file "
        "in your response. Images (.png, .jpg, .webp) are sent as photo "
        "attachments, audio as file attachments. You can also include image URLs "
        "in markdown format ![alt](url) and they will be sent as attachments."
    ),
    "slack": (
        "You are in a Slack workspace communicating with your user. "
        "You can send media files natively: include MEDIA:/absolute/path/to/file "
        "in your response. Images (.png, .jpg, .webp) are uploaded as photo "
        "attachments, audio as file attachments. You can also include image URLs "
        "in markdown format ![alt](url) and they will be uploaded as attachments."
    ),
    "signal": (
        "You are on a text messaging communication platform, Signal. "
        "Please do not use markdown as it does not render. "
        "You can send media files natively: to deliver a file to the user, "
        "include MEDIA:/absolute/path/to/file in your response. Images "
        "(.png, .jpg, .webp) appear as photos, audio as attachments, and other "
        "files arrive as downloadable documents. You can also include image "
        "URLs in markdown format ![alt](url) and they will be sent as photos."
    ),
    "email": (
        "You are communicating via email. Write clear, well-structured responses "
        "suitable for email. Use plain text formatting (no markdown). "
        "Keep responses concise but complete. You can send file attachments — "
        "include MEDIA:/absolute/path/to/file in your response. The subject line "
        "is preserved for threading. Do not include greetings or sign-offs unless "
        "contextually appropriate."
    ),
    "cron": (
        "You are running as a scheduled cron job. There is no user present — you "
        "cannot ask questions, request clarification, or wait for follow-up. Execute "
        "the task fully and autonomously, making reasonable decisions where needed. "
        "Your final response is automatically delivered to the job's configured "
        "destination — put the primary content directly in your response."
    ),
    "cli": (
        "You are a CLI AI Agent. Try not to use markdown but simple text "
        "renderable inside a terminal."
    ),
    "sms": (
        "You are communicating via SMS. Keep responses concise and use plain text "
        "only — no markdown, no formatting. SMS messages are limited to ~1600 "
        "characters, so be brief and direct."
    ),
}

CONTEXT_FILE_MAX_CHARS = 20_000
CONTEXT_TRUNCATE_HEAD_RATIO = 0.7
CONTEXT_TRUNCATE_TAIL_RATIO = 0.2


# =========================================================================
# Skills index
# =========================================================================

# Module-level cache for build_skills_system_prompt to avoid rescanning the
# filesystem on every turn.  Keyed by (available_tools, available_toolsets).
_skills_prompt_cache: dict = {
    "mtime": None,      # last observed skills dir mtime
    "timestamp": 0.0,   # time.monotonic() when cache was populated
    "result": "",        # cached prompt string
    "tools_key": None,   # (frozenset, frozenset) key for tool/toolset args
}

def _parse_skill_file(skill_file: Path) -> tuple[bool, dict, str]:
    """Read a SKILL.md once and return platform compatibility, frontmatter, and description.

    Returns (is_compatible, frontmatter, description). On any error, returns
    (True, {}, "") to err on the side of showing the skill.
    """
    try:
        from tools.skills_tool import _parse_frontmatter, skill_matches_platform

        raw = skill_file.read_text(encoding="utf-8")[:2000]
        frontmatter, _ = _parse_frontmatter(raw)

        if not skill_matches_platform(frontmatter):
            return False, {}, ""

        desc = ""
        raw_desc = frontmatter.get("description", "")
        if raw_desc:
            desc = str(raw_desc).strip().strip("'\"")
            if len(desc) > 60:
                desc = desc[:57] + "..."

        return True, frontmatter, desc
    except Exception as e:
        logger.debug("Failed to parse skill file %s: %s", skill_file, e)
        return True, {}, ""


def _read_skill_conditions(skill_file: Path) -> dict:
    """Extract conditional activation fields from SKILL.md frontmatter."""
    try:
        from tools.skills_tool import _parse_frontmatter
        raw = skill_file.read_text(encoding="utf-8")[:2000]
        frontmatter, _ = _parse_frontmatter(raw)
        hermes = frontmatter.get("metadata", {}).get("hermes", {})
        return {
            "fallback_for_toolsets": hermes.get("fallback_for_toolsets", []),
            "requires_toolsets": hermes.get("requires_toolsets", []),
            "fallback_for_tools": hermes.get("fallback_for_tools", []),
            "requires_tools": hermes.get("requires_tools", []),
        }
    except Exception as e:
        logger.debug("Failed to read skill conditions from %s: %s", skill_file, e)
        return {}


def _skill_should_show(
    conditions: dict,
    available_tools: "set[str] | None",
    available_toolsets: "set[str] | None",
) -> bool:
    """Return False if the skill's conditional activation rules exclude it."""
    if available_tools is None and available_toolsets is None:
        return True  # No filtering info — show everything (backward compat)

    at = available_tools or set()
    ats = available_toolsets or set()

    # fallback_for: hide when the primary tool/toolset IS available
    for ts in conditions.get("fallback_for_toolsets", []):
        if ts in ats:
            return False
    for t in conditions.get("fallback_for_tools", []):
        if t in at:
            return False

    # requires: hide when a required tool/toolset is NOT available
    for ts in conditions.get("requires_toolsets", []):
        if ts not in ats:
            return False
    for t in conditions.get("requires_tools", []):
        if t not in at:
            return False

    return True


def build_skills_system_prompt(
    available_tools: "set[str] | None" = None,
    available_toolsets: "set[str] | None" = None,
) -> str:
    """Build a compact skill index for the system prompt.

    Scans ~/.hermes/skills/ for SKILL.md files grouped by category.
    Includes per-skill descriptions from frontmatter so the model can
    match skills by meaning, not just name.
    Filters out skills incompatible with the current OS platform.
    """
    hermes_home = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes"))
    skills_dir = hermes_home / "skills"

    if not skills_dir.exists():
        return ""

    # --- mtime-based cache: avoid rescanning if nothing changed (<60s) ---
    tools_key = (
        frozenset(available_tools) if available_tools else frozenset(),
        frozenset(available_toolsets) if available_toolsets else frozenset(),
    )
    try:
        current_mtime = skills_dir.stat().st_mtime
    except OSError:
        current_mtime = None

    cache = _skills_prompt_cache
    now_mono = time.monotonic()
    if (
        cache["tools_key"] == tools_key
        and cache["mtime"] == current_mtime
        and current_mtime is not None
        and (now_mono - cache["timestamp"]) < 60
    ):
        return cache["result"]

    # Collect skills with descriptions, grouped by category.
    # Each entry: (skill_name, description)
    # Supports sub-categories: skills/mlops/training/axolotl/SKILL.md
    # -> category "mlops/training", skill "axolotl"
    # Load disabled skill names once for the entire scan
    try:
        from tools.skills_tool import _get_disabled_skill_names
        disabled = _get_disabled_skill_names()
    except Exception:
        disabled = set()

    skills_by_category: dict[str, list[tuple[str, str]]] = {}
    for skill_file in skills_dir.rglob("SKILL.md"):
        is_compatible, frontmatter, desc = _parse_skill_file(skill_file)
        if not is_compatible:
            continue
        rel_path = skill_file.relative_to(skills_dir)
        parts = rel_path.parts
        if len(parts) >= 2:
            skill_name = parts[-2]
            category = "/".join(parts[:-2]) if len(parts) > 2 else parts[0]
        else:
            category = "general"
            skill_name = skill_file.parent.name
        # Respect user's disabled skills config
        fm_name = frontmatter.get("name", skill_name)
        if fm_name in disabled or skill_name in disabled:
            continue
        # Skip skills whose conditional activation rules exclude them
        conditions = _read_skill_conditions(skill_file)
        if not _skill_should_show(conditions, available_tools, available_toolsets):
            continue
        skills_by_category.setdefault(category, []).append((skill_name, desc))

    if not skills_by_category:
        return ""

    # Read category-level descriptions from DESCRIPTION.md
    # Checks both the exact category path and parent directories
    category_descriptions = {}
    for category in skills_by_category:
        cat_path = Path(category)
        desc_file = skills_dir / cat_path / "DESCRIPTION.md"
        if desc_file.exists():
            try:
                content = desc_file.read_text(encoding="utf-8")
                match = re.search(r"^---\s*\n.*?description:\s*(.+?)\s*\n.*?^---", content, re.MULTILINE | re.DOTALL)
                if match:
                    category_descriptions[category] = match.group(1).strip()
            except Exception as e:
                logger.debug("Could not read skill description %s: %s", desc_file, e)

    index_lines = []
    for category in sorted(skills_by_category.keys()):
        cat_desc = category_descriptions.get(category, "")
        if cat_desc:
            index_lines.append(f"  {category}: {cat_desc}")
        else:
            index_lines.append(f"  {category}:")
        # Deduplicate and sort skills within each category
        seen = set()
        for name, desc in sorted(skills_by_category[category], key=lambda x: x[0]):
            if name in seen:
                continue
            seen.add(name)
            if desc:
                index_lines.append(f"    - {name}: {desc}")
            else:
                index_lines.append(f"    - {name}")

    result = (
        "## Skills (mandatory)\n"
        "Before replying, scan the skills below. If one clearly matches your task, "
        "load it with skill_view(name) and follow its instructions. "
        "If a skill has issues, fix it with skill_manage(action='patch').\n"
        "After difficult/iterative tasks, offer to save as a skill. "
        "If a skill you loaded was missing steps, had wrong commands, or needed "
        "pitfalls you discovered, update it before finishing.\n"
        "\n"
        "<available_skills>\n"
        + "\n".join(index_lines) + "\n"
        "</available_skills>\n"
        "\n"
        "If none match, proceed normally without loading a skill."
    )

    # Populate cache
    cache["mtime"] = current_mtime
    cache["timestamp"] = now_mono
    cache["result"] = result
    cache["tools_key"] = tools_key

    return result


# =========================================================================
# Context files (SOUL.md, AGENTS.md, .cursorrules)
# =========================================================================

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

    soul_path = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes")) / "SOUL.md"
    if not soul_path.exists():
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
    """AGENTS.md — hierarchical, recursive directory walk."""
    top_level_agents = None
    for name in ["AGENTS.md", "agents.md"]:
        candidate = cwd_path / name
        if candidate.exists():
            top_level_agents = candidate
            break

    if not top_level_agents:
        return ""

    agents_files = []
    for root, dirs, files in os.walk(cwd_path):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('node_modules', '__pycache__', 'venv', '.venv')]
        for f in files:
            if f.lower() == "agents.md":
                agents_files.append(Path(root) / f)
    agents_files.sort(key=lambda p: len(p.parts))

    total_content = ""
    for agents_path in agents_files:
        try:
            content = agents_path.read_text(encoding="utf-8").strip()
            if content:
                rel_path = agents_path.relative_to(cwd_path)
                content = _scan_context_content(content, str(rel_path))
                total_content += f"## {rel_path}\n\n{content}\n\n"
        except Exception as e:
            logger.debug("Could not read %s: %s", agents_path, e)

    if not total_content:
        return ""
    return _truncate_content(total_content, "AGENTS.md")


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
      2. AGENTS.md / agents.md   (recursive directory walk)
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


# =========================================================================
# Repository map (Aider-style code map)
# =========================================================================

_repo_map_cache: dict = {
    "root": None,
    "timestamp": 0.0,
    "result": "",
}


def build_repo_map_prompt(cwd: Optional[str] = None, max_tokens: int = 4000) -> str:
    """Build repository map prompt section if cwd is a git repo.

    Returns a formatted string with the repo map, or empty string if:
    - cwd is not a git repo
    - no source files found
    - repo_map is disabled in config
    """
    if cwd is None:
        cwd = os.getcwd()

    cwd_path = Path(cwd).resolve()

    # Only include for git repos
    if not _find_git_root(cwd_path):
        return ""

    # Cache: reuse if same root and < 60s old
    cache = _repo_map_cache
    now = time.monotonic()
    if (
        cache["root"] == str(cwd_path)
        and (now - cache["timestamp"]) < 60
        and cache["result"]
    ):
        return cache["result"]

    try:
        from agent.repo_map import build_repo_map
        repo_root = str(_find_git_root(cwd_path) or cwd_path)
        map_text = build_repo_map(repo_root, max_tokens=max_tokens)
    except Exception as e:
        logger.debug("Could not build repo map: %s", e)
        return ""

    if not map_text:
        return ""

    result = (
        "## Repository Map\n\n"
        "Key symbols in the codebase:\n\n"
        f"```\n{map_text}\n```"
    )

    cache["root"] = str(cwd_path)
    cache["timestamp"] = now
    cache["result"] = result

    return result
