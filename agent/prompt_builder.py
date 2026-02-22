"""System prompt assembly -- identity, platform hints, skills index, context files.

All functions are stateless. AIAgent._build_system_prompt() calls these to
assemble pieces, then combines them with memory and ephemeral prompts.
"""

import logging
import os
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# =========================================================================
# Constants
# =========================================================================

DEFAULT_AGENT_IDENTITY = (
    "You are Hermes Agent, an intelligent AI assistant created by Nous Research. "
    "You are helpful, knowledgeable, and direct. You assist users with a wide "
    "range of tasks including answering questions, writing and editing code, "
    "analyzing information, creative work, and executing actions via your tools. "
    "You communicate clearly, admit uncertainty when appropriate, and prioritize "
    "being genuinely useful over being verbose unless otherwise directed below."
)

MEMORY_GUIDANCE = (
    "You have persistent memory across sessions. Proactively save important things "
    "you learn (user preferences, environment details, useful approaches) and do "
    "(like a diary!) using the memory tool -- don't wait to be asked."
)

SESSION_SEARCH_GUIDANCE = (
    "When the user references something from a past conversation or you suspect "
    "relevant prior context exists, use session_search to recall it before asking "
    "them to repeat themselves."
)

PLATFORM_HINTS = {
    "whatsapp": (
        "You are on a text messaging communication platform, WhatsApp. "
        "Please do not use markdown as it does not render."
    ),
    "telegram": (
        "You are on a text messaging communication platform, Telegram. "
        "Please do not use markdown as it does not render."
    ),
    "discord": (
        "You are in a Discord server or group chat communicating with your user."
    ),
    "cli": (
        "You are a CLI AI Agent. Try not to use markdown but simple text "
        "renderable inside a terminal."
    ),
}

CONTEXT_FILE_MAX_CHARS = 20_000
CONTEXT_TRUNCATE_HEAD_RATIO = 0.7
CONTEXT_TRUNCATE_TAIL_RATIO = 0.2


# =========================================================================
# Skills index
# =========================================================================

def build_skills_system_prompt() -> str:
    """Build a compact skill index for the system prompt.

    Scans ~/.hermes/skills/ for SKILL.md files grouped by category so the
    model can match skills at a glance without extra tool calls.
    """
    hermes_home = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes"))
    skills_dir = hermes_home / "skills"

    if not skills_dir.exists():
        return ""

    skills_by_category = {}
    for skill_file in skills_dir.rglob("SKILL.md"):
        rel_path = skill_file.relative_to(skills_dir)
        parts = rel_path.parts
        if len(parts) >= 2:
            category = parts[0]
            skill_name = parts[-2]
        else:
            category = "general"
            skill_name = skill_file.parent.name
        skills_by_category.setdefault(category, []).append(skill_name)

    if not skills_by_category:
        return ""

    category_descriptions = {}
    for category in skills_by_category:
        desc_file = skills_dir / category / "DESCRIPTION.md"
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
        desc = category_descriptions.get(category, "")
        names = ", ".join(sorted(set(skills_by_category[category])))
        if desc:
            index_lines.append(f"  {category}: {desc}")
        else:
            index_lines.append(f"  {category}:")
        index_lines.append(f"    skills: {names}")

    return (
        "## Skills (mandatory)\n"
        "Before replying, scan the skills below. If one clearly matches your task, "
        "load it with skill_view(name) and follow its instructions. "
        "If a skill has issues, fix it with skill_manage(action='patch').\n"
        "\n"
        "<available_skills>\n"
        + "\n".join(index_lines) + "\n"
        "</available_skills>\n"
        "\n"
        "If none match, proceed normally without loading a skill."
    )


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


def build_context_files_prompt(cwd: Optional[str] = None) -> str:
    """Discover and load context files for the system prompt.

    Discovery: AGENTS.md (recursive), .cursorrules / .cursor/rules/*.mdc,
    SOUL.md (cwd then ~/.hermes/ fallback). Each capped at 20,000 chars.
    """
    if cwd is None:
        cwd = os.getcwd()

    cwd_path = Path(cwd).resolve()
    sections = []

    # AGENTS.md (hierarchical, recursive)
    top_level_agents = None
    for name in ["AGENTS.md", "agents.md"]:
        candidate = cwd_path / name
        if candidate.exists():
            top_level_agents = candidate
            break

    if top_level_agents:
        agents_files = []
        for root, dirs, files in os.walk(cwd_path):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('node_modules', '__pycache__', 'venv', '.venv')]
            for f in files:
                if f.lower() == "agents.md":
                    agents_files.append(Path(root) / f)
        agents_files.sort(key=lambda p: len(p.parts))

        total_agents_content = ""
        for agents_path in agents_files:
            try:
                content = agents_path.read_text(encoding="utf-8").strip()
                if content:
                    rel_path = agents_path.relative_to(cwd_path)
                    total_agents_content += f"## {rel_path}\n\n{content}\n\n"
            except Exception as e:
                logger.debug("Could not read %s: %s", agents_path, e)

        if total_agents_content:
            total_agents_content = _truncate_content(total_agents_content, "AGENTS.md")
            sections.append(total_agents_content)

    # .cursorrules
    cursorrules_content = ""
    cursorrules_file = cwd_path / ".cursorrules"
    if cursorrules_file.exists():
        try:
            content = cursorrules_file.read_text(encoding="utf-8").strip()
            if content:
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
                    cursorrules_content += f"## .cursor/rules/{mdc_file.name}\n\n{content}\n\n"
            except Exception as e:
                logger.debug("Could not read %s: %s", mdc_file, e)

    if cursorrules_content:
        cursorrules_content = _truncate_content(cursorrules_content, ".cursorrules")
        sections.append(cursorrules_content)

    # SOUL.md (cwd first, then ~/.hermes/ fallback)
    soul_path = None
    for name in ["SOUL.md", "soul.md"]:
        candidate = cwd_path / name
        if candidate.exists():
            soul_path = candidate
            break
    if not soul_path:
        global_soul = Path.home() / ".hermes" / "SOUL.md"
        if global_soul.exists():
            soul_path = global_soul

    if soul_path:
        try:
            content = soul_path.read_text(encoding="utf-8").strip()
            if content:
                content = _truncate_content(content, "SOUL.md")
                sections.append(
                    f"## SOUL.md\n\nIf SOUL.md is present, embody its persona and tone. "
                    f"Avoid stiff, generic replies; follow its guidance unless higher-priority "
                    f"instructions override it.\n\n{content}"
                )
        except Exception as e:
            logger.debug("Could not read SOUL.md from %s: %s", soul_path, e)

    if not sections:
        return ""
    return "# Project Context\n\nThe following project context files have been loaded and should be followed:\n\n" + "\n".join(sections)
