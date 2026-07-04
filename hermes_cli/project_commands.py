"""
Project-local slash commands (``.hermes/commands/*.md``).

Each ``.md`` file in the project-local ``.hermes/commands/`` directory
becomes a slash command whose name is the filename (without the ``.md``
extension).  YAML frontmatter provides metadata (description, args,
category); the markdown body is the prompt template injected into the
conversation when the command is invoked.

Template variables in the body (``{{var_name}}``) are substituted with
user-provided arguments or frontmatter defaults.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Limit individual command file size to prevent abuse.
_MAX_COMMAND_FILE_BYTES = 20_000


@dataclass
class ProjectCommand:
    """Parsed project-local slash command."""

    name: str                          # command name (filename without .md)
    description: str                   # from frontmatter (required)
    category: str = "项目命令"          # from frontmatter or default
    args: List[Dict[str, str]] = field(default_factory=list)
    override: bool = False             # if True, replaces built-in command
    template: str = ""                 # markdown body (without frontmatter)
    source_path: Optional[Path] = None


# ── Discovery ────────────────────────────────────────────────────────────


def _get_project_commands_dir() -> Optional[Path]:
    """Return the project-local commands directory, or None."""
    try:
        from agent.prompt_builder import _get_cached_hermes_project_dir
    except ImportError:
        return None
    project_dir = _get_cached_hermes_project_dir()
    if project_dir:
        return project_dir / "commands"
    return None


# ── Parsing ───────────────────────────────────────────────────────────────


def _parse_command_file(filepath: Path) -> Optional[ProjectCommand]:
    """Parse a single ``.hermes/commands/<name>.md`` file.

    Returns a ``ProjectCommand`` on success, or ``None`` if the file is
    missing required fields, too large, or otherwise invalid.
    """
    # Name = filename without .md extension
    name = filepath.stem
    if not name or not re.match(r"^[a-zA-Z0-9_-]+$", name):
        logger.debug("Skipping project command with invalid name: %s", filepath)
        return None

    # Size guard
    try:
        size = filepath.stat().st_size
        if size > _MAX_COMMAND_FILE_BYTES:
            logger.warning(
                "Project command %s is %d bytes (max %d), skipping",
                filepath, size, _MAX_COMMAND_FILE_BYTES,
            )
            return None
    except OSError:
        return None

    try:
        raw = filepath.read_text(encoding="utf-8").strip()
    except OSError as e:
        logger.debug("Could not read project command %s: %s", filepath, e)
        return None

    if not raw:
        return None

    # Parse YAML frontmatter
    try:
        from agent.skill_utils import parse_frontmatter
    except ImportError:
        logger.debug("Cannot import parse_frontmatter; project commands disabled")
        return None

    frontmatter, body = parse_frontmatter(raw)

    # description is required
    description = frontmatter.get("description", "").strip()
    if not description:
        logger.debug(
            "Project command %s missing required 'description' in frontmatter, skipping",
            filepath,
        )
        return None

    # args
    args_raw = frontmatter.get("args")
    args: List[Dict[str, str]] = []
    if isinstance(args_raw, list):
        for a in args_raw:
            if isinstance(a, dict) and "name" in a:
                args.append({
                    "name": str(a.get("name", "")),
                    "description": str(a.get("description", "")),
                    "default": str(a.get("default", "")),
                })

    # category
    category = str(frontmatter.get("category", "项目命令")).strip() or "项目命令"

    # override
    override = bool(frontmatter.get("override", False))

    return ProjectCommand(
        name=name,
        description=description,
        category=category,
        args=args,
        override=override,
        template=body.strip(),
        source_path=filepath,
    )


# ── Loading ───────────────────────────────────────────────────────────────

# In-process cache so we don't re-scan on every /help or tab-completion.
_commands_cache: Optional[List[ProjectCommand]] = None
_commands_cache_dir: Optional[str] = None


def load_project_commands() -> List[ProjectCommand]:
    """Scan ``.hermes/commands/*.md`` and return parsed project commands.

    Results are cached in-process.  The cache is invalidated when the
    commands directory path changes (e.g. a ``cd`` in the terminal tool).
    """
    global _commands_cache, _commands_cache_dir

    commands_dir = _get_project_commands_dir()
    dir_key = str(commands_dir) if commands_dir else None

    if _commands_cache is not None and _commands_cache_dir == dir_key:
        return _commands_cache

    if not commands_dir or not commands_dir.is_dir():
        _commands_cache = []
        _commands_cache_dir = dir_key
        return []

    commands: List[ProjectCommand] = []
    try:
        for md_file in sorted(commands_dir.glob("*.md")):
            cmd = _parse_command_file(md_file)
            if cmd:
                commands.append(cmd)
    except OSError:
        pass

    _commands_cache = commands
    _commands_cache_dir = dir_key
    return commands


# ── Lookup & rendering ────────────────────────────────────────────────────


def resolve_project_command(name: str) -> Optional[ProjectCommand]:
    """Find a project command by name.  Returns None if not found."""
    for cmd in load_project_commands():
        if cmd.name == name:
            return cmd
    return None


_TEMPLATE_RE = re.compile(r"\{\{(\w+)\}\}")


def render_command(command: ProjectCommand, args: Dict[str, str]) -> str:
    """Render a command template by substituting ``{{var}}`` placeholders.

    Variables come from *args* (user-provided) with fallback to frontmatter
    defaults.  Unmatched placeholders are left as-is so the LLM can ask
    the user for missing values.
    """
    # Build substitution map: user args > frontmatter defaults
    subs: Dict[str, str] = {}
    for arg_def in command.args:
        name = arg_def.get("name", "")
        if name:
            subs[name] = arg_def.get("default", "")
    subs.update(args)

    def _replace(match: re.Match) -> str:
        var = match.group(1)
        return subs.get(var, match.group(0))

    return _TEMPLATE_RE.sub(_replace, command.template)
