"""External plugin compatibility entries only; canonical metadata lives in commands."""
from __future__ import annotations

import logging
logger = logging.getLogger(__name__)

# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from typing import Any  # noqa: F401,E402
from collections.abc import Callable  # noqa: F401,E402
from typing import Dict  # noqa: F401,E402
from collections.abc import Mapping  # noqa: F401,E402
from typing import Optional  # noqa: F401,E402
from collections.abc import Sequence  # noqa: F401,E402
from typing import Tuple  # noqa: F401,E402
from dataclasses import field  # noqa: F401,E402
import os  # noqa: F401,E402
import shutil  # noqa: F401,E402
import subprocess  # noqa: F401,E402
import time  # noqa: F401,E402

def _requires_argument(args_hint: str) -> bool:
    """Return True when selecting a command without text would be incomplete."""
    return args_hint.strip().startswith("<")

_CMD_NAME_LIMIT = 32

def _clamp_command_names(
    entries: Sequence[tuple[str, ...]],
    reserved: set[str],
) -> list[tuple[str, ...]]:
    """Enforce 32-char command name limit with collision avoidance.

    Both Telegram and Discord cap slash command names at 32 characters.
    Names exceeding the limit are truncated.  If truncation creates a duplicate
    (against *reserved* names or earlier entries in the same batch), the name is
    shortened to 31 chars and a digit ``0``-``9`` is appended to differentiate.
    If all 10 digit slots are taken the entry is silently dropped.

    Accepts tuples of any length >= 2.  Extra elements beyond ``(name, desc)``
    (e.g. ``cmd_key``) are passed through unchanged, so callers can attach
    metadata that survives the rename.
    """
    used: set[str] = set(reserved)
    result: list[tuple] = []
    for entry in entries:
        name, desc, *extra = entry
        if len(name) > _CMD_NAME_LIMIT:
            candidate = name[:_CMD_NAME_LIMIT]
            if candidate in used:
                prefix = name[:_CMD_NAME_LIMIT - 1]
                for digit in range(10):
                    candidate = f"{prefix}{digit}"
                    if candidate not in used:
                        break
                else:
                    # All 10 digit slots exhausted — skip entry
                    continue
            name = candidate
        if name in used:
            continue
        used.add(name)
        result.append((name, desc, *extra))
    return result

def _collect_gateway_skill_entries(
    platform: str,
    max_slots: int | None,
    reserved_names: set[str],
    desc_limit: int = 100,
    sanitize_name: "Callable[[str], str] | None" = None,
) -> tuple[list[tuple[str, str, str, str]], int]:
    """Collect plugin + skill entries for a gateway platform.

    Priority order:
      1. Plugin slash commands (take precedence over skills)
      2. Built-in skill commands (fill remaining slots, alphabetical)

    Only skills are trimmed when the cap is reached.
    Hub-installed skills are excluded.  Per-platform disabled skills are
    excluded.

    Args:
        platform: Platform identifier for per-platform skill filtering
            (``"telegram"``, ``"discord"``, etc.).
        max_slots: Maximum number of entries to return (remaining slots after
            built-in/core commands), or ``None`` to return every eligible
            plugin and skill candidate for a caller that applies a global cap.
        reserved_names: Names already taken by built-in commands.  Mutated
            in-place as new names are added.
        desc_limit: Max description length (40 for Telegram, 100 for Discord).
        sanitize_name: Optional name transform applied before clamping, e.g.
            :func:`_sanitize_telegram_name` for Telegram.  May return an
            empty string to signal "skip this entry".

    Returns:
        ``(entries, hidden_count)`` where *entries* contains
        ``(name, description, cmd_key, raw_name)`` tuples. ``cmd_key`` is the
        original skill key (empty for plugins); ``raw_name`` is the sanitized
        pre-clamp name used for configured priority matching.
    """
    all_entries: list[tuple[str, str, str, str]] = []

    # --- Tier 1: Plugin slash commands (never trimmed) ---------------------
    plugin_pairs: list[tuple[str, str, str]] = []
    try:
        from plugin_runtime.api import get_plugin_commands
        plugin_cmds = get_plugin_commands()
        for cmd_name in sorted(plugin_cmds):
            if platform == "telegram":
                args_hint = str(plugin_cmds[cmd_name].get("args_hint") or "").strip()
                if _requires_argument(args_hint):
                    continue
            name = sanitize_name(cmd_name) if sanitize_name else cmd_name
            if not name:
                continue
            desc = plugin_cmds[cmd_name].get("description", "Plugin command")
            if len(desc) > desc_limit:
                desc = desc[:desc_limit - 3] + "..."
            plugin_pairs.append((name, desc, name))
    except Exception:
        pass

    plugin_pairs = [
        (name, desc, raw_name)
        for name, desc, raw_name in _clamp_command_names(plugin_pairs, reserved_names)
    ]
    reserved_names.update(n for n, _d, _raw_name in plugin_pairs)
    # Plugins have no cmd_key — use empty string as placeholder.
    for name, desc, raw_name in plugin_pairs:
        all_entries.append((name, desc, "", raw_name))

    # --- Tier 2: Built-in skill commands (trimmed at cap) -----------------
    _platform_disabled: set[str] = set()
    try:
        from agent.skill_utils import get_disabled_skill_names
        _platform_disabled = get_disabled_skill_names(platform=platform)
    except Exception:
        pass

    skill_entries: list[tuple[str, str, str, str]] = []
    try:
        from agent.skill_commands import get_skill_commands
        from tools.skills_tool import SKILLS_DIR
        from agent.skill_utils import get_external_skills_dirs, get_project_skills_dirs
        _skills_dir = str(SKILLS_DIR.resolve())
        _hub_dir = str((SKILLS_DIR / ".hub").resolve()).rstrip("/") + "/"
        # Build set of allowed directory prefixes: local skills dir + any
        # user-configured ``skills.external_dirs`` + trusted project dirs.
        # Ensure each prefix ends
        # with ``/`` so ``/my-skills`` does not also match ``/my-skills-extra``.
        # Without this widening, external skills are visible in
        # ``hermes skills list`` and the agent's ``/skill-name`` dispatch but
        # silently excluded from gateway slash menus (#8110).
        _allowed_prefixes = [_skills_dir.rstrip("/") + "/"]
        _allowed_prefixes.extend(
            str(d).rstrip("/") + "/" for d in get_external_skills_dirs()
        )
        _allowed_prefixes.extend(
            str(d).rstrip("/") + "/" for d in get_project_skills_dirs()
        )
        skill_cmds = get_skill_commands()
        for cmd_key in sorted(skill_cmds):
            info = skill_cmds[cmd_key]
            skill_path = info.get("skill_md_path", "")
            if not skill_path:
                continue
            if not any(skill_path.startswith(prefix) for prefix in _allowed_prefixes):
                continue
            if skill_path.startswith(_hub_dir):
                continue
            skill_name = info.get("name", "")
            if skill_name in _platform_disabled:
                continue
            raw_name = cmd_key.lstrip("/")
            name = sanitize_name(raw_name) if sanitize_name else raw_name
            if not name:
                continue
            desc = info.get("description", "")
            if len(desc) > desc_limit:
                desc = desc[:desc_limit - 3] + "..."
            skill_entries.append((name, desc, cmd_key, name))
    except Exception:
        pass

    # Clamp names; cmd_key and raw_name survive any clamp-induced rename.
    skill_entries = [
        (name, desc, cmd_key, raw_name)
        for name, desc, cmd_key, raw_name in _clamp_command_names(
            skill_entries, reserved_names
        )
    ]

    if max_slots is None:
        return all_entries + skill_entries, 0

    # Skills fill remaining slots — only tier that gets trimmed
    remaining = max(0, max_slots - len(all_entries))
    hidden_count = max(0, len(skill_entries) - remaining)
    for name, desc, cmd_key, raw_name in skill_entries[:remaining]:
        all_entries.append((name, desc, cmd_key, raw_name))

    return all_entries[:max_slots], hidden_count

def discord_skill_commands(
    max_slots: int,
    reserved_names: set[str],
) -> tuple[list[tuple[str, str, str]], int]:
    """Return skill entries for Discord slash command registration.

    Same priority and filtering logic as :func:`telegram_menu_commands`
    (plugins > skills, hub excluded, per-platform disabled excluded), but
    adapted for Discord's constraints:

    - Hyphens are allowed in names (no ``-`` → ``_`` sanitization)
    - Descriptions capped at 100 chars (Discord's per-field max)

    Args:
        max_slots: Available command slots (100 minus existing built-in count).
        reserved_names: Names of already-registered built-in commands.

    Returns:
        ``(entries, hidden_count)`` where *entries* is a list of
        ``(discord_name, description, cmd_key)`` triples.  ``cmd_key`` is
        the original ``/skill-name`` key needed for the slash handler callback.
    """
    entries, hidden_count = _collect_gateway_skill_entries(
        platform="discord",
        max_slots=max_slots,
        reserved_names=set(reserved_names),  # copy — don't mutate caller's set
        desc_limit=100,
    )
    return [
        (name, desc, cmd_key) for name, desc, cmd_key, _raw_name in entries
    ], hidden_count


_PLUGIN_COMPAT_LAZY = {
    'SlashCommandAutoSuggest': ('hermes_cli.commands_completion', 'SlashCommandAutoSuggest'),
    'SlashCommandCompleter': ('hermes_cli.commands_completion', 'SlashCommandCompleter'),
    'discord_skill_commands_by_category': ('gateway.command_platforms', 'discord_skill_commands_by_category'),
    'slack_app_manifest': ('gateway.command_platforms', 'slack_app_manifest'),
    'slack_native_slashes': ('gateway.command_platforms', 'slack_native_slashes'),
    'slack_subcommand_map': ('gateway.command_platforms', 'slack_subcommand_map'),
    'telegram_bot_commands': ('gateway.command_platforms', 'telegram_bot_commands'),
    'telegram_menu_commands': ('gateway.command_platforms', 'telegram_menu_commands'),
    'telegram_menu_max_commands': ('gateway.command_platforms', 'telegram_menu_max_commands'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from plugin_runtime.compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
