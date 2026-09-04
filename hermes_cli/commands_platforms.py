"""Gateway platform command derivations (Telegram / Discord / Slack) from ``COMMAND_REGISTRY``.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from hermes_cli.commands import (
    COMMAND_REGISTRY, _is_gateway_available, _iter_plugin_command_entries, _resolve_config_gates)

# Logger name parity with the origin module (tests capture "hermes_cli.commands").
logger = logging.getLogger("hermes_cli.commands")

_CMD_NAME_LIMIT = 32  # max command name length shared by Telegram and Discord

_TG_INVALID_CHARS = re.compile(r"[^a-z0-9_]")
_TG_MULTI_UNDERSCORE = re.compile(r"_{2,}")


def _gateway_available_commands() -> list:
    """Registry entries visible on gateway surfaces (config gates read once)."""
    overrides = _resolve_config_gates()
    return [cmd for cmd in COMMAND_REGISTRY if _is_gateway_available(cmd, overrides)]


def _requires_argument(args_hint: str) -> bool:
    """True when selecting a command without text would be incomplete."""
    return args_hint.strip().startswith("<")


def _sanitize_telegram_name(raw: str) -> str:
    """Telegram allows only ``[a-z0-9_]``: lowercase, hyphens -> ``_``, strip the rest,
    collapse/strip ``_``."""
    name = _TG_INVALID_CHARS.sub("", raw.lower().replace("-", "_"))
    return _TG_MULTI_UNDERSCORE.sub("_", name).strip("_")


def _truncate_desc(desc: str, limit: int) -> str:
    """Clamp a menu description to *limit* chars with a ``...`` tail."""
    return desc if len(desc) <= limit else desc[:limit - 3] + "..."


def _clamp_command_names(
    entries: Sequence[tuple[str, ...]], reserved: set[str]) -> list[tuple[str, ...]]:
    """Enforce the 32-char Telegram/Discord name limit: over-long names are truncated; on a
    collision with *reserved* or an earlier entry a 31-char prefix + digit ``0``-``9`` is tried,
    then the entry is silently dropped. Duplicates are dropped; extra tuple elements pass through.
    """
    used: set[str] = set(reserved)
    result: list = []
    for name, desc, *extra in entries:
        if len(name) > _CMD_NAME_LIMIT:
            candidate = name[:_CMD_NAME_LIMIT]
            if candidate in used:
                prefix = name[:_CMD_NAME_LIMIT - 1]
                for digit in range(10):
                    candidate = f"{prefix}{digit}"
                    if candidate not in used:
                        break
                else:
                    continue
            name = candidate
        if name in used:
            continue
        used.add(name)
        result.append((name, desc, *extra))
    return result


# --- Telegram ---------------------------------------------------------------

def telegram_bot_commands(*, include_plugins: bool = True) -> list[tuple[str, str]]:
    """(command_name, description) pairs for Telegram setMyCommands: sanitized canonical names
    only (no aliases). Built-ins needing arguments are included (their handlers show usage when
    selected bare); plugin commands needing arguments are excluded (may lack a no-arg fallback)."""
    pairs = [(cmd.name, cmd.description) for cmd in _gateway_available_commands()]
    if include_plugins:
        pairs += [(n, d) for n, d, hint in _iter_plugin_command_entries()
                  if not _requires_argument(hint)]
    return [(tg, desc) for name, desc in pairs if (tg := _sanitize_telegram_name(name))]


# Telegram allows 100 BotCommands; the 60-slot default keeps every built-in plus common skill
# commands under the ~4KB payload limit (platforms.telegram.extra.command_menu.max_commands).
_DEFAULT_TELEGRAM_MENU_MAX_COMMANDS = 60
_TELEGRAM_BOT_API_MAX_COMMANDS = 100
# priority_mode -> rank tables consulted in order ("configured" = user list,
# "default" = _TELEGRAM_MENU_PRIORITY); unranked candidates keep stable order after.
_TELEGRAM_PRIORITY_TIERS: dict[str, tuple[str, ...]] = {
    "prepend": ("configured", "default"), "append": ("default", "configured"),
    "replace": ("configured",)}

# Built-ins that must survive Telegram's small visible menu cap (everything else stays
# dispatchable when typed). Order = rank: everyday, maintenance, mid-turn control, operational.
_TELEGRAM_MENU_PRIORITY = (
    "help", "new", "stop", "status", "egress", "resume", "sessions", "model",
    "debug", "restart", "update", "verbose", "commands",
    "approve", "deny", "queue", "steer", "bg", "btw",
    "reasoning", "usage", "platforms", "platform", "profile", "whoami")


def _telegram_command_menu_config() -> dict[str, Any]:
    """Normalized ``platforms.telegram.extra.command_menu`` config with safe defaults."""
    try:
        from hermes_cli.config import read_raw_config
        node: Any = read_raw_config() or {}
    except Exception:
        node = {}
    for key in ("platforms", "telegram", "extra", "command_menu"):
        node = node.get(key) if isinstance(node, Mapping) else None
    menu_cfg: Mapping[str, Any] = node if isinstance(node, Mapping) else {}
    try:
        max_commands = int(menu_cfg.get("max_commands", _DEFAULT_TELEGRAM_MENU_MAX_COMMANDS))
    except (TypeError, ValueError):
        max_commands = _DEFAULT_TELEGRAM_MENU_MAX_COMMANDS
    priority_mode = str(menu_cfg.get("priority_mode") or "prepend").strip().lower()
    raw_priority = menu_cfg.get("priority")
    if isinstance(raw_priority, list):
        priority = [str(item) for item in raw_priority if str(item).strip()]
    else:
        priority = [raw_priority] if isinstance(raw_priority, str) and raw_priority.strip() else []
    return {
        "max_commands": max(1, min(_TELEGRAM_BOT_API_MAX_COMMANDS, max_commands)),
        "priority_mode": priority_mode if priority_mode in _TELEGRAM_PRIORITY_TIERS else "prepend",
        "priority": priority}


def telegram_menu_max_commands() -> int:
    """Return configured Telegram BotCommand menu cap with safe bounds."""
    return int(_telegram_command_menu_config()["max_commands"])


def _sanitized_rank(raw_names: Sequence[str]) -> dict[str, int]:
    """name -> rank for the deduped, Telegram-sanitized *raw_names* (first occurrence wins)."""
    rank: dict[str, int] = {}
    for raw in raw_names:
        name = _sanitize_telegram_name(str(raw))
        if name and name not in rank:
            rank[name] = len(rank)
    return rank


def _prioritize_telegram_menu_candidates(
    candidates: list[tuple[str, str, str, str]]) -> list[tuple[str, str, str, str]]:
    """Order ``(final_name, description, source, raw_name)`` candidates; the default priority
    applies to core only, "replace" mode ignores it. ``raw_name`` is the pre-clamp name so a
    configured long command stays addressable."""
    menu_cfg = _telegram_command_menu_config()
    configured_rank = _sanitized_rank(menu_cfg["priority"])
    default_rank = _sanitized_rank(_TELEGRAM_MENU_PRIORITY)
    tiers = _TELEGRAM_PRIORITY_TIERS[menu_cfg["priority_mode"]]

    def _rank(stable_index: int, candidate: tuple[str, str, str, str]) -> tuple[int, int, int]:
        final_name, _desc, source, raw_name = candidate
        indexes = {
            "configured": configured_rank.get(raw_name, configured_rank.get(final_name)),
            "default": default_rank.get(final_name) if source == "core" else None}
        for tier, table in enumerate(tiers):
            if indexes[table] is not None:
                return (tier, indexes[table], stable_index)
        return (len(tiers), 0, stable_index)

    return [c for _i, c in sorted(enumerate(candidates), key=lambda item: _rank(*item))]


# --- Shared skill/plugin collection for gateway platforms -------------------

def _iter_gateway_skills(platform: str):
    """Yield ``(cmd_key, info, rel_parts)`` for skills eligible as gateway slash commands.

    Scan roots: ``SKILLS_DIR`` plus ``skills.external_dirs`` / trusted project skills dirs;
    anything elsewhere or under ``SKILLS_DIR/.hub`` is skipped, as are skills disabled for
    *platform*. Paths are resolved on both sides (symlinked roots still match) and matched per
    path component (``/my-skills`` never admits ``/my-skills-extra``). ``rel_parts`` is the skill
    dir relative to its root (``("creative", "ascii-art")``). Alphabetical so first-wins
    collision handling is deterministic.
    """
    from pathlib import Path

    from agent.skill_commands import get_skill_commands
    from agent.skill_utils import (
        get_disabled_skill_names, get_external_skills_dirs, get_project_skills_dirs)
    from tools.skills_tool import SKILLS_DIR

    try:
        disabled = get_disabled_skill_names(platform=platform)
    except Exception:
        disabled = set()
    hub_dir = (SKILLS_DIR / ".hub").resolve()
    roots = [SKILLS_DIR.resolve()]
    for getter in (get_external_skills_dirs, get_project_skills_dirs):
        try:
            for d in getter():
                try:
                    roots.append(Path(d).resolve())
                except Exception:
                    continue
        except Exception:
            pass
    skill_cmds = get_skill_commands()
    for cmd_key in sorted(skill_cmds):
        info = skill_cmds[cmd_key]
        skill_path = info.get("skill_md_path", "")
        if not skill_path:
            continue
        sp = Path(skill_path).resolve()
        if sp.is_relative_to(hub_dir):
            continue
        root = next((r for r in roots if sp.is_relative_to(r)), None)
        if root is None or info.get("name", "") in disabled:
            continue
        yield cmd_key, info, sp.parent.relative_to(root).parts


def _collect_gateway_skill_entries(
    platform: str, max_slots: int | None, reserved_names: set[str], desc_limit: int = 100,
    sanitize_name: "Callable[[str], str] | None" = None,
) -> tuple[list[tuple[str, str, str, str]], int]:
    """Collect plugin + skill entries for a gateway platform.

    Plugin slash commands come first and are never trimmed; skill commands (alphabetical) fill
    the remaining *max_slots* (``None`` = every candidate, caller caps). *reserved_names* is
    mutated in place as names are claimed; *sanitize_name* runs before clamping and may return
    "" to skip. Returns ``(entries, hidden_count)``, entries ``(name, description, cmd_key,
    raw_name)`` — ``cmd_key`` "" for plugins; ``raw_name`` is the sanitized pre-clamp name used
    for configured-priority matching (both survive a clamp-induced rename).
    """
    sanitize = sanitize_name or (lambda n: n)

    def _entries(rows) -> list[tuple[str, str, str, str]]:
        """Sanitize + truncate ``(raw_name, description, cmd_key)`` rows, then clamp against
        ``reserved_names``; any failure in the lazy source yields the rows collected so far."""
        out: list[tuple[str, str, str, str]] = []
        try:
            for raw, desc, cmd_key in rows:
                if name := sanitize(raw):
                    out.append((name, _truncate_desc(desc, desc_limit), cmd_key, name))
        except Exception:
            pass
        return _clamp_command_names(out, reserved_names)

    def _plugin_rows():
        from hermes_cli.plugins import get_plugin_commands
        plugin_cmds = get_plugin_commands()
        for cmd_name in sorted(plugin_cmds):
            meta = plugin_cmds[cmd_name]
            if platform == "telegram" and _requires_argument(str(meta.get("args_hint") or "")):
                continue
            yield cmd_name, meta.get("description", "Plugin command"), ""

    plugin_entries = _entries(_plugin_rows())
    reserved_names.update(n for n, *_rest in plugin_entries)
    skill_entries = _entries(
        (cmd_key.lstrip("/"), info.get("description", ""), cmd_key)
        for cmd_key, info, _rel in _iter_gateway_skills(platform))

    if max_slots is None:
        return plugin_entries + skill_entries, 0
    remaining = max(0, max_slots - len(plugin_entries))
    hidden_count = max(0, len(skill_entries) - remaining)
    return (plugin_entries + skill_entries[:remaining])[:max_slots], hidden_count


def telegram_menu_commands(max_commands: int = 100) -> tuple[list[tuple[str, str]], int]:
    """``(menu_commands, hidden_count)`` for Telegram, capped to the Bot API limit. Tier order:
    core CommandDefs, plugin slash commands, skill commands (alphabetical; hub and
    telegram-disabled skills excluded). Tiers keep relative order unless named in
    ``platforms.telegram.extra.command_menu.priority`` — applied *before* the cap, so a
    prioritized dynamic command can displace an unprioritized core command."""
    core_commands = list(telegram_bot_commands(include_plugins=False))
    entries, hidden_count = _collect_gateway_skill_entries(
        platform="telegram", max_slots=None, reserved_names={n for n, _ in core_commands},
        desc_limit=40, sanitize_name=_sanitize_telegram_name)
    candidates = [(name, desc, "core", name) for name, desc in core_commands]
    candidates += [(name, desc, "skill" if cmd_key else "plugin", raw)
                   for name, desc, cmd_key, raw in entries]
    candidates = _prioritize_telegram_menu_candidates(candidates)
    overflow_count = max(0, len(candidates) - max_commands)
    menu = [(name, desc) for name, desc, _source, _raw_name in candidates[:max_commands]]
    return menu, hidden_count + overflow_count


# --- Discord ----------------------------------------------------------------

def discord_skill_commands_by_category(
    reserved_names: set[str],
) -> tuple[dict[str, list[tuple[str, str, str]]], list[tuple[str, str, str]], int]:
    """``(categories, uncategorized, hidden_count)`` for Discord ``/skill`` autocomplete.

    Skills nested >= 2 levels under a scan root (``creative/ascii-art/SKILL.md``) group under
    ``categories[top_level]``; root-level skills are *uncategorized*. Entries are
    ``(name, description, cmd_key)``, names clamped to 32 chars, descriptions to 100. No
    per-group cap (the caller flattens into one autocomplete callback); ``hidden_count`` only
    reports 32-char clamp collisions against reserved names or earlier skills.

    Scan roots include the local ``SKILLS_DIR`` **and** any configured ``skills.external_dirs`` — matching
    the widened filter applied to the flat ``discord_skill_commands()`` collector in #18741. Without this
    parity, external-dir skills are visible via ``hermes skills list`` and the agent's ``/skill-name``
    dispatch but silently absent from Discord's ``/skill`` autocomplete.
    The legacy 25-group × 25-subcommand caps (from the old nested ``/skill <cat> <name>`` layout) are
    **not** applied — the live caller (``_register_skill_group`` in ``gateway/platforms/discord.py``,
    refactored in PR #11580) flattens these results and feeds them into a single autocomplete callback,
    which scales to thousands of entries without any per-command payload concerns. ``hidden_count`` is
    retained in the return tuple for backward compatibility and still reports skills dropped for other
    reasons (32-char clamp collision vs a reserved name).
    """
    categories: dict[str, list[tuple[str, str, str]]] = {}
    uncategorized: list[tuple[str, str, str]] = []
    # clamped name -> origin; reserved names carry a sentinel so the warning distinguishes a
    # reserved-command collision from two skills colliding on the clamp (the rename-worthy case).
    names_used: dict[str, str] = dict.fromkeys(reserved_names, "<reserved>")
    hidden = 0
    try:
        for cmd_key, info, rel_parts in _iter_gateway_skills("discord"):
            # First (alphabetical) skill wins; the loser is dropped from the picker — warn loudly.
            discord_name = cmd_key.lstrip("/")[:32]
            prior = names_used.get(discord_name)
            if prior == "<reserved>":
                logger.warning(
                    "Discord /skill: %r (from %r) collides on its 32-char "
                    "clamp with a reserved gateway command name %r — the "
                    "skill will not appear in the /skill autocomplete. "
                    "Rename the skill's frontmatter ``name:`` to differ "
                    "in its first 32 chars.",
                    discord_name, cmd_key, discord_name)
            elif prior is not None:
                logger.warning(
                    "Discord /skill: %r and %r both clamp to %r on "
                    "Discord's 32-char command-name limit — only %r "
                    "will appear in the /skill autocomplete. Rename "
                    "one skill's frontmatter ``name:`` to differ in "
                    "its first 32 chars.",
                    prior, cmd_key, discord_name, prior)
            if prior is not None:
                hidden += 1
                continue
            names_used[discord_name] = cmd_key
            entry = (discord_name, _truncate_desc(info.get("description", ""), 100), cmd_key)
            if len(rel_parts) >= 2:
                categories.setdefault(rel_parts[0], []).append(entry)
            else:
                uncategorized.append(entry)
    except Exception:
        pass
    return categories, uncategorized, hidden


# --- Slack native slash commands --------------------------------------------

# Slack slash names: lowercase a-z, 0-9, hyphens, underscores, max 32 chars; an app manifest
# accepts up to 50 slash commands. Reserved = Slack built-ins apps cannot register
# (https://slack.com/help/articles/201259356-Use-built-in-slash-commands).
_SLACK_MAX_SLASH_COMMANDS = 50
_SLACK_NAME_LIMIT = 32
_SLACK_INVALID_CHARS = re.compile(r"[^a-z0-9_\-]")
_SLACK_RESERVED_COMMANDS = frozenset({
    "me", "status", "away", "dnd", "shrug", "remind", "msg", "feed", "who", "collapse", "expand",
    "leave", "join", "open", "search", "topic", "mute", "pro", "shortcuts"})

# Canonical commands deliberately routed through ``/hermes <command>`` on Slack only: the registry
# sits at Slack's 50-slash cap, so rather than let the clamp silently drop whichever command sorts
# last (breaking the Telegram-parity test), low-frequency ones are demoted here. Rule: when a new
# canonical command tips past the cap, demote a rarer one-off lookup (version, whoami, diff, ...)
# rather than a recurring interactive surface (context, loop, save, approvals). Keep TIGHT — the
# parity test reads this set. Aliases are never pinned ahead of canonicals.
_SLACK_VIA_HERMES_ONLY = frozenset({
    "topup", "moa", "debug", "egress", "init", "version", "diff", "update", "heartbeat",
    "refine", "review", "pause", "whoami", "platform", "insights", "group"})


def _sanitize_slack_name(raw: str) -> str:
    """Lowercase, strip chars outside ``[a-z0-9_-]`` and edge ``-_``, clamp to 32."""
    return _SLACK_INVALID_CHARS.sub("", raw.lower()).strip("-_")[:_SLACK_NAME_LIMIT]


def slack_native_slashes() -> list[tuple[str, str, str]]:
    """(slash_name, description, usage_hint) triples for Slack: every gateway-available command
    (canonical names first so they win slots at the cap, then aliases, then plugins) becomes a
    standalone slash, deduped and clamped to the 50-command cap; Slack built-ins and
    _SLACK_VIA_HERMES_ONLY are skipped. ``/hermes`` is always first for anything dropped."""
    available = _gateway_available_commands()
    wanted = [(cmd.name, cmd.description, cmd.args_hint or "") for cmd in available]
    wanted += [(alias, f"Alias for /{cmd.name} — {cmd.description}", cmd.args_hint or "")
               for cmd in available for alias in cmd.aliases]
    wanted += [(name, desc, hint or "") for name, desc, hint in _iter_plugin_command_entries()]

    entries: list[tuple[str, str, str]] = [
        ("hermes", "Talk to Hermes or run a subcommand", "[subcommand] [args]")]
    seen = {"hermes"}
    for name, desc, hint in wanted:
        slack_name = _sanitize_slack_name(name)
        if (not slack_name or slack_name in seen or slack_name in _SLACK_RESERVED_COMMANDS
                or slack_name in _SLACK_VIA_HERMES_ONLY
                or len(entries) >= _SLACK_MAX_SLASH_COMMANDS):
            continue
        # Slack description cap is 2000 chars; keep it short.
        entries.append((slack_name, desc[:140], hint[:100]))
        seen.add(slack_name)
    return entries


def slack_app_manifest(
    request_url: str = "https://hermes-agent.local/slack/commands") -> dict[str, Any]:
    """``features.slash_commands`` manifest portion only (decoupled from the rest of the manifest
    users configure in the Slack UI); ``request_url`` is schema-required, ignored in Socket Mode."""
    slashes = []
    for name, desc, usage in slack_native_slashes():
        entry = {"command": f"/{name}", "description": desc or f"Run /{name}",
                 "should_escape": False, "url": request_url}
        if usage:
            entry["usage_hint"] = usage
        slashes.append(entry)
    return {"features": {"slash_commands": slashes}}


def slack_subcommand_map() -> dict[str, str]:
    """name/alias -> "/command" for the Slack ``/hermes`` handler, plugin commands included."""
    mapping: dict[str, str] = {
        name: f"/{name}"
        for cmd in _gateway_available_commands() for name in (cmd.name, *cmd.aliases)}
    for name, _description, _args_hint in _iter_plugin_command_entries():
        mapping.setdefault(name, f"/{name}")
    return mapping
