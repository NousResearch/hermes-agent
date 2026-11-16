"""Registry-owned slash command execution (thin slice).

Invariant: an executor's output depends only on ``ctx.args`` / ``ctx.options`` — never on
``ctx.surface`` — so the core text is identical across surfaces for a fixed context (enforced by
tests/hermes_cli/test_commands_execute.py).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from agent.i18n import t

__all__ = [
    "EXECUTORS",
    "CommandContext",
    "CommandReply",
    "execute_command",
    "resolve_executor",
    "run_execute",
]


@dataclass(frozen=True)
class CommandContext:
    """Surface-provided inputs for a shared command executor."""
    surface: str = "cli"                # "cli" | "gateway" | "tui" — decoration only
    args: str = ""                      # raw argument string after the command word
    options: Mapping[str, Any] = field(default_factory=dict)  # surface params (page_size, ...)
    config_get: Callable[[str, Any], Any] | None = None       # optional config accessor


@dataclass(frozen=True)
class CommandReply:
    """Canonical result of a shared executor: surface-independent ``text`` plus the structured
    ``data`` it derived so a surface can re-render with its own decoration."""
    text: str
    data: Mapping[str, Any] = field(default_factory=dict)
    format: str = "plain"               # "plain" | "markdown" (hint, not a contract)


# Executors — pure formatters, no agent/session mutation.
def _exec_version(ctx: CommandContext) -> CommandReply:
    """Core /version text — the banner version label."""
    from hermes_cli.banner import format_banner_version_label
    return CommandReply(format_banner_version_label())


def _exec_egress(ctx: CommandContext) -> CommandReply:
    """Core /egress text — Docker egress proxy status."""
    from hermes_cli.proxy_cli import format_status_text
    return CommandReply(format_status_text())


def _profiles_for_display() -> list[Any]:
    """Every profile for ``/profile --list`` / ``/profile <name>``, name-only if a config won't load."""
    from hermes_cli.profiles import list_profile_names, list_profiles
    try:
        return list_profiles(lazy_skill_count=True)
    except Exception:
        return [SimpleNamespace(name=name, model=None, provider=None, is_default=name == "default",
                                path=None, display_name="") for name in list_profile_names()]


def _profile_roster(active: str) -> CommandReply:
    """``/profile --list`` — every profile, the active one marked, plus how a profile is entered."""
    rows = _profiles_for_display()
    if not rows:
        return CommandReply("No profiles found.", data={"profiles": [], "active": active})
    width = max(len(p.name) for p in rows)
    lines = [f"Profiles ({len(rows)}), ◆ = active here:"]
    for p in rows:
        model = f"  {p.model}" if p.model else ""
        lines.append(f"  {'◆' if p.name == active else ' '} {p.name:<{width}}{model}")
    lines.append("Switch: hermes -p <name>   (or `hermes profile use <name>` to change the default)")
    return CommandReply("\n".join(lines),
                        data={"profiles": [p.name for p in rows], "active": active})


def _profile_detail(requested: str, active: str) -> CommandReply:
    """``/profile <name>`` — that profile's model/path and the command that actually switches.

    A profile is bound at process launch, so this reports; it never claims a mid-session swap.
    """
    from hermes_cli.profiles import format_profile_label, normalize_profile_name
    try:
        wanted = normalize_profile_name(requested)
    except ValueError:
        wanted = requested
    rows = _profiles_for_display()
    match = next((p for p in rows if p.name == wanted), None)
    if match is None:
        known = ", ".join(p.name for p in rows) or "none"
        return CommandReply(f"No profile named '{requested}'. Known profiles: {known}.",
                            data={"error": "unknown_profile", "requested": requested,
                                  "profiles": [p.name for p in rows]})
    lines = [f"Profile: {format_profile_label(match.name, getattr(match, 'display_name', ''))}"]
    if match.path is not None:
        lines.append(f"Path:    {match.path}")
    if match.model:
        lines.append(f"Model:   {match.model}" + (f" ({match.provider})" if match.provider else ""))
    is_active = match.name == active
    lines.append(f"Active:  {'yes' if is_active else f'no — this session runs {active}'}")
    switch = f"hermes -p {match.name}"
    if is_active:
        lines.append(f"Switch:  {switch}   (already active here; starts a new session on it)")
    else:
        lines.append(f"Switch:  {switch}   (or `hermes profile use {match.name}` to change the default)")
    return CommandReply("\n".join(lines),
                        data={"profile": match.name, "active": is_active,
                              "model": match.model or "", "path": str(match.path or "")})


def _exec_profile(ctx: CommandContext) -> CommandReply:
    """Core /profile data — the active profile, the roster, or one profile's details.

    Bare: active profile name + home directory. A multiplexed gateway may pre-resolve the
    per-source profile/home via ``options`` (``profile_name`` / ``home_display``); otherwise
    process-level values are used. ``--list``/``-l`` reads the roster, a name reads that
    profile — ``/profile`` cannot switch, because the profile is bound when the process starts.
    """
    from hermes_cli.profiles import get_active_profile_name
    from hermes_constants import display_hermes_home
    profile_name = str(ctx.options.get("profile_name") or "").strip() or get_active_profile_name()
    home_display = str(ctx.options.get("home_display") or "").strip() or display_hermes_home()
    # Presentation-only display name (profile.yaml); `data.profile` stays the canonical id.
    label = profile_name
    try:
        from hermes_cli.profiles import format_profile_label, get_profile_dir, read_profile_meta
        display = read_profile_meta(get_profile_dir(profile_name)).get("display_name", "")
        label = format_profile_label(profile_name, display)
    except Exception:
        pass
    args = (ctx.args or "").strip()
    if args:
        # Flags, not words: ``PROFILE_ID_RE`` is ``^[a-z0-9][a-z0-9_-]{0,63}$``, so no profile can
        # be named ``--list``/``-l`` and ``/profile list`` stays a real profile.
        if args in {"--list", "-l"}:
            return _profile_roster(profile_name)
        return _profile_detail(args, profile_name)
    return CommandReply(t("gateway.profile.plain", profile=label, home=home_display),
                        data={"profile": profile_name, "home": home_display})


def _exec_autoprune(ctx: CommandContext) -> CommandReply:
    """Core /autoprune text — record the auto-prune decision or show its state."""
    from hermes_cli.prune_advisor import run_command
    return CommandReply(run_command(ctx.args))


def _exec_bundles(ctx: CommandContext) -> CommandReply:
    """Core /bundles data — installed skill bundles listing."""
    try:
        from agent.skill_bundles import _bundles_dir, list_bundles
    except Exception as exc:  # pragma: no cover - env-specific
        return CommandReply(t("gateway.bundles.unavailable", error=exc), data={"error": str(exc)})
    bundles = list_bundles()
    bundles_dir = str(_bundles_dir())
    if not bundles:
        return CommandReply(t("gateway.bundles.none_plain", dir=bundles_dir), data={"bundles": [], "dir": bundles_dir})
    lines = [t("gateway.bundles.header_plain", count=len(bundles))]
    for info in bundles:
        skills = info.get("skills", [])
        desc = info.get("description") or t("gateway.bundles.default_desc", count=len(skills))
        lines.append(t("gateway.bundles.item_plain", slug=info["slug"], desc=desc, count=len(skills)))
        lines.extend(f"    · {s}" for s in skills)
    lines.append(t("gateway.bundles.invoke_hint_plain"))
    return CommandReply("\n".join(lines), data={"bundles": bundles, "dir": bundles_dir})


def _skill_commands() -> dict:
    """Registered skill commands, or ``{}`` when the skill subsystem is unavailable."""
    try:
        from agent.skill_commands import get_skill_commands
        return get_skill_commands() or {}
    except Exception:
        return {}


def _exec_help(ctx: CommandContext) -> CommandReply:
    """Core gateway /help body (pre platform mention decoration)."""
    from hermes_cli.commands import gateway_help_lines
    # ``allowed_commands`` (gateway, non-admin caller): only the commands the slash-access
    # policy lets this user run; skill commands are hidden too since the gate refuses them.
    allowed = ctx.options.get("allowed_commands")
    lines = [t("gateway.help.header"), *gateway_help_lines(allowed)]
    skill_cmds = _skill_commands() if allowed is None else {}
    try:
        if skill_cmds:
            lines.append(t("gateway.help.skill_header", count=len(skill_cmds)))
            sorted_cmds = sorted(skill_cmds)  # first 10, then point to /commands for the rest
            lines.extend(f"`{cmd}` — {skill_cmds[cmd]['description']}"
                         for cmd in sorted_cmds[:10])
            if len(sorted_cmds) > 10:
                lines.append(t("gateway.help.more_use_commands", count=len(sorted_cmds) - 10))
    except Exception:
        pass
    return CommandReply("\n".join(lines), format="markdown")


def _exec_commands(ctx: CommandContext) -> CommandReply:
    """Core gateway /commands body — paginated command + skill listing.

    ``ctx.options["page_size"]`` is a surface parameter (Telegram uses 15, everything else 20).
    """
    from hermes_cli.commands import gateway_help_lines
    try:
        requested_page = int((ctx.args or "").strip() or 1)
    except ValueError:
        return CommandReply(t("gateway.commands.usage"), format="markdown")

    allowed = ctx.options.get("allowed_commands")
    entries = list(gateway_help_lines(allowed))
    skill_cmds = _skill_commands() if allowed is None else {}
    try:
        if skill_cmds:
            entries.extend(["", t("gateway.commands.skill_header")])
            for cmd in sorted(skill_cmds):
                desc = skill_cmds[cmd].get("description", "").strip() or t("gateway.commands.default_desc")
                entries.append(f"`{cmd}` — {desc}")
        # Skills kept off the menu because a built-in owns the name (agent.skill_commands guard).
        from agent.skill_commands import skill_command_collision_note
        from tools.skills_tool import find_all_skills
        entries.extend(f"⚠ {note}" for note in filter(None, map(
            skill_command_collision_note, sorted(s["name"] for s in find_all_skills()))))
    except Exception:
        pass

    if not entries:
        return CommandReply(t("gateway.commands.none"), format="markdown")

    try:
        page_size = max(1, int(ctx.options.get("page_size", 20)))
    except (TypeError, ValueError):
        page_size = 20
    total_pages = max(1, (len(entries) + page_size - 1) // page_size)
    page = max(1, min(requested_page, total_pages))
    start = (page - 1) * page_size
    lines = [t("gateway.commands.header", total=len(entries), page=page, total_pages=total_pages),
             "", *entries[start:start + page_size]]
    if total_pages > 1:
        nav_parts = ([t("gateway.commands.nav_prev", page=page - 1)] if page > 1 else []) + (
            [t("gateway.commands.nav_next", page=page + 1)] if page < total_pages else [])
        lines.extend(["", " | ".join(nav_parts)])
    if page != requested_page:
        lines.append(t("gateway.commands.out_of_range", requested=requested_page, page=page))
    return CommandReply("\n".join(lines), format="markdown")


EXECUTORS: dict[str, Callable[[CommandContext], CommandReply]] = {
    "version": _exec_version,
    "egress": _exec_egress,
    "profile": _exec_profile,
    "bundles": _exec_bundles,
    "autoprune": _exec_autoprune,
    "gateway_help": _exec_help,
    "gateway_commands": _exec_commands}


def resolve_executor(cmd_def: Any) -> Callable[[CommandContext], CommandReply] | None:
    """Return the shared executor for ``cmd_def`` (or None when not migrated)."""
    return EXECUTORS.get(getattr(cmd_def, "execute", None) or "")


def run_execute(cmd_def: Any, ctx: CommandContext) -> CommandReply | None:
    """Run ``cmd_def``'s registry-owned executor, if any."""
    fn = resolve_executor(cmd_def)
    return None if fn is None else fn(ctx)


def execute_command(name: str, ctx: CommandContext) -> CommandReply:
    """Run the shared executor for ``name``; ``LookupError`` when unknown or not migrated."""
    from hermes_cli.commands import resolve_command
    cmd_def = resolve_command(name)
    reply = run_execute(cmd_def, ctx) if cmd_def is not None else None
    if reply is None:
        raise LookupError(f"no registry-owned executor for /{name}")
    return reply
