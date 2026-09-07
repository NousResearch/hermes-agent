"""Registry-owned slash command execution (thin slice).

Invariant: an executor's output depends only on ``ctx.args`` / ``ctx.options`` — never on
``ctx.surface`` — so the core text is identical across surfaces for a fixed context (enforced by
tests/hermes_cli/test_commands_execute.py).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = ["CommandContext", "CommandReply", "EXECUTORS", "execute_command", "resolve_executor",
           "run_execute"]


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


def _exec_profile(ctx: CommandContext) -> CommandReply:
    """Core /profile data — active profile name + home directory.

    A multiplexed gateway may pre-resolve the per-source profile/home via ``options``
    (``profile_name`` / ``home_display``); otherwise process-level values are used.
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
    return CommandReply(f"Profile: {label}\nHome: {home_display}",
                        data={"profile": profile_name, "home": home_display})


def _exec_edition(ctx: CommandContext) -> CommandReply:
    """Core /edition text — North Forge tier + pinned-edition status and switch menu.

    Read-only (the executor invariant): switching an edition changes ``HERMES_HOME``
    and needs a relaunch, so this prints the exact ``hermes profile use`` command
    rather than mutating state. On a Basic-tier drive there is nothing to switch to.
    """
    try:
        from hermes_cli import nf_tier
    except Exception as exc:  # pragma: no cover
        return CommandReply(f"North Forge tier subsystem unavailable: {exc}", data={"error": str(exc)})

    p = nf_tier.load(use_cache=False)
    arg = (ctx.args or "").strip()

    if p.state == nf_tier.STATE_UNPROVISIONED:
        return CommandReply(
            "This drive has no North Forge provisioning record — all editions are "
            "available. Use `hermes profile list` / `hermes profile use <name>`.",
            data={"state": p.state})

    if p.state == nf_tier.STATE_TAMPERED:
        return CommandReply(nf_tier._tampered_message(p),
                            data={"state": p.state, "error": p.error})

    try:
        from hermes_cli.profiles import get_active_profile_name
        current = get_active_profile_name()
    except Exception:
        current = "default"
    current_label = current if current != "default" else "North Forge (chassis)"
    pin_label = p.pinned_edition or "North Forge (chassis)"

    if p.locked:
        text = (
            f"Edition: {pin_label}\n"
            f"Tier:    Basic — locked to this edition on this drive.\n"
            f"There is no switcher: no other edition is installed or reachable here."
        )
        return CommandReply(text, data={
            "state": p.state, "tier": p.tier, "current": current,
            "pinned_edition": p.pinned_edition, "locked": True, "switchable": []})

    # Full tier — the pin is only a default; list what can be switched to.
    try:
        from hermes_cli.profiles import list_profile_names
        names = list_profile_names()
    except Exception:
        names = ["default"]
    menu = []
    for n in names:
        label = n if n != "default" else "default  (North Forge chassis)"
        mark = "→ " if (n == current or (n == "default" and current == "default")) else "  "
        pinmark = "  [pinned default]" if nf_tier.normalize_edition(n) == (p.pinned_edition or "default") else ""
        menu.append(f"  {mark}{label}{pinmark}")
    text = (
        f"Edition: {current_label}\n"
        f"Tier:    Full — pinned default is '{pin_label}'; any edition is available.\n\n"
        f"Installed editions:\n" + "\n".join(menu) +
        f"\n\nSwitch with:  hermes profile use <name>   (then relaunch)"
    )
    return CommandReply(text, data={
        "state": p.state, "tier": p.tier, "current": current,
        "pinned_edition": p.pinned_edition, "locked": False, "switchable": names,
        "requested": arg or None})


def _exec_bundles(ctx: CommandContext) -> CommandReply:
    """Core /bundles data — installed skill bundles listing."""
    try:
        from agent.skill_bundles import _bundles_dir, list_bundles
    except Exception as exc:  # pragma: no cover - env-specific
        return CommandReply(f"Bundles subsystem unavailable: {exc}", data={"error": str(exc)})
    bundles = list_bundles()
    bundles_dir = str(_bundles_dir())
    if not bundles:
        return CommandReply(
            "No skill bundles installed.\n"
            "Create one with: hermes bundles create <name> --skill <s1> --skill <s2>\n"
            f"Directory: {bundles_dir}",
            data={"bundles": [], "dir": bundles_dir})
    lines = [f"Skill Bundles ({len(bundles)} installed):"]
    for info in bundles:
        skills = info.get("skills", [])
        desc = info.get("description") or f"Load {len(skills)} skills"
        lines.append(f"/{info['slug']} — {desc} ({len(skills)} skills)")
        lines.extend(f"    · {s}" for s in skills)
    lines.append("Invoke a bundle with /<slug> to load all its skills.")
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
    from agent.i18n import t
    from hermes_cli.commands import gateway_help_lines
    lines = [t("gateway.help.header"), *gateway_help_lines()]
    skill_cmds = _skill_commands()
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
    from agent.i18n import t
    from hermes_cli.commands import gateway_help_lines
    try:
        requested_page = int((ctx.args or "").strip() or 1)
    except ValueError:
        return CommandReply(t("gateway.commands.usage"), format="markdown")

    entries = list(gateway_help_lines())
    skill_cmds = _skill_commands()
    try:
        if skill_cmds:
            entries.extend(["", t("gateway.commands.skill_header")])
            for cmd in sorted(skill_cmds):
                desc = skill_cmds[cmd].get("description", "").strip() or t("gateway.commands.default_desc")
                entries.append(f"`{cmd}` — {desc}")
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
    "edition": _exec_edition,
    "bundles": _exec_bundles,
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
