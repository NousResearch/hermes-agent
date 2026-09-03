"""Registry-owned slash command execution (thin slice).

Shared, surface-independent executors for informational slash commands.
``CommandDef.execute`` (hermes_cli/commands.py) names a key in
:data:`EXECUTORS`; each surface (CLI REPL, gateway, TUI slash worker via the
CLI) resolves that key through :func:`run_execute` and applies only its own
decoration (Rich markup, emoji/markdown, ``_telegramize_command_mentions``)
to the canonical :class:`CommandReply`.

Invariant: an executor's output depends only on ``ctx.args`` / ``ctx.options``
— never on ``ctx.surface`` — so the core text is identical across surfaces
for a fixed context (enforced by tests/hermes_cli/test_commands_execute.py).

Import discipline: this module imports nothing heavy at module level and
``hermes_cli.commands`` does NOT import this module (the ``execute`` field is
a plain string), so the gateway can keep importing ``commands.py`` without
prompt_toolkit and without cycles.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "CommandContext",
    "CommandReply",
    "EXECUTORS",
    "execute_command",
    "resolve_executor",
    "run_execute",
]


# ---------------------------------------------------------------------------
# Context / reply dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CommandContext:
    """Surface-provided inputs for a shared command executor."""

    surface: str = "cli"                # "cli" | "gateway" | "tui" — decoration only
    args: str = ""                      # raw argument string after the command word
    options: Mapping[str, Any] = field(default_factory=dict)  # surface params (page_size, ...)
    config_get: Callable[[str, Any], Any] | None = None       # optional config accessor


@dataclass(frozen=True)
class CommandReply:
    """Canonical result of a shared executor.

    ``text`` is the surface-independent core text.  ``data`` carries the
    structured values the executor derived so a surface may re-render them
    with its own decoration (Rich columns, markdown bullets) without
    duplicating the computation.  ``format`` is a rendering hint only.
    """

    text: str
    data: Mapping[str, Any] = field(default_factory=dict)
    format: str = "plain"               # "plain" | "markdown" (hint, not a contract)


# ---------------------------------------------------------------------------
# Executors — pure formatters, no agent/session mutation
# ---------------------------------------------------------------------------

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

    A multiplexed gateway may pre-resolve the per-source profile/home and pass
    them via ``options`` (``profile_name`` / ``home_display``); otherwise the
    process-level values are used (identical to the old CLI + non-multiplex
    gateway behavior).
    """
    profile_name = str(ctx.options.get("profile_name") or "").strip()
    home_display = str(ctx.options.get("home_display") or "").strip()

    if not profile_name:
        from hermes_cli.profiles import get_active_profile_name

        profile_name = get_active_profile_name()
    if not home_display:
        from hermes_constants import display_hermes_home

        home_display = display_hermes_home()

    # Presentation-only display name (profile.yaml). `data.profile` stays
    # the canonical id — consumers route on it; only the text gets the label.
    label = profile_name
    try:
        from hermes_cli.profiles import (
            format_profile_label,
            get_profile_dir,
            read_profile_meta,
        )

        display = read_profile_meta(get_profile_dir(profile_name)).get("display_name", "")
        label = format_profile_label(profile_name, display)
    except Exception:
        pass

    return CommandReply(
        f"Profile: {label}\nHome: {home_display}",
        data={"profile": profile_name, "home": home_display},
    )


def _exec_bundles(ctx: CommandContext) -> CommandReply:
    """Core /bundles data — installed skill bundles listing."""
    try:
        from agent.skill_bundles import _bundles_dir, list_bundles
    except Exception as exc:  # pragma: no cover - env-specific
        return CommandReply(
            f"Bundles subsystem unavailable: {exc}",
            data={"error": str(exc)},
        )

    bundles = list_bundles()
    bundles_dir = str(_bundles_dir())
    if not bundles:
        return CommandReply(
            "No skill bundles installed.\n"
            "Create one with: hermes bundles create <name> --skill <s1> --skill <s2>\n"
            f"Directory: {bundles_dir}",
            data={"bundles": [], "dir": bundles_dir},
        )

    lines = [f"Skill Bundles ({len(bundles)} installed):"]
    for info in bundles:
        skill_count = len(info.get("skills", []))
        desc = info.get("description") or f"Load {skill_count} skills"
        lines.append(f"/{info['slug']} — {desc} ({skill_count} skills)")
        for s in info.get("skills", []):
            lines.append(f"    · {s}")
    lines.append("Invoke a bundle with /<slug> to load all its skills.")
    return CommandReply(
        "\n".join(lines),
        data={"bundles": bundles, "dir": bundles_dir},
    )


def _exec_help(ctx: CommandContext) -> CommandReply:
    """Core gateway /help body (pre platform mention decoration)."""
    from agent.i18n import t
    from hermes_cli.commands import gateway_help_lines

    lines = [
        t("gateway.help.header"),
        *gateway_help_lines(),
    ]
    try:
        from agent.skill_commands import get_skill_commands
        skill_cmds = get_skill_commands()
        if skill_cmds:
            lines.append(t("gateway.help.skill_header", count=len(skill_cmds)))
            # Show first 10, then point to /commands for the rest
            sorted_cmds = sorted(skill_cmds)
            for cmd in sorted_cmds[:10]:
                lines.append(f"`{cmd}` — {skill_cmds[cmd]['description']}")
            if len(sorted_cmds) > 10:
                lines.append(t("gateway.help.more_use_commands", count=len(sorted_cmds) - 10))
    except Exception:
        pass
    return CommandReply("\n".join(lines), format="markdown")


def _exec_commands(ctx: CommandContext) -> CommandReply:
    """Core gateway /commands body — paginated command + skill listing.

    ``ctx.options["page_size"]`` is a surface parameter (Telegram uses 15,
    everything else 20) — for a fixed context the text is surface-invariant.
    """
    from agent.i18n import t
    from hermes_cli.commands import gateway_help_lines

    raw_args = (ctx.args or "").strip()
    if raw_args:
        try:
            requested_page = int(raw_args)
        except ValueError:
            return CommandReply(t("gateway.commands.usage"), format="markdown")
    else:
        requested_page = 1

    # Build combined entry list: built-in commands + skill commands
    entries = list(gateway_help_lines())
    try:
        from agent.skill_commands import get_skill_commands
        skill_cmds = get_skill_commands()
        if skill_cmds:
            entries.append("")
            entries.append(t("gateway.commands.skill_header"))
            for cmd in sorted(skill_cmds):
                desc = skill_cmds[cmd].get("description", "").strip() or t("gateway.commands.default_desc")
                entries.append(f"`{cmd}` — {desc}")
    except Exception:
        pass

    if not entries:
        return CommandReply(t("gateway.commands.none"), format="markdown")

    try:
        page_size = int(ctx.options.get("page_size", 20))
    except (TypeError, ValueError):
        page_size = 20
    page_size = max(1, page_size)
    total_pages = max(1, (len(entries) + page_size - 1) // page_size)
    page = max(1, min(requested_page, total_pages))
    start = (page - 1) * page_size
    page_entries = entries[start:start + page_size]

    lines = [
        t("gateway.commands.header", total=len(entries), page=page, total_pages=total_pages),
        "",
        *page_entries,
    ]
    if total_pages > 1:
        nav_parts = []
        if page > 1:
            nav_parts.append(t("gateway.commands.nav_prev", page=page - 1))
        if page < total_pages:
            nav_parts.append(t("gateway.commands.nav_next", page=page + 1))
        lines.extend(["", " | ".join(nav_parts)])
    if page != requested_page:
        lines.append(t("gateway.commands.out_of_range", requested=requested_page, page=page))
    return CommandReply("\n".join(lines), format="markdown")


# ---------------------------------------------------------------------------
# Registry + resolution
# ---------------------------------------------------------------------------

EXECUTORS: dict[str, Callable[[CommandContext], CommandReply]] = {
    "version": _exec_version,
    "egress": _exec_egress,
    "profile": _exec_profile,
    "bundles": _exec_bundles,
    "gateway_help": _exec_help,
    "gateway_commands": _exec_commands,
}


def resolve_executor(cmd_def: Any) -> Callable[[CommandContext], CommandReply] | None:
    """Return the shared executor for ``cmd_def`` (or None when not migrated)."""
    key = getattr(cmd_def, "execute", None)
    if not key:
        return None
    return EXECUTORS.get(key)


def run_execute(cmd_def: Any, ctx: CommandContext) -> CommandReply | None:
    """Run ``cmd_def``'s registry-owned executor, if any."""
    fn = resolve_executor(cmd_def)
    if fn is None:
        return None
    return fn(ctx)


def execute_command(name: str, ctx: CommandContext) -> CommandReply:
    """Run the shared executor for the command named ``name``.

    Raises ``LookupError`` when the command is unknown or not migrated —
    call sites use this only for commands they know carry ``execute``.
    """
    from hermes_cli.commands import resolve_command

    cmd_def = resolve_command(name)
    reply = run_execute(cmd_def, ctx) if cmd_def is not None else None
    if reply is None:
        raise LookupError(f"no registry-owned executor for /{name}")
    return reply
