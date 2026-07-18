"""Shared `/profile` slash-command behavior."""

from __future__ import annotations

from hermes_constants import display_hermes_home
from hermes_cli.profiles import (
    get_active_profile_name,
    list_profiles,
    set_active_profile,
)


def handle_profile_slash(args: list[str], *, markdown: bool = False) -> str:
    """Handle in-session profile status/list/switch commands.

    The switch changes the sticky default used by future Hermes launches and
    new backend sessions. It intentionally does not mutate the current process's
    ``HERMES_HOME``.
    """
    if not args:
        profile_name = get_active_profile_name()
        display = display_hermes_home()
        if markdown:
            return "\n".join([
                f"**Profile:** `{profile_name}`",
                f"**Home:** `{display}`",
                "",
                "Use `/profile list` to see profiles or `/profile switch <name>` to set the default for new sessions.",
            ])
        return "\n".join([
            "",
            f"  Profile: {profile_name}",
            f"  Home:    {display}",
            "",
            "  Use /profile list to see profiles.",
            "  Use /profile switch <name> to set the default for new sessions.",
            "",
        ])

    subcommand = args[0].lower()
    if subcommand in {"list", "ls"}:
        profiles = list_profiles()
        active = get_active_profile_name()
        if not profiles:
            return "No profiles found."
        lines: list[str] = []
        for profile in profiles:
            is_active = profile.name == active or (
                active == "default" and profile.is_default
            )
            marker = "*" if is_active else "-"
            label = f"`{profile.name}`" if markdown else profile.name
            details = []
            if profile.model:
                details.append(profile.model)
            details.append(
                "gateway running" if profile.gateway_running else "gateway stopped"
            )
            lines.append(f"{marker} {label} ({', '.join(details)})")
        return "\n".join(lines)

    if subcommand in {"switch", "use"}:
        name = " ".join(args[1:]).strip()
        if not name:
            return "Usage: /profile switch <name>"
    else:
        name = " ".join(args).strip()

    set_active_profile(name)
    label = "default (~/.hermes)" if name == "default" else name
    return (
        f"Default profile set to {label}. "
        "Start a new session or restart Hermes to run under that profile."
    )
