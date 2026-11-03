"""``hermes profile create NAME --from-prompt TEXT``: draft the bot, show it, save on confirm."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from hermes_cli.profile_draft import ProfileDraft, build_catalog, draft_profile


def _catalog_source(args) -> Path:
    """Skills/MCP come from the profile the new one starts from (clone source, else the active one)."""
    from hermes_cli.profiles import get_active_profile_name, get_profile_dir
    return Path(get_profile_dir(getattr(args, "clone_from", None) or get_active_profile_name()))


def _source_catalog(source: Path):
    """The catalog under ``source``'s home, so plugin toolsets are the ones installed THERE."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    token = set_hermes_home_override(source)
    try:
        return build_catalog(source)
    finally:
        reset_hermes_home_override(token)


def render_draft(name: str, draft: ProfileDraft) -> str:
    lines = [f"Name:        {name}", f"Title:       {draft.title or '(none)'}",
             f"Description: {draft.description or '(none)'}", "", "SOUL.md", "-------", draft.soul.rstrip(), ""]
    for kind, label in (("skills", "Suggested skills"), ("toolsets", "Suggested toolsets"),
                        ("mcp_servers", "Suggested MCP servers")):
        picks = getattr(draft, kind)
        lines.append(f"{label}: {', '.join(picks) if picks else '(none)'}")
    if draft.dropped:
        lines.append("Not installed here, left out: " + ", ".join(f"{d['name']} ({d['kind']})" for d in draft.dropped))
    return "\n".join(lines)


def confirm_draft(args, die) -> Optional[ProfileDraft]:
    """Draft from ``--from-prompt`` and get the user's yes, or ``die`` with a reason (None without
    the flag). Nothing is created before this returns. Non-interactive runs must pass ``--yes``:
    a hidden prompt would hang."""
    if not getattr(args, "from_prompt", None):
        return None
    interactive = sys.stdin.isatty() and sys.stdout.isatty()
    if not interactive and not getattr(args, "yes", False):
        die("Error: --from-prompt shows a draft and asks before creating; re-run in a terminal or add --yes.")
    print("Drafting the profile with the auxiliary model…", flush=True)
    draft = draft_profile(args.from_prompt, catalog=_source_catalog(_catalog_source(args)))
    if not draft.ok:
        die(f"Error: could not draft the profile: {draft.reason}")
    print("\n" + render_draft(args.profile_name, draft) + "\n")
    if getattr(args, "yes", False):
        return draft
    answer = input(f"Create profile '{args.profile_name}' with this draft? [y/N] ").strip().lower()
    if answer not in ("y", "yes"):
        die("Cancelled — nothing was created.")
    return draft


def draft_identity(draft: Optional[ProfileDraft], description: Optional[str]) -> dict:
    """``create_profile`` kwargs: the confirmed draft's SOUL, description (an explicit
    ``--description`` wins) and title go into the staging tree, so the profile is published
    with them in one rename — never seeded first and rewritten after."""
    if draft is None:
        return {"description": description}
    return {"description": description or draft.description, "soul": draft.soul, "display_name": draft.title or None}


def print_suggestions(name: str, draft: Optional[ProfileDraft]) -> None:
    """Capabilities stay suggestions: skills are already seeded, toolset/MCP changes are the user's call."""
    if draft and draft.toolsets:
        print(f"Suggested toolsets: {', '.join(draft.toolsets)} — enable with `hermes -p {name} tools`.")
    if draft and draft.mcp_servers:
        print(f"Suggested MCP servers: {', '.join(draft.mcp_servers)} — add with `hermes -p {name} mcp`.")
