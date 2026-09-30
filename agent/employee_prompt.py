"""Knowledge pointers added to the native prompt; final offering wording is open."""

def prompt_parts(agent):
    from agent.knowledge import render
    from hermes_constants import get_hermes_home
    from responsibilities.common import get_responsibilities_root, ResponsibilityFilesystemError
    from responsibilities.packages import scan_workspace_responsibilities
    from responsibilities.roster import render_responsibility_roster
    parts = [
        render("For any service connection work, read {guides_root}/employee/references/service-connections.md and the existing manual.md under {profile_home}/connections/<service>/ before operating. Document every connection there, including its account, access method and verification result; keep reusable references and scripts alongside the manual, never secrets. Update the manual when access or operating knowledge changes. The listing records operating knowledge, not current access."),
    ]
    manual_root = get_hermes_home() / "connections"
    manuals = [p.parent.name for p in sorted(manual_root.glob("*/manual.md")) if not p.is_symlink()]
    parts.append("Service manuals: " + (", ".join(manuals) or "none yet"))
    try:
        snapshot = scan_workspace_responsibilities(get_responsibilities_root())
    except (ResponsibilityFilesystemError, OSError) as exc:
        import logging
        logging.getLogger(__name__).warning("Responsibility index unavailable: %s", exc)
        parts.append(f"Responsibility index unavailable: {exc}. Existing responsibilities may still be present; repair the knowledge directory before assuming it is empty.")
    else:
        roster = render_responsibility_roster({"responsibilities": [entry.to_dict() for entry in snapshot.entries]})
        parts.append(render(roster) if roster else render("No responsibilities yet. When taking ownership of work, read {guides_root}/responsibility-authoring/guide.md."))
    return [part for part in parts if part]
