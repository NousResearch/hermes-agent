"""Guide pointers and the existing responsibility roster for native prompt slots."""


def connection_guidance():
    from agent.knowledge import render
    return render("For service connection work, read {guides_root}/employee/references/service-connections.md and the existing {profile_home}/connections/<service>/manual.md, if present. Document every connection in that manual; keep reusable references and scripts alongside it.")


def file_keeping_guidance():
    from agent.knowledge import render
    return render("Keep lasting documents, data and deliverables in {profile_home}/documents/ and Git checkouts in {profile_home}/repos/. Before filing or reorganizing lasting material, read {guides_root}/file-keeping/guide.md. Search documents before saying a file is unavailable. Use native scratch storage for disposable work.")


def responsibility_prompt():
    from agent.knowledge import render
    from responsibilities.common import get_responsibilities_root, ResponsibilityFilesystemError
    from responsibilities.packages import scan_workspace_responsibilities
    from responsibilities.roster import render_responsibility_roster
    try:
        snapshot = scan_workspace_responsibilities(get_responsibilities_root())
    except (ResponsibilityFilesystemError, OSError) as exc:
        import logging
        logging.getLogger(__name__).warning("Responsibility index unavailable: %s", exc)
        return f"Responsibility index unavailable: {exc}. Existing responsibilities may still be present; repair the knowledge directory before assuming it is empty."
    else:
        roster = render_responsibility_roster({"responsibilities": [entry.to_dict() for entry in snapshot.entries]})
        return render(roster) if roster else render("No responsibilities yet. When taking ownership of work, read {guides_root}/responsibility-authoring/guide.md.")
