"""What a session picker shows for each ``session.list`` row (``picker: true``), and the folder filter (#150091).

Location (``cwd``, ``workspace``) comes from the conversation's first row: the row ``workspace`` filters
by (the listing filters root rows, as ``hermes -c`` does), and the row a workspace move rewrites with every
other one. The rest comes from its current segment (a local receipt's entry, else the compression tip), as
the listing projects a compressed chain onto its tip: the first prompt there (``preview``), its branch,
pin, newest activity and newest reply. The stored feed passes the listing entry it already read; the live
feed reads the columns here. Either way a page is one store read in a worker thread
(``sessions.picker_session_facts``).
"""
from __future__ import annotations

from persistence.conversation.sessions import workspace_key

from gateway.session_persistence import authority_persistence


def workspace_members(store, session_ids, workspace):
    """The ids among *session_ids* whose row is in *workspace* (``sessions.sessions_in_workspace``: the rule
    the listing filters the stored feed by, ``hermes -c``'s)."""
    return store.sessions.sessions_in_workspace(session_ids, workspace)


def fill_picker_fields(store, items):
    """Set the picker fields on each ``(row, root, segment, listed)``: *listed* is the segment's listing entry
    (``SessionListEntry``: its ``preview`` / ``last_active`` / branch / pin), None to read them."""
    facts = store.sessions.picker_session_facts(
        [root for _, root, _, _ in items], [segment for _, _, segment, listed in items if listed is None],
        [segment for _, _, segment, _ in items])
    for row, root, segment, listed in items:
        if listed is None:
            columns = facts.segments.get(segment)
            if columns is None:
                continue  # deleted since it was listed: the row keeps what the feed read
            preview, branch, last_active, pinned = columns.preview, columns.git_branch, columns.last_active, columns.pinned
        else:
            preview, branch, last_active, pinned = (listed.preview, listed.session.git_branch, listed.last_active,
                                                    listed.session.pinned)
        first = facts.roots.get(root)
        location = {'cwd': first.cwd, 'git_repo_root': first.git_repo_root} if first is not None else {}
        row.update(preview=preview or None, cwd=location.get('cwd') or None, workspace=workspace_key(location),
                   git_branch=branch or None, last_active_at=last_active, pinned=bool(pinned),
                   last_reply=facts.last_replies.get(segment))


def current_segments(authority, session_ids):
    """``{logical id: current physical segment}`` (``lineage.current_segments``: a local receipt's entry, else
    the compression tip) in one store read."""
    return authority_persistence(authority).lineage.current_segments(session_ids)


def live_picker_page(authority, sessions, limit, workspace=None, picker=False):
    """The live feed's page: *sessions* (newest first) in *workspace*, cut to *limit*, with the picker
    fields when asked."""
    store = authority_persistence(authority)
    if workspace is not None:
        members = workspace_members(store, [row['session_id'] for row in sessions], workspace)
        sessions = [row for row in sessions if row['session_id'] in members]
    page = sessions[:limit]
    if picker and page:
        segments = current_segments(authority, [row['session_id'] for row in page])
        fill_picker_fields(store, [(row, row['session_id'], segments.get(row['session_id'], row['session_id']),
                                    None) for row in page])
    return page
