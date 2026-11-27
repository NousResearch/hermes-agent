"""ACP historical catalog without storage initialization or an execution owner."""
from pathlib import Path

from persistence_assembly import open_persistence_reader, profile_scope, store_exists
from acp_adapter.session import (
    _normalize_cwd_for_compare, _parse_model_config, _session_info, _updated_at_sort_key,
)


def _store_scope(home):
    """The profile scope of *home*, a profile's folder."""
    return profile_scope(home=Path(home))


def read_catalog_rows(home, include_hidden=False):
    """Every ACP row of this profile's store, keyed by id.

    ``include_hidden`` lifts the listing's own exclusion (``SessionQuery.include_hidden``). Hidden
    rows still carry ``hidden`` as a cell, so a caller can tell them apart from the rest.
    """
    scope = _store_scope(home)
    if not store_exists(scope):
        return {}
    from hermes_cli.gateway_session_store import SurfaceLabelReader
    from persistence.conversation.sessions import SessionQuery, SessionWindow
    # The owner stores an editor thread as ``local`` with ``created_source='acp'``; the surface
    # labels match both in SQL, before the LIMIT.
    with open_persistence_reader(scope.profile_dir) as persistence:
        expanded = SurfaceLabelReader(persistence)._expand(["acp"])
        if expanded is None:
            query = SessionQuery(sources=("acp",), include_hidden=include_hidden)
        else:
            query = SessionQuery(sources=tuple(expanded[0]),
                                 local_created_sources=tuple(expanded[1]),
                                 include_hidden=include_hidden)
        entries = persistence.sessions.list_sessions(query, SessionWindow(limit=1000))
        rows = [entry.as_list_row() for entry in entries]
        return {str(row["id"]): dict(row) for row in rows}


def _local_owners(persistence):
    """Physical id -> retained logical owner, from each private local creation receipt's lineage.
    A canonical reset or compression advances the receipt's physical target but keeps policy/FIFO
    on the creation id. Legacy reset histories have no receipt and stay independent chats."""
    owners = {}
    for receipt in persistence.bindings.list_lineages():
        for physical in receipt.lineage:
            owners[physical] = receipt.owner
    return owners


def read_local_owners(home):
    scope = _store_scope(home)
    if not store_exists(scope):
        return {}
    with open_persistence_reader(scope.profile_dir) as persistence:
        return _local_owners(persistence)


def logical_session_id(home, session_id):
    """The authority's logical id for an id an editor already holds: compression and a canonical
    reset advance the physical transcript, never the session identity (the local receipt's owner,
    else ``SessionAuthority.logical_owner``'s compression root)."""
    scope = _store_scope(home)
    if not store_exists(scope):
        return session_id
    with open_persistence_reader(scope.profile_dir) as persistence:
        lineage = persistence.lineage.lineage(session_id)
        root = lineage[0] if lineage else session_id
        return _local_owners(persistence).get(root, root)


def catalog_sessions(home, cwd=None):
    normalized = _normalize_cwd_for_compare(cwd) if cwd else None
    owners = read_local_owners(home)
    results = {}
    for sid, row in read_catalog_rows(home).items():
        # The listing projects a compressed chat's tip; the editor must hold the logical id.
        sid = row.get("_lineage_root_id") or sid
        sid = owners.get(sid, sid)
        count = int(row.get("message_count") or 0)
        session_cwd = row.get('cwd') or _parse_model_config(row.get("model_config")).get("cwd", ".")
        if count <= 0 or (normalized and _normalize_cwd_for_compare(session_cwd) != normalized):
            continue
        info = _session_info(sid, session_cwd, row.get("model") or "", count, row.get("title"),
                             row.get("preview"), row.get("last_active") or row.get("started_at"))
        # One row per logical conversation: its most recently active segment represents it.
        held = results.get(sid)
        if held is None or _updated_at_sort_key(info.get("updated_at")) > _updated_at_sort_key(held.get("updated_at")):
            results[sid] = info
    return sorted(results.values(), key=lambda row: _updated_at_sort_key(row.get("updated_at")), reverse=True)


def catalog_with_hidden(home, cwd=None):
    """Every ACP session, hidden ones included, each row carrying its ``hidden`` flag.

    ``catalog_sessions`` is the projection ``session/list`` shows and keeps that listing's rule: a
    hidden (or archived) conversation is not in it. A side panel's tabs are the inverse projection —
    a tab is an *unhidden* session — so it needs both halves and the flag: the same rows out of the
    same store, read with ``include_hidden`` on the query.

    A row carries what a tab renders (``message_count``, epoch ``updated_at``) rather than
    ``session/list``'s ``history_len`` and formatted stamp, and the same logical id
    ``catalog_sessions`` reports: the editor loads a tab by that id, and compression advanced the
    physical transcript.
    """
    normalized = _normalize_cwd_for_compare(cwd) if cwd else None
    owners = read_local_owners(home)
    results = {}
    for sid, row in read_catalog_rows(home, include_hidden=True).items():
        sid = row.get("_lineage_root_id") or sid
        sid = owners.get(sid, sid)
        count = int(row.get("message_count") or 0)
        session_cwd = row.get("cwd") or _parse_model_config(row.get("model_config")).get("cwd", ".")
        if count <= 0 or (normalized and _normalize_cwd_for_compare(session_cwd) != normalized):
            continue
        info = {
            "session_id": sid,
            "cwd": session_cwd,
            "title": row.get("title") or "",
            "hidden": bool(int(row.get("hidden") or 0)),
            "message_count": count,
            "updated_at": float(row.get("last_active") or row.get("started_at") or 0.0),
        }
        # One row per logical conversation: its most recently active segment represents it.
        held = results.get(sid)
        if held is None or _updated_at_sort_key(info["updated_at"]) > _updated_at_sort_key(held["updated_at"]):
            results[sid] = info
    return sorted(results.values(), key=lambda item: _updated_at_sort_key(item["updated_at"]), reverse=True)
