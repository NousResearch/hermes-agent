"""ACP historical catalog without storage initialization or an execution owner."""
from pathlib import Path

from persistence_assembly import open_persistence_reader
from acp_adapter.session import (
    _normalize_cwd_for_compare, _parse_model_config, _session_info, _updated_at_sort_key,
)


def read_catalog_rows(path):
    path = Path(path)
    if not path.exists():
        return {}
    from hermes_cli.gateway_session_store import SurfaceLabelReader
    from persistence.conversation.sessions import SessionQuery, SessionWindow
    # The owner stores an editor thread as ``local`` with ``created_source='acp'``; the surface
    # labels match both in SQL, before the LIMIT.
    with open_persistence_reader(path) as persistence:
        expanded = SurfaceLabelReader(persistence)._expand(["acp"])
        query = (SessionQuery(sources=("acp",)) if expanded is None
                 else SessionQuery(sources=tuple(expanded[0]), local_created_sources=tuple(expanded[1])))
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


def read_local_owners(path):
    path = Path(path)
    if not path.exists():
        return {}
    with open_persistence_reader(path) as persistence:
        return _local_owners(persistence)


def logical_session_id(path, session_id):
    """The authority's logical id for an id an editor already holds: compression and a canonical
    reset advance the physical transcript, never the session identity (the local receipt's owner,
    else ``SessionAuthority.logical_owner``'s compression root)."""
    path = Path(path)
    if not path.exists():
        return session_id
    with open_persistence_reader(path) as persistence:
        lineage = persistence.lineage.lineage(session_id)
        root = lineage[0] if lineage else session_id
        return _local_owners(persistence).get(root, root)


def catalog_sessions(path, cwd=None):
    normalized = _normalize_cwd_for_compare(cwd) if cwd else None
    owners = read_local_owners(path)
    results = {}
    for sid, row in read_catalog_rows(path).items():
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
