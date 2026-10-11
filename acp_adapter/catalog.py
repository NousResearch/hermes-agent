"""ACP historical catalog without storage initialization or an execution owner."""
import json
from pathlib import Path

from hermes_state import SessionDB
from acp_adapter.session import (
    _normalize_cwd_for_compare, _parse_model_config, _session_info, _updated_at_sort_key,
)


def read_catalog_rows(path):
    path = Path(path)
    if not path.exists():
        return {}
    with SessionDB(db_path=path, read_only=True) as db:
        return {str(row["id"]): dict(row) for row in db.list_sessions_rich(source="acp", limit=1000)}


def _local_owners(db):
    """Physical id -> retained logical owner, from each private local creation receipt's lineage.
    A canonical reset or compression advances the receipt's physical target but keeps policy/FIFO
    on the creation id. Legacy reset histories have no receipt and stay independent chats."""
    from hermes_state_local import POLICY_PREFIX
    owners = {}
    with db._read_ctx() as conn:
        rows = conn.execute('SELECT value FROM state_meta WHERE key GLOB ?', (POLICY_PREFIX + '*',)).fetchall()
    for (raw,) in rows:
        try:
            receipt = json.loads(raw)
        except (TypeError, ValueError):
            continue
        owner = receipt.get('session_id') if isinstance(receipt, dict) else None
        lineage = receipt.get('lineage') if isinstance(owner, str) else None
        for physical in lineage if isinstance(lineage, list) else ():
            if isinstance(physical, str):
                owners[physical] = owner
    return owners


def read_local_owners(path):
    path = Path(path)
    if not path.exists():
        return {}
    with SessionDB(db_path=path, read_only=True) as db:
        return _local_owners(db)


def logical_session_id(path, session_id):
    """The authority's logical id for an id an editor already holds: compression and a canonical
    reset advance the physical transcript, never the session identity (the local receipt's owner,
    else ``SessionAuthority.logical_owner``'s compression root)."""
    path = Path(path)
    if not path.exists():
        return session_id
    with SessionDB(db_path=path, read_only=True) as db:
        lineage = db.get_compression_lineage(session_id)
        root = lineage[0] if lineage else session_id
        return _local_owners(db).get(root, root)


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
