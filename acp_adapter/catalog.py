"""ACP historical catalog without storage initialization or an execution owner."""
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


def catalog_sessions(path, cwd=None):
    normalized = _normalize_cwd_for_compare(cwd) if cwd else None
    results = []
    for sid, row in read_catalog_rows(path).items():
        count = int(row.get("message_count") or 0)
        session_cwd = row.get('cwd') or _parse_model_config(row.get("model_config")).get("cwd", ".")
        if count <= 0 or (normalized and _normalize_cwd_for_compare(session_cwd) != normalized):
            continue
        results.append(_session_info(sid, session_cwd, row.get("model") or "", count, row.get("title"),
                       row.get("preview"), row.get("last_active") or row.get("started_at")))
    return sorted(results, key=lambda row: _updated_at_sort_key(row.get("updated_at")), reverse=True)
