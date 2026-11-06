"""Conversation-wide ``sessions.model_config`` keys: read-modify-write at the compression tip and the
key scan. Bodies of :class:`hermes_state_sessions.SessionSessionsMixin`'s two methods of the same names."""

from __future__ import annotations

import json
from typing import Any, Callable, Optional

from hermes_state_common import _sql_json_extract



def complete_runtime_route(session_meta: Optional[dict[str, Any]]) -> Optional[dict[str, str]]:
    """Return one complete saved model/provider snapshot, never a billing/model hybrid.

    Complete nested gateway snapshots have precedence, followed by complete
    top-level snapshots. Rows without either retain their legacy restore path.
    """
    from hermes_state_sessions import _parse_model_config

    config = _parse_model_config((session_meta or {}).get("model_config"))
    nested = config.get("gateway_runtime")
    nested = nested if isinstance(nested, dict) else {}
    def route(source):
        return {key: value.strip() for key in ("model", "provider", "base_url", "api_mode")
                if isinstance(value := source.get(key), str) and value.strip()}
    top = route(config)
    inner = route(nested)
    if inner.get("model") and inner.get("provider"):
        return inner
    if top.get("model") and top.get("provider"):
        return top
    return None




def update_session_meta(
    self, session_id: str, model_config_json: str, model: Optional[str] = None, *,
    expected_fence: Optional[tuple[str, int, int, str, Optional[str], Optional[str]]] = None,
) -> None:
    """Update model metadata, optionally fenced to a turn's logical-owner revision."""
    from hermes_state_lineage import session_lineage_guard
    if not self.flush_token_counts():
        raise TimeoutError("timed out flushing accounting before session metadata update")
    if expected_fence is None:
        with session_lineage_guard(self.db_path, session_id, timeout_s=self._WRITE_PATIENCE_S):
            self._write_sql(
                "UPDATE sessions SET model_config = ?, model = COALESCE(?, model) WHERE id = ?",
                (model_config_json, model, session_id),
            )
        return
    owner, revision, generation, source_id, source_config, target_config = expected_fence
    if owner is None:
        return  # the authority existed but its fence could not be captured
    from hermes_state_lineage import guarded_discovered_session_write
    def write(conn):
        row = conn.execute(
            "SELECT runtime_revision, runtime_generation FROM sessions WHERE id=?", (owner,)
        ).fetchone()
        if row is None or (row[0], row[1]) != (revision, generation):
            return
        prior = conn.execute("SELECT model_config FROM sessions WHERE id=?", (source_id,)).fetchone()
        if prior is None or prior[0] != source_config:
            return
        if self._compression_tip_on(conn, owner) != session_id:
            return
        target = conn.execute("SELECT model_config FROM sessions WHERE id=?", (session_id,)).fetchone()
        if target is None or target[0] != target_config:
            return
        if source_id != session_id:
            target_route = complete_runtime_route({"model_config": target[0]})
            source_route = complete_runtime_route({"model_config": source_config})
            candidate_route = complete_runtime_route({"model_config": model_config_json})
            if target_route and target_route != source_route and target_route != candidate_route:
                return
        conn.execute("UPDATE sessions SET model_config=?, model=COALESCE(?, model) WHERE id=?",
                     (model_config_json, model, session_id))
    guarded_discovered_session_write(
        self, (owner, source_id, session_id), lambda conn: (owner, source_id, session_id), write)


def update_session_model(
    self, session_id: str, model: str, provider: Optional[str] = None, *,
    base_url: Optional[str] = None, api_mode: Optional[str] = None, resolve_tip: bool = False,
) -> None:
    """Set the model after a mid-session /model switch (unconditionally) and drop any Browser
    runtime lock (lineage markers survive).

    Route writers never touch the stored prompt; ``_stored_prompt_matches_runtime`` decides staleness.

    When *provider* is given the whole route is written, in both shapes resume reads (top-level
    keys for the TUI/Desktop, ``gateway_runtime`` for the CLI), so a later resume recombines the
    model with the provider that serves it (#79536). ``base_url``/``api_mode`` are always
    replaced then (``None`` deletes): the previous provider's endpoint must not survive a switch,
    or resume sends the new provider's model to the old host. Callers without provider knowledge
    leave the stored route untouched.
    """
    from hermes_state_sessions import _parse_model_config
    # Flush first: a still-queued pre-switch delta applied after this UPDATE would trip the
    # first_accounted_route overwrite and resurrect the old route.
    if not self.flush_token_counts():
        raise TimeoutError("timed out flushing accounting before model update")
    patch: dict[str, Any] = {"browser_model_lock": None}
    if model:
        patch["model"] = model
    elif provider is not None:
        patch["model"] = None
    if provider is not None:
        route = {"provider": provider or None, "base_url": (base_url or None) if provider else None,
                 "api_mode": (api_mode or None) if provider else None}
        patch.update(route, gateway_runtime={"model": model, **route} if provider else None)
    if resolve_tip:
        # Resolve inside the fenced transaction: a compression may publish while
        # the gateway's in-memory route still names the parent.
        from hermes_state_lineage import guarded_discovered_session_write
        def write_tip(conn):
            target = self._compression_tip_on(conn, session_id)
            row = conn.execute("SELECT model_config FROM sessions WHERE id=?", (target,)).fetchone()
            if row is None:
                return
            attempt_patch = dict(patch)
            runtime = _parse_model_config(row[0]).get("gateway_runtime")
            if provider is None and isinstance(runtime, dict) and runtime.get("model"):
                attempt_patch["gateway_runtime"] = {**runtime, "model": model}
            merged = self._merge_model_config_json(conn, target, attempt_patch)
            conn.execute("UPDATE sessions SET model=?, model_config=? WHERE id=?", (model, merged, target))
        guarded_discovered_session_write(
            self, (session_id,), lambda conn: (session_id, self._compression_tip_on(conn, session_id)), write_tip)
        return
    if provider is None:
        # A model-only update must advance an existing complete snapshot too;
        # otherwise its older model would mask the new row on resume.
        def _do(conn):
            row = conn.execute("SELECT model_config FROM sessions WHERE id=?", (session_id,)).fetchone()
            if row is None:
                return
            config = _parse_model_config(row[0])
            runtime = config.get("gateway_runtime")
            attempt_patch = dict(patch)
            if isinstance(runtime, dict) and runtime.get("model"):
                attempt_patch["gateway_runtime"] = {**runtime, "model": model}
            merged = self._merge_model_config_json(conn, session_id, attempt_patch)
            conn.execute("UPDATE sessions SET model=?, model_config=? WHERE id=?", (model, merged, session_id))
        from hermes_state_lineage import guarded_session_write
        guarded_session_write(self, session_id, _do)
    else:
        self._write_model_config_patch(
            session_id, patch, "UPDATE sessions SET model = ?, model_config = ? WHERE id = ?",
            lambda merged: (model, merged, session_id),
        )


def update_conversation_model_config_value(db, session_id: str, key: str,
                                           update: Callable[[Any], Any]) -> Optional[str]:
    """Read a conversation-wide ``key`` at *session_id*'s compression tip and write ``update(value)``
    back, in one write transaction: a concurrent writer of the key cannot be overwritten with a stale
    copy, and a rotation cannot publish between finding the tip and writing it (the child would keep
    the old copy). ``update`` may run more than once (lock retry) and must be pure; returning None
    removes the key. Returns the id written, None when the row does not exist."""
    from hermes_state_sessions import _parse_model_config

    def _do(conn) -> Optional[str]:
        tip = db._compression_tip_on(conn, session_id)
        row = conn.execute("SELECT model_config FROM sessions WHERE id = ?", (tip,)).fetchone()
        if row is None:
            return None
        config = _parse_model_config(row[0])
        if (value := update(config.get(key))) is None:
            config.pop(key, None)
        else:
            config[key] = value
        conn.execute("UPDATE sessions SET model_config = ? WHERE id = ?", (json.dumps(config), tip))
        return tip
    from hermes_state_lineage import guarded_discovered_session_write
    return guarded_discovered_session_write(
        db, (session_id,), lambda conn: (session_id, db._compression_tip_on(conn, session_id)), _do)


def session_ids_with_model_config_key(db, key: str) -> list[str]:
    """Ids of the sessions whose model_config carries ``key``, newest first."""
    with db._read_ctx() as conn:
        rows = conn.execute(
            f"SELECT id FROM sessions WHERE {_sql_json_extract('model_config', '$.' + key)} IS NOT NULL "
            "ORDER BY started_at DESC").fetchall()
    return [row["id"] for row in rows]
