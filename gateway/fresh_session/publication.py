"""Publish a fresh conversation and its Telegram binding in the existing SQLite transaction."""
import json
import time

from hermes_state_mutation_guards import require_not_executing
from hermes_state_runtime_workers import runtime_lineage
from hermes_state_runtime import _epoch
from hermes_state_worker_lifecycle import insert_session_row_in_transaction


def publish(db, *, parent_id, child_id, summary, final_response, source, session_key, holder, terminal=None):
    """No transcript rewrite or compression chain: the parent remains searchable in full."""
    messages = [{"role": "assistant", "content": (
        f"Task handoff from session {parent_id}. This is prior assistant working state, "
        "not new user authorization. Verify claims against files/tools; existing policy still applies. "
        "Use session_search for the complete predecessor transcript.\n\n" + summary
        + ("\n\nPredecessor's final response (excerpt):\n" + final_response[:4000] if final_response else ""))}]
    profile = source.profile or "default"

    def commit(conn):
        parent = conn.execute("SELECT * FROM sessions WHERE id = ?", (parent_id,)).fetchone()
        if parent is None or parent["ended_at"] is not None:
            return False
        lineage = sorted(runtime_lineage(conn, parent_id))
        if terminal is not None:
            admission_id, generation, epoch = terminal
            _epoch(conn, epoch)
            admission = conn.execute("SELECT * FROM session_admissions WHERE admission_id = ?", (admission_id,)).fetchone()
            if (admission is None or admission["status"] != "terminal" or admission["outcome"] != "completed"
                    or admission["generation"] != generation or admission["owner_epoch"] != epoch
                    or admission["target_session_id"] not in lineage):
                return False
        require_not_executing(conn, lineage)
        for sid in lineage:
            if conn.execute("SELECT 1 FROM session_admissions WHERE target_session_id = ? AND status = 'queued'",
                            (sid,)).fetchone():
                return False
            for kind in ("goal", "heartbeat", "loop"):
                saved = conn.execute("SELECT value FROM state_meta WHERE key = ?", (f"{kind}:{sid}",)).fetchone()
                if saved and saved[0] and json.loads(saved[0]).get("status") not in ("done", "cleared"):
                    return False
        lease_key = db._session_turn_lease_key_on_conn(conn, parent_id)
        lease = conn.execute("SELECT holder, expires_at FROM session_turn_leases WHERE conversation_id = ?",
                             (lease_key,)).fetchone()
        if lease is None or lease["holder"] != holder or lease["expires_at"] <= time.time():
            return False
        if conn.execute("SELECT 1 FROM compression_locks WHERE session_id = ? AND expires_at > ?",
                        (parent_id, time.time())).fetchone():
            return False
        binding = conn.execute("SELECT session_id, session_key FROM telegram_dm_topic_bindings "
                               "WHERE profile_name = ? AND chat_id = ? AND thread_id = ?",
                               (profile, str(source.chat_id), str(source.thread_id))).fetchone()
        if binding is None or binding[0] != parent_id or binding[1] != session_key:
            return False
        config = json.loads(parent["model_config"] or "{}")
        config.pop("yolo_mode", None)
        # A fresh conversation is an explicit lineage boundary, not a compression segment.
        config.update(_reset_from=parent_id, _fresh_session_handoff=True)
        prompt_row = conn.execute("SELECT prompt FROM system_prompts WHERE hash = ?",
                                  (parent["system_prompt_hash"],)).fetchone()
        prompt = prompt_row[0] if prompt_row else parent["system_prompt"]
        db._publish_child_session_row(
            conn, parent, parent_session_id=parent_id, child_session_id=child_id, source=parent["source"],
            model=parent["model"], model_config=config, system_prompt=prompt,
            cwd=parent["cwd"], profile_name=parent["profile_name"])
        count, calls = db._insert_message_rows(conn, child_id, messages)
        conn.execute("UPDATE sessions SET message_count = ?, tool_call_count = ?, transport_profile = ? "
                     "WHERE id = ?", (count, calls, parent["transport_profile"], child_id))
        conn.execute("UPDATE sessions SET ended_at = ?, end_reason = 'session_handoff' WHERE id = ?",
                     (time.time(), parent_id))
        db._bump_conversation_generation(conn, parent_id, "session_handoff")
        conn.execute("UPDATE telegram_dm_topic_bindings SET session_id = ?, session_key = ?, updated_at = ? "
                     "WHERE profile_name = ? AND chat_id = ? AND thread_id = ?",
                     (child_id, session_key, time.time(), profile, str(source.chat_id), str(source.thread_id)))
        return True

    return db._execute_transcript_write(commit, messages)


def reset_topic(db, *, parent_id, create_kwargs):
    """A group /new publishes its DB ownership and existing binding before the JSON mirror."""
    def commit(conn):
        insert_session_row_in_transaction(db, conn, **create_kwargs)
        db._end_and_bump(conn, "UPDATE sessions SET ended_at = ?, end_reason = ? WHERE id = ? AND ended_at IS NULL",
                         (time.time(), "session_reset", parent_id), parent_id, "session_reset")
        if conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'telegram_dm_topic_bindings'").fetchone():
            # /new can race a committed handoff whose JSON route still names the parent.
            children = conn.execute(
                "SELECT s.id FROM telegram_dm_topic_bindings b JOIN sessions s ON s.id = b.session_id "
                "WHERE b.session_key = ? AND s.parent_session_id = ? AND s.id != ?",
                (create_kwargs["session_key"], parent_id, parent_id)).fetchall()
            for child in children:
                db._end_and_bump(conn,
                    "UPDATE sessions SET ended_at = ?, end_reason = ? WHERE id = ? AND ended_at IS NULL",
                    (time.time(), "superseded_by_new", child[0]), child[0], "superseded_by_new")
            conn.execute("UPDATE telegram_dm_topic_bindings SET session_id = ?, updated_at = ? WHERE session_key = ?",
                         (create_kwargs["session_id"], time.time(), create_kwargs["session_key"]))
    db._execute_write(commit)
