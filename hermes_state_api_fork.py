"""Create an API branch, copy its active transcript and end its parent atomically."""

import time

from hermes_state_lineage import guarded_discovered_session_write
from hermes_state_mutation_ownership import mutation_write_ids
from hermes_state_runtime import RuntimeStoreError
from hermes_state_titles import lineage_title_on_conn
from hermes_state_worker_lifecycle import insert_session_row_in_transaction


def fork_api_session(db, source_id, child_id, title):
    """Fence parent, new child and title owner through the one branch commit."""
    def desired_title(conn):
        if title is not None:
            return str(title)
        row = conn.execute('SELECT title FROM sessions WHERE id=?', (source_id,)).fetchone()
        return lineage_title_on_conn(conn, (row['title'] if row else None) or 'fork')

    def discover(conn):
        return {source_id, child_id} | mutation_write_ids(
            db, conn, child_id, 'rename', {'title': desired_title(conn)})

    def write(conn):
        source = conn.execute(
            'SELECT s.*,COALESCE(sp.prompt,s.system_prompt) AS resolved_prompt '
            'FROM sessions s LEFT JOIN system_prompts sp ON sp.hash=s.system_prompt_hash WHERE s.id=?',
            (source_id,)).fetchone()
        if source is None:
            raise RuntimeStoreError('not_found')
        if conn.execute('SELECT 1 FROM sessions WHERE id=?', (child_id,)).fetchone():
            raise RuntimeStoreError('session_exists')
        messages = [db._row_to_message_dict(row, warn_context='api fork', summary_flag=True)
                    for row in conn.execute(
                        'SELECT * FROM messages WHERE session_id=? AND active=1 ORDER BY id', (source_id,))]
        chosen_title = desired_title(conn)
        insert_session_row_in_transaction(db, conn, child_id, 'api_server',
            model=source['model'], system_prompt=source['resolved_prompt'],
            parent_session_id=source_id, model_config={'_branched_from': source_id})
        db._end_and_bump(conn,
            'UPDATE sessions SET ended_at=?,end_reason=? WHERE id=? AND ended_at IS NULL',
            (time.time(), 'branched', source_id), source_id, 'branched')
        db.replace_messages_in_transaction(conn, child_id, messages)
        db._set_session_title_in_transaction(conn, child_id, chosen_title, source=db.TITLE_SOURCE_USER)
        return child_id

    return guarded_discovered_session_write(db, (source_id, child_id), discover, write,
                                            patience_s=db._TRANSCRIPT_WRITE_PATIENCE_S)
