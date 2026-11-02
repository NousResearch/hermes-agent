"""Queue-order writes on a session's admission FIFO: an edit keeps its place, a reorder moves rows.

A FIFO runs in ``seq`` order. Both writes only permute ``seq`` values the session already holds, so
the AUTOINCREMENT high-water mark never moves and no other session's order changes. ``seq`` is the
table's primary key: a swap parks one row at a negative value inside the same transaction.
"""
from hermes_state_runtime import RuntimeStoreError, _admission, _epoch, _row


def _swap_seq(conn, first, second):
    a, b = first['seq'], second['seq']
    conn.execute('UPDATE session_admissions SET seq=? WHERE seq=?', (-a, a))
    conn.execute('UPDATE session_admissions SET seq=? WHERE seq=?', (a, b))
    conn.execute('UPDATE session_admissions SET seq=? WHERE seq=?', (b, -a))


def take_queue_place(conn, admitted: dict, replaced_id: str):
    """Inside a replacement's admission write, after its INSERT: the new row takes the FIFO place of
    the row it retired (cancelled earlier in the same transaction), so an edit never sends a prompt
    to the back. Returns the admitted row as re-read."""
    old = _admission(conn, replaced_id)
    new = _admission(conn, admitted['admission_id'])
    if old['seq'] < new['seq']:
        _swap_seq(conn, old, new)
    return _row(_admission(conn, admitted['admission_id']))


def reorder_queued_input(db, *, epoch: int, session_id: str, admission_id: str, position: int,
                         movable) -> list[str]:
    """Move one queued admission to ``position`` (0-based) among the session's ``movable`` queued
    rows. Rows outside ``movable`` keep their places; the movable rows trade ``seq`` values among
    themselves. Refuses ``admission_started`` when the row left the queue, ``invalid_params`` for a
    row that is not movable. Returns the session's queued admission ids in their new run order."""
    if type(position) is not int or position < 0:
        raise RuntimeStoreError('invalid_params', field='position')
    movable = set(movable)

    def write(conn):
        _epoch(conn, epoch)
        row = _admission(conn, admission_id)
        if row['target_session_id'] != session_id:
            raise RuntimeStoreError('not_found')
        if row['status'] != 'queued':
            raise RuntimeStoreError('admission_started')
        if admission_id not in movable:
            raise RuntimeStoreError('unsupported_control')
        queued = conn.execute("SELECT admission_id, seq FROM session_admissions WHERE target_session_id=? "
                              "AND status='queued' ORDER BY seq", (session_id,)).fetchall()
        slots = [r for r in queued if r['admission_id'] in movable]
        order = [r['admission_id'] for r in slots if r['admission_id'] != admission_id]
        order.insert(min(position, len(order)), admission_id)
        seqs = [r['seq'] for r in slots]
        current = {r['admission_id']: r['seq'] for r in slots}
        if [current[a] for a in order] != seqs:
            for aid in order:  # park every moved row first: the targets are each other's keys
                conn.execute('UPDATE session_admissions SET seq=? WHERE admission_id=?', (-current[aid], aid))
            for aid, seq in zip(order, seqs):
                conn.execute('UPDATE session_admissions SET seq=? WHERE admission_id=?', (seq, aid))
            conn.execute('UPDATE sessions SET runtime_revision=runtime_revision+1 WHERE id=?', (session_id,))
        return [r['admission_id'] for r in conn.execute(
            "SELECT admission_id FROM session_admissions WHERE target_session_id=? AND status='queued' "
            "ORDER BY seq", (session_id,))]
    return db._execute_write(write)
