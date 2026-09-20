"""Exact Output retirement on an already-held owner connection; never initialize.

The caller commits intent before invoking cleanup in a second transaction.
This is not input spool disposal and does not acknowledge a published message.
"""
import json
import os
import re
import time

from gateway.hosted_room_artifacts import RoomArtifactError


def require_retired(conn, scope):
    fence = conn.execute('SELECT retired_generation FROM hosted_room_output_generation_fences '
                         'WHERE lineage_identity=?', (scope.lineage_json,)).fetchone()
    if fence is None or fence[0] < scope.execution_generation:
        raise RoomArtifactError('Output retirement fence is unavailable')


def retire_exact(outbox, conn, scope, items, *, authorize):
    """Stage one immutable manifest's fence/intent; caller owns the commit."""
    if not conn.in_transaction:
        raise RoomArtifactError('Output retirement requires its owner transaction')
    rows = conn.execute('SELECT * FROM hosted_room_output_artifacts WHERE scope_key=? '
                        'ORDER BY created_at, artifact_id', (scope.key,)).fetchall()
    if (not rows or [outbox._manifest(row) for row in rows] != items
            or any(json.loads(row['scope_json']) != scope.as_mapping() or row['acknowledged_at'] is not None
                   or row['cleanup_required_at'] is not None for row in rows)):
        raise RoomArtifactError('Output retirement manifest changed')
    authorize(conn)
    outbox._retire_generation(conn, scope)
    # Retired bytes are no longer readable, even if unlink fails. Keep each
    # blob's cleanup obligation until the physical operation can be retried.
    now = time.time()
    conn.execute('UPDATE hosted_room_output_artifacts SET cleanup_required_at=?, acknowledged_at=? '
                 'WHERE scope_key=?', (now, now, scope.key))
    return len(rows)


def cleanup_exact(outbox, conn, scope, items, *, authorize):
    """Remove only already-retired blobs; failure preserves committed intent."""
    if not conn.in_transaction:
        raise RoomArtifactError('Output cleanup requires its owner transaction')
    require_retired(conn, scope)
    rows = conn.execute('SELECT * FROM hosted_room_output_artifacts WHERE scope_key=?', (scope.key,)).fetchall()
    expected = {item['artifact_id']: item for item in items}
    if any(json.loads(row['scope_json']) != scope.as_mapping() or row['cleanup_required_at'] is None
           or row['acknowledged_at'] is None or row['ack_message_event_id'] is not None
           or outbox._manifest(row) != expected.get(row['artifact_id'])
           or not re.fullmatch(r'blob_[0-9a-f]{32}', row['blob_name']) for row in rows):
        raise RoomArtifactError('Output cleanup commitment changed')
    # Directory-relative unlink cannot follow a replaced blob directory or a
    # blob symlink to foreign bytes. Never discover targets by scanning disk.
    authorize(conn)
    directory = os.open(outbox.blob_root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for row in rows:
            try:
                os.unlink(row['blob_name'], dir_fd=directory)
            except FileNotFoundError:
                pass  # A prior interrupted cleanup may already have unlinked it.
        os.fsync(directory)
    finally:
        os.close(directory)
    conn.execute('DELETE FROM hosted_room_output_artifacts WHERE scope_key=?', (scope.key,))
