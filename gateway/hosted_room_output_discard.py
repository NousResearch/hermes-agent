"""Exact Output retirement on an already-held owner connection; never initialize.

The caller commits per-blob intent before cleanup in a second transaction.
Artifact rows are not completion evidence: pruning or an older writer can lose
those rows. The retained Run owns the exact names until a synced cleanup commits.
"""
import json
import os
import re
import time

from gateway.hosted_room_artifacts import RoomArtifactError


class OutputCleanupUnavailable(RuntimeError):
    """Retirement exists, but its physical completion cannot yet be proven."""


def require_retired(conn, scope):
    fence = conn.execute('SELECT retired_generation FROM hosted_room_output_generation_fences '
                         'WHERE lineage_identity=?', (scope.lineage_json,)).fetchone()
    if fence is None or fence[0] < scope.execution_generation:
        raise OutputCleanupUnavailable('Output retirement fence is unavailable')


def require_cleanup_blobs(blobs, items):
    if (type(blobs) is not list or len(blobs) != len(items) or not blobs
            or any(type(blob) is not dict or set(blob) != {'artifact_id', 'blob_name'}
                   or type(blob['blob_name']) is not str
                   or not re.fullmatch(r'blob_[0-9a-f]{32}', blob['blob_name']) for blob in blobs)
            or [blob['artifact_id'] for blob in blobs] != [item['artifact_id'] for item in items]
            or len({blob['blob_name'] for blob in blobs}) != len(blobs)):
        raise OutputCleanupUnavailable('Output per-blob cleanup evidence is unavailable')


def require_record(record, commitment, items):
    from tui_gateway.hosted_room_peer_artifacts import require_discard_receipt
    from tui_gateway.hosted_room_peer_http import PeerRunsHTTPError
    if (type(record) is not dict or set(record) != {'commitment', 'receipt', 'state', 'blobs'}
            or record['state'] not in ('pending', 'completed')):
        # Old success-shaped records cannot distinguish lost rows from removed
        # bytes. Never upgrade those ambiguous receipts by guessing.
        raise OutputCleanupUnavailable('Output completion evidence is unavailable')
    if record['commitment'] != commitment:
        raise RoomArtifactError('Output retirement commitment changed')
    try:
        require_discard_receipt(record['receipt'])
    except PeerRunsHTTPError as exc:
        raise OutputCleanupUnavailable('Output retirement count is unavailable') from exc
    if record['receipt']['removed'] != len(items):
        raise OutputCleanupUnavailable('Output retirement count changed')
    if record['state'] == 'pending':
        require_cleanup_blobs(record['blobs'], items)
    elif record['blobs'] != []:
        raise OutputCleanupUnavailable('Output completion evidence changed')


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
    blobs = [dict(artifact_id=row['artifact_id'], blob_name=row['blob_name']) for row in rows]
    require_cleanup_blobs(blobs, items)
    authorize(conn)
    outbox._retire_generation(conn, scope)
    now = time.time()
    conn.execute('UPDATE hosted_room_output_artifacts SET cleanup_required_at=?, acknowledged_at=? '
                 'WHERE scope_key=?', (now, now, scope.key))
    return blobs


def unlink_blob_names(outbox, names):
    """Confirm exact directory-relative removal, including prior missing blobs."""
    if any(type(name) is not str or not re.fullmatch(r'blob_[0-9a-f]{32}', name) for name in names):
        raise OutputCleanupUnavailable('Output blob identity changed')
    directory = os.open(outbox.blob_root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for name in names:
            try:
                os.unlink(name, dir_fd=directory)
            except FileNotFoundError:
                pass  # Prior interrupted cleanup is confirmed by the fsync below.
        os.fsync(directory)
    finally:
        os.close(directory)


def cleanup_exact(outbox, conn, scope, items, blobs, *, authorize):
    """Use retained per-blob authority, never the cardinality of surviving rows."""
    if not conn.in_transaction:
        raise RoomArtifactError('Output cleanup requires its owner transaction')
    require_retired(conn, scope)
    require_cleanup_blobs(blobs, items)
    names = {blob['artifact_id']: blob['blob_name'] for blob in blobs}
    expected = {item['artifact_id']: item for item in items}
    placeholders = ','.join('?' for _ in blobs)
    rows = conn.execute('SELECT * FROM hosted_room_output_artifacts WHERE scope_key=? '
                        f'OR blob_name IN ({placeholders})', (scope.key, *names.values())).fetchall()
    if any(row['scope_key'] != scope.key or json.loads(row['scope_json']) != scope.as_mapping()
           or row['cleanup_required_at'] is None or row['acknowledged_at'] is None
           or row['ack_message_event_id'] is not None
           or outbox._manifest(row) != expected.get(row['artifact_id'])
           or row['blob_name'] != names.get(row['artifact_id']) for row in rows):
        raise OutputCleanupUnavailable('Output cleanup commitment changed')
    authorize(conn)
    # Missing rows do not remove names from this durable exact obligation.
    unlink_blob_names(outbox, list(names.values()))
    conn.execute('DELETE FROM hosted_room_output_artifacts WHERE scope_key=?', (scope.key,))
