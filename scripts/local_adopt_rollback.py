#!/usr/bin/env python3
"""Undo a viewer adoption of a historical messaging row (local divergence, NousResearch/hermes-agent#134889).

``gateway/session_viewer_adoption.py`` binds such a row as a local session, rewriting its identity
columns (and its first turn reopens an ended row), after storing the originals in
``state_meta['local_adopt_preimage:<sid>']``. This restores those columns, end stamp included, and drops
what the adoption added: the local receipt, the legacy binding marker, the local route and the preimage.
Messages written after the adoption stay in the transcript.

Stop the gateway serving this ``state.db`` first: a live authority keeps the adopted session mounted
and its session store would write the local route back.

    python scripts/local_adopt_rollback.py --db <profile home>/state.db --sid <session id> [--dry-run]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hermes_state_local import POLICY_PREFIX
from hermes_state_local_migration import IDENTITY_COLUMNS, LEGACY_PREFIX, PREIMAGE_PREFIX


def plan(conn, sid):
    """``(column restore, state_meta keys, local route)`` for *sid*; SystemExit when not adopted."""
    preimage = conn.execute('SELECT value FROM state_meta WHERE key=?', (PREIMAGE_PREFIX + sid,)).fetchone()
    if preimage is None:
        raise SystemExit(f'{sid}: no adoption preimage, nothing to roll back')
    receipt = conn.execute('SELECT value FROM state_meta WHERE key=?', (POLICY_PREFIX + sid,)).fetchone()
    route = json.loads(receipt[0])['route'] if receipt is not None else None
    keys = [PREIMAGE_PREFIX + sid, POLICY_PREFIX + sid, LEGACY_PREFIX + sid]
    saved = json.loads(preimage[0])['columns']
    return {column: saved[column] for column in IDENTITY_COLUMNS}, keys, route


def rollback(db_path, sid, *, dry_run=False):
    """Restore *sid*'s pre-adoption identity in one transaction; returns the plan applied (or not)."""
    conn = sqlite3.connect(db_path)
    try:
        columns, keys, route = plan(conn, sid)
        if dry_run:
            return {'sid': sid, 'restore': columns, 'delete_state_meta': keys, 'delete_route': route}
        with conn:
            assignments = ', '.join(f'{column}=?' for column in columns)
            conn.execute(f'UPDATE sessions SET {assignments} WHERE id=?', (*columns.values(), sid))
            conn.executemany('DELETE FROM state_meta WHERE key=?', [(key,) for key in keys])
            if route is not None:
                conn.execute('DELETE FROM gateway_routing WHERE session_key=?', (route,))
        return {'sid': sid, 'restore': columns, 'delete_state_meta': keys, 'delete_route': route}
    finally:
        conn.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--db', required=True, help='the profile state.db holding the adopted row')
    parser.add_argument('--sid', required=True, help='the adopted session id')
    parser.add_argument('--dry-run', action='store_true', help='print the plan, write nothing')
    args = parser.parse_args(argv)
    result = rollback(args.db, args.sid, dry_run=args.dry_run)
    print(json.dumps({'dry_run': args.dry_run, **result}, indent=2))


if __name__ == '__main__':
    main()
