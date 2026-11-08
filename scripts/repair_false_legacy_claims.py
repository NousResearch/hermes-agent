#!/usr/bin/env python3
"""Replace false legacy claims on receipts with the profile home their id was minted under.

A receipt only carries ``legacy_session_id`` when the store adopted a pre-gateway row, and
``bind_legacy_target`` writes the matching ``gateway.local_legacy.v1:`` row in the same
transaction. A receipt that pairs a *gateway-minted* id (``local-<sha256>``) with a
self-referencing claim cannot have been written by the store: the id derives from
``local_identity(profile_id, principal_id, request_id)``, so it was minted by a create, whose row
is routed and which the bind path refuses. That shape comes from a hand rewrite of ``state_meta``
after a store was moved between profiles: the ids no longer derive from the new home, so the
rewrite stamped a claim to get past the identity check, and the claim then refuses every resume at
``legacy_lineage_root`` once the binding row is missing (or hides a lie once it is added).

This tool replaces the lie with the truth: it recovers the home each id was minted under (the id
is a hash of that home, the stored principal and the stored request id, so the home is verifiable,
not guessed), records it as ``adopted_profile_id``, and removes the false claim and the binding
row that claim implied. ``validate_local_lineage`` accepts a receipt whose id derives from its
``adopted_profile_id``, so the conversation becomes resumable again with no claim at all.

Dry run by default; it only reads. ``--apply`` is one transaction, re-validates every rewritten
receipt with the shipped validator before committing, and refuses to touch anything it cannot
prove.

    python scripts/repair_false_legacy_claims.py --db profiles/nico/state.db
    python scripts/repair_false_legacy_claims.py --db ... --apply --backup-dir ... --gateway-stopped
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

POLICY_PREFIX = 'gateway.local_policy.v1:'
LEGACY_PREFIX = 'gateway.local_legacy.v1:'


def _minted(session_id: str) -> bool:
    return isinstance(session_id, str) and session_id.startswith('local-')


def _candidate_homes(receipt: dict, db_path: Path) -> list[str]:
    """Homes the id could have been minted under: what the receipt remembers, then its siblings."""
    homes: list[str] = []
    ref = receipt['policy'].get('config_secret_ref') or ''
    if ref.startswith('profile-config-v1:'):
        try:
            home = json.loads(ref[len('profile-config-v1:'):]).get('home')
        except (TypeError, ValueError):
            home = None
        if isinstance(home, str) and home:
            homes.append(home.rstrip('\\/'))
    profiles = None
    for parent in db_path.resolve().parents:
        if parent.name == 'profiles':
            profiles = parent
            break
        if (parent / 'profiles').is_dir():
            profiles = parent / 'profiles'
            break
    if profiles is not None and profiles.is_dir():
        for child in sorted(profiles.iterdir()):
            if child.is_dir():
                homes.append(str(child))
    return list(dict.fromkeys(homes))


def _minting_home(receipt: dict, db_path: Path):
    """The home ``local_identity(home, principal, request)`` reproduces this id, else None."""
    from hermes_state_keys import local_identity
    for home in _candidate_homes(receipt, db_path):
        if local_identity(home, receipt['principal_id'], receipt['request_id']) == receipt['session_id']:
            return home
    return None


def survey(conn, db_path: Path) -> tuple[list[tuple[str, dict]], list[tuple[str, str]]]:
    """``(repairable, unrepairable)`` false-claim receipts: ``(session_id, receipt)`` and why."""
    repairable, unrepairable = [], []
    rows = conn.execute('SELECT key,value FROM state_meta WHERE key GLOB ?', (POLICY_PREFIX + '*',)).fetchall()
    for key, value in rows:
        try:
            receipt = json.loads(value)
        except (TypeError, ValueError):
            continue
        sid = key[len(POLICY_PREFIX):]
        if not isinstance(receipt, dict) or receipt.get('legacy_session_id') != sid:
            continue
        if not _minted(sid):
            continue  # a real adopted pre-gateway row: its id is a timestamp id
        home = _minting_home(receipt, db_path)
        if home is None:
            unrepairable.append((sid, 'no profile home reproduces this id from principal and request_id'))
        else:
            repairable.append((sid, receipt | {'_minted_under': home}))
    return repairable, unrepairable


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--db', required=True, help='state.db to inspect (read-only unless --apply)')
    parser.add_argument('--apply', action='store_true', help='write the repair (default: dry run)')
    parser.add_argument('--backup-dir', help='directory the pre-write copy was made in (recorded in the report)')
    parser.add_argument('--gateway-stopped', action='store_true',
                        help='confirm nothing else is writing this store (required with --apply)')
    args = parser.parse_args(argv)
    db_path = Path(args.db)
    if not db_path.is_file():
        print(f'no such store: {db_path}', file=sys.stderr)
        return 2
    if args.apply and not (args.backup_dir and args.gateway_stopped):
        print('--apply needs --backup-dir and --gateway-stopped', file=sys.stderr)
        return 2
    mode = 'rw' if args.apply else 'ro'
    uri = f'file:{db_path.as_posix()}?mode={mode}'
    conn = sqlite3.connect(uri, uri=True, timeout=60)
    try:
        from hermes_state_local_lineage import validate_local_lineage
        repairable, unrepairable = survey(conn, db_path)
        print(f'false legacy claims: {len(repairable) + len(unrepairable)} '
              f'({len(repairable)} repairable, {len(unrepairable)} not)')
        for sid, why in unrepairable:
            print(f'  UNREPAIRABLE {sid}: {why}')
        for sid, receipt in repairable[:5]:
            print(f'  {sid[:24]} minted under {receipt["_minted_under"]}')
        if not args.apply:
            print('dry run: nothing written')
            return 0 if not unrepairable else 1
        from hermes_state_runtime import RuntimeStoreError, _json
        repaired = []

        def write(c):
            for sid, receipt in repairable:
                stored = c.execute('SELECT value FROM state_meta WHERE key=?', (POLICY_PREFIX + sid,)).fetchone()
                current = json.loads(stored[0])
                if current.get('legacy_session_id') != sid:
                    raise SystemExit(f'{sid} changed under us: no longer a false claim')
                updated = dict(current, adopted_profile_id=receipt['_minted_under'])
                updated.pop('legacy_session_id', None)
                try:
                    validate_local_lineage(c, updated)
                except (RuntimeStoreError, KeyError, ValueError) as exc:
                    raise SystemExit(f'{sid} would not validate after the repair: {exc}')
                c.execute('UPDATE state_meta SET value=? WHERE key=?', (_json(updated), POLICY_PREFIX + sid))
                binding = c.execute('SELECT value FROM state_meta WHERE key=?', (LEGACY_PREFIX + sid,)).fetchone()
                if binding is not None and json.loads(binding[0]).get('legacy_session_id') == sid:
                    c.execute('DELETE FROM state_meta WHERE key=?', (LEGACY_PREFIX + sid,))
                    repaired.append(sid)
            return repaired
        from hermes_state import SessionDB
        db = SessionDB(db_path=db_path)
        try:
            repaired = db._execute_write(write)
        finally:
            db.close()
        print(f'repaired {len(repaired)} receipt(s); backup recorded at {args.backup_dir}')
        return 0
    finally:
        conn.close()


if __name__ == '__main__':
    raise SystemExit(main())
