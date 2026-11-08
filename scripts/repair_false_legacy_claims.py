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
prove. It takes the backup itself into ``--backup-dir`` and verifies it with ``PRAGMA
quick_check`` before the first write, so the backup path it reports is a file it wrote and
checked, never a path it was merely handed.

    python scripts/repair_false_legacy_claims.py --db profiles/nico/state.db
    python scripts/repair_false_legacy_claims.py --db ... --apply --backup-dir ... --gateway-stopped
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
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


def _binding_state(conn, sid: str) -> str:
    """``'remove'`` (the binding row names this session), ``'absent'``, or ``'foreign'``."""
    row = conn.execute('SELECT value FROM state_meta WHERE key=?', (LEGACY_PREFIX + sid,)).fetchone()
    if row is None:
        return 'absent'
    try:
        named = json.loads(row[0]).get('legacy_session_id')
    except (TypeError, ValueError, AttributeError):
        return 'foreign'
    return 'remove' if named == sid else 'foreign'


def survey(conn, db_path: Path) -> tuple[list[tuple[str, dict]], list[tuple[str, str]]]:
    """``(repairable, unrepairable)`` false-claim receipts: ``(session_id, receipt)`` and why.

    A receipt is repairable when its id is gateway-minted, some profile home reproduces it, and
    its binding row either names this session or is absent. A binding row that names something
    else is reported, not deleted: this tool does not touch a row it cannot tie to the claim it
    is repairing.
    """
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
            continue
        binding = _binding_state(conn, sid)
        if binding == 'foreign':
            unrepairable.append((sid, 'a legacy binding row exists but does not name this session; left untouched'))
            continue
        repairable.append((sid, receipt | {'_minted_under': home, '_binding': binding}))
    return repairable, unrepairable


def backup_store(conn, db_path: Path, backup_dir: Path) -> Path:
    """A verified single-file copy of the store, written before anything is repaired."""
    stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    target = backup_dir / f'{db_path.stem}-{stamp}-{os.getpid()}.db'
    dst = sqlite3.connect(target)
    try:
        conn.backup(dst)
        check = dst.execute('PRAGMA quick_check').fetchall()
    finally:
        dst.close()
    if check != [('ok',)]:
        raise RuntimeError(f'backup failed quick_check: {check}')
    if not target.is_file() or target.stat().st_size == 0:
        raise RuntimeError(f'backup was not written: {target}')
    return target


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--db', required=True, help='state.db to inspect (read-only unless --apply)')
    parser.add_argument('--apply', action='store_true', help='write the repair (default: dry run)')
    parser.add_argument('--backup-dir',
                        help='existing directory this tool writes the pre-write copy into (required with --apply)')
    parser.add_argument('--gateway-stopped', action='store_true',
                        help='confirm nothing else is writing this store (required with --apply)')
    args = parser.parse_args(argv)
    db_path = Path(args.db)
    if not db_path.is_file():
        print(f'no such store: {db_path}', file=sys.stderr)
        return 2
    backup_dir = None
    if args.apply:
        if not args.gateway_stopped:
            print('--apply needs --gateway-stopped', file=sys.stderr)
            return 2
        if not args.backup_dir:
            print('--apply needs --backup-dir: the pre-write copy is written there', file=sys.stderr)
            return 2
        backup_dir = Path(args.backup_dir)
        if not backup_dir.is_dir():
            print(f'--apply needs --backup-dir to be an existing directory: {backup_dir}', file=sys.stderr)
            return 2
    mode = 'rw' if args.apply else 'ro'
    uri = f'file:{db_path.as_posix()}?mode={mode}'
    conn = sqlite3.connect(uri, uri=True, timeout=60)
    try:
        from hermes_state_local_lineage import validate_local_lineage
        repairable, unrepairable = survey(conn, db_path)
        print(f'false legacy claims: {len(repairable) + len(unrepairable)} '
              f'({len(repairable)} repairable, {len(unrepairable)} not)')
        print(f'  bindings: {sum(1 for _sid, r in repairable if r["_binding"] == "remove")} to remove, '
              f'{sum(1 for _sid, r in repairable if r["_binding"] == "absent")} already absent')
        for sid, why in unrepairable:
            print(f'  UNREPAIRABLE {sid}: {why}')
        for sid, receipt in repairable[:5]:
            print(f'  {sid[:24]} minted under {receipt["_minted_under"]}')
        if not args.apply:
            print('dry run: nothing written')
            return 0 if not unrepairable else 1
        if unrepairable:
            # A partial repair is not the repair: leave every receipt untouched and let the
            # operator look at what could not be proved before anything in the store changes.
            print('refusing to apply a partial repair; nothing written', file=sys.stderr)
            return 1
        assert backup_dir is not None  # set above whenever --apply is allowed through
        try:
            backup_path = backup_store(conn, db_path, backup_dir)
        except (OSError, sqlite3.Error, RuntimeError) as exc:
            print(f'backup failed, nothing written: {exc}', file=sys.stderr)
            return 2
        print(f'backup: {backup_path} ({backup_path.stat().st_size} bytes, quick_check ok)')
        from hermes_state_runtime import RuntimeStoreError, _json

        def write(c):
            # Local to each invocation: _execute_write retries the callback on lock contention.
            repaired, removed, absent = [], [], []
            for sid, receipt in repairable:
                stored = c.execute('SELECT value FROM state_meta WHERE key=?', (POLICY_PREFIX + sid,)).fetchone()
                if stored is None:
                    raise SystemExit(f'{sid} vanished under us')
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
                repaired.append(sid)  # every rewrite is counted, with or without a binding row
                if receipt['_binding'] == 'remove':
                    c.execute('DELETE FROM state_meta WHERE key=?', (LEGACY_PREFIX + sid,))
                    removed.append(sid)
                else:
                    absent.append(sid)
            if len(repaired) != len(repairable):
                raise SystemExit(f'repaired {len(repaired)} of {len(repairable)} surveyed receipts')
            return repaired, removed, absent

        from hermes_state import SessionDB
        db = SessionDB(db_path=db_path)
        try:
            repaired, removed, absent = db._execute_write(write)
        finally:
            db.close()
        print(f'repaired {len(repaired)} receipt(s) of {len(repairable)} surveyed; backup {backup_path}')
        print(f'  bindings: {len(removed)} removed, {len(absent)} absent')
        return 0
    finally:
        conn.close()


if __name__ == '__main__':
    raise SystemExit(main())
