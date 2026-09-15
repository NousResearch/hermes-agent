"""Conditional Codex quota recovery, not OAuth refresh or request accounting.

A recovery epoch fences snapshots predating a clear. Failure IDs distinguish
separate events even when their wall-clock timestamps coincide. Legacy rows
have an absent epoch and acquire one only on a proven recovery.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from uuid import uuid4

PROVIDER = 'openai-codex'
EPOCH = '_codex_recovery_epoch'
FAILURE = '_codex_failure_id'
STATUS_FIELDS = ('last_status', 'last_status_at', 'last_error_code',
                 'last_error_reason', 'last_error_message', 'last_error_reset_at',
                 'failure_reason', FAILURE)
IDENTITY_FIELDS = ('id', 'source', 'auth_type', 'access_token', 'refresh_token',
                   'base_url', EPOCH, *STATUS_FIELDS)


def _semantic(row):
    # Use exactly the supported reader defaults/legacy timestamp conversion.
    # This is recognition only: Observation retains raw bytes' data for CAS.
    from agent.credential_pool import PooledCredential
    return PooledCredential.from_dict(PROVIDER, row).to_dict()


def fence_snapshot(incoming, disk, *, failure_event=None):
    """Return the authoritative row when an automatic-clear epoch was missed.

    Never merge stale tokens or labels back while preserving only status: that
    would silently regress the credential whose quota was actually checked.
    Explicit administrative resets retain their existing bypass contract.
    """
    if disk and failure_event and failure_event == incoming.get(FAILURE):
        # A newly reported HTTP failure is an operation, not a stale snapshot.
        # It may arrive after a clear from an already-running request. Apply
        # only status to the still-identical grant, preserving the owner's epoch.
        identity = ('id', 'source', 'auth_type', 'access_token', 'refresh_token', 'base_url')
        mem, stored = _semantic(incoming), _semantic(disk)
        if any(mem.get(k) != stored.get(k) for k in identity):
            return dict(disk)
        # DEAD is terminal for this grant, regardless of request completion
        # order. Among nonterminal events, keep the already-committed event
        # on ties; a UUID establishes identity, not temporal precedence.
        if disk.get('last_status') == 'dead':
            return dict(disk)
        if (incoming.get('last_status') != 'dead' and disk.get(FAILURE)
                and disk.get(FAILURE) != failure_event
                and (stored.get('last_status_at') or 0) >= (mem.get('last_status_at') or 0)):
            return dict(disk)
        updated = dict(disk)
        for key in STATUS_FIELDS:
            updated.pop(key, None)
            if key in incoming:
                updated[key] = incoming[key]
        return updated
    if disk and (incoming.get(EPOCH) != disk.get(EPOCH)
                 or incoming.get(FAILURE) != disk.get(FAILURE)):
        return dict(disk)
    return None


@dataclass(frozen=True)
class Observation:
    owner: Path
    # Atomic replacement (including remove/reinsert of an identical row) must
    # invalidate the proof. Unrelated owner writes conservatively retry later.
    version: tuple
    row: dict


def _version(path):
    st = path.stat()
    return st.st_dev, st.st_ino, st.st_mtime_ns, st.st_ctime_ns, st.st_size


@contextmanager
def _owner_store():
    from agent import credential_pool as cp
    from hermes_cli import auth
    # Same profile -> owner order as existing auth transactions. Holding the
    # profile lock also prevents a login moving authority during commit.
    with auth._auth_store_lock():
        path = auth._auth_file_path()
        if not cp._profile_owns_pool_provider(PROVIDER):
            root = cp._borrowed_single_use_pool_root()
            if root is not None:
                with auth._auth_store_lock(target_path=root):
                    yield root, auth._load_auth_store(root)
                return
        yield path, auth._load_auth_store(path)


def _row(store, entry_id):
    return next((r for r in store.get('credential_pool', {}).get(PROVIDER, [])
                 if isinstance(r, dict) and r.get('id') == entry_id), None)


def reconcile(entry):
    """Read the current owner row, without writing or manufacturing a proof.

    Removal is unavailable, not a reason to recreate a local/global row.
    Owner/material changes can be adopted only as a new observation later.
    """
    from agent.credential_pool import PooledCredential
    with _owner_store() as (_, store):
        row = _row(store, entry.id)
        if row is None:
            return None
        normalized, payload = _semantic(row), entry.to_dict()
        if all(normalized.get(k) == payload.get(k) for k in (*IDENTITY_FIELDS, 'quota_scope')):
            # Selection counters/order are process-local until the next write;
            # a read of identical authority must not reset them every request.
            return entry
        return PooledCredential.from_dict(PROVIDER, row)


def observe(entry) -> Optional[Observation]:
    """Capture owner, material and failure before doing any quota I/O."""
    payload = entry.to_dict()
    with _owner_store() as (path, store):
        row = _row(store, entry.id)
        if row is None:
            return None
        normalized = _semantic(row)
        if any(normalized.get(k) != payload.get(k) for k in IDENTITY_FIELDS):
            return None
        if row.get('last_status') != 'exhausted':
            return None
        return Observation(path, _version(path), deepcopy(row))


def observe_token(token: str, *, entry_id=None) -> Optional[Observation]:
    """Capture the one pool-only grant that the singleton fallback will probe."""
    from hermes_cli import auth
    with _owner_store() as (path, store):
        matches = [row for row in store.get('credential_pool', {}).get(PROVIDER, [])
                   if isinstance(row, dict) and row.get('access_token') == token
                   and (entry_id is None or row.get('id') == entry_id)]
        # Token-only callers must not select arbitrarily between duplicate IDs.
        if len(matches) == 1:
            row = matches[0]
            if (row.get('access_token') == token
                    and row.get('last_status') == 'exhausted'
                    and auth._is_codex_rate_limit_shaped(row.get('last_error_code'),
                        row.get('last_error_reason'), row.get('last_error_message'))):
                return Observation(path, _version(path), deepcopy(row))
    return None


def recover(observed: Observation, *, proof: str):
    """Update ONLY the observed owner row; return it only after durable save.

    The caller supplies only the existing aggregate quota probe or elapsed
    cooldown policy, never inference/accounting success or a token rotation.
    Unknown legacy quota scope keeps the existing aggregate-probe semantics;
    explicitly model-scoped failures are not cleared by an aggregate probe.
    """
    from hermes_cli import auth
    if proof not in ('quota', 'elapsed'):
        raise ValueError('unsupported recovery proof')
    if proof == 'quota' and observed.row.get('quota_scope') not in (None, 'unknown', 'provider'):
        return None
    with _owner_store() as (path, store):
        if path != observed.owner or _version(path) != observed.version:
            return None
        row = _row(store, observed.row['id'])
        if row is None or row != observed.row:
            return None
        updated = dict(row)
        for key in STATUS_FIELDS:
            updated.pop(key, None)
        updated['last_status'] = 'ok'
        updated[EPOCH] = uuid4().hex
        rows = store['credential_pool'][PROVIDER]
        rows[rows.index(row)] = updated
        auth._save_auth_store(store, target_path=path)
        return updated
