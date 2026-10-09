"""Explicit opt-in Kanban execution-policy seam (not wired by this module)."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import importlib.util
from pathlib import Path

REQUIRED_TABLE = 'kanban_execution_policy_required'


class PolicyViolation(PermissionError):
    """An opted-in board cannot prove its lifecycle policy is effective."""


def _load_policy(conn, *, required=False, marker=None):
    database = next(row[2] for row in conn.execute('PRAGMA database_list') if row[1] == 'main')
    registered = conn.execute('SELECT 1 FROM sqlite_master WHERE type=\'table\' AND name=?', (REQUIRED_TABLE,)).fetchone()
    default_marker = Path(database + '.policy-required.json') if database else None
    path = Path(marker) if marker is not None else default_marker
    if not (required or registered or marker is not None or (path and path.exists())):
        return None
    try:
        if path is None:
            raise ValueError('file-backed database required')
        if registered:
            rows = conn.execute('SELECT marker FROM ' + REQUIRED_TABLE).fetchall()
            if len(rows) != 1 or Path(rows[0][0]).resolve() != path.resolve():
                raise ValueError('required marker registration mismatch')
        data = json.loads(path.read_text())
        fields = {'version', 'database', 'required', 'capabilities', 'protected_tables'}
        if set(data) not in (fields, fields | {'provider_sha256'}):
            raise ValueError('invalid marker fields')
        if data['version'] != '1' or data['required'] is not True or Path(data['database']).resolve() != Path(database).resolve():
            raise ValueError('marker identity/version mismatch')
        capabilities = data['capabilities']
        protected = data['protected_tables']
        if not isinstance(capabilities, dict) or not capabilities or not isinstance(protected, list) or not protected:
            raise ValueError('missing capability contract')
        if REQUIRED_TABLE not in capabilities or any(t not in capabilities for t in protected):
            raise ValueError('missing marker/policy table capability')
        for name, digest in capabilities.items():
            row = conn.execute('SELECT type,sql FROM sqlite_master WHERE name=?', (name,)).fetchone()
            if not row or row[0] not in ('trigger', 'table') or not row[1] or hashlib.sha256(row[1].encode()).hexdigest() != digest:
                raise ValueError('missing or modified capability: ' + str(name))
        if not any(conn.execute('SELECT type FROM sqlite_master WHERE name=?', (n,)).fetchone()[0] == 'trigger' for n in capabilities):
            raise ValueError('no persistent lifecycle trigger')
        return data
    except (OSError, ValueError, TypeError, sqlite3.Error) as exc:
        raise PolicyViolation('Execution policy: ' + str(exc)) from exc


def policy_provider(conn):
    """Load an explicitly installed, hash-pinned board extension, never a JSON path.

    Like a plugin, the fixed board-local execution_policy.py is trusted OS-owner
    code. Native records/worker tools cannot install it or edit its hash grant.
    """
    data = _load_policy(conn)
    if data is None:
        return None
    try:
        path = Path(data['database']).parent / 'execution_policy.py'
        digest = data.get('provider_sha256')
        if not isinstance(digest, str) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('missing or changed pinned provider')
        spec = importlib.util.spec_from_file_location('_kanban_policy_' + digest, path)
        if spec is None or spec.loader is None:
            raise ValueError('provider loader unavailable')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not callable(module.validate_receipt) or not callable(module.runtime_snapshot):
            raise ValueError('provider interface unavailable')
        return module
    except Exception as exc:
        raise PolicyViolation('Execution policy: provider unavailable: ' + str(exc)) from exc


def provider_worker_context(conn, task_id):
    """Opt-in replacement task context from the verified board-local provider.

    None preserves the SDK renderer. A provider-supplied string is the entire
    task context: the caller must not append implicit history or comments.
    The provider owns stable rendering; malformed/oversized output is refused,
    never truncated or silently replaced with the broader default context.
    """
    provider = policy_provider(conn)
    missing = object()
    hook = getattr(provider, 'worker_context', missing)
    if hook is missing:
        return None
    try:
        if not callable(hook):
            raise TypeError('hook must be callable')
        context = hook(conn, task_id)
        if context is not None and (
            type(context) is not str or not context.strip() or len(context) > 128_000
        ):
            raise ValueError('expected None or nonblank string of at most 128000 characters')
        return context
    except Exception as exc:
        raise PolicyViolation('Execution policy: worker context unavailable: ' + str(exc)) from exc


def validate_before_claim(conn, task_id, profile, *, required=False, marker=None, verify_receipt=None):
    """Return None for unguarded boards, True for validated opted-in boards.

    Caller owns a BEGIN IMMEDIATE transaction including the actual claim. The
    trusted project verifier is supplied by the caller, never imported from JSON.
    """
    return _validate_receipt(conn, task_id, profile, 'claim', run_id=None, effective=None,
                             required=required, marker=marker, verify_receipt=verify_receipt)


def profile_requires_policy(profile):
    """An opted-in profile cannot escape to an unguarded/renamed board."""
    if not profile:
        return False
    from hermes_cli.profiles import resolve_profile_env
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    from hermes_cli.config import load_config_readonly
    try:
        home = resolve_profile_env(profile)
    except FileNotFoundError:
        return False
    token = set_hermes_home_override(home)
    try:
        config = load_config_readonly()
    finally:
        reset_hermes_home_override(token)
    return config.get('agent', {}).get('require_execution_receipt', False) is True


def _validate_receipt(conn, task_id, profile, phase, *, run_id, effective, required, marker, verify_receipt):
    data = _load_policy(conn, required=required, marker=marker)
    if data is None and phase == 'claim' and profile_requires_policy(profile):
        data = _load_policy(conn, required=True, marker=marker)
    if data is None:
        return None
    if not conn.in_transaction:
        raise PolicyViolation('Execution policy: validation requires a write transaction')
    provider = None
    if verify_receipt is None:
        provider = policy_provider(conn)
        if provider is None:
            raise PolicyViolation('Execution policy: receipt verifier unavailable')
        verify_receipt = provider.validate_receipt

    if not callable(verify_receipt):
        raise PolicyViolation('Execution policy: receipt verifier unavailable')
    try:
        if provider is not None and phase == 'claim':
            if verify_receipt(conn, task_id, profile, phase, run_id=run_id, effective=None) is not True:
                raise ValueError('receipt rejected')
            effective = provider.runtime_snapshot(conn, task_id)
        if verify_receipt(conn, task_id, profile, phase, run_id=run_id, effective=effective) is not True:
            raise ValueError('receipt rejected')
    except (ValueError, TypeError, KeyError, sqlite3.Error) as exc:
        raise PolicyViolation('Execution policy: ' + str(exc)) from exc
    return True


def validate_before_spawn(conn, task_id, profile, run_id, *, effective=None, required=False, marker=None, verify_receipt=None):
    """Final gate using freshly resolved actual effective tools/context, never config ACKs.

    Caller must hold its write transaction through validation and the final spawn
    decision. The verifier owns project-specific snapshot/one-spawn semantics.
    """
    if _load_policy(conn, required=required, marker=marker) is None:
        return None
    if type(run_id) is not int or run_id <= 0 or not isinstance(effective, dict) or not effective:
        raise PolicyViolation('Execution policy: exact run and latest effective assertion required')
    return _validate_receipt(conn, task_id, profile, 'spawn', run_id=run_id, effective=effective,
                             required=required, marker=marker, verify_receipt=verify_receipt)


def guard_destructive_operation(conn, operation, *, required=False, marker=None):
    """Call BEFORE ordinary import/replace/delete/recovery drops any board state."""
    if _load_policy(conn, required=required, marker=marker) is not None:
        raise PolicyViolation('Execution policy: ordinary ' + str(operation) + ' cannot remove or replace guarded board state')


def protect_connection(conn, *, required=False, marker=None):
    """Install the managed connection's authorizer (caller owns this authorizer).

    Block schema replacement and direct edits to declared policy tables. Persistent
    verified triggers may update their own revision/consumption tables. Trusted
    receipt minting uses a separate explicit administrative connection. Not an OS
    sandbox: raw SQLite admins can remove triggers/files outside this connection.
    """
    data = _load_policy(conn, required=required, marker=marker)
    if data is None:
        return None
    protected = set(data['protected_tables'])
    trusted_triggers = {name for name in data['capabilities']
                        if conn.execute('SELECT type FROM sqlite_master WHERE name=?', (name,)).fetchone()[0] == 'trigger'}
    ddl = {sqlite3.SQLITE_DROP_TABLE, sqlite3.SQLITE_DROP_TRIGGER, sqlite3.SQLITE_ALTER_TABLE,
           sqlite3.SQLITE_CREATE_TABLE, sqlite3.SQLITE_CREATE_TRIGGER, sqlite3.SQLITE_ATTACH,
           sqlite3.SQLITE_DETACH, sqlite3.SQLITE_DROP_INDEX}
    writes = {sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE, sqlite3.SQLITE_DELETE}
    def authorize(action, arg1, arg2, database, source):
        if action in ddl:
            return sqlite3.SQLITE_DENY
        if action in writes and arg1 in protected and source not in trusted_triggers:
            return sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_PRAGMA and arg2 is not None and arg1 in ('writable_schema', 'recursive_triggers', 'ignore_check_constraints'):
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK
    conn.set_authorizer(authorize)
    return True
