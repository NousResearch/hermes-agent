"""Transactional legacy binding; never a second transcript writer."""
import functools
import json
import time

from hermes_state_runtime import RuntimeStoreError, _json

LEGACY_PREFIX = 'gateway.local_legacy.v1:'


# Sources the pre-authority local CLI/TUI/GUI wrote. ``oneshot`` (finite ``chat -q`` / ``-z``) and ``tool``
# (``chat --source tool``) are CLI runs that printed ``hermes --resume <id>``: adopted for resume like
# ``cli`` and keeping their label, so human pickers still hide them (INTERNAL_LISTING_SOURCES).
# ``desktop`` is the label main's Desktop stored; it is adopted as the ``gui`` creation source.
# ``acp`` is main's in-process editor server (Zed, VS Code, JetBrains): adopted under its own source.
LEGACY_LOCAL_SOURCES = frozenset({'cli', 'tui', 'gui', 'desktop', 'oneshot', 'tool', 'acp', 'subagent'})
# Delegated history is admitted only after ancestry validation, never as a creation surface.
LEGACY_CREATION_SOURCES = {'desktop': 'gui', 'subagent': 'cli'}
# Labels a routeless row can carry that never name a chat app: automation and recovery rows, a
# retired platform, and a third-party WebUI that runs its own agent on this state.db (adopting its rows
# would legitimise a second writer, so they stay read-only: gateway/session_history_mount.py).
NOT_CLIENT_LABELS = frozenset({'cron', 'subagent', 'kanban', 'bot_room', 'a2a', 'webhook', 'api_server',
                               'import', 'recovered', 'curator', 'batch', 'homeassistant', 'webui',
                               'local', 'unknown'})


@functools.lru_cache(maxsize=1)
def _messaging_labels():
    """Every platform Hermes ships: the enum and the bundled platform plugins. A runtime-registered
    catalog plugin (hermes-mobile's ``mobile``) is not in it: its rows are routed, which excludes them."""
    from gateway.config import Platform
    bundled, aliases = Platform._scan_bundled_plugin_platforms()
    return frozenset({platform.value for platform in Platform} | set(bundled) | set(aliases))


def not_client_labels():
    """Labels :func:`client_label` refuses, for queries that pre-filter rows."""
    return LEGACY_LOCAL_SOURCES | NOT_CLIENT_LABELS | _messaging_labels()


def client_label(source):
    """Whether *source* is a client app's own label (``mobile``, ``ios``): not a local surface, a
    shipped messaging platform, or an automation / foreign-writer label."""
    return isinstance(source, str) and bool(source.strip()) and source not in not_client_labels()


def client_history_row(row):
    """A chat a client app wrote under its own label before ``session.create`` took only surfaces.

    The pre-authority ``hermes serve`` stored whatever ``source`` a client sent, so phone apps signed
    in through the dashboard login left routeless rows labelled ``mobile``, ``ios`` and the like. They
    are the owner's Desktop-style chats: adopted as ``gui``. A routed row is never one, which keeps a
    plugin platform sharing the label (hermes-mobile registers ``mobile``) on its messaging route."""
    return client_label(row.get('source')) and not any(row.get(k) for k in ('session_key', 'chat_id', 'origin_json'))


def legacy_history_row(row):
    """Whether resume adopts *row* as old local history: a local surface's label, or a client app's."""
    return row.get('source') in LEGACY_LOCAL_SOURCES or client_history_row(row)


def legacy_creation_source(row):
    """The creation source an adopted legacy row resumes on (session_policy.SURFACES)."""
    return 'gui' if client_history_row(row) else LEGACY_CREATION_SOURCES.get(row['source'], row['source'])


def delegated_history(row):
    """Literal children and inherited-source children share the same trust boundary."""
    from hermes_state_sessions import _parse_model_config
    return row.get('source') == 'subagent' or '_delegate_from' in _parse_model_config(row.get('model_config'))


def _delegated_local_receipt(conn, row, profile_id, subject):
    from gateway.owner_identity import same_owner
    from hermes_state_local import POLICY_PREFIX, local_lineage_owner
    owner = row.get('chat_id')
    if not owner:
        raise RuntimeStoreError('not_found')
    saved = conn.execute('SELECT value FROM state_meta WHERE key=?', (POLICY_PREFIX + owner,)).fetchone()
    try:
        if local_lineage_owner(conn, row['id']) != owner:
            raise ValueError('foreign local lineage')
        receipt = json.loads(saved[0]) if saved else {}
        if (receipt['profile_id'] != profile_id or not same_owner(receipt['principal_id'], subject)
                or receipt['session_id'] != owner or receipt['route'] != row.get('session_key')
                or not same_owner(row.get('user_id'), subject)):
            raise ValueError('foreign local receipt')
        if delegated_history(row):
            legacy = conn.execute('SELECT value FROM state_meta WHERE key=?', (LEGACY_PREFIX + owner,)).fetchone()
            expected = {'session_id': owner, 'legacy_session_id': owner,
                        'profile_id': profile_id, 'principal_id': receipt['principal_id']}
            if legacy is None or json.loads(legacy[0]) != expected or receipt.get('legacy_session_id') != owner:
                raise ValueError('missing delegated adoption receipt')
        origin = json.loads(row.get('origin_json') or '{}')
        if origin != receipt['entry']['origin'] or origin.get('platform') != 'local':
            raise ValueError('foreign local origin')
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        raise RuntimeStoreError('not_found') from exc


def _delegated_registry_idle(conn, session_ids):
    """Inspect the DB profile's liveness registry without pruning or creating files."""
    from pathlib import Path
    from hermes_cli.active_sessions import ActiveSessionRegistryError, _prune_dead, _read_entries
    path = next((r[2] for r in conn.execute('PRAGMA database_list') if r[1] == 'main'), '')
    if not path:
        raise RuntimeStoreError('not_found')
    try:
        entries = _read_entries(Path(path).parent / 'runtime' / 'active_sessions.json', strict=True)
        # Scope unknown-owner refusals to this ancestry; unrelated leases grant no authority.
        relevant = [entry for entry in entries if entry['session_id'] in session_ids]
        if _prune_dead(relevant, strict=True):
            raise RuntimeStoreError('not_found')
    except ActiveSessionRegistryError as exc:
        raise RuntimeStoreError('not_found') from exc


def validate_delegated_history(conn, row, *, subject, profile_id, profile_name=None):
    """Read-only, bounded ancestry authorization, also used inside the adoption transaction.

    Markers establish provenance, never ownership. Every ancestor independently passes the
    same local-owner boundary; an existing import binding cannot override a foreign ancestor.
    """
    row = dict(row)
    if not delegated_history(row):
        return
    from hermes_state_mutation_binding import BINDING_PREFIX, binding_names
    from hermes_state_sessions import _parse_model_config
    if profile_name is None:
        from pathlib import Path
        path = next((r[2] for r in conn.execute('PRAGMA database_list') if r[1] == 'main'), '')
        home = Path(path).parent
        profile_name = home.name if home.parent.name == 'profiles' else 'default'
    seen = set()
    registry_ids = set()
    for hop in range(33):
        sid = row['id']
        if sid in seen or row.get('profile_name') not in (None, '', profile_name):
            raise RuntimeStoreError('not_found')
        seen.add(sid)
        child = delegated_history(row)
        if (not (legacy_history_row(row) or row.get('source') == 'local')
                or (child and row.get('source') in {'tool', 'oneshot'})):
            raise RuntimeStoreError('not_found')
        owner = row.get('user_id')
        # Delegation deliberately does not inherit routing/user stamps. Unstamped local
        # history follows the native claim boundary; any explicit owner must still match.
        if owner and owner != subject and not login_owned_by(row, subject):
            raise RuntimeStoreError('not_found')
        saved = conn.execute('SELECT value FROM state_meta WHERE key=?', (BINDING_PREFIX + sid,)).fetchone()
        try:
            if saved is not None and not binding_names(saved[0], profile_id, subject):
                raise ValueError('foreign control binding')
        except (ValueError, TypeError) as exc:
            raise RuntimeStoreError('not_found') from exc
        routed = any(row.get(k) for k in ('session_key', 'chat_id', 'origin_json'))
        if routed:
            _delegated_local_receipt(conn, row, profile_id, subject)
        # A live parent is not a settled authorization anchor, nor can a live child be mounted.
        if (conn.execute("SELECT 1 FROM session_admissions WHERE target_session_id=? AND status IN ('queued','started','unknown')", (sid,)).fetchone()
                or conn.execute("SELECT 1 FROM worker_executions WHERE session_id=? AND status IN ('registered','running','unknown')", (sid,)).fetchone()
                or conn.execute('SELECT 1 FROM session_turn_leases WHERE conversation_id IN (?,?) AND expires_at>?',
                                (sid, row.get('chat_id') or sid, time.time())).fetchone()):
            raise RuntimeStoreError('not_found')
        registry_ids.update((sid, row.get('chat_id') or sid))
        if child and row.get('ended_at') is None:
            raise RuntimeStoreError('not_found')
        if not child:
            _delegated_registry_idle(conn, registry_ids)
            return
        marker = _parse_model_config(row.get('model_config')).get('_delegate_from')
        if (hop == 32 or not isinstance(marker, str) or not marker.strip()
                or marker == sid or row.get('parent_session_id') != marker):
            raise RuntimeStoreError('not_found')
        parent = conn.execute('SELECT * FROM sessions WHERE id=?', (marker,)).fetchone()
        if parent is None:
            raise RuntimeStoreError('not_found')
        row = dict(parent)
    raise RuntimeStoreError('not_found')


def legacy_account_label(row):
    # main's Desktop stamped the signed-in Nous account on its chats. That is a label, not a route or
    # an authenticated owner, so it does not block adoption of an otherwise unrouted Desktop row.
    user_id = row.get('user_id')
    return row['source'] == 'desktop' and isinstance(user_id, str) and user_id.startswith('nous:')


def login_owned_by(row, subject):
    """Whether *subject* owns the login stamp a served dashboard put on this row.

    Older ``hermes serve`` builds stamped the WS-upgrade login as ``<provider>:<user id>``
    (tui_gateway/session_transports.py), and the gateway stored a Webapp login as its
    ``auth:v1:[...]`` subject. Hermes is single-user and sign-in is owner-only, so both name the
    machine owner: the host owner's subject adopts them (gateway.owner_identity.same_owner).
    """
    user_id = row.get('user_id')
    if not isinstance(user_id, str) or not subject:
        return False
    from gateway.owner_identity import same_owner
    return same_owner(user_id, subject)


def validate_legacy_row(db, row, *, subject=None, profile_id=None, conn=None):
    # Unrouted old local history only. Existing identity, foreign profile,
    # API, messaging, delegate and canonical rows require their own authority.
    # ``subject``: the adopting caller, whose own login stamp is not a foreign identity. A ``user_id``
    # stamp equal to ``subject`` itself names that principal: it is already its history by
    # ``authorize_history``'s rule, so it adopts like an unstamped row.
    user_id = row.get('user_id')
    if (not legacy_history_row(row)
            or any(row.get(k) for k in ('session_key', 'chat_id', 'origin_json'))
            or (user_id and not legacy_account_label(row) and not login_owned_by(row, subject))
            or row.get('profile_name') not in (None, '', db._own_profile_name() or 'default')):
        raise RuntimeStoreError('not_found')

    if delegated_history(row):
        if subject is None:
            raise RuntimeStoreError('not_found')
        if not profile_id:
            raise RuntimeStoreError('not_found')
        kwargs = {'subject': subject, 'profile_id': profile_id,
                  'profile_name': db._own_profile_name() or 'default'}
        if conn is not None:
            validate_delegated_history(conn, row, **kwargs)
        else:
            with db._read_ctx() as read_conn:
                validate_delegated_history(read_conn, row, **kwargs)


def bind_legacy_target(db, conn, receipt):
    target = receipt['legacy_session_id']
    if target != receipt['session_id']:
        raise RuntimeStoreError('storage_unavailable')
    row = conn.execute('SELECT * FROM sessions WHERE id=?', (target,)).fetchone()
    if row is None:
        raise RuntimeStoreError('not_found')
    validate_legacy_row(db, dict(row), subject=receipt['principal_id'],
                        profile_id=receipt['profile_id'], conn=conn)
    from hermes_state_mutation_binding import BINDING_PREFIX, binding_names
    imported = conn.execute('SELECT value FROM state_meta WHERE key=?', (BINDING_PREFIX + target,)).fetchone()
    if imported and not binding_names(imported[0], receipt['profile_id'], receipt['principal_id']):
        raise RuntimeStoreError('permission_denied')
    from hermes_state_compression import _CHAIN_STEP_SQL
    if conn.execute(_CHAIN_STEP_SQL, (target,)).fetchone() is not None:
        raise RuntimeStoreError('admission_conflict')
    from hermes_state_mutation_guards import require_idle
    require_idle(db, conn, [target])
    root = db._session_turn_lease_key_on_conn(conn, target)
    if conn.execute('SELECT 1 FROM session_turn_leases WHERE conversation_id=? AND expires_at>?',
                    (root, time.time())).fetchone():
        raise RuntimeStoreError('runtime_coordination_required')
    key = LEGACY_PREFIX + target
    if conn.execute('SELECT 1 FROM state_meta WHERE key=?', (key,)).fetchone():
        raise RuntimeStoreError('storage_unavailable')
    source = receipt['entry']['origin']
    # Binding is a read-side claim: the row keeps main's ``source`` label and its end stamp. The first
    # admitted turn reopens it (``reopen_local_session``), exactly as the classic surfaces did. A client
    # app's own label is the exception: bound, the row has a route, so the label alone would read it as
    # a messaging chat; it becomes the Desktop chat it resumes as (``created_source`` keeps provenance).
    label = 'desktop' if client_history_row(dict(row)) else row['source']
    conn.execute('UPDATE sessions SET source=?,session_key=?,chat_id=?,user_id=?,chat_type=?,origin_json=? WHERE id=?',
                 (label, receipt['route'], receipt['session_id'], receipt['principal_id'], 'dm', _json(source), target))
    conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?)', (key, _json({
        'session_id': receipt['session_id'], 'profile_id': receipt['profile_id'],
        'principal_id': receipt['principal_id'], 'legacy_session_id': target})))


def legacy_lineage_root(conn, receipt):
    root = receipt.get('legacy_session_id', receipt['session_id'])
    if 'legacy_session_id' in receipt:
        if root != receipt['session_id']:
            raise RuntimeStoreError('storage_unavailable')
        row = conn.execute('SELECT value FROM state_meta WHERE key=?', (LEGACY_PREFIX + root,)).fetchone()
        expected = {k: receipt[k] for k in ('session_id', 'profile_id', 'principal_id', 'legacy_session_id')}
        if row is None or json.loads(row[0]) != expected:
            raise RuntimeStoreError('storage_unavailable')
    return root
