"""Transactional legacy binding; never a second transcript writer."""
import json
import time

from hermes_state_runtime import RuntimeStoreError, _json

LEGACY_PREFIX = 'gateway.local_legacy.v1:'


# Sources the pre-authority local CLI/TUI/GUI wrote. ``oneshot`` (finite ``chat -q`` / ``-z``) and ``tool``
# (``chat --source tool``) are CLI runs that printed ``hermes --resume <id>``: adopted for resume like
# ``cli`` and keeping their label, so human pickers still hide them (INTERNAL_LISTING_SOURCES).
# ``desktop`` is the label main's Desktop stored; it is adopted as the ``gui`` creation source.
# ``acp`` is main's in-process editor server (Zed, VS Code, JetBrains): adopted under its own source.
LEGACY_LOCAL_SOURCES = frozenset({'cli', 'tui', 'gui', 'desktop', 'oneshot', 'tool', 'acp'})
# Legacy labels that are not creation sources of their own (session_policy.SURFACES).
LEGACY_CREATION_SOURCES = {'desktop': 'gui'}


def legacy_account_label(row):
    # main's Desktop stamped the signed-in Nous account on its chats. That is a label, not a route or
    # an authenticated owner, so it does not block adoption of an otherwise unrouted Desktop row.
    user_id = row.get('user_id')
    return row['source'] == 'desktop' and isinstance(user_id, str) and user_id.startswith('nous:')


def login_owned_by(row, subject):
    """Whether *subject* is the login a served remote backend stamped on this Desktop row.

    ``hermes serve`` stamps the WS-upgrade login as ``<provider>:<user id>``
    (tui_gateway/session_transports.py). Unlike the Nous label that stamp IS an owner: once a
    login's dashboard writes reach the gateway as that login, reading it as no-owner let login bob
    rename, claim and delete alice's chats (#140183 review F1). So it adopts only for the caller
    whose authenticated subject spells exactly that label (``_login_label``: issuer-less logins
    only); every other caller, the host's own native peer included, still meets a foreign owner.
    """
    user_id = row.get('user_id')
    if row['source'] != 'desktop' or not isinstance(user_id, str) or not subject:
        return False
    from hermes_state_mutation_binding import _login_label
    return _login_label(subject) == user_id


def canonical_bot_chat_row(row):
    """Whether *row* holds the canonical ``Bot Chat`` title — the profile's forever-chat.

    Bot Mode identifies that chat by NAME on every open (Desktop re-resolves the exact title and
    fails closed rather than mint a second one), so the chat belongs to the PROFILE, not to the
    principal or the surface that happened to write it. The surfaces that are not local creation
    lanes stamp their own provenance label and their own login on the row they write; neither names
    an owner of the Bot Chat.
    """
    from tools.bot_mode_probe import BOT_CHAT_TITLE
    return str(row.get('title') or '').strip() == BOT_CHAT_TITLE


def validate_legacy_row(db, row, *, subject=None, operator=False):
    # Unrouted old local history only. Existing identity, foreign profile,
    # API, messaging, delegate and canonical rows require their own authority.
    # ``subject``: the adopting caller, whose own login stamp is not a foreign identity. A ``user_id``
    # stamp equal to ``subject`` itself names that principal: it is already its history by
    # ``authorize_history``'s rule, so it adopts like an unstamped row.
    # ``operator``: a verified transport gate of this profile (never identity data). The canonical
    # ``Bot Chat`` belongs to the PROFILE — its identity is the title, resolved on every open — so
    # the operator may adopt it whoever wrote it, including a surface with no creation lane of its
    # own (the dashboard's basic-auth login, a mobile client) whose only trace is a provenance
    # label and a login stamp. Without this, such a row is refused by every caller, the restore
    # chain has nothing to rebuild from, and the forever-chat can never open on any surface again.
    # A non-operator caller still meets a login-stamped chat as its owner's (#140183 F1).
    user_id = row.get('user_id')
    foreign = bool(user_id) and user_id != subject and not legacy_account_label(row) and not login_owned_by(row, subject)
    canonical = canonical_bot_chat_row(row) and (operator or not foreign)
    if (row.get('profile_name') not in (None, '', db._own_profile_name() or 'default')
            or any(row.get(k) for k in ('session_key', 'chat_id', 'origin_json'))
            or (not canonical and (row['source'] not in LEGACY_LOCAL_SOURCES or foreign))):
        raise RuntimeStoreError('not_found')


def bind_legacy_target(db, conn, receipt):
    target = receipt['legacy_session_id']
    if target != receipt['session_id']:
        raise RuntimeStoreError('storage_unavailable')
    row = conn.execute('SELECT * FROM sessions WHERE id=?', (target,)).fetchone()
    if row is None:
        raise RuntimeStoreError('not_found')
    validate_legacy_row(db, dict(row), subject=receipt['principal_id'],
                        operator=bool(receipt.get('canonical_adoption')))
    from hermes_state_mutation_binding import BINDING_PREFIX
    imported = conn.execute('SELECT value FROM state_meta WHERE key=?', (BINDING_PREFIX + target,)).fetchone()
    if imported and json.loads(imported[0]) != [receipt['profile_id'], receipt['principal_id']]:
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
    # admitted turn reopens it (``reopen_local_session``), exactly as the classic surfaces did.
    conn.execute('UPDATE sessions SET session_key=?,chat_id=?,user_id=?,chat_type=?,origin_json=? WHERE id=?',
                 (receipt['route'], receipt['session_id'], receipt['principal_id'], 'dm', _json(source), target))
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
