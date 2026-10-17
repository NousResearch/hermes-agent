"""Control-only ownership for unbound legacy history; never an execution policy."""
import json
from hermes_state_runtime import RuntimeStoreError

BINDING_PREFIX = 'gateway.history_control.v1.'


def _history_sources():
    """Local surfaces whose unrouted rows are owner history: every source the native resume adopts
    (``LEGACY_LOCAL_SOURCES``: old ``desktop`` chats and ``oneshot``/``tool`` CLI runs too) plus imports.
    A row resume can adopt must also take a sidebar edit (pin/archive/hide/rename)."""
    from hermes_state_local_migration import LEGACY_LOCAL_SOURCES
    return LEGACY_LOCAL_SOURCES | {'import'}


def _owner_user(row):
    """The row's owner identity; main's Desktop Nous-account label is not one (see validate_legacy_row)."""
    from hermes_state_local_migration import legacy_account_label
    return None if legacy_account_label(dict(row)) else row['user_id']


def _login_label(subject):
    """The ``<provider>:<user>`` label Desktop stamps on its rows (tui_gateway/session_transports.py) when it
    names exactly the dashboard subject ``auth:v1:[provider, "", user]`` (gateway/session_identity.py).
    Only an issuer-less login maps: the label drops the issuer, so an issuer-bearing subject must not claim it."""
    from gateway.session_identity import parse_authenticated_subject
    parts = parse_authenticated_subject(subject)
    if parts is None:
        return None
    provider, issuer, user = parts
    if issuer or ':' in provider or not provider.strip() or not user.strip():
        return None
    label = f'{provider.strip()}:{user.strip()}'
    # Never yield something that reads as a subject itself (a provider named ``auth`` + user ``v1:[...]``).
    return None if label.startswith('auth:') else label


def import_history_control(conn, actor, session_ids):
    binding = json.dumps([actor.profile_id, actor.subject], separators=(',', ':'))
    for sid in session_ids:
        if conn.execute('SELECT 1 FROM sessions WHERE id=?', (sid,)).fetchone() is None:
            conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?)', (BINDING_PREFIX + sid, binding))


def authorize_history(conn, actor, session_id, *, claim=False):
    row = conn.execute('SELECT * FROM sessions WHERE id=?', (session_id,)).fetchone()
    if row is None:
        return False
    # Import intentionally strips routes. Never reinterpret a cold native/API
    # transcript as imported history, even if stale import metadata exists.
    if row['session_key'] or row['chat_id'] or row['origin_json']:
        return False
    key = BINDING_PREFIX + session_id
    binding = json.dumps([actor.profile_id, actor.subject], separators=(',', ':'))
    saved = conn.execute('SELECT value FROM state_meta WHERE key=?', (key,)).fetchone()
    if saved is not None:
        if saved[0] != binding:
            raise RuntimeStoreError('permission_denied')
        return True
    if row['source'] not in _history_sources():
        return False
    if _owner_user(row) and row['user_id'] not in (actor.subject, _login_label(actor.subject)):
        raise RuntimeStoreError('permission_denied')
    if 'session:create' not in actor.capabilities:
        raise RuntimeStoreError('permission_denied')
    if claim:
        conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?)', (key, binding))
    return True


def history_binding(conn, session_id):
    saved = conn.execute('SELECT value FROM state_meta WHERE key=?', (BINDING_PREFIX + session_id,)).fetchone()
    return saved[0] if saved else None


def same_history_owner(conn, anchor_id, session_id):
    """Whether chain member ``session_id`` is history the authorizer of ``anchor_id`` controls.

    A real compression continuation copies its parent's route columns, so a routed member must
    carry the anchor's exact (user_id, session_key, chat_id). A routeless member needs the
    anchor's exact import binding; unbound, it must also pass :func:`authorize_history`'s local
    rule (local source, no foreign user_id) — e.g. the pre-adoption ancestors of an adopted
    legacy CLI conversation. A bound (imported) anchor therefore never reaches unbound rows.
    """
    columns = 'SELECT source,user_id,session_key,chat_id,origin_json FROM sessions WHERE id=?'
    anchor, row = (conn.execute(columns, (sid,)).fetchone() for sid in (anchor_id, session_id))
    if anchor is None or row is None:
        return False
    if row['session_key'] or row['chat_id'] or row['origin_json']:
        return tuple(row)[1:4] == tuple(anchor)[1:4]
    binding = history_binding(conn, session_id)
    if binding != history_binding(conn, anchor_id):
        return False
    return binding is not None or (
        row['source'] in _history_sources() and _owner_user(row) in (None, '', anchor['user_id']))
