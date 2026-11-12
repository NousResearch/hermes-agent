"""Control-only ownership for unbound legacy history; never an execution policy."""
import json
from hermes_agent.hermes_state_runtime import RuntimeStoreError

BINDING_PREFIX = 'gateway.history_control.v1.'


def _history_row(row):
    """Local surfaces' unrouted rows are owner history: every row the native resume adopts
    (``legacy_history_row``: old ``desktop`` chats, ``oneshot``/``tool`` CLI runs and phone apps'
    self-labelled chats too) plus imports. A row resume can adopt must also take a sidebar edit
    (pin/archive/hide/rename)."""
    from hermes_agent.hermes_state_local_migration import legacy_history_row
    return row['source'] == 'import' or legacy_history_row(dict(row))


def _owner_user(row):
    """The row's owner identity; main's Desktop Nous-account label is not one (see validate_legacy_row)."""
    from hermes_agent.hermes_state_local_migration import legacy_account_label
    return None if legacy_account_label(dict(row)) else row['user_id']


def _login_label(subject):
    """The ``<provider>:<user>`` label Desktop stamps on its rows (hermes_agent/tui_gateway/session_transports.py) when it
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


def _binding_owner(saved):
    """The owner half of a stored ``[profile_id, subject]`` binding, or None when unreadable.

    The profile half is the home path that wrote the binding into this profile's own store, so it
    never names another profile; a store that moved with its profile keeps its bindings."""
    try:
        profile, subject = json.loads(saved)
    except (TypeError, ValueError):
        return None
    return subject if isinstance(profile, str) else None


def binding_names(saved, subject):
    """Whether a stored binding names *subject*: the same owner (a binding a Webapp login stored
    belongs to the machine owner, gateway.owner_identity.same_owner). Every reader of a binding asks
    this, never a string compare."""
    from gateway.owner_identity import same_owner
    owner = _binding_owner(saved)
    return owner is not None and same_owner(owner, subject)


def bindings_agree(first, second):
    """Two stored bindings name the same owner (both absent agrees; one absent does not)."""
    if first == second or first is None or second is None:
        return first == second
    from gateway.owner_identity import owners_agree
    first_owner, second_owner = _binding_owner(first), _binding_owner(second)
    return first_owner is not None and second_owner is not None and owners_agree(first_owner, second_owner)


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
        if not binding_names(saved[0], actor.subject):
            raise RuntimeStoreError('permission_denied')
        return True
    if not _history_row(row):
        return False
    from gateway.owner_identity import same_owner
    if _owner_user(row) and not same_owner(row['user_id'], actor.subject):
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
    legacy CLI conversation — or share an unbound routeless anchor's source (an old cron run's or
    subagent's own compression chain, mounted as one chat). A bound (imported) anchor therefore
    never reaches unbound rows.
    """
    columns = 'SELECT source,user_id,session_key,chat_id,origin_json FROM sessions WHERE id=?'
    anchor, row = (conn.execute(columns, (sid,)).fetchone() for sid in (anchor_id, session_id))
    if anchor is None or row is None:
        return False
    if row['session_key'] or row['chat_id'] or row['origin_json']:
        return tuple(row)[1:4] == tuple(anchor)[1:4]
    binding = history_binding(conn, session_id)
    if not bindings_agree(binding, history_binding(conn, anchor_id)):
        return False
    routeless_peer = row['source'] == anchor['source'] and not any(tuple(anchor)[2:5])
    from gateway.owner_identity import owners_agree
    return binding is not None or (
        (_history_row(row) or routeless_peer)
        and (_owner_user(row) in (None, '') or owners_agree(_owner_user(row), anchor['user_id'])))
