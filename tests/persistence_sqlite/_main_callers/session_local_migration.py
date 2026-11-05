"""Main's ``gateway/session_local_migration.py`` bodies (as of the pinned main), the legacy side of the backend parity checks."""

from persistence.core.errors import AuthorityRefusal


def require_history_claim(authority, conn, actor, sid):
    """Import receipts confer ownership; merely knowing an old ID does not."""
    from hermes_state_mutation_binding import BINDING_PREFIX, _login_label, _owner_user
    row = conn.execute('SELECT * FROM sessions WHERE id=?', (sid,)).fetchone()
    if row is None or row['session_key'] or row['chat_id'] or row['origin_json']:
        return
    if conn.execute('SELECT 1 FROM state_meta WHERE key=?', (BINDING_PREFIX + sid,)).fetchone():
        return  # authorize_history checks the already committed importer identity.
    own_profile = row['profile_name'] in (None, '', authority.db._own_profile_name() or 'default')
    if own_profile and _owner_user(row) and row['user_id'] in (actor.subject, _login_label(actor.subject)):
        return  # The login that created it (a served remote Desktop); authorize_history re-checks it.
    if getattr(authority, '_native_legacy_transports', {}).get(actor.transport_id) != actor:
        raise AuthorityRefusal('permission_denied')
