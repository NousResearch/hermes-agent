"""A cold capability call adopts a routeless client-app row, exactly as ``session.resume`` does.

The read-only history mount used to claim such a row before any adoption path saw it, so the same
chat was writable after a resume and refused ``session_read_only`` for a client whose first call on
it was a capability (a metadata read, a submit, a file drop) instead. ``session_authority.authorize``
now adopts first (``session_local_migration.adopt_client_history``) and mounts only what it declines.

A row carrying a ``parent_session_id`` is a subagent transcript: evidence inside a turn, never a chat
of its own, so it stays read-only however it is labelled (``client_history_row``).

The adoption path is label-agnostic, so most cases drive it with a synthetic client label. The
``webui`` cases are the live ones on this install and are gated: the dashboard's own label is client
history only from t_33cf038a's ``NOT_CLIENT_LABELS`` change, which is in the running checkout but not
in this branch (that hunk is that card's to land).
"""
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.session_contract import SessionRef
from hermes_state_errors import SessionTurnLeaseLostError
from hermes_state_local_migration import not_client_labels
from hermes_state_runtime import RuntimeStoreError

_CAPS = ['session:create', 'session:read', 'session:submit', 'session:control']
# A client app's own label, as the pre-authority ``hermes serve`` stored whatever a client sent. Not a
# platform, not a local surface, not an automation label, so it is client history at every tip: the
# adoption path under test never depends on the dashboard's own label.
CLIENT_LABEL = 'phoneapp'
_WEBUI_IS_CLIENT_HISTORY = 'webui' not in not_client_labels()
webui_only = pytest.mark.skipif(
    not _WEBUI_IS_CLIENT_HISTORY,
    reason="'webui' is still in NOT_CLIENT_LABELS on this branch: t_33cf038a owns removing it, and "
           "that change is not committed here. The same code path is covered by the CLIENT_LABEL "
           "cases; this case re-proves it for the live dashboard label wherever that hunk is present.")


async def _owner(tmp_path, monkeypatch):
    """A fresh gateway over a state.db in *tmp_path*, and a native client connection to it."""
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.owner_identity import host_owner_subject
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(run, '_load_gateway_config',
                        lambda: {'model': {'default': 'm1', 'provider': 'openrouter'},
                                 'platform_toolsets': {'gui': []}})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             _cached_agent_for=lambda route: None, _restored_source=lambda entry: entry.origin,
                             _delivery_adapter_for=lambda source: None)
    runner._adapter_for_source = lambda source: None
    authority = await initialize_session_authority(runner, profile_id=str(tmp_path), instance_id='first')
    desktop = AuthorityConnection(authority, object(), {
        'provider': 'local', 'user_id': host_owner_subject(), 'profile_id': str(tmp_path),
        'instance_id': 'first', 'capabilities': _CAPS})
    return store, desktop


def _chat(store, sid, source, *, parent=None):
    """A routeless stored row (no route, no chat_id, no origin) with one message."""
    from gateway.owner_identity import host_owner_subject
    db = store._db
    if parent is not None:
        db.create_session(parent, source=CLIENT_LABEL, model='m-old', cwd=str(Path(db.db_path).parent))
    db.create_session(sid, source=source, model='m-old', cwd=str(Path(db.db_path).parent),
                      user_id=host_owner_subject())
    db.append_message(sid, role='user', content='hi from ' + sid)
    if parent is not None:
        db._write_sql('UPDATE sessions SET parent_session_id=? WHERE id=?', (parent, sid))
    return db.get_session(sid)


def _ref(conn, sid):
    return SessionRef(conn.authority.profile_id, sid)


def _lease(db, sid):
    db._write_sql('INSERT INTO session_turn_leases(conversation_id,holder,acquired_at,expires_at) VALUES(?,?,?,?)',
                  (sid, 'pid:4242', time.time(), time.time() + 300))


async def _submit(conn, sid):
    return await conn.dispatch({'id': 1, 'method': 'prompt.submit',
                                'params': {'session_id': sid, 'text': 'go on', 'submission_id': 's1'}})


async def _cold_submit_adopts(tmp_path, monkeypatch, source):
    store, desktop = await _owner(tmp_path, monkeypatch)
    db = store._db
    _chat(store, 'cold-chat', source)

    desktop.authority.authorize(desktop.actor, _ref(desktop, 'cold-chat'), 'session:submit')

    row = db.get_session('cold-chat')
    assert row['source'] == 'desktop'
    assert row['session_key'] and row['chat_id'], row
    with db._read_ctx() as conn:
        keys = {r[0] for r in conn.execute('SELECT key FROM state_meta WHERE key LIKE ?', ('%cold-chat%',))}
    assert {'gateway.local_legacy.v1:cold-chat', 'gateway.local_policy.v1:cold-chat'} <= keys, keys
    # The whole submit path then runs on that chat, never resumed first. A WS client is subscribed
    # to the chat it has open (resume makes this exact assignment); the subscription gate is not
    # what this test is about.
    desktop.subscriptions['cold-chat'] = 'sub-1'
    sent = await _submit(desktop, 'cold-chat')
    assert 'error' not in sent, sent


async def _cold_read_opens_writable(tmp_path, monkeypatch, source):
    from gateway.session_history_mount import is_history_mount
    store, desktop = await _owner(tmp_path, monkeypatch)
    db = store._db
    _chat(store, 'read-chat', source)

    desktop.authority.authorize(desktop.actor, _ref(desktop, 'read-chat'), 'session:read')

    assert not is_history_mount(desktop.authority.sessions['read-chat'])
    assert db.get_session('read-chat')['source'] == 'desktop'


@pytest.mark.asyncio
async def test_a_cold_session_submit_adopts_a_routeless_client_row(tmp_path, monkeypatch):
    """The submit capability, asked before any resume, adopts the chat instead of mounting it."""
    await _cold_submit_adopts(tmp_path, monkeypatch, CLIENT_LABEL)


@pytest.mark.asyncio
@webui_only
async def test_a_cold_session_submit_adopts_a_routeless_webui_row(tmp_path, monkeypatch):
    """The live case: the dashboard's own routeless chat adopts on a cold submit."""
    await _cold_submit_adopts(tmp_path, monkeypatch, 'webui')


@pytest.mark.asyncio
async def test_a_cold_metadata_read_opens_the_row_writable(tmp_path, monkeypatch):
    """``session:read`` behaves as before, and the row it opens is a chat, not a read-only mount."""
    await _cold_read_opens_writable(tmp_path, monkeypatch, CLIENT_LABEL)


@pytest.mark.asyncio
@webui_only
async def test_a_cold_metadata_read_opens_a_routeless_webui_row_writable(tmp_path, monkeypatch):
    """Same for the dashboard's own label, wherever t_33cf038a's change is present."""
    await _cold_read_opens_writable(tmp_path, monkeypatch, 'webui')


@pytest.mark.asyncio
async def test_a_routeless_row_under_a_live_turn_lease_keeps_its_refusal(tmp_path, monkeypatch):
    """A leased row is refused exactly as a resume refuses it: never adopted, never mounted."""
    store, desktop = await _owner(tmp_path, monkeypatch)
    _chat(store, 'leased-chat', CLIENT_LABEL)
    _lease(store._db, 'leased-chat')
    with pytest.raises(SessionTurnLeaseLostError) as resumed:
        await desktop.dispatch({'id': 1, 'method': 'session.resume', 'params': {'session_id': 'leased-chat'}})

    _chat(store, 'cold-leased', CLIENT_LABEL)
    _lease(store._db, 'cold-leased')
    with pytest.raises(SessionTurnLeaseLostError) as cold:
        desktop.authority.authorize(desktop.actor, _ref(desktop, 'cold-leased'), 'session:submit')

    assert str(cold.value).replace('cold-leased', 'X') == str(resumed.value).replace('leased-chat', 'X')
    assert store._db.get_session('cold-leased')['source'] == CLIENT_LABEL


@pytest.mark.asyncio
async def test_a_cold_read_on_a_leased_row_keeps_its_read_only_view(tmp_path, monkeypatch):
    """A lease fences transcript mutation, not a read: the cold read opens the mount it always did."""
    from gateway.session_history_mount import is_history_mount
    store, desktop = await _owner(tmp_path, monkeypatch)
    _chat(store, 'leased-read', CLIENT_LABEL)
    _lease(store._db, 'leased-read')

    desktop.authority.authorize(desktop.actor, _ref(desktop, 'leased-read'), 'session:read')

    assert is_history_mount(desktop.authority.sessions['leased-read'])
    assert store._db.get_session('leased-read')['source'] == CLIENT_LABEL


@pytest.mark.asyncio
async def test_a_second_writer_refusal_is_not_hidden_by_the_mount(tmp_path, monkeypatch):
    """``runtime_coordination_required`` keeps its refusal for a write; a read still gets the mount."""
    from gateway.session_history_mount import is_history_mount
    import gateway.session_local_migration as local
    store, desktop = await _owner(tmp_path, monkeypatch)
    _chat(store, 'held-chat', CLIENT_LABEL)

    def held(*_args, **_kwargs):
        raise RuntimeStoreError('runtime_coordination_required')
    monkeypatch.setattr(local, 'resolve_local_target', held)

    with pytest.raises(RuntimeStoreError, match='runtime_coordination_required'):
        desktop.authority.authorize(desktop.actor, _ref(desktop, 'held-chat'), 'session:submit')
    assert 'held-chat' not in desktop.authority.sessions  # never degraded to the read-only mount
    assert store._db.get_session('held-chat')['source'] == CLIENT_LABEL

    desktop.authority.authorize(desktop.actor, _ref(desktop, 'held-chat'), 'session:read')
    assert is_history_mount(desktop.authority.sessions['held-chat'])


@pytest.mark.asyncio
@pytest.mark.parametrize('source', ['cron', 'curator'])
async def test_a_routeless_row_that_is_not_client_history_stays_read_only(tmp_path, monkeypatch, source):
    """A cron or curator row is nobody's chat: it reads, and every other capability is refused."""
    from gateway.session_history_mount import is_history_mount
    store, desktop = await _owner(tmp_path, monkeypatch)
    _chat(store, 'auto-row', source)

    desktop.authority.authorize(desktop.actor, _ref(desktop, 'auto-row'), 'session:read')
    assert is_history_mount(desktop.authority.sessions['auto-row'])
    with pytest.raises(RuntimeStoreError, match='session_read_only'):
        desktop.authority.authorize(desktop.actor, _ref(desktop, 'auto-row'), 'session:submit')
    assert store._db.get_session('auto-row')['source'] == source


@pytest.mark.asyncio
async def test_a_subagent_transcript_is_never_client_history(tmp_path, monkeypatch):
    """A transcript with a ``parent_session_id`` is evidence: not adoptable, read-only on submit."""
    from gateway.session_history_mount import is_history_mount
    from hermes_state_local_migration import client_history_row, validate_legacy_row
    store, desktop = await _owner(tmp_path, monkeypatch)
    plain = _chat(store, 'plain-chat', CLIENT_LABEL)
    row = _chat(store, 'sub-chat', CLIENT_LABEL, parent='parent-chat')

    # The label alone IS client history: the parent stamp is what refuses the transcript.
    assert client_history_row(plain) is True
    assert client_history_row(row) is False
    with pytest.raises(RuntimeStoreError, match='not_found'):
        validate_legacy_row(store._db, row, subject=desktop.actor.subject)

    desktop.authority.authorize(desktop.actor, _ref(desktop, 'sub-chat'), 'session:read')
    assert is_history_mount(desktop.authority.sessions['sub-chat'])
    with pytest.raises(RuntimeStoreError, match='session_read_only'):
        desktop.authority.authorize(desktop.actor, _ref(desktop, 'sub-chat'), 'session:submit')
    assert store._db.get_session('sub-chat')['source'] == CLIENT_LABEL
    assert store._db.get_session('sub-chat')['parent_session_id'] == 'parent-chat'
