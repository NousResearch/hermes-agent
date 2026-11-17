"""A served profile's LOCAL receipts from before the host multiplexed it restore under its namespace.

A named profile's own gateway runs with multiplexing off, so the sessions it creates (Desktop, TUI,
cron, Kanban workers) are keyed in the legacy ``agent:main:`` namespace. Once
``hermes gateway migrate --multiplex`` (or the implicit default) folds that home into the host, its
authority recomputes ``agent:<profile>:`` and every such receipt was refused as ``route mismatch`` on
every boot and resume. The receipt is bound to its home (``profile_id`` and the identity hash), so the
restore re-keys every copy the creation wrote to the served spelling instead; it never serves the
session under ``agent:main:``, which the multiplex store resolves to the LAUNCH profile's ``state.db``.
"""
import json
import logging

import pytest

from gateway.session_persistence import authority_persistence
from persistence import SessionId
from persistence.conversation.messages import ConversationPeer
from persistence_assembly import close_stores_under, open_profile_repair, profile_scope
from tests.fixtures.storage_holder import policy_key, raw_holder
from tests.gateway.test_session_authorities_multiplex import _reserve_homes, _runner


def _stored(store, sid):
    """*sid*'s creation receipt exactly as stored."""
    return json.loads(store.state_meta.get(policy_key(sid)))


def _as_standalone_gateway_wrote_it(store, receipt):
    """Every copy the local reservation wrote for *receipt* (the receipt, its ``sessions`` row and its
    routing row), spelled the way alpha's own gateway (multiplex off) spelled it: ``agent:main:`` keys and
    origins without ``profile``. These are exactly the copies a real pre-fold receipt has on disk."""
    route = receipt['route']
    legacy = route.replace('agent:alpha:', 'agent:main:', 1)
    aged = json.loads(json.dumps(receipt))
    aged['route'] = aged['entry']['session_key'] = legacy
    aged['entry']['origin'].pop('profile')

    def write(conn):
        conn.execute('UPDATE state_meta SET value=? WHERE key=?', (json.dumps(aged), policy_key(receipt['session_id'])))
        entry = json.loads(conn.execute("SELECT entry_json FROM gateway_routing WHERE scope='' AND session_key=?",
                                        (route,)).fetchone()[0])
        entry['session_key'] = legacy
        entry['origin'].pop('profile', None)
        conn.execute("UPDATE gateway_routing SET session_key=?, entry_json=? WHERE scope='' AND session_key=?",
                     (legacy, json.dumps(entry), route))
        for sid, origin_json in conn.execute('SELECT id, origin_json FROM sessions WHERE session_key=?', (route,)).fetchall():
            origin = json.loads(origin_json)
            origin.pop('profile', None)
            conn.execute('UPDATE sessions SET session_key=?, origin_json=? WHERE id=?', (legacy, json.dumps(origin), sid))
        # One `/new` boundary that gateway crossed on this route (its per-key reset generation).
        conn.execute("INSERT INTO conversation_generations(source, session_key, generation) VALUES ('gui', ?, 1)", (legacy,))
    raw_holder(store)._execute_write(write)
    return aged


def _close(runner):
    for owner in list(getattr(runner, 'session_authorities', ()) or ()):
        close_stores_under(owner.persistence.scope.profile_dir)


@pytest.mark.asyncio
@pytest.mark.parametrize('variant', ['standalone', 'ended', 'half_rekeyed', 'occupied'])
async def test_a_standalone_era_receipt_restores_under_the_served_route(tmp_path, monkeypatch, caplog, variant):
    from gateway import run
    from gateway.run import _profile_runtime_scope
    from gateway.run_runtime import initialize_gateway_runtime
    from gateway.runtime_ownership import process_ownership
    from gateway.session_contract import SessionActor
    from gateway.session_local import create_local_session
    from gateway.session_local_recovery import restore_local_session
    from persistence.core.errors import AuthorityRefusal

    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})
    root, homes = _reserve_homes(tmp_path, monkeypatch, names=('alpha',))
    process_ownership.reserve([home for _, home in homes])
    runners = []
    try:
        first = _runner(root, homes)
        runners.append(first)
        await initialize_gateway_runtime(first)
        home = homes[1][1]
        authority = first.session_authorities.for_home(home)
        actor = SessionActor('owner', authority.profile_id,
                             frozenset({'session:create', 'session:read', 'session:control'}), 'socket')
        with _profile_runtime_scope(home, {}):
            ref = create_local_session(authority, actor, {'request_id': 'kanban-worker', 'source': 'gui',
                'cwd': str(tmp_path), 'model': 'frozen', 'toolsets': []})
        store = authority_persistence(authority)
        original = _stored(store, ref.session_id)
        aged = _as_standalone_gateway_wrote_it(store, original)
        if variant == 'ended':
            # Closed before the fold: boot leaves it unrestored and its first touch restores it.
            store.sessions.end(SessionId(original['entry']['session_id']), 'closed')
        if variant == 'half_rekeyed':
            # `hermes sessions repair-profiles --legacy-main rekey` before this fix: the rows alone.
            assert open_profile_repair(profile_scope(home=home), read_only=False).rekey_legacy_main_sessions(
                [ref.session_id], 'alpha') == 1
        if variant == 'occupied':
            # Something already holds the served route: not provably this receipt's, so nothing moves.
            raw_holder(store)._write_sql(
                "INSERT INTO gateway_routing(scope,session_key,entry_json,updated_at) VALUES('',?,?,0)",
                (original['route'], json.dumps({**original['entry'], 'session_id': 'someone-else'})))
        generation = store.messages.latest_conversation_boundary(ConversationPeer(aged['route'], 'gui'))
        _close(first)

        with caplog.at_level(logging.INFO, logger='gateway'):
            second = _runner(root, homes)
            runners.append(second)
            await initialize_gateway_runtime(second)
        restored = second.session_authorities.for_home(home)
        store = authority_persistence(restored)
        boot = [r.getMessage() for r in caplog.records if ref.session_id in r.getMessage()]
        launch = authority_persistence(second.session_authorities.launch)
        assert launch.sessions.get(ref.session_id) is None
        if variant == 'occupied':
            assert ref.session_id not in restored.sessions
            assert _stored(store, ref.session_id) == aged
            assert any('route mismatch' in message for message in boot), boot
            with pytest.raises(AuthorityRefusal, match='storage_unavailable'):
                restore_local_session(restored, ref.session_id)
            return

        assert not any('refused' in message or 'paused' in message for message in boot), boot
        assert any('agent:main:' in message and original['route'] in message for message in boot), boot
        if variant == 'ended':
            assert ref.session_id not in restored.sessions
        else:
            assert ref.session_id in restored.sessions, f'standalone-era receipt still refused: {boot}'
        with _profile_runtime_scope(home, {}):
            assert restore_local_session(restored, ref.session_id) == ref
        live = restored.sessions[ref.session_id]
        assert live.route == original['route'] and live.source.profile == 'alpha'
        # Every copy now carries the served spelling, and the legacy key is gone from the home's store.
        assert _stored(store, ref.session_id) == original
        with raw_holder(store)._read_ctx() as conn:
            row = conn.execute('SELECT session_key, origin_json FROM sessions WHERE id=?', (ref.session_id,)).fetchone()
            keys = [r[0] for r in conn.execute('SELECT session_key FROM gateway_routing WHERE session_key LIKE ?',
                                               ('%' + ref.session_id,))]
        assert row[0] == original['route']
        assert json.loads(row[1])['profile'] == 'alpha'
        assert keys == [original['route']]
        # The route's reset generation moved with it.
        assert generation is not None
        assert store.messages.latest_conversation_boundary(ConversationPeer(original['route'], 'gui')) == generation
        assert store.messages.latest_conversation_boundary(ConversationPeer(aged['route'], 'gui')) is None
        # The served route resolves to alpha's store, never the launch store's agent:main namespace.
        assert launch.sessions.get(ref.session_id) is None
    finally:
        for runner in runners:
            _close(runner)
        for _, home in homes:
            process_ownership.release(home)


@pytest.mark.asyncio
async def test_a_multiplex_era_receipt_restores_under_the_standalone_route(tmp_path, monkeypatch, caplog):
    """The reverse switch (``gateway.standalone: true``): the profile's own gateway serves the receipts the
    multiplex host keyed ``agent:alpha:``, re-keyed to its ``agent:main:`` spelling with the profile dropped
    from every origin."""
    from types import SimpleNamespace

    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.run import _profile_runtime_scope
    from gateway.run_runtime import initialize_gateway_runtime
    from gateway.runtime_ownership import process_ownership
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_contract import SessionActor
    from gateway.session_local import create_local_session
    from gateway.session_local_recovery import restore_local_session
    from persistence_assembly import open_persistence

    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})
    root, homes = _reserve_homes(tmp_path, monkeypatch, names=('alpha',))
    home = homes[1][1]
    process_ownership.reserve([home for _, home in homes])
    try:
        host = _runner(root, homes)
        await initialize_gateway_runtime(host)
        served = host.session_authorities.for_home(home)
        actor = SessionActor('owner', served.profile_id, frozenset({'session:create', 'session:read'}), 'socket')
        with _profile_runtime_scope(home, {}):
            ref = create_local_session(served, actor, {'request_id': 'tui', 'source': 'gui',
                'cwd': str(tmp_path), 'model': 'frozen', 'toolsets': []})
        multiplexed = _stored(authority_persistence(served), ref.session_id)
        assert multiplexed['route'].startswith('agent:alpha:') and multiplexed['entry']['origin']['profile'] == 'alpha'
        _close(host)
    finally:
        for _, reserved_home in homes:
            process_ownership.release(reserved_home)

    monkeypatch.setenv('HERMES_HOME', str(home))  # alpha's own gateway
    store = SessionStore(tmp_path / 'standalone-sessions', GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store.persistence, adapters={}, _draining=False)
    persistence = open_persistence(profile_scope(home=home))
    try:
        with caplog.at_level(logging.INFO, logger='gateway'):
            standalone = await initialize_session_authority(runner, profile_id=served.profile_id,
                                                            instance_id='standalone', persistence=persistence)
        boot = [r.getMessage() for r in caplog.records if ref.session_id in r.getMessage()]
        legacy = 'agent:main:local:dm:' + ref.session_id
        assert not any('refused' in message or 'paused' in message for message in boot), boot
        assert any(multiplexed['route'] in message and legacy in message for message in boot), boot
        assert ref.session_id in standalone.sessions
        assert restore_local_session(standalone, ref.session_id) == ref
        receipt = _stored(persistence, ref.session_id)
        assert (receipt['route'], receipt['entry']['session_key']) == (legacy, legacy)
        assert 'profile' not in receipt['entry']['origin']
        with raw_holder(persistence)._read_ctx() as conn:
            row = conn.execute('SELECT session_key, origin_json FROM sessions WHERE id=?', (ref.session_id,)).fetchone()
        assert row[0] == legacy and 'profile' not in json.loads(row[1])
    finally:
        persistence.close()
        close_stores_under(home)
