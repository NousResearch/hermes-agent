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

from tests.gateway.test_session_authorities_multiplex import _reserve_homes, _runner


def _as_standalone_gateway_wrote_it(db, receipt):
    """Every copy ``commit_local_session`` wrote for *receipt* (the receipt, its ``sessions`` row and its
    routing row), spelled the way alpha's own gateway (multiplex off) spelled it: ``agent:main:`` keys and
    origins without ``profile``. These are exactly the copies a real pre-fold receipt has on disk."""
    from hermes_state_local import POLICY_PREFIX
    route = receipt['route']
    legacy = route.replace('agent:alpha:', 'agent:main:', 1)
    aged = json.loads(json.dumps(receipt))
    aged['route'] = aged['entry']['session_key'] = legacy
    aged['entry']['origin'].pop('profile')

    def write(conn):
        conn.execute('UPDATE state_meta SET value=? WHERE key=?', (json.dumps(aged), POLICY_PREFIX + receipt['session_id']))
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
    db._execute_write(write)
    return aged


@pytest.mark.asyncio
@pytest.mark.parametrize('variant', ['standalone', 'half_rekeyed', 'occupied'])
async def test_a_standalone_era_receipt_restores_under_the_served_route(tmp_path, monkeypatch, caplog, variant):
    from gateway import run
    from gateway.run import _profile_runtime_scope
    from gateway.run_runtime import initialize_gateway_runtime
    from gateway.runtime_ownership import process_ownership
    from gateway.session_contract import Principal
    from gateway.session_local import create_local_session
    from gateway.session_local_recovery import restore_local_session
    from hermes_state_local import local_receipt
    from hermes_state_runtime import RuntimeStoreError

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
        actor = Principal('owner', authority.profile_id,
                          frozenset({'session:create', 'session:read', 'session:control'}), 'socket')
        with _profile_runtime_scope(home, {}):
            ref = create_local_session(authority, actor, {'request_id': 'kanban-worker', 'source': 'gui',
                'cwd': str(tmp_path), 'model': 'frozen', 'toolsets': []})
        original = local_receipt(authority.db, ref.session_id)
        aged = _as_standalone_gateway_wrote_it(authority.db, original)
        if variant == 'half_rekeyed':
            # `hermes sessions repair-profiles --legacy-main rekey` before this fix: the rows alone.
            assert authority.db.rekey_legacy_main_sessions([ref.session_id], 'alpha') == 1
        if variant == 'occupied':
            # Something already holds the served route: not provably this receipt's, so nothing moves.
            authority.db._write_sql("INSERT INTO gateway_routing(scope,session_key,entry_json,updated_at) VALUES('',?,?,0)",
                                    (original['route'], json.dumps({**original['entry'], 'session_id': 'someone-else'})))
        for owner in first.session_authorities:
            owner.db.close()

        with caplog.at_level(logging.INFO, logger='gateway'):
            second = _runner(root, homes)
            runners.append(second)
            await initialize_gateway_runtime(second)
        restored = second.session_authorities.for_home(home)
        boot = [r.getMessage() for r in caplog.records if ref.session_id in r.getMessage()]
        launch_db = second.session_authorities.launch.db
        assert launch_db.get_session(ref.session_id) is None
        if variant == 'occupied':
            assert ref.session_id not in restored.sessions
            assert local_receipt(restored.db, ref.session_id) == aged
            assert any('route mismatch' in message for message in boot), boot
            with pytest.raises(RuntimeStoreError, match='storage_unavailable'):
                restore_local_session(restored, ref.session_id)
            return

        assert ref.session_id in restored.sessions, f'standalone-era receipt still refused: {boot}'
        assert not any('refused' in message or 'paused' in message for message in boot), boot
        assert any('agent:main:' in message and original['route'] in message for message in boot), boot
        with _profile_runtime_scope(home, {}):
            assert restore_local_session(restored, ref.session_id) == ref
        live = restored.sessions[ref.session_id]
        assert live.route == original['route'] and live.source.profile == 'alpha'
        # Every copy now carries the served spelling, and the legacy key is gone from the home's store.
        assert local_receipt(restored.db, ref.session_id) == original
        row = restored.db.get_session(ref.session_id)
        assert row['session_key'] == original['route']
        assert json.loads(row['origin_json'])['profile'] == 'alpha'
        with restored.db._read_ctx() as conn:
            keys = [r[0] for r in conn.execute('SELECT session_key FROM gateway_routing WHERE session_key LIKE ?',
                                               ('%' + ref.session_id,))]
        assert keys == [original['route']]
        assert restored.db.latest_conversation_boundary(original['route'], 'gui') == 1
        # The served route resolves to alpha's store, never the launch store's agent:main namespace.
        assert launch_db.get_session(ref.session_id) is None
    finally:
        for runner in runners:
            for owner in getattr(runner, 'session_authorities', ()):
                owner.db.close()
        for _, home in homes:
            process_ownership.release(home)
