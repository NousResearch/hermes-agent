"""Regression for #140099: old local receipts omit the served profile in origin."""
import json
import time

import pytest

from tests.gateway.test_session_authorities_multiplex import _reserve_homes, _runner

# How each variant ages the receipt once its origin lost ``profile`` (``missing`` keeps just that).
_AGED = {
    'missing': lambda receipt, root: None,
    'null': lambda receipt, root: receipt['entry']['origin'].update(profile=None),
    'foreign': lambda receipt, root: receipt['entry']['origin'].update(profile='beta'),
    'route': lambda receipt, root: receipt.update(route=receipt['route'].replace('agent:alpha:', 'agent:main:')),
    'home': lambda receipt, root: receipt.update(profile_id=str(root)),
    'user': lambda receipt, root: receipt['entry']['origin'].update(user_id='another'),
}


def _legacy_main(receipt, root):
    """A pre-multiplex standalone receipt: route, entry key and origin all under agent:main."""
    receipt['route'] = receipt['route'].replace('agent:alpha:', 'agent:main:')
    receipt['entry']['session_key'] = receipt['route']
    receipt['entry']['origin']['profile'] = 'default'


_AGED['legacy_main'] = _legacy_main


@pytest.mark.asyncio
@pytest.mark.parametrize('variant', ['current', 'missing', 'null', 'foreign', 'route', 'home', 'user',
                                     'legacy_main', 'legacy_main_ended'])
async def test_secondary_receipt_restores_without_losing_its_owner(tmp_path, monkeypatch, variant):
    from gateway import run
    from gateway.run import _profile_runtime_scope
    from gateway.run_runtime import initialize_gateway_runtime
    from gateway.runtime_ownership import process_ownership
    from gateway.session_contract import SessionActor
    from gateway.session_local import create_local_session
    from gateway.session_local_recovery import restore_local_session
    from hermes_state_local import POLICY_PREFIX, local_receipt

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
            ref = create_local_session(authority, actor, {'request_id': 'old-desktop', 'source': 'gui',
                'cwd': str(tmp_path), 'model': 'frozen', 'toolsets': []})
        receipt = local_receipt(authority.db, ref.session_id)
        original = json.loads(json.dumps(receipt))
        if variant != 'current':
            receipt['entry']['origin'].pop('profile')
            (_legacy_main if variant == 'legacy_main_ended' else _AGED[variant])(receipt, root)
            authority.db._execute_write(lambda conn: conn.execute(
                'UPDATE state_meta SET value=? WHERE key=?',
                (json.dumps(receipt), POLICY_PREFIX + ref.session_id)))
            if variant == 'legacy_main_ended':
                authority.db._write_sql(
                    "UPDATE sessions SET ended_at=?, end_reason='done' WHERE id=?",
                    (time.time(), ref.session_id))
        for owner in first.session_authorities:
            owner.db.close()

        second = _runner(root, homes)
        runners.append(second)
        await initialize_gateway_runtime(second)
        restored = second.session_authorities.for_home(home)
        if variant not in ('current', 'missing', 'legacy_main', 'legacy_main_ended'):
            from hermes_state_runtime import RuntimeStoreError
            assert ref.session_id not in restored.sessions
            with pytest.raises(RuntimeStoreError):
                restore_local_session(restored, ref.session_id)
            assert local_receipt(restored.db, ref.session_id) == receipt
            assert second.session_authorities.launch.db.get_session(ref.session_id) is None
            return
        if variant == 'legacy_main_ended':
            # Boot no longer restores ended receipts (#143547 LC-36); a Desktop resume restores on
            # first touch, which is the path an old chat actually takes.
            assert ref.session_id not in restored.sessions
        else:
            assert ref.session_id in restored.sessions, 'owned legacy receipt was paused on cold recovery'
        with _profile_runtime_scope(home, {}):
            assert restore_local_session(restored, ref.session_id) == ref
        live = restored.sessions[ref.session_id]
        assert live.source.profile == 'alpha'
        assert live.route == original['route']
        assert second.session_authorities.launch.db.get_session(ref.session_id) is None
        assert local_receipt(restored.db, ref.session_id)['policy'] == original['policy']
        if variant.startswith('legacy_main'):
            # The promotion is a read-time decision; the durable receipt stays as the old build
            # wrote it so a downgrade still owns the row.
            assert local_receipt(restored.db, ref.session_id)['route'].startswith('agent:main:local:')
    finally:
        for runner in runners:
            for owner in getattr(runner, 'session_authorities', ()):
                owner.db.close()
        for _, home in homes:
            process_ownership.release(home)
