"""Regression for #140099: old local receipts omit the served profile in origin."""
import json

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


@pytest.mark.asyncio
@pytest.mark.parametrize('variant', ['current', 'missing', 'null', 'foreign', 'route', 'home', 'user'])
async def test_secondary_receipt_restores_without_losing_its_owner(tmp_path, monkeypatch, variant):
    from gateway import run
    from gateway.run import _profile_runtime_scope
    from gateway.run_runtime import initialize_gateway_runtime
    from gateway.runtime_ownership import process_ownership
    from gateway.session_contract import Principal
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
        actor = Principal('owner', authority.profile_id,
                          frozenset({'session:create', 'session:read', 'session:control'}), 'socket')
        with _profile_runtime_scope(home, {}):
            ref = create_local_session(authority, actor, {'request_id': 'old-desktop', 'source': 'gui',
                'cwd': str(tmp_path), 'model': 'frozen', 'toolsets': []})
        receipt = local_receipt(authority.db, ref.session_id)
        original = json.loads(json.dumps(receipt))
        if variant != 'current':
            receipt['entry']['origin'].pop('profile')
            _AGED[variant](receipt, root)
            authority.db._execute_write(lambda conn: conn.execute(
                'UPDATE state_meta SET value=? WHERE key=?',
                (json.dumps(receipt), POLICY_PREFIX + ref.session_id)))
        for owner in first.session_authorities:
            owner.db.close()

        second = _runner(root, homes)
        runners.append(second)
        await initialize_gateway_runtime(second)
        restored = second.session_authorities.for_home(home)
        if variant not in ('current', 'missing'):
            from hermes_state_runtime import RuntimeStoreError
            assert ref.session_id not in restored.sessions
            with pytest.raises(RuntimeStoreError):
                restore_local_session(restored, ref.session_id)
            assert local_receipt(restored.db, ref.session_id) == receipt
            assert second.session_authorities.launch.db.get_session(ref.session_id) is None
            return
        assert ref.session_id in restored.sessions, 'owned legacy receipt was paused on cold recovery'
        with _profile_runtime_scope(home, {}):
            assert restore_local_session(restored, ref.session_id) == ref
        live = restored.sessions[ref.session_id]
        assert live.source.profile == 'alpha'
        assert live.route == original['route']
        assert second.session_authorities.launch.db.get_session(ref.session_id) is None
        assert local_receipt(restored.db, ref.session_id)['policy'] == original['policy']
    finally:
        for runner in runners:
            for owner in getattr(runner, 'session_authorities', ()):
                owner.db.close()
        for _, home in homes:
            process_ownership.release(home)
