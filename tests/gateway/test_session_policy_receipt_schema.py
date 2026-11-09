"""A stored local receipt written by another build still restores (#151667).

``restore_policy`` built ``LocalSessionPolicy(**data)`` from the stored dict verbatim, so a receipt
carrying one key this build's dataclass does not define (``fallback_providers``, written by an
earlier build) raised ``TypeError`` and was answered ``storage_unavailable`` on every restart: the
session never came back and every Bot Mode delivery into it failed. Unknown keys are now dropped
(named in the log, never their values: a receipt holds routes and credentials), while a receipt
that is invalid for THIS build still refuses.
"""
import json
import logging
from types import SimpleNamespace

import pytest

from gateway.session_policy import restore_policy
from hermes_state_runtime import RuntimeStoreError


async def _authority(tmp_path, monkeypatch, instance_id):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             _cached_agent_for=lambda route: None)
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    return await initialize_session_authority(runner, profile_id=str(tmp_path / 'fixture'),
                                              instance_id=instance_id)


@pytest.mark.asyncio
async def test_a_receipt_carrying_a_key_this_build_dropped_restores_on_restart(tmp_path, monkeypatch, caplog):
    from gateway.session_controls import AuthorityConnection
    from gateway.session_local import create_local_session
    from gateway.session_local_recovery import restore_local_session
    from hermes_state_local import POLICY_PREFIX, local_receipt
    from gateway import session_policy
    monkeypatch.setattr(session_policy, '_ADOPTED_KEY_SETS', set(), raising=False)

    first = await _authority(tmp_path, monkeypatch, 'first')
    viewer = AuthorityConnection(first, object(), {'user_id': 'human', 'profile_id': str(tmp_path / 'fixture')})
    ref = create_local_session(first, viewer.actor, {'request_id': 'r', 'source': 'cli',
                                                     'cwd': str(tmp_path), 'model': 'm', 'toolsets': []})
    receipt = local_receipt(first.db, ref.session_id)
    # An earlier build's receipt: an extra policy key, plus extra entry/origin keys (those already passed).
    receipt['policy']['fallback_providers'] = [{'base_url': 'https://secret-route.invalid', 'api_key': 'sk-x'}]
    receipt['entry']['future_entry_field'] = 1
    receipt['entry']['origin']['future_origin_field'] = 'x'
    first.db._write_sql('UPDATE state_meta SET value=? WHERE key=?',
                        (json.dumps(receipt), POLICY_PREFIX + ref.session_id))
    first.db.close()

    with caplog.at_level(logging.WARNING):
        second = await _authority(tmp_path, monkeypatch, 'second')
        assert ref.session_id in second.sessions, 'cold recovery refused the receipt'
        assert restore_local_session(second, ref.session_id) == ref
    lines = [record.getMessage() for record in caplog.records]
    assert not any('receipt refused' in line for line in lines), lines
    assert any('fallback_providers' in line for line in lines), lines
    assert not any('secret-route' in line or 'sk-x' in line for line in lines), lines
    second.db.close()


def test_unknown_keys_are_dropped_but_an_invalid_receipt_still_refuses(tmp_path):
    good = {'source': 'tui', 'platform': 'tui', 'cwd': str(tmp_path), 'model': 'fixture',
            'toolsets': ['terminal'], 'config_json': '{}', 'request_json': '{}',
            'terminal_json': json.dumps({'TERMINAL_CWD': str(tmp_path)})}
    policy = restore_policy({**good, 'fallback_providers': [], 'some_future_field': 1})
    assert (policy.source, policy.cwd, policy.model) == ('tui', str(tmp_path), 'fixture')
    assert not hasattr(policy, 'fallback_providers')
    missing = {k: v for k, v in good.items() if k != 'model'}
    for invalid in ({**missing, 'model_name': 'fixture'},  # a required field missing, renamed or not
                    {**good, 'fallback_providers': [], 'toolsets': 'terminal'},  # a known field mistyped
                    {**good, 'safe_mode': 'yes'}, None, [], 'x'):
        with pytest.raises(RuntimeStoreError) as caught:
            restore_policy(invalid)
        assert caught.value.reason == 'storage_unavailable', invalid
