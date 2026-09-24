"""Native current-policy fencing, including effective overlay recovery and commit boundary."""
import asyncio
import importlib.util
from pathlib import Path
import pytest
import yaml
spec = importlib.util.spec_from_file_location('policy_decision_fixture', Path(__file__).with_name('test_task_decisions_native.py'))
assert spec is not None and spec.loader is not None
n = importlib.util.module_from_spec(spec)
spec.loader.exec_module(n)
from hermes_cli.config_effective import load_user_config_effective
rig = n.rig


@pytest.mark.asyncio
@pytest.mark.parametrize('layer', ['user-null', 'user-list', 'managed-malformed', 'managed-list', 'managed-null'])
async def test_invalid_policy_denies_and_repair_requires_new_pending_control(rig, monkeypatch, layer):
    r = rig
    if layer.startswith('managed'):
        managed = r.home / 'managed'
        managed.mkdir()
        monkeypatch.setenv('HERMES_MANAGED_DIR', str(managed))
        policy = managed / 'config.yaml'
        policy.write_text('kanban: {}\n')
    else:
        policy = r.home / 'config.yaml'
    await n.tick(r, monkeypatch)
    before = n.readback(r)
    token = before[2][0]['token']
    valid = policy.read_text()
    if layer.startswith('user'):
        cfg = yaml.safe_load(valid)
        cfg['kanban'] = None if layer == 'user-null' else ['invalid']
        broken = yaml.safe_dump(cfg)
    else:
        broken = {'managed-malformed': 'kanban: [unterminated',
                  'managed-list': '- invalid\n', 'managed-null': 'kanban: null\n'}[layer]
    policy.write_text(broken)
    load_user_config_effective()  # ordinary recovery must not prime authorization
    service = r.manager._task_card_registration.decisions
    source = next(iter(service.bindings.values()))
    assert service.controls(source, source.data['snapshot']) == ()
    await r.app.process_update(n.callback(r, token))
    assert n.readback(r) == before and 'No change' in n.answers(r)[-1]
    policy.write_text(valid)
    await r.app.process_update(n.callback(r, token))
    assert n.readback(r) == before
    await n.tick(r, monkeypatch)
    fresh = next(x['token'] for x in n.readback(r)[2] if x['token'] != token)
    await r.app.process_update(n.callback(r, fresh))
    assert n.readback(r)[0] == 'ready' and len(n.readback(r)[1]) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('boundary', ['await', 'precommit'])
async def test_policy_invalidation_rolls_back_all_canonical_effects(rig, monkeypatch, boundary):
    r = rig
    await n.tick(r, monkeypatch)
    before = n.readback(r)
    token = before[2][0]['token']
    def revoke():
        (r.home / 'config.yaml').write_text('kanban: [unterminated')
    if boundary == 'await':
        service = r.manager._task_card_registration.decisions
        await service.mutation_lock.acquire()
        pending = asyncio.create_task(r.app.process_update(n.callback(r, token)))
        await asyncio.sleep(0)
        revoke()
        service.mutation_lock.release()
        await pending
    else:
        from hermes_cli import kanban_db_actions as actions
        real = actions.record_action_outcome
        def boundary_write(*args, **kwargs):
            result = real(*args, **kwargs)
            revoke()  # after transaction-local mutation, before the commit guard
            return result
        monkeypatch.setattr(actions, 'record_action_outcome', boundary_write)
        await r.app.process_update(n.callback(r, token))
    assert n.readback(r) == before
    assert 'No change' in n.answers(r)[-1] and 'unterminated' not in n.answers(r)[-1]
