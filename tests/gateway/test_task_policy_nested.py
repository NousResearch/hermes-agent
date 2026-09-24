"""Managed malformed sections cannot expose lower-priority native read/action grants."""
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from hermes_cli.config_effective import load_user_config_effective


def fixture_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


n = fixture_module('nested_decision_fixture', 'test_task_decisions_native.py')
f = fixture_module('nested_read_fixture', 'test_task_detail_native.py')
decision_rig = n.rig
read_rig = f.rig


def overlay(r, monkeypatch, value):
    root = r.home / 'managed'
    root.mkdir(exist_ok=True)
    monkeypatch.setenv('HERMES_MANAGED_DIR', str(root))
    path = root / 'config.yaml'
    path.write_text(yaml.safe_dump(value))
    return path


def policy(kind, shape):
    if shape.startswith('section-'):
        return {'kanban': {'section-null': None, 'section-list': [],
                           'section-scalar': 'invalid'}[shape]}
    return {'kanban': {kind + '_grants': {'null': None, 'scalar': 'invalid',
                                         'mapping': {'invalid': True},
                                         'invalid-items': [None, 'invalid', {}],
                                         'empty-deny': []}[shape]}}


SHAPES = ['section-null', 'section-list', 'section-scalar', 'null',
          'scalar', 'mapping', 'invalid-items', 'empty-deny']


@pytest.mark.asyncio
@pytest.mark.parametrize('warm', [False, True])
@pytest.mark.parametrize('shape', SHAPES)
async def test_managed_decision_denial_survives_malformed_edit(decision_rig, monkeypatch, shape, warm):
    r = decision_rig
    await n.tick(r, monkeypatch)
    before = n.readback(r)
    token = before[2][0]['token']
    path = overlay(r, monkeypatch, {'kanban': {'decision_grants': []}})
    await r.app.process_update(n.callback(r, token))
    assert n.readback(r) == before
    path.write_text(yaml.safe_dump(policy('decision', shape)))
    if warm:
        load_user_config_effective()
    # Use the existing native callback, not a restarted watcher: its unrelated startup
    # notification-settings reader assumes a mapping for list/scalar sections.
    service = r.manager._task_card_registration.decisions
    source = next(iter(service.bindings.values()))
    assert service._grant(source) is None
    assert service.controls(source, source.data['snapshot']) == ()
    await r.app.process_update(n.callback(r, token))
    assert n.readback(r) == before
    assert 'No change' in n.answers(r)[-1]


@pytest.mark.asyncio
@pytest.mark.parametrize('warm', [False, True])
@pytest.mark.parametrize('shape', SHAPES)
async def test_managed_read_denial_survives_malformed_edit(read_rig, monkeypatch, shape, warm):
    r = read_rig
    path = overlay(r, monkeypatch, {'kanban': {'read_grants': []}})
    assert (await f.http(r.app, headers=r.headers)).status == 403
    before = f.db_state(r)
    path.write_text(yaml.safe_dump(policy('read', shape)))
    if warm:
        load_user_config_effective()
    response = await f.http(r.app, headers=r.headers)
    assert response.status == 403 and json.loads(response.text) == {'error': 'unavailable'}
    assert f.db_state(r) == before


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['absent', 'empty', 'current', 'read-invalid'])
async def test_valid_decision_policy_and_read_independence(decision_rig, monkeypatch, mode):
    r = decision_rig
    cfg = yaml.safe_load((r.home / 'config.yaml').read_text())
    value = {'empty': {'kanban': {}}, 'current': {'kanban': cfg['kanban']},
             'read-invalid': {'kanban': {'read_grants': None}}}
    if mode != 'absent':
        overlay(r, monkeypatch, value[mode])
    load_user_config_effective()
    await n.tick(r, monkeypatch)
    token = n.readback(r)[2][0]['token']
    await r.app.process_update(n.callback(r, token))
    after = n.readback(r)
    assert after[0] == 'ready' and len(after[1]) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['absent', 'empty', 'current', 'decision-invalid'])
async def test_valid_read_policy_and_action_independence(read_rig, monkeypatch, mode):
    r = read_rig
    cfg = yaml.safe_load((r.home / 'config.yaml').read_text())
    value = {'empty': {'kanban': {}}, 'current': {'kanban': cfg['kanban']},
             'decision-invalid': {'kanban': {'decision_grants': None}}}
    if mode != 'absent':
        overlay(r, monkeypatch, value[mode])
    load_user_config_effective()
    before = f.db_state(r)
    assert (await f.http(r.app, headers=r.headers)).status == 200
    assert f.db_state(r) == before
