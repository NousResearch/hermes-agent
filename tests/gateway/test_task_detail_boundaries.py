"""Additional native read lifecycle, resource and scope boundaries; no listener."""
import importlib.util
import json
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

import pytest
pytest.importorskip(
    "hermes_telegram_experience",
    reason="optional external-plugin integration; exercised by plugin repository CI",
)

spec = importlib.util.spec_from_file_location('detail_fixture', Path(__file__).with_name('test_task_detail_native.py'))
f = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f)
rig = f.rig


@pytest.mark.asyncio
@pytest.mark.parametrize('case', ['host', 'fetch_site', 'duplicate_header', 'symlink', 'pinned', 'missing_credential', 'grant_in_plugin'])
async def test_additional_denials(rig, monkeypatch, case):
    r = rig
    headers = dict(r.headers)
    if case == 'host': headers['Host'] = 'foreign.example.test'
    if case == 'fetch_site': headers['Sec-Fetch-Site'] = 'cross-site'
    if case == 'duplicate_header':
        from multidict import CIMultiDict
        headers = CIMultiDict(headers)
        headers.add('X-Telegram-Init-Data', f.signed())
    if case == 'symlink':
        target = r.path.with_name('original.db')
        r.path.rename(target)
        r.path.symlink_to(target)
    if case == 'pinned': monkeypatch.setenv('HERMES_KANBAN_DB', str(r.path))
    if case == 'missing_credential': r.telegram.config.token = ''
    if case == 'grant_in_plugin':
        old = r.grants
        r.grants = []
        f.config(r, settings={'read_grants': old})
    before = f.db_state(r)
    response = await f.http(r.app, headers=headers)
    assert response.status == 403 and json.loads(response.text) == {'error': 'unavailable'}
    assert f.db_state(r) == before


@pytest.mark.asyncio
async def test_config_disable_unload_reenable_and_idempotent_wiring(rig):
    r = rig
    before = f.db_state(r)
    assert (await f.http(r.app, headers=r.headers)).status == 200
    routes = len(list(r.app.router.routes()))
    r.api._wire_plugin_handlers(r.app)
    assert len(list(r.app.router.routes())) == routes
    f.config(r, settings={'task_detail': False})
    assert (await f.http(r.app, headers=r.headers)).status == 403
    f.config(r)
    assert (await f.http(r.app, headers=r.headers)).status == 200
    r.manager.unload()
    assert (await f.http(r.app, headers=r.headers)).status == 403
    r.manager.discover_and_load(force=True)
    assert (await f.http(r.app, headers=r.headers)).status == 403
    new_app = r.api.create_http_application()
    new_app.freeze()
    assert (await f.http(new_app, headers=r.headers)).status == 200
    assert (await f.http(r.app, headers=r.headers)).status == 403
    assert f.db_state(r) == before


@pytest.mark.asyncio
async def test_two_homes_context_a_b_a_and_no_secondary_fallback(rig):
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    r = rig
    other = r.home/'profiles/other'
    other.mkdir(parents=True)
    (other/'config.yaml').write_text((r.home/'config.yaml').read_text())
    assert (await f.http(r.app, headers=r.headers)).status == 200
    scope = set_hermes_home_override(other)
    try:
        assert (await f.http(r.app, headers=r.headers)).status == 403
    finally:
        reset_hermes_home_override(scope)
    assert (await f.http(r.app, headers=r.headers)).status == 200
    # Host delivery lookup may resolve default even if primary ownership changed;
    # the read boundary must not borrow that credential.
    r.runner._primary_profile_name = 'other'
    assert (await f.http(r.app, headers=r.headers)).status == 403


@pytest.mark.asyncio
async def test_same_task_id_different_board_and_symlink_free_explicit_grant(rig):
    r = rig
    other_path = f.kb.kanban_db_path(board='other')
    with closing(f.kbc.connect(other_path)) as conn:
        other_id = f.kb.create_task(conn, title='Different board private title')
        conn.execute('UPDATE tasks SET id=? WHERE id=?', (r.tid, other_id))
        conn.execute('UPDATE task_events SET task_id=? WHERE task_id=?', (r.tid, other_id))
        conn.commit()
        incarnation = f.source.get_task_source(conn, r.tid).task_incarnation
    encoded = f.selector('default', 'other', r.tid, incarnation)
    headers = {**r.headers, 'X-Hermes-Task-Selector': encoded}
    assert (await f.http(r.app, headers=headers)).status == 403
    r.grants.append(dict(actor=42, bot_id=123456, profile='default', board='other', task_id=r.tid,
                         task_incarnation=incarnation, permissions=['read']))
    f.config(r)
    assert json.loads((await f.http(r.app, headers=headers)).text)['title'] == 'Different board private title'
    assert 'A synthetic task' in (await f.http(r.app, headers=r.headers)).text


@pytest.mark.asyncio
async def test_public_static_navigation_and_no_prefixed_data_route(rig):
    r = rig
    assert (await f.http(r.app, f.PREFIX, headers={'Sec-Fetch-Site': 'cross-site'})).status == 200
    response = await f.http(r.app, '/p/default'+f.PREFIX+'detail', headers=r.headers)
    assert response.status != 200 and 'A synthetic task' not in (response.text or '')


def test_missing_read_capability_registers_nothing():
    import hermes_telegram_experience as plugin
    calls = []
    settings = {'enabled': True, 'task_detail': True, 'scope': {
        'routes': [dict(profile='default', platform='telegram', chat_id='-100', thread_id='7')],
        'task_resources': [dict(board='default', task_id='t_12345678')],
    }}
    ctx = SimpleNamespace(live_todo_capability=2, register_live_todo=lambda *a, **kw: calls.append(a),
                          get_config=lambda key, default=None: settings.get(key, default))
    with pytest.raises(RuntimeError, match='task_read capability'):
        plugin.register(ctx)
    assert not calls


@pytest.mark.parametrize('missing', ['register_platform_handler', 'task_card_capability', 'task_decision_capability'])
def test_all_capabilities_checked_before_registration(missing):
    import hermes_telegram_experience as plugin
    calls = []
    settings = dict(enabled=True, task_detail=True, durable_cards=True, decisions=True,
                    public_origin=f.ORIGIN, bot_username='SyntheticTaskBot', app_short_name='taskdetail',
                    scope=dict(
                        routes=[dict(profile='default', platform='telegram', chat_id='-100', thread_id='7')],
                        task_resources=[dict(board='default', task_id='t_12345678')],
                    ))
    ctx = SimpleNamespace(**{name: 2 for name in ('live_todo_capability', 'task_read_capability', 'task_card_capability', 'task_decision_capability')},
                          **{name: lambda *a, **kw: calls.append((a, kw)) for name in ('register_live_todo', 'register_task_detail', 'register_task_cards', 'register_task_decisions', 'register_platform_handler')},
                          get_config=lambda key, default=None: settings.get(key, default))
    setattr(ctx, missing, None)
    with pytest.raises(RuntimeError, match='capability'):
        plugin.register(ctx)
    assert not calls


@pytest.mark.asyncio
@pytest.mark.parametrize('header', ['Host', 'Origin', 'X-Hermes-Task-Selector'])
async def test_ambiguous_headers_are_denied(rig, header):
    from multidict import CIMultiDict
    headers = CIMultiDict(rig.headers)
    headers.setdefault('Host', 'tasks.example.test')
    headers.add(header, headers[header])
    assert (await f.http(rig.app, headers=headers)).status == 403


@pytest.mark.asyncio
async def test_two_registered_profiles_same_canonical_id_and_distinct_credentials(rig):
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    from agent.secret_scope import set_multiplex_active
    import yaml
    r = rig
    other = r.home/'profiles/other'
    other.mkdir(parents=True)
    cfg = json.loads(json.dumps(r.config))
    for grant in cfg['kanban']['read_grants']:
        grant.update(profile='other', bot_id=654321)
    settings = cfg['plugins']['entries']['hermes-telegram-experience']['settings']
    for route in settings['scope']['routes']:
        route['profile'] = 'other'
    (other/'config.yaml').write_text(yaml.safe_dump(cfg))
    token = '654321:another-synthetic-token-for-slice-four'
    adapter = f.TelegramAdapter(f.PlatformConfig(enabled=True, token=token, typing_indicator=False))
    r.runner.config.multiplex_profiles = True
    r.runner._profile_adapters = {'other': {f.Platform.TELEGRAM: adapter}}
    before = f.db_state(r)
    set_multiplex_active(True)
    manager = api = None
    try:
        assert (await f.http(r.app, headers=r.headers)).status == 200
        scope = set_hermes_home_override(other)
        try:
            manager = f.get_plugin_manager()
            assert manager is not r.manager
            manager.discover_and_load(force=True)
            assert manager._task_read_registration.profile == 'other'
            api = f.APIServerAdapter(f.PlatformConfig(enabled=True, extra=dict(key='synthetic-api-key', cors_origins=[f.ORIGIN])))
            api.gateway_runner = r.runner
            app = api.create_http_application(); app.freeze()
            encoded = f.selector('other', 'default', r.tid, r.incarnation)
            headers = {**r.headers, 'X-Hermes-Task-Selector': encoded, 'X-Telegram-Init-Data': f.signed(token=token)}
            assert (await f.http(app, headers=headers)).status == 200
            assert (await f.http(app, headers=r.headers)).status == 403
            assert (await f.http(app, headers={**headers, 'X-Telegram-Init-Data': f.signed()})).status == 403
            assert (await f.http(r.app, headers=r.headers)).status == 403
            # Exercise the actual shared-delivery fallback, not a mocked verifier.
            r.runner._profile_adapters['other'] = {}
            r.runner.config.profile_routes = [SimpleNamespace(enabled=True, profile='other', bot_profile=None)]
            assert r.runner._authorization_adapter(f.Platform.TELEGRAM, 'other') is r.telegram
            cfg['kanban']['read_grants'][0]['bot_id'] = 123456
            (other/'config.yaml').write_text(yaml.safe_dump(cfg))
            assert (await f.http(app, headers={**headers, 'X-Telegram-Init-Data': f.signed()})).status == 403
        finally:
            if manager: manager.unload()
            if api: api._response_store.close()
            reset_hermes_home_override(scope)
        assert (await f.http(r.app, headers=r.headers)).status == 200
        assert f.db_state(r) == before
    finally:
        set_multiplex_active(False)


@pytest.mark.asyncio
async def test_core_auth_success_and_read_projection_does_not_initialize(rig, monkeypatch):
    r = rig
    # Native auth remains useful, not merely a permanent 401.
    response = await f.http(r.app, '/api/sessions', headers={'Authorization': 'Bearer synthetic-api-key-no-task-authority'})
    assert response.status == 200
    def forbidden(*args, **kwargs):
        pytest.fail('GET used the canonical write/initialization connection')
    monkeypatch.setattr(f.kbc, 'connect', forbidden)
    before = f.db_state(r)
    assert (await f.http(r.app, headers=r.headers)).status == 200
    assert f.db_state(r) == before


@pytest.mark.asyncio
async def test_malformed_policy_cannot_reuse_last_good_grant(rig):
    r = rig
    assert (await f.http(r.app, headers=r.headers)).status == 200
    (r.home/'config.yaml').write_text('kanban: [unterminated')
    response = await f.http(r.app, headers=r.headers)
    assert response.status == 403 and json.loads(response.text) == {'error': 'unavailable'}


@pytest.mark.asyncio
async def test_invalid_init_data_does_not_create_good_config_backup(rig):
    r = rig
    backups = r.home / 'backups' / 'config'
    backups.mkdir(parents=True, exist_ok=True)
    for path in backups.glob('config.yaml.good.*'):
        path.rename(path.with_name(path.name + '.retained'))
    f.config(r, review_synthetic_revision='changed before side-effect-free policy read')
    before = {path.name for path in backups.iterdir()}
    response = await f.http(r.app, headers={
        'X-Telegram-Init-Data': 'invalid',
        'X-Hermes-Task-Selector': r.encoded,
    })
    after = {path.name for path in backups.iterdir()}
    assert response.status == 403 and json.loads(response.text) == {'error': 'unavailable'}
    assert after == before
