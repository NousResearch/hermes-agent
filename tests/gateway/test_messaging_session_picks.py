"""Desktop/TUI model and reasoning picks on a messaging chat (#141152).

``session.mutate`` ``model`` / ``reasoning`` on a Telegram chat a viewer opened was refused with
``runtime_coordination_required`` (only LOCAL sessions have a frozen policy), and the reasoning pill
never got a level. The pick is the chat's own ``/model`` / ``/reasoning`` override, so the chat's next
turn runs it and every viewer reads it back, also when the chat was compressed and the viewer named
the compression segment. Real GatewayRunner, SessionAuthority and SQLite store.
"""
import pytest

from gateway.run import GatewayRunner


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))


async def _compressed_chat(tmp_path):
    """A Telegram chat compressed once; the viewer resumes it by its segment id, as history pickers do."""
    import json
    from gateway.config import GatewayConfig, Platform
    from gateway.session import SessionSource
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection
    runner = GatewayRunner(GatewayConfig())
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='owner')
    source = SessionSource(platform=Platform.TELEGRAM, chat_id='4242', user_id='u-4242', chat_type='dm')
    root = authority.register(source).session_id
    route = runner.session_store._generate_session_key(source)
    db = authority.db
    db.append_message(root, 'user', 'first')
    db.publish_compression_child(parent_session_id=root, child_session_id='seg-1', source='telegram',
                                 messages=[{'role': 'user', 'content': 'summary'}], require_compression_lease=False)
    assert runner.session_store.advance_compression_session(route, root, 'seg-1') is not None
    db.record_gateway_session_peer('seg-1', source='telegram', user_id=source.user_id, session_key=route,
                                   chat_id=source.chat_id, chat_type='dm', origin_json=json.dumps(source.to_dict()))
    # The segment's own fences moved on (its turns ran there); the viewer holds the mount's.
    db._execute_write(lambda conn: conn.execute(
        "UPDATE sessions SET runtime_revision=7, runtime_generation=3 WHERE id='seg-1'"))
    viewer = AuthorityConnection(authority, object(), {
        'provider': 'local', 'user_id': 'viewer', 'profile_id': 'default', 'instance_id': 'owner',
        'capabilities': ['session:read', 'session:control']})
    resumed = await viewer.dispatch({'id': 1, 'method': 'session.resume', 'params': {'session_id': 'seg-1'}})
    assert resumed.get('result', {}).get('session_id') == root, resumed
    return runner, source, route, viewer, root, resumed['result']


def _mutate(sid, fences, operation, payload, request_id):
    return {'id': request_id, 'method': 'session.mutate', 'params': {
        'session_id': sid, 'request_id': request_id, 'operation': operation, 'payload': payload,
        'expected_revision': fences['revision'], 'expected_generation': fences['execution_generation']}}


@pytest.mark.asyncio
async def test_a_reasoning_pick_on_a_compressed_messaging_chat_is_the_chats_next_turn_level(tmp_path):
    runner, source, route, viewer, root, fences = await _compressed_chat(tmp_path)
    try:
        before = await viewer.dispatch({'id': 2, 'method': 'session.info', 'params': {'session_id': root}})
        assert isinstance(before.get('result', {}).get('reasoning_effort'), str), before  # the pill resolves

        picked = await viewer.dispatch(_mutate('seg-1', fences, 'reasoning', {'effort': 'high'}, 'pick-high'))

        assert picked.get('result', {}).get('effort') == 'high', picked
        level = runner._resolve_session_reasoning_config(source=source, session_key=route, model='')
        assert level and level.get('effort') == 'high', level
        info = await viewer.dispatch({'id': 3, 'method': 'session.info', 'params': {'session_id': root}})
        assert info['result']['reasoning_effort'] == 'high', info
    finally:
        await viewer.close()
        runner.session_store.close_all_db_handles()


@pytest.mark.asyncio
async def test_a_model_pick_on_a_messaging_chat_is_the_chats_next_turn_model(tmp_path, monkeypatch):
    from hermes_cli import model_selection_guards, model_switch
    monkeypatch.setattr(model_switch, 'switch_model', lambda **k: model_switch.ModelSwitchResult(
        success=True, new_model=k['raw_input'], target_provider='custom', base_url='http://127.0.0.1:9/v1'))
    monkeypatch.setattr(model_selection_guards, 'selection_warnings', lambda *a, **k: [])
    runner, source, route, viewer, root, fences = await _compressed_chat(tmp_path)
    try:
        picked = await viewer.dispatch(_mutate(root, fences, 'model', {'model': 'picked', 'reasoning': 'low'}, 'pick'))

        assert picked.get('result', {}).get('model') == 'picked', picked
        assert runner._session_model_override(route)['model'] == 'picked'
        assert runner.session_store.get_model_override(route)['model'] == 'picked'  # survives a restart
        level = runner._resolve_session_reasoning_config(source=source, session_key=route, model='picked')
        assert level and level.get('effort') == 'low', level
        info = await viewer.dispatch({'id': 3, 'method': 'session.info', 'params': {'session_id': root}})
        assert (info['result']['model'], info['result']['reasoning_effort']) == ('picked', 'low'), info
    finally:
        await viewer.close()
        runner.session_store.close_all_db_handles()
