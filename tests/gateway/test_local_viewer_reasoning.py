"""Reasoning a local viewer's turn sends Ink and Desktop: the live stream by default, the final reasoning on
``message.complete``, and never the messaging platforms' reasoning block in the reply text."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from gateway.config import Platform
from tests.gateway.test_local_viewer_events_turn import _turn


@pytest.mark.parametrize(('source', 'config', 'flag', 'streams'), [
    ('tui', {}, None, True),  # unset: the launch's display.show_reasoning, on by default
    ('gui', {}, None, True),
    ('gui', {'display': {'show_reasoning': False}}, None, False),
    ('gui', {}, False, False),  # /reasoning hide or config.set from a viewer wins
    ('tui', {'display': {'show_reasoning': False}}, True, True),
])
def test_a_local_turn_streams_reasoning_unless_the_session_or_its_config_hides_it(tmp_path, source, config, flag,
                                                                                    streams):
    from gateway import local_viewer_events
    from gateway.session_policy import build_policy, policy_scope
    published = []
    live = SimpleNamespace(tool_progress_mode=None, **({} if flag is None else {'show_reasoning': flag}))
    turn = _turn(Platform.LOCAL, published, live=live)
    agent = SimpleNamespace(tool_gen_callback=None, _print_fn=None, thinking_callback=None,
                            reaction_callback=None, reasoning_callback=None)
    with policy_scope(build_policy({'source': source, 'cwd': str(tmp_path), 'model': 'fixture'}, config)):
        local_viewer_events.wire(turn, agent)
        if agent.reasoning_callback is not None:
            agent.reasoning_callback('step 1')
    assert (('reasoning.delta', {'text': 'step 1'}) in published) is streams


def test_completion_carries_the_turns_reasoning_only_to_local_viewers():
    from gateway.session_settlement_recovery import completion_payload
    from tui_gateway.contracts.registry import EVENTS
    row = {'admission_id': 'a1', 'request_id': 'r1'}
    local = completion_payload(row, {'outcome': 'completed'}, 'answer', {'result': {
        'local_viewer': True, 'last_reasoning': '  weigh both options  '}})
    assert local['reasoning'] == 'weigh both options'
    EVENTS['message.complete'].payload.model_validate({'text': 'answer', 'reasoning': local['reasoning']})
    messaging = completion_payload(row, {'outcome': 'completed'}, 'answer', {'result': {
        'last_reasoning': 'weigh both options'}})
    assert 'reasoning' not in messaging
    blank = completion_payload(row, {'outcome': 'completed'}, 'answer', {'result': {
        'local_viewer': True, 'last_reasoning': '   '}})
    assert 'reasoning' not in blank


def _wired(tmp_path, source, config, flag):
    from gateway import local_viewer_events
    from gateway.session_policy import build_policy, policy_scope
    published = []
    live = SimpleNamespace(tool_progress_mode=None, **({} if flag is None else {'show_reasoning': flag}))
    turn = _turn(Platform.LOCAL, published, live=live)
    agent = SimpleNamespace(tool_gen_callback=None, _print_fn=None, thinking_callback=None,
                            reaction_callback=None, reasoning_callback=None)
    scope = policy_scope(build_policy({'source': source, 'cwd': str(tmp_path), 'model': 'fixture'}, config))
    scope.__enter__()
    local_viewer_events.wire(turn, agent)
    return live, agent, published, scope


def test_reasoning_show_and_hide_apply_mid_turn_on_ink_and_desktop(tmp_path):
    live, agent, published, scope = _wired(tmp_path, 'gui', {}, None)
    try:
        agent.reasoning_callback('one')
        live.show_reasoning = False
        agent.reasoning_callback('two')
        live.show_reasoning = True
        agent.reasoning_callback('three')
    finally:
        scope.__exit__(None, None, None)
    assert [p['text'] for kind, p in published if kind == 'reasoning.delta'] == ['one', 'three']


@pytest.mark.parametrize(('config', 'streams'), [
    ({'display': {'show_reasoning': None}}, False),  # main read bool(display.get('show_reasoning', True))
    ({'display': 'not a mapping'}, True),  # a malformed display section counts as empty
])
def test_the_launch_default_reads_display_show_reasoning_as_main_did(tmp_path, config, streams):
    _live, agent, published, scope = _wired(tmp_path, 'tui', config, None)
    try:
        agent.reasoning_callback('step')
    finally:
        scope.__exit__(None, None, None)
    assert bool(published) is streams


@pytest.mark.parametrize('source', ['cli', 'acp'])
def test_other_local_launches_still_wait_for_the_session_flag(tmp_path, source):
    _live, agent, published, scope = _wired(tmp_path, source, {}, None)
    try:
        if agent.reasoning_callback is not None:
            agent.reasoning_callback('step')
    finally:
        scope.__exit__(None, None, None)
    assert published == []


def test_an_empty_reply_still_carries_the_turns_reasoning_to_local_viewers():
    from gateway.run_turn_runner import TurnRunner
    from gateway.session import SessionSource
    from gateway.session_results import execution_result
    from gateway.session_settlement_recovery import completion_payload
    from gateway.turn_context import TurnContext
    from tests.gateway.test_local_viewer_events_submit import _gateway_runner, _SteeredAgent
    ctx = TurnContext(
        source=SessionSource(platform=Platform.LOCAL, chat_id='sid', user_id='u'),
        message='hi', history=[], session_id='sid', session_key='key', user_config={},
        AIAgent=_SteeredAgent, resolve_display_setting=lambda *_a: False,
        _run_still_current=lambda: True, _hooks_ref=SimpleNamespace(loaded_hooks=False))
    captured = {}
    token = execution_result.set(captured)
    try:
        turn = TurnRunner(_gateway_runner(), ctx)
        authority = SimpleNamespace(sessions={'sid': SimpleNamespace(tool_progress_mode=None)}, profile_id='p',
                                    adopt_agent=lambda *a: None)
        turn._approval_owner = (authority, 'sid', 1)
        delivered = turn.run_sync()
    finally:
        execution_result.reset(token)
    complete = completion_payload({'admission_id': 'a1', 'request_id': 'r1'}, {'outcome': 'failed'},
                                  delivered['final_response'], captured)
    assert complete['reasoning'] == 'thought'


@pytest.mark.asyncio
@pytest.mark.parametrize(('display', 'value'), [('', '1'), ('display: {show_reasoning: false}\n', '0')])
async def test_config_get_reports_the_reasoning_display_the_session_streams(tmp_path, monkeypatch, display, value):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection
    home = tmp_path / 'home'
    home.mkdir()
    (home / 'config.yaml').write_text(
        'model: {default: m, provider: custom, base_url: "http://127.0.0.1:9/v1"}\nplatform_toolsets: {cli: []}\n'
        + display)
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setattr(run, '_hermes_home', home)
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             _cached_agent_for=lambda route: None)
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    authority = await initialize_session_authority(runner, profile_id=str(home), instance_id='first')
    connection = AuthorityConnection(authority, object(), {'user_id': 'human'})

    async def call(method, **params):
        reply = await connection.dispatch({'id': 1, 'method': method, 'params': params})
        assert 'result' in reply, reply
        return reply['result']

    try:
        sid = (await call('session.create', request_id='r', source='gui', cwd=str(tmp_path)))['session_id']
        assert (await call('config.get', session_id=sid, key='show_reasoning'))['value'] == value
    finally:
        await connection.close()


@pytest.mark.parametrize(('source', 'projected'), [('acp', True), ('gui', False)])
def test_only_acp_keeps_the_reasoning_block_in_its_reply_text(monkeypatch, source, projected):
    import gateway.run as gateway_run
    from gateway import session_setting_commands
    from gateway.run_turn_persistence import GatewayTurnPersistenceMixin
    monkeypatch.setattr(gateway_run, '_load_gateway_config', lambda: {'display': {'show_reasoning': True}})
    monkeypatch.setattr(session_setting_commands, 'local_policy', lambda runner, src: SimpleNamespace(source=source))
    reply = GatewayTurnPersistenceMixin()._hmwa_prepend_reasoning(
        {'last_reasoning': 'weigh it'}, 'final answer', SimpleNamespace(platform=Platform.LOCAL), False)
    assert ('weigh it' in reply) is projected
    assert reply.endswith('final answer')
