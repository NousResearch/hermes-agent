"""Local-viewer turn events the classic CLI renders through its own callbacks (CALLMAP W1).

On the session authority the agent runs in the gateway, so every callback main's HermesCLI
passed to its in-process AIAgent must reach the viewer as an execution event: the reply-box
segment boundary, the tool-generating line, every ``_print_fn`` status line and the tool
duration. A messaging route must not see any of it.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.run_turn_runner import TurnRunner


def _turn(platform, published, *, live=None):
    turn = object.__new__(TurnRunner)
    turn._ctx = SimpleNamespace(
        source=SimpleNamespace(platform=platform, chat_id='chat'),
        mute_notification_reply=False, scheduled_heartbeat=False,
        streaming_tts_consumer_holder=[None], stream_consumer_holder=[None],
        interim_assistant_messages_enabled=False, user_config={},
        resolve_display_setting=lambda *a: None, _run_still_current=lambda: True,
        _voice_ack_guild=[None], _native_slack_task_cards=False, progress_mode='all',
        session_id='s', progress_queue=None, _live_status_adapter=None, _live_status_mode='off',
        log_queue=None,
    )
    turn._runner = SimpleNamespace(config=None, _delivery_adapter_for=lambda source: None)
    live = live or SimpleNamespace(tool_progress_mode=None)
    authority = SimpleNamespace(sessions={'s': live})
    turn._approval_owner = (authority, 's', 1)
    turn._publish_execution = lambda kind, payload: published.append((kind, payload)) or True
    turn._publish_api_tool = lambda *a, **k: None
    return turn


def _delta_cb(turn):
    _consumer, delta_cb, _interim, _want = turn._setup_stream_consumer('cli')
    return delta_cb


def test_local_viewer_gets_the_segment_boundary_as_null_delta():
    published = []
    delta_cb = _delta_cb(_turn(Platform.LOCAL, published))
    delta_cb('hello')
    delta_cb(None)
    assert published == [('message.delta', {'text': 'hello'}), ('message.delta', {'text': None})]


def test_messaging_route_never_publishes_the_boundary():
    published = []
    delta_cb = _delta_cb(_turn(Platform.TELEGRAM, published))
    delta_cb('hello')
    delta_cb(None)
    assert published == [('message.delta', {'text': 'hello'})]


def test_turn_sinks_publish_and_restore_on_the_shared_agent():
    from gateway import local_viewer_events
    published = []
    turn = _turn(Platform.LOCAL, published)
    sentinel = object()
    agent = SimpleNamespace(tool_gen_callback=sentinel, _print_fn=None)
    previous = local_viewer_events.wire(turn, agent)
    agent.tool_gen_callback('terminal')
    agent._print_fn('  ✓ [1/2]', 'done')
    assert published == [('tool.generating', {'name': 'terminal'}),
                         ('agent.print', {'text': '  ✓ [1/2] done'})]
    local_viewer_events.restore(agent, previous)
    assert agent.tool_gen_callback is sentinel and agent._print_fn is None


def test_status_lines_the_agent_prints_reach_the_viewer():
    """``_safe_print`` (subagent completion, compaction, errors) routes through ``_print_fn``."""
    from agent.status_output import StatusOutputMixin
    from gateway import local_viewer_events
    published = []
    agent = type('A', (StatusOutputMixin,), {})()
    agent._print_fn = None
    local_viewer_events.wire(_turn(Platform.LOCAL, published), agent)
    agent._safe_print('🗜️ compacted')
    assert published == [('agent.print', {'text': '🗜️ compacted'})]


@pytest.mark.parametrize('platform', [Platform.LOCAL, Platform.TELEGRAM])
def test_tool_complete_carries_duration_only_for_local_viewers(platform):
    published = []
    turn = _turn(platform, published)
    turn.combined_tool_start_callback('c1', 'read_file', {'path': 'x'})
    turn.combined_tool_complete_callback('c1', 'read_file', {'path': 'x'}, '{"content": "hi"}')
    complete = next(payload for kind, payload in published if kind == 'tool.complete')
    if platform == Platform.LOCAL:
        assert isinstance(complete['duration_s'], float) and complete['duration_s'] >= 0
    else:
        assert 'duration_s' not in complete


def test_wire_contract_accepts_the_boundary_frame():
    from tui_gateway.contracts.registry import EVENTS
    EVENTS['message.delta'].payload.model_validate({'text': None})
    EVENTS['tool.generating'].payload.model_validate({'name': 'terminal'})
    EVENTS['agent.print'].payload.model_validate({'text': 'line'})
    EVENTS['tool.complete'].payload.model_validate({'tool_id': 'c', 'name': 'n', 'duration_s': 0.1})


def test_thinking_and_reaction_publish_and_reasoning_waits_for_the_session_flag():
    from gateway import local_viewer_events
    published = []
    live = SimpleNamespace(tool_progress_mode=None)
    turn = _turn(Platform.LOCAL, published, live=live)
    agent = SimpleNamespace(tool_gen_callback=None, _print_fn=None, thinking_callback=None,
                            reaction_callback=None, reasoning_callback=None)
    previous = local_viewer_events.wire(turn, agent)
    agent.thinking_callback('(◕‿◕) pondering...')
    agent.thinking_callback('')
    agent.reaction_callback('vibe')
    # No sink unless the viewer asked: a reasoning callback changes the provider request.
    assert agent.reasoning_callback is None
    local_viewer_events.restore(agent, previous)
    assert (agent.thinking_callback, agent.reaction_callback) == (None, None)
    live.show_reasoning = True
    local_viewer_events.wire(turn, agent)
    agent.reasoning_callback('step 1')
    assert published == [('thinking.delta', {'text': '(◕‿◕) pondering...'}), ('thinking.delta', {'text': ''}),
                         ('reaction', {'kind': 'vibe'}), ('reasoning.delta', {'text': 'step 1'})]


@pytest.mark.parametrize('classic', [True, False])
def test_progress_projects_reasoning_available_and_moa(monkeypatch, classic):
    from gateway import session_cli_turn
    monkeypatch.setattr(session_cli_turn, 'classic_cli_turn', lambda turn: classic)
    published = []
    live = SimpleNamespace(tool_progress_mode=None)
    turn = _turn(Platform.LOCAL, published, live=live)
    turn.progress_callback('reasoning.available', '_thinking', 'because', None)
    turn.progress_callback('moa.reference', 'gpt-x', 'ref answer', None, moa_index=1, moa_count=2)
    turn.progress_callback('moa.progress', 'gpt-x', None, None, moa_refs_done=1, moa_refs_total=2)
    turn.progress_callback('moa.phase', 'agg', None, None, moa_phase='aggregator', moa_refs_done=2, moa_refs_total=2)
    turn.progress_callback('moa.aggregating', 'agg', None, None, moa_ref_count=2)
    moa = [('moa.reference', {'label': 'gpt-x', 'text': 'ref answer', 'index': 1, 'count': 2}),
           ('moa.progress', {'label': 'gpt-x', 'refs_done': 1, 'refs_total': 2}),
           ('moa.phase', {'aggregator': 'agg', 'phase': 'aggregator', 'refs_done': 2, 'refs_total': 2}),
           ('moa.aggregating', {'aggregator': 'agg'})]
    # main's CLI renders MoA references whatever its reasoning display; reasoning.available only when shown.
    assert published == (moa if classic else [])
    published.clear()
    live.show_reasoning = True
    turn.progress_callback('reasoning.available', '_thinking', 'because', None)
    assert published == [('reasoning.available', {'text': 'because'})]
    from tui_gateway.contracts.registry import EVENTS
    for kind, payload in moa:
        EVENTS[kind].payload.model_validate(payload)


def test_messaging_progress_never_publishes_reasoning():
    published = []
    turn = _turn(Platform.TELEGRAM, published, live=SimpleNamespace(tool_progress_mode=None, show_reasoning=True))
    turn.progress_callback('reasoning.available', '_thinking', 'because', None)
    turn.progress_callback('moa.aggregating', 'agg', None, None)
    assert published == []


@pytest.mark.asyncio
async def test_show_reasoning_is_a_session_scoped_config_key(tmp_path, monkeypatch):
    from gateway.session_controls import AuthorityConnection
    from tests.gateway.test_surface_admission import _authority

    async def answer(event):
        return 'ok'
    authority = await _authority(tmp_path, monkeypatch, answer)
    connection = AuthorityConnection(authority, object(), {'user_id': 'owner'})
    try:
        await connection.dispatch({'id': 1, 'method': 'session.resume', 'params': {'session_id': 's'}})
        get = {'id': 2, 'method': 'config.get', 'params': {'session_id': 's', 'key': 'show_reasoning'}}
        assert (await connection.dispatch(get))['result']['value'] == '0'
        set_on = await connection.dispatch({'id': 3, 'method': 'config.set', 'params': {
            'session_id': 's', 'key': 'show_reasoning', 'value': 'on'}})
        assert set_on['result'] == {'key': 'show_reasoning', 'value': '1', 'scope': 'session'}, set_on
        assert authority.sessions['s'].show_reasoning is True
        bad = await connection.dispatch({'id': 4, 'method': 'config.set', 'params': {
            'session_id': 's', 'key': 'show_reasoning', 'value': 'loud'}})
        assert bad['error']['message'] == 'invalid_params', bad
    finally:
        await connection.close()


def test_a_local_write_ships_its_inline_diff_snapshotted_before_the_tool_ran(tmp_path):
    published = []
    turn = _turn(Platform.LOCAL, published)
    target = tmp_path / 'notes.txt'
    target.write_text('alpha\nbeta\n')
    args = {'path': str(target), 'content': 'alpha\ngamma\n'}
    turn.combined_tool_start_callback('w1', 'write_file', args)
    target.write_text(args['content'])  # the tool writes after the start callback
    turn.combined_tool_complete_callback('w1', 'write_file', args, '{"bytes_written": 12}')
    complete = next(payload for kind, payload in published if kind == 'tool.complete')
    diff = complete['inline_diff']
    assert isinstance(diff, str) and '-beta' in diff and '+gamma' in diff, diff
    from tui_gateway.contracts.registry import EVENTS
    EVENTS['tool.complete'].payload.model_validate({k: complete[k] for k in ('tool_id', 'name', 'duration_s', 'inline_diff')})


def test_the_todo_tool_publishes_the_full_snapshot():
    import json
    published = []
    turn = _turn(Platform.LOCAL, published)
    todos = [{'id': '1', 'content': 'ship', 'status': 'in_progress'}]
    turn.combined_tool_start_callback('t1', 'todo', {'todos': todos})
    turn.combined_tool_complete_callback('t1', 'todo', {'todos': todos}, json.dumps({'todos': todos, 'revision': 3}))
    kinds = [kind for kind, _ in published]
    assert kinds == ['tool.start', 'tool.complete', 'todo.updated']
    assert published[-1][1] == {'todos': todos, 'revision': 3}
    assert published[1][1]['inline_diff'] is None and published[1][1]['todos'] == todos
    from tui_gateway.contracts.registry import EVENTS
    EVENTS['todo.updated'].payload.model_validate(published[-1][1])


def test_messaging_tool_events_carry_no_diff_or_todo_frame():
    import json
    published = []
    turn = _turn(Platform.TELEGRAM, published)
    todos = [{'id': '1', 'content': 'ship', 'status': 'pending'}]
    turn.combined_tool_start_callback('t1', 'todo', {'todos': todos})
    turn.combined_tool_complete_callback('t1', 'todo', {'todos': todos}, json.dumps({'todos': todos, 'revision': 1}))
    assert [kind for kind, _ in published] == ['tool.start', 'tool.complete']
    assert 'inline_diff' not in published[1][1] and 'todos' not in published[1][1]


@pytest.mark.parametrize('platform', [Platform.LOCAL, Platform.TELEGRAM])
def test_prompts_carry_the_gateway_wait_window_only_for_local_viewers(monkeypatch, platform):
    from gateway.session_pending_controls import PendingControls
    import threading
    monkeypatch.setattr('tools.approval_context._get_approval_timeout', lambda: 90)
    monkeypatch.setattr('tools.clarify_gateway.get_clarify_timeout', lambda: 600)
    published = []
    events = SimpleNamespace(lock=threading.RLock(), publish=lambda sid, prompt, event_type: published.append((event_type, prompt)))
    controls = PendingControls(events)
    authority = SimpleNamespace(
        check_approval_generation=lambda *a: None,
        register_approval=lambda sid, gen, route, data, timeout_s=None: controls.register(sid, route, gen, data, timeout_s=timeout_s),
        register_clarify=lambda sid, gen, entry, timeout_s=None: controls.register_clarify(sid, gen, entry, timeout_s=timeout_s))
    turn = _turn(platform, [])
    turn._approval_owner = (authority, 's', 1)
    turn._render_approval_sync = lambda data: None
    turn._ctx.session_key = 'k'
    turn._approval_notify_sync({'request_id': 'a1', 'command': 'rm -rf build', 'description': 'delete'})
    entry = SimpleNamespace(clarify_id='c1', question='Which?', choices=['x'], multi_select=False)
    from gateway import local_viewer_events
    authority.register_clarify('s', 1, entry, timeout_s=local_viewer_events.prompt_timeout_s(turn, 'clarify'))
    approval, clarify = published[0][1], published[1][1]
    if platform == Platform.LOCAL:
        assert (approval['timeout_s'], clarify['timeout_s']) == (90, 600)
        from tui_gateway.contracts.canonical_sessions import CanonicalPendingPrompt
        CanonicalPendingPrompt.model_validate(approval)
        CanonicalPendingPrompt.model_validate(clarify)
    else:
        assert 'timeout_s' not in approval and 'timeout_s' not in clarify


@pytest.mark.asyncio
async def test_service_tier_is_a_session_scoped_fast_override(tmp_path, monkeypatch):
    """The classic CLI's /fast: config.set service_tier writes the session's tier override (what the
    next turn's agent reads), never config.yaml; an unknown tier is refused."""
    from gateway.session_controls import AuthorityConnection
    from gateway.session_state import SERVICE_TIER_UNSET
    from tests.gateway.test_surface_admission import _authority

    async def answer(event):
        return 'ok'
    authority = await _authority(tmp_path, monkeypatch, answer)
    # The fixture runner is a bare namespace: give it the real runner's per-session state methods.
    from types import MethodType
    from gateway.run import GatewayRunner
    for name in ('_sessions_map', '_session_state', '_peek_session_state', '_set_session_service_tier_override'):
        setattr(authority.runner, name, MethodType(getattr(GatewayRunner, name), authority.runner))
    connection = AuthorityConnection(authority, object(), {'user_id': 'owner'})
    try:
        await connection.dispatch({'id': 1, 'method': 'session.resume', 'params': {'session_id': 's'}})
        route = authority.sessions['s'].route

        def override():
            state = authority.runner._peek_session_state(route)
            return SERVICE_TIER_UNSET if state is None else state.conversation.service_tier_override
        res = await connection.dispatch({'id': 2, 'method': 'config.set', 'params': {
            'session_id': 's', 'key': 'service_tier', 'value': 'priority'}})
        assert res['result'] == {'key': 'service_tier', 'value': 'priority', 'scope': 'session'}, res
        assert override() == 'priority'
        await connection.dispatch({'id': 3, 'method': 'config.set', 'params': {
            'session_id': 's', 'key': 'service_tier', 'value': 'normal'}})
        assert override() is None
        bad = await connection.dispatch({'id': 4, 'method': 'config.set', 'params': {
            'session_id': 's', 'key': 'service_tier', 'value': 'warp'}})
        assert bad['error']['message'] == 'invalid_params', bad
    finally:
        await connection.close()
