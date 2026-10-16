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


def test_completion_payload_forwards_billing_failure_and_response_flags():
    """TUI-6: ``message.complete`` carries ``billing``, ``failure_reason``, ``warning``, ``response_previewed`` and ``response_transformed`` from the turn result on local-viewer turns only."""
    from agent.billing_links import build_billing_block
    from gateway.session import SessionSource
    from gateway.session_results import execution_result
    from gateway.session_settlement_recovery import completion_payload
    from gateway.turn_context import TurnContext
    from tests.gateway.test_local_viewer_events_submit import _gateway_runner, _SteeredAgent
    from tui_gateway.contracts.registry import EVENTS

    billing = build_billing_block(
        provider='openrouter', base_url='https://openrouter.ai/api/v1', model='gpt-x', message='Out of credits'
    ).to_dict()
    interim_probes = []

    class _ViewerTurnAgent(_SteeredAgent):
        def _interim_text_was_delivered(self, text):
            interim_probes.append(text)
            return True

        def run_conversation(self, message, **kwargs):
            out = super().run_conversation(message, **kwargs)
            return {**out, 'final_response': 'partial answer', 'billing_block': billing,
                    'warning': 'History changed during this turn'}

    runner = _gateway_runner()
    live = SimpleNamespace(tool_progress_mode=None)
    authority = SimpleNamespace(sessions={'sid': live}, profile_id='p', adopt_agent=lambda *a: None)
    ctx = TurnContext(
        source=SessionSource(platform=Platform.LOCAL, chat_id='sid', user_id='u'),
        message='hi', history=[], session_id='sid', session_key='key', user_config={},
        AIAgent=_ViewerTurnAgent, resolve_display_setting=lambda *_a: False,
        _run_still_current=lambda: True, _hooks_ref=SimpleNamespace(loaded_hooks=False))
    captured = {}
    token = execution_result.set(captured)
    try:
        turn = TurnRunner(runner, ctx)
        turn._approval_owner = (authority, 'sid', 1)
        delivered = turn.run_sync()
    finally:
        execution_result.reset(token)

    assert interim_probes == ['partial answer']
    assert captured['result']['billing_block'] == billing
    assert captured['result']['warning'] == 'History changed during this turn'
    assert captured['result']['response_previewed'] is True
    complete = completion_payload(
        {'admission_id': 'a1', 'request_id': 'r1'}, {'outcome': 'failed'}, delivered['final_response'], captured)
    assert complete['billing'] == billing
    assert complete['failure_reason'] == 'billing'
    assert complete['warning'] == 'History changed during this turn'
    assert complete['response_previewed'] is True
    assert complete['response_transformed'] is True
    EVENTS['message.complete'].payload.model_validate(
        {k: v for k, v in complete.items() if k not in ('content', 'admission_id', 'outcome')})

    # Non-viewer (messaging) turns do not probe _interim_text_was_delivered, attach viewer_meta, or emit TUI-6 completion fields.
    interim_probes.clear()
    msg_ctx = TurnContext(
        source=SessionSource(platform=Platform.TELEGRAM, chat_id='sid', user_id='u'),
        message='hi', history=[], session_id='sid', session_key='key', user_config={},
        AIAgent=_ViewerTurnAgent, resolve_display_setting=lambda *_a: False,
        _run_still_current=lambda: True, _hooks_ref=SimpleNamespace(loaded_hooks=False))
    msg_captured = {}
    token = execution_result.set(msg_captured)
    try:
        msg_delivered = TurnRunner(_gateway_runner(), msg_ctx).run_sync()
    finally:
        execution_result.reset(token)
    assert interim_probes == []
    assert 'billing_block' not in msg_captured['result'] and 'warning' not in msg_captured['result']
    assert msg_captured['result']['response_previewed'] is False
    msg_complete = completion_payload(
        {'admission_id': 'a2', 'request_id': 'r2'}, {'outcome': 'failed'}, msg_delivered['final_response'], msg_captured)
    for viewer_key in ('billing', 'failure_reason', 'warning', 'response_previewed', 'response_transformed'):
        assert viewer_key not in msg_complete


def test_progress_publishes_subagent_lifecycle_events_for_local_viewers():
    """TUI-4: ``_ChildProgressRelay`` events reach the viewer stream on ``Platform.LOCAL`` (including registered background children descending from the spawning turn's agent after the turn settles, without borrowing a later turn's execution identity) and never on messaging platforms."""
    import weakref
    from gateway.session_events import SessionEvents
    from tools.delegate_tool_progress import _build_child_progress_callback
    from tools.delegate_tool_registry import _register_subagent, _unregister_subagent
    from tui_gateway.contracts.registry import EVENTS

    class _Parent:
        pass

    published = []
    live = SimpleNamespace(tool_progress_mode=None, show_reasoning=False)
    turn = _turn(Platform.LOCAL, published, live=live)
    parent = _Parent()
    parent._delegate_spinner = None
    parent.tool_progress_callback = turn.progress_callback
    turn._ctx.agent_holder = [parent]
    ref = {'session_id': 'child-sess-1', 'delegation_id': 'deleg_12345678'}
    relay = _build_child_progress_callback(
        0, 'inspect viewer payloads', parent, 2,
        subagent_id='sa-0-abc12345', parent_id='sa-root', depth=1, model='fake-model',
        toolsets=['terminal'], session_ref=ref)

    relay('subagent.spawn_requested', preview='inspect viewer payloads')
    relay('subagent.start')
    relay('tool.started', 'read_file', 'README.md', {'path': 'README.md'})
    relay('reasoning.available', preview='hidden chain of thought')
    live.show_reasoning = True
    relay('reasoning.available', preview='visible chain of thought')
    relay('subagent.text', preview='streamed child token')
    relay._flush()
    relay('subagent.complete', status='completed', summary='all clear', duration_seconds=1.5)

    kinds = [kind for kind, _ in published]
    assert kinds == [
        'subagent.spawn_requested', 'subagent.start', 'subagent.tool',
        'subagent.thinking', 'subagent.thinking', 'subagent.progress', 'subagent.complete',
    ]
    for kind, payload in published:
        EVENTS[kind].payload.model_validate(payload)
        assert payload['subagent_id'] == 'sa-0-abc12345'
        assert payload['goal'] == 'inspect viewer payloads'
        assert payload['task_index'] == 0 and payload['task_count'] == 2
        assert payload['child_session_id'] == 'child-sess-1'
        assert payload['delegation_id'] == 'deleg_12345678'
    tool_payload = published[2][1]
    assert tool_payload['tool_name'] == 'read_file' and tool_payload['tool_preview'] == 'README.md'
    assert 'text' not in published[3][1]
    assert published[4][1]['text'] == 'visible chain of thought'
    assert published[6][1]['status'] == 'completed' and published[6][1]['summary'] == 'all clear'

    # Registered background child descending from the turn's agent publishes without borrowing the next turn's execution identity; unregistered or foreign-parent children (even with matching owner_agent_session_id) are rejected.
    stream_frames = []
    live.event_stream = SessionEvents()
    live.event_stream.execution = {'admission_id': 'turn-b', 'execution_generation': 2}
    live.event_stream.observers.add(stream_frames.append)
    turn._approval_owner[0].executing = {'s': 2}
    stale_relay = _build_child_progress_callback(0, 'stale goal', parent, subagent_id='sa-stale')
    stale_relay('subagent.complete', status='completed', summary='stale finish')
    assert stream_frames == []
    foreign_parent = _Parent()
    foreign_child = SimpleNamespace(_delegate_parent_ref=weakref.ref(foreign_parent))
    owned_child = SimpleNamespace(_delegate_parent_ref=weakref.ref(parent))
    _register_subagent({'subagent_id': 'sa-foreign-same-sid', 'agent': foreign_child, 'owner_agent_session_id': 's'})
    _register_subagent({'subagent_id': 'sa-0-abc12345', 'agent': owned_child, 'owner_agent_session_id': 's'})
    try:
        foreign_relay = _build_child_progress_callback(0, 'foreign goal', parent, subagent_id='sa-foreign-same-sid')
        foreign_relay('subagent.complete', status='completed', summary='foreign finish')
        assert stream_frames == []
        relay('subagent.complete', status='completed', summary='late finish')
    finally:
        _unregister_subagent('sa-foreign-same-sid')
        _unregister_subagent('sa-0-abc12345')
    assert len(stream_frames) == 1
    params = stream_frames[0]['params']
    assert params['type'] == 'subagent.complete' and 'admission_id' not in params and 'execution_generation' not in params
    assert live.event_stream.execution == {'admission_id': 'turn-b', 'execution_generation': 2}

    # In-turn event with no live viewer: recorded once with the turn's identity, never retried as an out-of-turn frame.
    authority = turn._approval_owner[0]
    authority.executing = {'s': 1}
    publish_calls = []
    turn._publish_execution = lambda kind, payload: publish_calls.append(kind) or False
    _register_subagent({'subagent_id': 'sa-0-abc12345', 'agent': owned_child, 'owner_agent_session_id': 's'})
    try:
        stream_frames.clear()
        relay('subagent.start')
    finally:
        _unregister_subagent('sa-0-abc12345')
    assert publish_calls == ['subagent.start'] and stream_frames == []

    # Out of turn, a queued grandchild's spawn_requested is admitted only through a registered descendant parent.
    authority.executing = {'s': 2}
    grandchild_relay = _build_child_progress_callback(
        0, 'grandchild goal', parent, subagent_id='sa-1-grand', parent_id='sa-0-abc12345', depth=2)
    orphan_relay = _build_child_progress_callback(
        0, 'orphan goal', parent, subagent_id='sa-1-orphan', parent_id='sa-foreign-same-sid', depth=2)
    _register_subagent({'subagent_id': 'sa-0-abc12345', 'agent': owned_child, 'owner_agent_session_id': 's'})
    _register_subagent({'subagent_id': 'sa-foreign-same-sid', 'agent': foreign_child, 'owner_agent_session_id': 's'})
    try:
        stream_frames.clear()
        orphan_relay('subagent.spawn_requested', preview='orphan goal')
        grandchild_relay('subagent.start')
        assert stream_frames == []
        grandchild_relay('subagent.spawn_requested', preview='grandchild goal')
    finally:
        _unregister_subagent('sa-0-abc12345')
        _unregister_subagent('sa-foreign-same-sid')
    assert [f['params']['type'] for f in stream_frames] == ['subagent.spawn_requested']
    assert stream_frames[0]['params']['payload']['subagent_id'] == 'sa-1-grand'

    # Messaging routes never publish subagent.* viewer frames.
    msg_published = []
    msg_turn = _turn(Platform.TELEGRAM, msg_published)
    msg_parent = SimpleNamespace(_delegate_spinner=None, tool_progress_callback=msg_turn.progress_callback)
    msg_relay = _build_child_progress_callback(0, 'quiet goal', msg_parent, subagent_id='sa-msg')
    msg_relay('subagent.start')
    msg_relay('tool.started', 'read_file', 'README.md', {})
    msg_relay('subagent.complete', status='completed')
    assert msg_published == []


def test_a_failed_viewer_publish_is_logged_and_still_reaches_the_failure_notice(monkeypatch, caplog):
    """A viewer publish error stays best-effort: logged, and the failed-subagent notice still runs."""
    import logging
    from gateway import local_viewer_events

    def boom(*_a, **_k):
        raise RuntimeError('viewer down')

    monkeypatch.setattr(local_viewer_events, 'publish_progress', boom)
    turn = _turn(Platform.LOCAL, [])
    notices = []
    turn._progress_subagent_notice = lambda preview, kwargs: notices.append(kwargs.get('status'))
    with caplog.at_level(logging.DEBUG, logger='gateway.run'):
        turn.progress_callback('subagent.complete', None, 'child', None, status='failed')
    assert notices == ['failed']
    assert any('Local viewer progress publish failed for subagent.complete' in r.getMessage() and r.exc_info for r in caplog.records)


def test_subagent_roster_retains_running_child_after_fallback_evicts_parent():
    """TUI-4 fallback roster: when ``_run_agent_evict_on_fallback`` evicts the parent after a successful provider fallback, ``_owner_objects`` and ``subagent_records`` keep the child visible for the owning profile DB and route without transferring control/tail ownership or leaking across independent DB or route mismatches."""
    import weakref
    from gateway.run_turn import GatewayTurnMixin
    from gateway.session_ancillary import _owner_objects, subagent_records
    from tools.delegate_tool_registry import _register_subagent, _unregister_subagent

    class Parent:
        pass

    db_a, db_b = object(), object()
    original = Parent()
    original.session_id = 'same-parent'
    original._session_db = db_a
    original.gateway_session_key = 'route-a'
    original.model = 'fallback-model'
    original._provider_fallback_active = True

    replacement = Parent()
    replacement.session_id = original.session_id
    replacement._session_db = db_a
    replacement.gateway_session_key = 'route-a'

    wrong_db_parent = Parent()
    wrong_db_parent.session_id = original.session_id
    wrong_db_parent._session_db = db_b
    wrong_db_parent.gateway_session_key = 'route-a'
    wrong_db_parent.model = 'fallback-model'
    wrong_db_parent._provider_fallback_active = True

    wrong_route_parent = Parent()
    wrong_route_parent.session_id = original.session_id
    wrong_route_parent._session_db = db_a
    wrong_route_parent.gateway_session_key = 'route-b'
    wrong_route_parent.model = 'fallback-model'
    wrong_route_parent._provider_fallback_active = True

    unevicted = Parent()
    unevicted.session_id = original.session_id
    unevicted._session_db = db_a

    child = SimpleNamespace(_delegate_parent_ref=weakref.ref(original))
    wrong_db_child = SimpleNamespace(_delegate_parent_ref=weakref.ref(wrong_db_parent))
    wrong_route_child = SimpleNamespace(_delegate_parent_ref=weakref.ref(wrong_route_parent))
    stale_child = SimpleNamespace(_delegate_parent_ref=weakref.ref(unevicted))
    _register_subagent({'subagent_id': 'fallback-child', 'agent': child, 'owner_agent_session_id': original.session_id})
    _register_subagent({'subagent_id': 'wrong-db-child', 'agent': wrong_db_child, 'owner_agent_session_id': original.session_id})
    _register_subagent({'subagent_id': 'wrong-route-child', 'agent': wrong_route_child, 'owner_agent_session_id': original.session_id})
    _register_subagent({'subagent_id': 'stale-child', 'agent': stale_child, 'owner_agent_session_id': unevicted.session_id})
    try:
        assert [r['subagent_id'] for r in subagent_records(original)] == ['fallback-child']
        evicted = []
        for route_key, parent_obj in (('route-a', original), ('route-a', wrong_db_parent), ('route-b', wrong_route_parent)):
            GatewayTurnMixin._run_agent_evict_on_fallback(
                SimpleNamespace(_evict_cached_agent=evicted.append),
                SimpleNamespace(session_key=route_key, agent_holder=[parent_obj], result_holder=[{'failed': False}]),
            )
        assert evicted == ['route-a', 'route-a', 'route-b']
        # After a replacement parent is built, and through _owner_objects when no resident parent is cached, only the matching (db_a, route-a) fallback child is returned.
        assert [r['subagent_id'] for r in subagent_records(replacement)] == ['fallback-child']
        authority_a = SimpleNamespace(
            db=db_a, runner=SimpleNamespace(), agent=lambda ref: None,
            store=SimpleNamespace(_entries={'route-a': SimpleNamespace(session_id='same-parent')}),
            sessions={'same-parent': SimpleNamespace(route='route-a', source=SimpleNamespace(platform=Platform.TELEGRAM))})
        ref = SimpleNamespace(profile_id='p', session_id='same-parent')
        assert [r['subagent_id'] for r in _owner_objects(authority_a, ref, include_fallback_evicted=True)[1]] == ['fallback-child']
        assert _owner_objects(authority_a, ref, include_fallback_evicted=False)[1] == []
        assert subagent_records(replacement, include_fallback_evicted=False) == []
        # A rotating /compress after the fallback moves the conversation to a new stored id; the lineage keeps the child listed, still scoped by DB and route.
        class _LineageDB:
            def get_compression_lineage(self, session_id):
                return ['same-parent', 'same-parent-c1'] if session_id in ('same-parent', 'same-parent-c1') else [session_id]
        lineage_db = _LineageDB()
        for agent_obj in (original, wrong_route_parent, replacement):
            agent_obj._session_db = lineage_db
        rotated = Parent()
        rotated.session_id = 'same-parent-c1'
        rotated._session_db = lineage_db
        rotated.gateway_session_key = 'route-a'
        assert [r['subagent_id'] for r in subagent_records(rotated)] == ['fallback-child']
        assert subagent_records(None, session_id='unrelated', session_db=lineage_db, route='route-a') == []
        # Ancestry is walked backward from the live session on the real store, so an orphaned closed sibling under a compressed parent never hides the evicted parent's children.
        import tempfile
        from pathlib import Path
        from hermes_state import SessionDB
        with tempfile.TemporaryDirectory() as tmp:
            real_db = SessionDB(db_path=Path(tmp) / 'state.db')
            with real_db:
                real_db.create_session('same-parent', source='cli')
                real_db.end_session('same-parent', 'compression')
                real_db.create_session('orphan', source='cli', parent_session_id='same-parent')
                real_db.end_session('orphan', 'ws_orphan_reap')
                real_db.create_session('live', source='cli', parent_session_id='same-parent')
                real_db.end_session('live', 'compression')
                real_db.create_session('tip', source='cli', parent_session_id='live')
                real_db.create_session('elsewhere', source='cli')
                original._session_db = real_db
                assert [r['subagent_id'] for r in subagent_records(None, session_id='tip', session_db=real_db, route='route-a')] == ['fallback-child']
                assert subagent_records(None, session_id='elsewhere', session_db=real_db, route='route-a') == []
    finally:
        for sid in ('fallback-child', 'wrong-db-child', 'wrong-route-child', 'stale-child'):
            _unregister_subagent(sid)
