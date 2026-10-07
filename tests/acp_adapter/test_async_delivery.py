"""Host-owned ACP delivery capability uses the existing delegation lifecycle."""
import argparse
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from acp_adapter import entry
from acp_adapter.server import HermesACPAgent
from gateway.session_context import async_delivery_supported
from hermes_cli.main_agent_cmds import cmd_acp
from hermes_cli.subcommands.acp import build_acp_parser


def turn(agent, callback):
    state = SimpleNamespace(agent=SimpleNamespace(run_conversation=lambda **kw: callback()), cwd='/tmp', history=[])
    return agent._run_agent_turn(state=state, session_id='delivery-test', user_text='background:false',
                                user_content='background:false', conn=None, loop=None,
                                approval_cb=None, edit_approval_requester=None)


@pytest.mark.parametrize('argv,enabled', [([], True), (['--async-delivery=on'], True), (['--async-delivery=off'], False)])
def test_launch_option_reaches_turn_binding(monkeypatch, argv, enabled):
    import acp
    observed = []
    async def serve(agent, **kwargs):
        observed.append(turn(agent, async_delivery_supported))
    monkeypatch.setattr(entry, '_setup_logging', lambda: None)
    monkeypatch.setattr(entry, '_load_env', lambda: None)
    monkeypatch.setenv('HERMES_ACP_SKIP_CONFIGURED_MCP', '1')
    monkeypatch.setattr(acp, 'run_agent', serve)
    entry.main(argv)
    assert observed == [enabled]


@pytest.mark.parametrize('mode', ['on', 'off'])
def test_hermes_subcommand_forwards_host_option(monkeypatch, mode):
    parser = argparse.ArgumentParser()
    build_acp_parser(parser.add_subparsers(dest='command'), cmd_acp=cmd_acp)
    args = parser.parse_args(['acp', '--async-delivery='+mode])
    forwarded = []
    monkeypatch.setattr(entry, 'main', lambda argv: forwarded.append(entry._parse_args(argv).async_delivery))
    args.func(args)
    assert forwarded == [mode]


def test_invalid_launch_value_is_refused():
    with pytest.raises(SystemExit):
        entry._parse_args(['--async-delivery=maybe'])


def test_default_constructor_preserves_async_delivery():
    assert turn(HermesACPAgent(session_manager=MagicMock()), async_delivery_supported) is True


def test_context_local_modes_do_not_leak_across_overlapping_turns():
    barrier = threading.Barrier(2)
    def observe(enabled):
        agent = HermesACPAgent(session_manager=MagicMock(), async_delivery=enabled)
        def callback():
            before = async_delivery_supported()
            barrier.wait(timeout=5)
            return before, async_delivery_supported()
        return turn(agent, callback)
    with ThreadPoolExecutor(max_workers=2) as pool:
        on, off = pool.submit(observe, True), pool.submit(observe, False)
        assert on.result(timeout=10) == (True, True)
        assert off.result(timeout=10) == (False, False)


def delegation_probe(monkeypatch, child):
    """Keep real model dispatch, context decision and batch join; stub only child construction/execution."""
    from run_agent import AIAgent
    from tools import delegate_tool as dt, delegate_tool_dispatch as dispatch
    def build(**kw):
        tasks = kw['tasks']
        batch = dispatch._Batch(task_list=tasks, children=[(i, t, object()) for i, t in enumerate(tasks)],
            parent_agent=kw['parent_agent'], creds={}, context=None, top_role='leaf', max_children=2,
            live_deleg_id=None, live_writers=[], live_paths=[], origin_wake_sid='session',
            origin_ui_session_id='', origin_owner_transport=None, origin_owner_session_record=None,
            origin_session_history_delivery=False, overall_start=0)
        return dispatch._run_batch(batch, kw['background'])
    monkeypatch.setattr(dt, 'delegate_task', build)
    monkeypatch.setattr(dt, '_run_single_child', child)
    monkeypatch.setattr(dispatch, '_finalize_child_results', lambda *a: None)
    monkeypatch.setattr(dispatch, '_report_child_done', lambda *a: None)
    monkeypatch.setattr(dispatch, '_dispatch_unit', lambda *a: {'status': 'dispatched', 'delegation_id': 'probe'})
    parent = SimpleNamespace(_delegate_depth=0, session_id='session', _interrupt_requested=False)
    return lambda depth=0: AIAgent._dispatch_delegate_task(
        SimpleNamespace(**{**vars(parent), '_delegate_depth': depth}),
        {'tasks': [{'goal': 'leaf'}] * (2 if depth else 1), 'background': False})


def test_default_model_background_false_still_dispatches_background(monkeypatch):
    calls = []
    invoke = delegation_probe(monkeypatch, lambda **kw: calls.append(True))
    result = turn(HermesACPAgent(session_manager=MagicMock()), invoke)
    import json
    assert json.loads(result)['mode'] == 'background'
    assert calls == []


@pytest.mark.asyncio
async def test_off_mode_joins_nested_children_before_turn_returns(monkeypatch):
    entered, release, completed = threading.Event(), threading.Event(), []
    started, lock = [], threading.Lock()
    invoke = None
    def child(**kw):
        if kw['parent_agent']._delegate_depth == 0:
            nested = invoke(1)
            completed.append('orchestrator')
            return {'task_index': 0, 'status': 'completed', 'summary': nested}
        with lock:
            started.append(kw['task_index'])
            if len(started) == 2:
                entered.set()
        assert release.wait(timeout=5)
        completed.append('leaf')
        return {'task_index': kw['task_index'], 'status': 'completed', 'summary': 'joined'}
    invoke = delegation_probe(monkeypatch, child)
    from acp_adapter.session import SessionManager
    from acp.schema import TextContentBlock
    manager = SessionManager(agent_factory=lambda: MagicMock(name='OfflineModel'))
    agent = HermesACPAgent(session_manager=manager, async_delivery=False)
    session = await agent.new_session(cwd='/tmp')
    state = manager.get_session(session.session_id)
    state.agent.run_conversation = lambda **kw: {'final_response': invoke(), 'messages': []}
    future = asyncio.create_task(agent.prompt(
        prompt=[TextContentBlock(type='text', text='background:false')], session_id=session.session_id))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        assert not future.done()
    finally:
        release.set()
    result = await asyncio.wait_for(future, timeout=5)
    assert result.stop_reason == 'end_turn'
    assert completed == ['leaf', 'leaf', 'orchestrator']


@pytest.mark.asyncio
async def test_session_config_cannot_override_host_policy():
    manager = MagicMock()
    manager.get_session.return_value = SimpleNamespace(mode='default')
    agent = HermesACPAgent(session_manager=manager, async_delivery=False)
    await agent.set_config_option('async_delivery', 'session', 'on')
    assert turn(agent, async_delivery_supported) is False


@pytest.mark.asyncio
async def test_cancel_propagates_to_attached_joined_descendants(monkeypatch):
    from agent.interrupt_control import InterruptControlMixin
    from tools.delegate_tool_child_run import _attach_child
    class OwnedAgent(InterruptControlMixin):
        def __init__(self):
            self._execution_thread_id = None
            self._active_children_lock = threading.RLock()
            self._active_children = []
            self._interrupt_requested = False
            self._hard_interrupt_requested = threading.Event()
            self.quiet_mode = True
    parent, child, leaf = OwnedAgent(), OwnedAgent(), OwnedAgent()
    _attach_child(parent, child)
    _attach_child(child, leaf)
    state = SimpleNamespace(agent=parent, cancel_event=threading.Event(), runtime_lock=threading.RLock(),
                            is_running=True, current_prompt_text='delegating')
    manager = MagicMock()
    manager.get_session.return_value = state
    monkeypatch.setattr('tools.async_delegation.interrupt_for_session', lambda **kw: None)
    await HermesACPAgent(session_manager=manager, async_delivery=False).cancel('session')
    assert state.cancel_event.is_set()
    assert all(a._hard_interrupt_requested.is_set() for a in (parent, child, leaf))
