"""A plugin tool reads the current messaging turn, not the daemon's or its parent's route."""
import contextvars
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from agent.delegation_context import DELEGATED_CHILD_ENV_MARKER, delegated_child_context, delegated_child_subprocess_env
from gateway.platforms.event import MessageEvent
from gateway.run import _profile_runtime_scope
from gateway.run_turn_runner import TurnRunner
from gateway.session_context import _SESSION_TURN_EVENT, clear_session_vars, get_session_env, set_session_vars
from gateway.session_contract import SessionActor, SessionRef, Submission
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from hermes_constants import get_hermes_home
from model_tools import handle_function_call
from tests.gateway.test_native_admission_named_refusal import _authority
from tests.waiting import async_wait_until


def _context(home):
    return PluginContext(PluginManifest(name="origin-reader"), PluginManager(scope_key=str(home)))


def test_delegated_subprocess_cannot_reconstruct_origin_from_exported_session():
    with delegated_child_context('child'):
        env = delegated_child_subprocess_env({**os.environ, 'HERMES_SESSION_ID': 'child'})
    subprocess.run([sys.executable, '-c', '''
from gateway.session_context import get_session_env
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
ctx = PluginContext(PluginManifest(name='origin-reader'), PluginManager())
sid = get_session_env('HERMES_SESSION_ID')
assert sid == 'child' and ctx.messaging_turn_origin(sid) is None
'''], cwd=Path(__file__).resolve().parents[2], env=env, check=True, timeout=20)


@pytest.mark.asyncio
async def test_plugin_reads_only_its_own_native_messaging_turn(tmp_path, monkeypatch):
    from persistence_assembly import close_stores_under

    home = get_hermes_home()
    other_home = tmp_path / 'other'
    ctx, other = _context(home), _context(other_home)
    authority, source = await _authority(tmp_path, monkeypatch)
    source.shared_session, source.chat_type, source.thread_id = True, 'group', 'topic'
    source.scope_id, source.parent_chat_id = 'scope', 'parent'
    adapter = authority.runner.adapters[source.platform]
    seen = {}

    def handler(args, session_id='', **kwargs):
        return json.dumps(ctx.messaging_turn_origin(session_id))

    ctx.register_tool(name="read_turn_origin", toolset="origin-test", handler=handler,
                      schema={"name": "read_turn_origin", "parameters": {"type": "object", "properties": {}}})

    def run(turn):
        sid = turn._ctx.session_id
        event = _SESSION_TURN_EVENT.get()
        if event.message_id == 'to-merge':
            event.absorb_turn_input(MessageEvent(text='follow-up', source=source, user_id='second'))
            seen['merged_actor'] = event.user_id, event.source.user_id
            seen['mixed'] = ctx.messaging_turn_origin(sid)
            return {"final_response": "ok", "messages": []}
        if 'sid' in seen:  # a viewer's own input on the *same* messaging session
            seen['viewer'] = ctx.messaging_turn_origin(sid)
            return {"final_response": "ok", "messages": []}
        seen['sid'], seen['key'] = sid, turn._ctx.session_key
        seen['origin'] = json.loads(handle_function_call("read_turn_origin", {}, session_id=sid))
        seen['wrong_id'] = ctx.messaging_turn_origin(sid + " ")
        seen['wrong_plugin'] = other.messaging_turn_origin(sid)
        with _profile_runtime_scope(other_home, {}, hydrate_secrets=False):
            seen['wrong_scope'] = ctx.messaging_turn_origin(sid)
        with delegated_child_context('child'):
            seen['child'] = ctx.messaging_turn_origin('child')
            marker = delegated_child_subprocess_env({})[DELEGATED_CHILD_ENV_MARKER]
        with delegated_child_context(sid):
            seen['child_same_id'] = ctx.messaging_turn_origin(sid)
        with monkeypatch.context() as m:
            m.setenv(DELEGATED_CHILD_ENV_MARKER, marker)
            seen['descendant'] = ctx.messaging_turn_origin(sid)
        seen['after_child'] = ctx.messaging_turn_origin(sid)
        seen['copied'] = contextvars.copy_context()
        with monkeypatch.context() as m:
            state = turn._runner._session_state(seen['key']).persistent
            m.setattr(state, 'run_generation', state.run_generation + 1)
            seen['superseded'] = ctx.messaging_turn_origin(sid)
        cleared = contextvars.copy_context()
        cleared.run(clear_session_vars, [])
        seen['cleared'] = cleared.run(ctx.messaging_turn_origin, sid)
        for platform, cron in [('whatsapp', '1'), ('webhook', '')]:
            nested = contextvars.copy_context()
            nested.run(set_session_vars, platform=platform, chat_id=source.chat_id, session_key=seen['key'],
                       session_id=sid, cron_session=cron, actor_id=event.sender_id(), turn_event=event)
            seen[platform, cron] = nested.run(ctx.messaging_turn_origin, sid)
        seen['route_user'] = get_session_env('HERMES_SESSION_USER_ID')
        return {"final_response": "ok", "messages": []}

    monkeypatch.setattr(TurnRunner, '_run_sync_scoped', run)
    try:
        await adapter.handle_message(MessageEvent(text='read here', source=source, message_id='incoming'))
        await async_wait_until(lambda: not adapter._active_sessions, timeout=15, what='the native turn')
        assert seen['origin'] == seen['after_child'] == {
            'session_id': seen['sid'], 'session_key': seen['key'], 'profile': 'default',
            'platform': 'whatsapp', 'chat_id': source.chat_id, 'actor_id': source.user_id,
            'chat_type': 'group', 'thread_id': 'topic', 'scope_id': 'scope', 'parent_chat_id': 'parent',
        }
        assert seen['route_user'] == ''
        for name in ('wrong_id', 'wrong_plugin', 'wrong_scope', 'child', 'child_same_id', 'descendant',
                     'cleared', 'superseded', ('whatsapp', '1'), ('webhook', '')):
            assert seen[name] is None, name
        assert seen['copied'].run(ctx.messaging_turn_origin, seen['sid']) is None
        monkeypatch.setenv('HERMES_SESSION_ID', seen['sid'])
        empty = contextvars.Context()
        assert empty.run(get_session_env, 'HERMES_SESSION_ID') == seen['sid']
        assert empty.run(ctx.messaging_turn_origin, seen['sid']) is None

        await adapter.handle_message(MessageEvent(text='merge here', source=source, message_id='to-merge'))
        await async_wait_until(lambda: not adapter._active_sessions, timeout=15, what='the merged turn')
        assert seen['merged_actor'] == ('', source.user_id)
        assert seen['mixed'] is None

        ref = SessionRef(authority.profile_id, seen['sid'])
        viewer = SessionActor('viewer', ref.profile_id, frozenset({'session:submit'}), 'viewer')
        await authority.submit(viewer, Submission('viewer-input', ref, {'text': 'read from viewer'}, 'queue'))
        await authority.sessions[ref.session_id].task
        assert seen['viewer'] is None
    finally:
        close_stores_under(authority.persistence.scope.profile_dir)
