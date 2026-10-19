"""Execution-owned terminal consent uses the canonical pending-control queue."""
import asyncio
import json
import shlex

import pytest

from gateway.session_policy import build_policy, policy_scope
from gateway.session_pending_controls import PendingControls
from gateway.session_events import SessionEvents


@pytest.mark.platforms("posix")
@pytest.mark.asyncio
@pytest.mark.parametrize('dangerous', [False, True])
@pytest.mark.parametrize('choice', ['once', 'deny'])
async def test_terminal_policy_uses_one_execution_owned_consent(tmp_path, dangerous, choice):
    from tools import terminal_tool
    from tools.approval import register_gateway_notify, unregister_gateway_notify
    from tools.approval_context import set_current_session_key, reset_current_session_key
    from tools.registry import registry
    marker = tmp_path / 'executed.txt'
    command = 'printf approved > ' + shlex.quote(str(marker))
    if dangerous:
        target = tmp_path / 'empty-target'
        target.mkdir()
        command = 'rm -rf -- ' + shlex.quote(str(target)) + '; ' + command
    route, generation = 'owned-terminal-route', 1
    controls = PendingControls(SessionEvents())
    notified = asyncio.Event()
    loop = asyncio.get_running_loop()
    requests = []

    def notify(data):
        requests.append(data)
        controls.register('editor', route, generation, data)
        loop.call_soon_threadsafe(notified.set)

    policy = build_policy({'source': 'acp', 'cwd': str(tmp_path), 'model': 'fixture'},
                          {'approvals': {'mode': 'manual'}, 'terminal': {'env_type': 'local'}})

    def execute():
        token = set_current_session_key(route)
        try:
            with policy_scope(policy):
                return json.loads(registry.dispatch('terminal', {'command': command}, task_id=route))
        finally:
            reset_current_session_key(token)

    register_gateway_notify(route, notify)
    task = asyncio.create_task(asyncio.to_thread(execute))
    try:
        await asyncio.wait_for(notified.wait(), 10)
        assert not marker.exists() and not task.done()
        pending = controls.snapshot('editor', generation)[0]
        assert pending['command'] == command
        controls.respond('editor', generation, pending['prompt_id'], {'choice': choice})
        result = await asyncio.wait_for(task, 15)
        assert marker.exists() == (choice == 'once'), result
        assert len(requests) == 1
        assert result.get('exit_code') == 0 if choice == 'once' else result.get('status') == 'blocked'
    finally:
        unregister_gateway_notify(route)
        await task
