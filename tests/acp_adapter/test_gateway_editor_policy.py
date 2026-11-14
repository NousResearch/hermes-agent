"""Frozen editor policy and execution-owned edits, without an editor process."""
import asyncio
from dataclasses import asdict, replace
import os

import pytest

from gateway.session_policy import build_policy, policy_scope, restore_policy


def test_acp_policy_uses_its_own_tools_and_restores_frozen_cwd(tmp_path):
    from hermes_cli.tools_config import _get_platform_tools
    cfg = {'platform_toolsets': {'cli': ['terminal'], 'acp': ['file']}}
    params = {'source': 'acp', 'cwd': str(tmp_path), 'model': 'fixture',
              'editor': {'mcp_servers': [], 'edit_approval_policy': 'ask'}}
    before = dict(os.environ)
    policy = build_policy(params, cfg)
    assert policy.platform == 'acp'
    assert set(policy.toolsets) == _get_platform_tools(cfg, 'acp')
    assert 'terminal' not in policy.toolsets and 'project' not in policy.toolsets
    restored = restore_policy(asdict(policy))
    assert restored == policy
    cfg['platform_toolsets']['acp'] = ['terminal']
    assert 'terminal' not in restored.toolsets
    assert dict(os.environ) == before


@pytest.mark.asyncio
async def test_editor_guard_uses_shared_waiter_without_leaking_to_cli(tmp_path):
    from acp_adapter.edit_approval import maybe_require_edit_approval
    from gateway.session_events import SessionEvents
    from gateway.session_pending_controls import PendingControls
    from tools.approval import register_gateway_notify, unregister_gateway_notify
    from tools.approval_context import set_current_session_key, reset_current_session_key

    route = 'owned-editor-route'
    controls = PendingControls(SessionEvents())
    notified = asyncio.Event()
    loop = asyncio.get_running_loop()

    def notify(data):
        controls.register('editor', route, 1, data)
        loop.call_soon_threadsafe(notified.set)

    register_gateway_notify(route, notify)
    policy = build_policy({'source': 'acp', 'cwd': str(tmp_path), 'model': 'fixture'}, {})
    path = tmp_path / 'edit.txt'
    path.write_text('original')

    def guard():
        token = set_current_session_key(route)
        try:
            with policy_scope(policy):
                return maybe_require_edit_approval('write_file', {'path': str(path), 'content': 'approved'})
        finally:
            reset_current_session_key(token)

    task = asyncio.create_task(asyncio.to_thread(guard))
    try:
        await asyncio.wait_for(notified.wait(), 5)
        pending = controls.snapshot('editor', 1)[0]
        assert pending['choices'] == ['once', 'deny']
        assert pending['edit']['old_text'] == 'original'
        assert pending['edit']['new_text'] == 'approved'
        assert path.read_text() == 'original' and not task.done()
        # An opposing session/context never inherits the editor's callback.
        with policy_scope(replace(policy, source='cli', platform='cli')):
            assert maybe_require_edit_approval('write_file', {'path': str(path), 'content': 'cli'}) is None
        controls.respond('editor', 1, pending['prompt_id'], {'choice': 'deny'})
        assert 'denied' in await asyncio.wait_for(task, 5)
        assert maybe_require_edit_approval('write_file', {'path': str(path), 'content': 'after'}) is None
    finally:
        unregister_gateway_notify(route)
        await task
