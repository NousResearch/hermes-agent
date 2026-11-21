"""Real gateway/agent/tool loop, with only Discord and model output synthetic."""
import asyncio
import json
import os
import sys
from pathlib import Path
import weakref


async def main():
    from gateway.config import Platform, PlatformConfig
    from gateway.platforms.base import BasePlatformAdapter, SendResult
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource
    from gateway.session_authority import initialize_session_authority

    class Adapter(BasePlatformAdapter):
        def __init__(self):
            super().__init__(PlatformConfig(enabled=True, token='fixture'), Platform.DISCORD)
            self._is_allowed_user = lambda user, **_: user == '42'
            self.sent = []
        async def connect(self, **kwargs):
            return True

        async def disconnect(self):
            pass

        async def send(self, chat_id, content, **kwargs):
            self.sent.append(content)
            return SendResult(success=True, message_id='1')

        async def send_typing(self, *args, **kwargs):
            pass

        async def get_chat_info(self, chat_id):
            return {'id': chat_id}

    root = Path(os.environ['HERMES_HOME'])
    runner = GatewayRunner()
    adapter = Adapter()
    await initialize_session_authority(runner, profile_id='default', instance_id='exact-test')
    runner.adapters[Platform.DISCORD] = adapter
    handler = runner._make_default_profile_message_handler()
    home = root
    adapter.set_message_handler(handler)
    source = SessionSource(Platform.DISCORD, '700', chat_type='dm', user_id='42', guild_id='1')
    source._transport_adapter_ref = weakref.ref(adapter)
    source._authorization_profile_home = home
    import threading
    from run_agent import AIAgent
    from agent.turn_api_call import bind_completion
    child_started, child_release, steer_observed = threading.Event(), threading.Event(), threading.Event()
    children = []
    original_run = AIAgent.run_conversation
    def controlled_run(agent, *args, **kwargs):
        if not getattr(agent, '_subagent_id', None):
            if args and args[0] == 'Initialize the test conversation.':
                from gateway.plugin_submission.execution import _response
                def initialize(parent, payload):
                    response = _response(parent.api_mode, [])
                    if parent.api_mode == 'codex_responses':
                        response.output[0].content[0].text = 'Initialized.'
                    else:
                        response.choices[0].message.content = 'Initialized.'
                    return response
                with bind_completion(agent, initialize):
                    return original_run(agent, *args, **kwargs)
            return original_run(agent, *args, **kwargs)
        children.append(agent)
        def complete(child, payload):
            first = not child_started.is_set()
            child_started.set()
            assert child_release.wait(60), 'child was never released'
            from gateway.plugin_submission.execution import _response
            if first:
                return _response(child.api_mode, [dict(call_id='child-boundary', name='terminal',
                                                       arguments={'command': 'echo child-boundary'})])
            assert 'Finish with a concise summary.' in json.dumps(payload), payload
            steer_observed.set()
            return _response(child.api_mode, [])
        with bind_completion(agent, complete):
            return original_run(agent, *args, **kwargs)
    AIAgent.run_conversation = controlled_run
    from tests.waiting import async_wait_until
    from gateway.platforms.event import MessageEvent
    await adapter._message_handler(MessageEvent(text='Initialize the test conversation.', source=source,
                                                message_id='initialize'))
    runner._gateway_loop = asyncio.get_running_loop()
    runner._running = True
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
    manager = PluginManager(scope_key=str(home))
    manager.set_gateway_message_injector(runner, runner._schedule_plugin_message_injection)
    context = PluginContext(PluginManifest(name='hybrid-test', key='hybrid-test', source='user'), manager)
    entry = runner.session_store.get_or_create_session(source)
    identity = dict(session_key=entry.session_key, expected_session_id=entry.session_id)
    discovered = await asyncio.to_thread(context.conversation_tools, **identity)
    assert discovered['ok'], discovered
    assert any(tool['name'] == 'terminal' for tool in discovered['tools'])
    async def execute(index, tool_name, arguments):
        calls = [dict(call_id=f'call-{index}', name=tool_name, arguments=arguments)]
        def observe():
            return context.execute_tools(calls, request_id=f'request-{index}',
                text=f'Perform requested operation {index}.', require_background=True, **identity)
        receipt = await asyncio.to_thread(observe)
        assert receipt['ok'], receipt
        async def completed():
            value = await asyncio.to_thread(observe)
            assert value['ok'], value
            return value if value['status'] == 'terminal' else None
        receipt = await async_wait_until(completed, timeout=30)
        assert receipt['outcome'] == 'completed', receipt
        assert receipt['outputs'], receipt
        assert all('Tool execution completed' not in text and '[SILENT]' not in text for text in adapter.sent), adapter.sent
        return receipt['outputs']
    try:
        dispatch = await execute(1, 'delegate_task', {'tasks': [{'goal': 'Wait for the fixture release.',
            'group': 'fixture'}], 'background': True})
        payload = json.loads(dispatch[0]['output'])
        assert payload['mode'] == 'background', payload
        await async_wait_until(child_started.is_set, timeout=10)
        child_id = payload['subagent_ids'][0]
        assert children[0]._subagent_id == child_id
        direct = await execute(2, 'terminal', {'command': 'echo direct-while-child-runs'})
        assert 'direct-while-child-runs' in direct[0]['output'], direct
        assert not child_release.is_set()
        listed = await execute(3, 'delegate_task', {'action': 'list'})
        assert child_id in listed[0]['output'], listed
        steered = await execute(4, 'delegate_task', {'action': 'steer', 'subagent_id': child_id,
            'message': 'Finish with a concise summary.'})
        assert 'error' not in json.loads(steered[0]['output']), steered
        assert not child_release.is_set()
        from tools import approval
        from hermes_cli.plugins import get_plugin_manager
        policy = PluginContext(PluginManifest(name='approval-test', source='user'), get_plugin_manager())
        approval_hook = policy.register_hook('pre_tool_call', lambda **_: {'action': 'approve', 'message': 'Test policy'})
        denied_task = asyncio.create_task(execute(5, 'terminal', {'command': 'echo forbidden > denied-marker'}))
        def pending_approvals():
            with approval._lock:
                return [key for key, entries in approval._gateway_queues.items() if entries]
        pending = await async_wait_until(pending_approvals, timeout=15)
        assert len(pending) == 1
        assert approval.list_gateway_approval_authority(pending[0])[0]['requester_id'] == '42'
        approval.resolve_gateway_approval(pending[0], 'deny')
        denied = await denied_task
        assert 'den' in denied[0]['output'].lower(), denied
        assert not (home / 'denied-marker').exists()
        approval_hook.dispose()
        child_release.set()
        await async_wait_until(steer_observed.is_set, timeout=15)
        Path(os.environ['HERMES_HOME'], 'hybrid-receipt.json').write_text(json.dumps({
            'dispatch': payload, 'direct': direct, 'list': listed, 'steer': steered,
            'child_session': children[0].session_id,
            'parent_session': children[0]._parent_session_id,
            'direct_completed_while_child_running': True, 'steer_observed_by_child': True}), encoding='utf-8')
    finally:
        child_release.set()
        runner._running = False
        AIAgent.run_conversation = original_run


if __name__ == '__main__':
    os.environ.update(OPENAI_API_KEY='offline-only', OPENAI_BASE_URL='http://127.0.0.1:9/v1',
                      DISCORD_ALLOWED_USERS='42', DISCORD_ALLOW_ALL_USERS='true')
    Path(os.environ['HERMES_HOME'], 'config.yaml').write_text(
        'model:\n  default: offline-model\n  provider: offline\n'
        'custom_providers:\n- name: offline\n  base_url: http://127.0.0.1:9/v1\n  key_env: OPENAI_API_KEY\n'
        f'  api_mode: {sys.argv[3] if len(sys.argv) > 3 else "chat_completions"}\n'
        'auxiliary:\n  title_generation:\n    enabled: false\n'
        'background_review:\n  enabled: false\n'
        'memory:\n  memory_enabled: false\n  user_profile_enabled: false\n'
        'toolsets:\n  discord: [terminal, delegation]\n'
        'plugins:\n  entries:\n    hybrid-test:\n      allow_gateway_injection: true\n'
        f'terminal:\n  cwd: {os.environ["HERMES_HOME"]}\n'
        'display:\n  tool_progress: off\n  thinking_progress: false\n', encoding='utf-8')
    asyncio.run(main())
