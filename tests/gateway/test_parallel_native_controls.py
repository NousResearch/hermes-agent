"""Parallel controls through ordinary ingress, real TurnRunner and a loopback model."""
from contextlib import asynccontextmanager
import os
from pathlib import Path
import threading

import pytest

from gateway.config import GatewayConfig, Platform, StreamingConfig
from gateway.run import GatewayRunner
from gateway.session_authority import initialize_session_authority
from gateway.session_parallel import physical_count
from tests.e2e.core.history._helpers import NO_BACKGROUND_REVIEW, OFFLINE_CONFIG
from tests.fakes.fake_llm_provider import FakeLLMServer, Text, ToolCall, write_hermes_home
from tests.gateway.test_parallel_native_ingress import DiscordFixture, admissions, event
from tests.waiting import async_wait_until


@asynccontextmanager
async def native_owner(monkeypatch, respond):
    import gateway.run as gateway_run
    home = Path(os.environ['HERMES_HOME'])
    monkeypatch.setenv('DISCORD_ALLOWED_USERS', 'alice,bob')
    with FakeLLMServer(respond, aux=lambda request: Text('Auxiliary')) as server:
        write_hermes_home(home, server.base_url, extra_config=OFFLINE_CONFIG + NO_BACKGROUND_REVIEW + '''
display:
  busy_input_mode: parallel
  tool_progress: off
  interim_assistant_messages: false
approvals:
  mode: manual
  approvers: []
  timeout: 30
platform_toolsets:
  discord: [terminal, clarify]
''')
        monkeypatch.setattr(gateway_run, '_hermes_home', home)
        monkeypatch.setattr(gateway_run, '_resolve_runtime_agent_kwargs', lambda: {
            'api_key': 'sk-fake-e2e', 'base_url': server.base_url,
            'provider': 'custom', 'api_mode': 'chat_completions'})
        runner = GatewayRunner(GatewayConfig())
        runner.config.streaming = StreamingConfig(enabled=False)
        adapter = DiscordFixture()
        adapter.gateway_runner = runner
        runner.adapters = {Platform.DISCORD: adapter}
        authority = await initialize_session_authority(runner, profile_id='default', instance_id='controls')
        runner._wire_adapter_handlers(adapter)
        runner._busy_input_mode = 'parallel'
        try:
            yield runner, authority, adapter
        finally:
            from tools import approval, clarify_gateway
            for live in authority.sessions.values():
                approval.clear_session(live.route)
                clarify_gateway.clear_session(live.route)
            await async_wait_until(lambda: not adapter._background_tasks, timeout=30, what='native control cleanup')
            authority.db.close()


def user_text(request):
    return str([m.get('content', '') for m in request['body']['messages'] if m['role'] == 'user'])


def confirmed_prompt(authority, message_id):
    # A fixture's send exposes the wire message before the gateway has received
    # and durably indexed its acknowledgement. Don't race that confirmation.
    with authority.db._read_ctx() as conn:
        return conn.execute('SELECT 1 FROM parallel_work_anchors WHERE message_id=? AND prompt_id IS NOT NULL',
                            (message_id,)).fetchone() is not None


@pytest.mark.asyncio
async def test_native_stop_targets_one_work_and_waits_for_physical_return(monkeypatch):
    entered = {name: threading.Event() for name in ('STOP-ONE', 'KEEP-TWO')}
    release = {name: threading.Event() for name in entered}

    def respond(request):
        name = next(name for name in entered if name in user_text(request))
        entered[name].set()
        assert release[name].wait(20)
        return Text('ANSWER-' + name)

    try:
        async with native_owner(monkeypatch, respond) as (runner, authority, adapter):
            for mid, text in [('m1', 'STOP-ONE'), ('m2', 'KEEP-TWO')]:
                await adapter.handle_message(event(adapter, mid, text))
            await async_wait_until(lambda: all(e.is_set() for e in entered.values()), timeout=20, what='two model calls')
            rows = {r['request_id']: r for r in admissions(authority)}
            routes = {mid: authority.sessions[row['target_session_id']].route for mid, row in rows.items()}
            agents = {mid: runner._running_agents[route] for mid, route in routes.items()}
            await adapter.handle_message(event(adapter, 'ambiguous', '/stop'))
            await adapter.handle_message(event(adapter, 'foreign', '/stop', reply='m1', user='bob'))
            assert not any(agent._interrupt_requested for agent in agents.values())
            await adapter.handle_message(event(adapter, 'stop-one', '/stop', reply='m1'))
            assert agents['m1']._interrupt_requested
            assert not agents['m2']._interrupt_requested
            assert physical_count(authority) == 2
            assert runner._drain_work_counts()[0] >= 2
            release['STOP-ONE'].set()
            await async_wait_until(lambda: physical_count(authority) == 1, timeout=20, what='stopped thread physically returned')
            assert not agents['m2']._interrupt_requested
            release['KEEP-TWO'].set()
            await async_wait_until(lambda: any('ANSWER-KEEP-TWO' in s for s in adapter.sent), timeout=20, what='sibling survives')
            await async_wait_until(lambda: physical_count(authority) == 0, timeout=20, what='all physical work done')
            assert len(admissions(authority)) == 2
    finally:
        for gate in release.values():
            gate.set()


@pytest.mark.asyncio
async def test_native_approval_and_clarify_are_sender_request_and_generation_scoped(monkeypatch):
    from tools import approval, clarify_gateway
    from hermes_state_runtime import RuntimeStoreError

    def respond(request):
        messages = request['body']['messages']
        name = 'APPROVAL-ONE' if 'APPROVAL-ONE' in user_text(request) else 'CLARIFY-TWO'
        if any(m['role'] == 'tool' for m in messages):
            return Text('DONE-' + name)
        if name == 'APPROVAL-ONE':
            return ToolCall('terminal', {'command': 'rm -rf /parallel-test-never-execute'})
        return ToolCall('clarify', {'questions': [{'question': 'Choose CLARIFY-TWO', 'choices': ['blue', 'green']}]})

    async with native_owner(monkeypatch, respond) as (_runner, authority, adapter):
        for mid, text in [('m1', 'APPROVAL-ONE'), ('m2', 'CLARIFY-TWO')]:
            await adapter.handle_message(event(adapter, mid, text))
        await async_wait_until(lambda: any('approve' in s.lower() for s in adapter.sent)
                               and any('Choose CLARIFY-TWO' in s for s in adapter.sent),
                               timeout=20, what='both real tool prompts delivered')
        rows = {r['request_id']: r for r in admissions(authority)}
        sid1, sid2 = (rows[mid]['target_session_id'] for mid in ('m1', 'm2'))
        route1, route2 = (authority.sessions[sid].route for sid in (sid1, sid2))
        await async_wait_until(lambda: bool(approval.list_gateway_approvals(route1)), what='approval registered')
        pending = approval.list_gateway_approvals(route1)[0]
        generation = authority.db.get_session(sid1)['runtime_generation']
        with pytest.raises(RuntimeStoreError):
            authority.check_approval_generation(sid1, generation - 1)
        assert not approval.resolve_gateway_approval_card(route1, 'once', 'wrong-request')
        await adapter.handle_message(event(adapter, 'ambiguous', 'yes'))
        await adapter.handle_message(event(adapter, 'foreign', '/approve', reply='m1', user='bob'))
        assert approval.has_blocking_approval(route1)
        assert clarify_gateway.get_pending_for_session(route2, include_choice_prompts=True) is not None
        approval_card = next(mid for _, text, _, mid in adapter.deliveries if 'approve' in text.lower())
        clarify_card = next(mid for _, text, _, mid in adapter.deliveries if 'Choose CLARIFY-TWO' in text)
        await async_wait_until(lambda: confirmed_prompt(authority, approval_card), what='approval receipt committed')
        await adapter.handle_message(event(adapter, 'deny-one', '/deny', reply=approval_card))
        await async_wait_until(lambda: any('DONE-APPROVAL-ONE' in s for s in adapter.sent), timeout=20, what='only approval resumes')
        assert clarify_gateway.get_pending_for_session(route2, include_choice_prompts=True) is not None
        assert not approval.resolve_gateway_approval_card(route1, 'once', pending['request_id'])
        await adapter.handle_message(event(adapter, 'answer-two', 'blue', reply=clarify_card))
        await async_wait_until(lambda: any('DONE-CLARIFY-TWO' in s for s in adapter.sent), timeout=20, what='clarify resumes')
        assert len(admissions(authority)) == 2
