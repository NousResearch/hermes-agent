"""Real prompt callbacks in two served profiles never answer each other's work."""
import pytest

from gateway.config import Platform, StreamingConfig
from tests.e2e.core.history._helpers import NO_BACKGROUND_REVIEW, OFFLINE_CONFIG
from tests.fakes.fake_llm_provider import FakeLLMServer, Text, ToolCall, write_hermes_home
from tests.gateway.test_parallel_native_ingress import DiscordFixture, admissions, event
from tests.gateway.test_parallel_native_controls import confirmed_prompt
from tests.gateway.test_session_authorities_multiplex import _reserve_homes, _runner
from tests.waiting import async_wait_until


@pytest.mark.asyncio
async def test_real_multiplex_approval_and_clarify_callbacks_are_profile_scoped(tmp_path, monkeypatch):
    import gateway.run as gateway_run
    from gateway.profile_routing import parse_profile_routes
    from gateway.run_runtime import initialize_gateway_runtime
    from gateway.runtime_ownership import process_ownership
    from tools import approval, clarify_gateway
    root, homes = _reserve_homes(tmp_path, monkeypatch, names=('alpha',))
    monkeypatch.setattr(gateway_run, '_hermes_home', root)

    def respond(request):
        messages = request['body']['messages']
        users = str([m['content'] for m in messages if m['role'] == 'user'])
        name = 'ALPHA' if 'ALPHA' in users else 'DEFAULT'
        n = sum(m['role'] == 'tool' for m in messages)
        if n == 0:
            return ToolCall('terminal', {'command': 'rm -rf /parallel-multiplex-never-execute'})
        if n == 1:
            return ToolCall('clarify', {'questions': [{'question': 'PICK-' + name, 'choices': ['blue', 'green']}]})
        return Text('DONE-' + name)

    with FakeLLMServer(respond, aux=lambda request: Text('Auxiliary')) as server:
        for _, home in homes:
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
            (home / '.env').write_text('DISCORD_ALLOWED_USERS=alice,bob\n')
        monkeypatch.setattr(gateway_run, '_resolve_runtime_agent_kwargs', lambda: {
            'api_key': 'sk-fake-e2e', 'base_url': server.base_url,
            'provider': 'custom', 'api_mode': 'chat_completions'})
        process_ownership.reserve([home for _, home in homes])
        runner = _runner(root, homes)
        runner.config.streaming = StreamingConfig(enabled=False)
        runner.config.profile_routes = parse_profile_routes([
            {'name': 'alpha-room', 'platform': 'discord', 'profile': 'alpha', 'chat_id': 'alpha-room'}])
        runner._snapshot_profile_busy_modes('alpha', {'display': {'busy_input_mode': 'parallel'}})
        runner._busy_input_mode = 'parallel'
        adapter = DiscordFixture()
        adapter.gateway_runner = runner
        runner.adapters = {Platform.DISCORD: adapter}
        try:
            await initialize_gateway_runtime(runner)
            runner._wire_adapter_handlers(adapter)
            for chat, text in [('room', 'DEFAULT'), ('alpha-room', 'ALPHA')]:
                await adapter.handle_message(event(adapter, 'same-opener', text, chat=chat))
            owners = [runner.session_authority, runner.session_authorities.for_home(homes[1][1])]
            await async_wait_until(lambda: all(len(admissions(a)) == 1 for a in owners), timeout=20, what='both profiles admitted')
            routes = [a.sessions[admissions(a)[0]['target_session_id']].route for a in owners]
            await async_wait_until(lambda: all(approval.has_blocking_approval(r) for r in routes), timeout=20, what='two real approvals')
            await async_wait_until(lambda: len([d for d in adapter.deliveries if 'approve' in d[1].lower()]) == 2,
                                   timeout=20, what='two prompt receipts')
            cards = {chat: mid for chat, text, _, mid in adapter.deliveries if 'approve' in text.lower()}
            await async_wait_until(lambda: all(confirmed_prompt(a, cards[chat]) for a, chat in zip(owners, ('room', 'alpha-room'))),
                                   what='profile prompt receipts committed')
            # A valid other-profile receipt must not become a current-profile answer.
            await adapter.handle_message(event(adapter, 'wrong-profile', '/deny', reply=cards['room'], chat='alpha-room'))
            assert all(approval.has_blocking_approval(r) for r in routes)
            await adapter.handle_message(event(adapter, 'deny-default', '/deny', reply=cards['room']))
            await async_wait_until(lambda: any('PICK-DEFAULT' in d[1] for d in adapter.deliveries), timeout=20, what='default clarify')
            assert approval.has_blocking_approval(routes[1])
            await adapter.handle_message(event(adapter, 'deny-alpha', '/deny', reply=cards['alpha-room'], chat='alpha-room'))
            await async_wait_until(lambda: any('PICK-ALPHA' in d[1] for d in adapter.deliveries), timeout=20, what='alpha clarify')
            questions = {chat: mid for chat, text, _, mid in adapter.deliveries if 'PICK-' in text}
            await adapter.handle_message(event(adapter, 'wrong-question-profile', 'blue', reply=questions['room'], chat='alpha-room'))
            assert all(clarify_gateway.get_pending_for_session(r, include_choice_prompts=True) for r in routes)
            await adapter.handle_message(event(adapter, 'answer-default', 'blue', reply=questions['room']))
            await async_wait_until(lambda: any('DONE-DEFAULT' in s for s in adapter.sent), timeout=20, what='default done')
            assert clarify_gateway.get_pending_for_session(routes[1], include_choice_prompts=True)
            await adapter.handle_message(event(adapter, 'answer-alpha', 'green', reply=questions['alpha-room'], chat='alpha-room'))
            await async_wait_until(lambda: any('DONE-ALPHA' in s for s in adapter.sent), timeout=20, what='alpha done')
            assert [len(admissions(a)) for a in owners] == [1, 1]
        finally:
            for a in list(getattr(runner, 'session_authorities', ())):
                for live in a.sessions.values():
                    approval.clear_session(live.route)
                    clarify_gateway.clear_session(live.route)
            await async_wait_until(lambda: not adapter._background_tasks, timeout=30, what='multiplex prompt cleanup')
            for a in list(getattr(runner, 'session_authorities', ())):
                a.db.close()
            for _, home in homes:
                process_ownership.release(home)
