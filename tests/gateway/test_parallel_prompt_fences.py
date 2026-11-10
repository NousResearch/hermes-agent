"""Independent exact-prompt probes; loopback only, all commands denied."""
import pytest

from tests.gateway.test_parallel_native_controls import confirmed_prompt, native_owner
from tests.gateway.test_parallel_native_ingress import admissions, event
from tests.fakes.fake_llm_provider import Text, ToolCall
from tests.waiting import async_wait_until


@pytest.mark.asyncio
@pytest.mark.parametrize('later_generation', [False, True])
async def test_old_approval_card_cannot_answer_a_new_request(monkeypatch, later_generation):
    from tools import approval
    calls = []

    def respond(request):
        calls.append(request)
        n = sum(m['role'] == 'tool' for m in request['body']['messages'])
        if n == 0 or (not later_generation and n == 1):
            return ToolCall('terminal', {'command': 'rm -rf /parallel-review-never-execute'})
        if later_generation:
            users = [m for m in request['body']['messages'] if m['role'] == 'user']
            tools_after_user = request['body']['messages'][request['body']['messages'].index(users[-1]) + 1:]
            if len(users) > 1 and not any(m['role'] == 'tool' for m in tools_after_user):
                return ToolCall('terminal', {'command': 'rm -rf /parallel-review-never-execute'})
        return Text('DONE')

    async with native_owner(monkeypatch, respond) as (_, authority, adapter):
        await adapter.handle_message(event(adapter, 'm1', 'FIRST'))
        await async_wait_until(lambda: bool(admissions(authority)), timeout=20, what='admitted')
        sid = admissions(authority)[0]['target_session_id']
        route = authority.sessions[sid].route
        await async_wait_until(lambda: bool(approval.list_gateway_approvals(route)), timeout=20, what='first approval')
        first = approval.list_gateway_approvals(route)[0]['request_id']
        await async_wait_until(lambda: any('approve' in text.lower() for _, text, _, _ in adapter.deliveries), timeout=20, what='first card')
        first_card = next(mid for _, text, _, mid in adapter.deliveries if 'approve' in text.lower())
        await async_wait_until(lambda: confirmed_prompt(authority, first_card), what='first prompt receipt committed')
        generation1 = authority.db.get_session(sid)['runtime_generation']
        await adapter.handle_message(event(adapter, 'deny-first', '/deny', reply=first_card))
        if later_generation:
            await async_wait_until(lambda: not adapter._background_tasks, timeout=20, what='first turn ends')
            await adapter.handle_message(event(adapter, 'm2', 'SECOND', reply='m1'))
        await async_wait_until(lambda: any(p['request_id'] != first for p in approval.list_gateway_approvals(route)), timeout=20, what='new approval')
        pending = approval.list_gateway_approvals(route)[0]['request_id']
        generation2 = authority.db.get_session(sid)['runtime_generation']
        assert (generation2 > generation1) == later_generation
        await adapter.handle_message(event(adapter, 'stale-answer', '/deny', reply=first_card))
        remaining = [p['request_id'] for p in approval.list_gateway_approvals(route)]
        print({'later_generation': later_generation, 'generation1': generation1, 'generation2': generation2,
               'old_request': first, 'new_request': pending, 'remaining_after_stale_reply': remaining})
        assert pending in remaining, 'stale card reply resolved a different pending request'


@pytest.mark.asyncio
async def test_old_clarify_card_cannot_answer_a_new_question(monkeypatch):
    from tools import clarify_gateway

    def respond(request):
        n = sum(m['role'] == 'tool' for m in request['body']['messages'])
        if n < 2:
            return ToolCall('clarify', {'questions': [{'question': 'QUESTION-' + str(n), 'choices': ['blue', 'green']}]})
        return Text('DONE')

    async with native_owner(monkeypatch, respond) as (_, authority, adapter):
        await adapter.handle_message(event(adapter, 'm1', 'CLARIFY'))
        await async_wait_until(lambda: any('QUESTION-0' in text for _, text, _, _ in adapter.deliveries), timeout=20, what='first clarify card')
        first_card = next(mid for _, text, _, mid in adapter.deliveries if 'QUESTION-0' in text)
        sid = admissions(authority)[0]['target_session_id']
        route = authority.sessions[sid].route
        await adapter.handle_message(event(adapter, 'answer-first', 'blue', reply=first_card))
        await async_wait_until(lambda: any('QUESTION-1' in text for _, text, _, _ in adapter.deliveries), timeout=20, what='second clarify card')
        second = clarify_gateway.get_pending_for_session(route, include_choice_prompts=True)
        second_card = next(mid for _, text, _, mid in adapter.deliveries if 'QUESTION-1' in text)
        await adapter.handle_message(event(adapter, 'foreign', 'blue', reply=second_card, user='bob'))
        assert not second.event.is_set(), 'foreign sender answered the question'
        await adapter.handle_message(event(adapter, 'stale-answer', 'blue', reply=first_card))
        print({'second_clarify_id': second.clarify_id, 'answered_by_old_card': second.event.is_set()})
        assert not second.event.is_set(), 'stale clarify card reply answered a different question'
