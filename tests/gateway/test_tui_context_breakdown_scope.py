"""``session.context_breakdown`` (Desktop context popover) and TUI ``/context`` measure the compression
tip's model history, and the popover's categories survive a session with no resident agent."""
from types import SimpleNamespace

import pytest

from tests.gateway import test_tui_session_reads as reads

owner = reads.owner


@pytest.mark.asyncio
async def test_context_breakdown_without_a_resident_agent_estimates_the_stored_freeze(owner, monkeypatch):
    """A session not run since the gateway started (or reaped) still gets the agent path's categories,
    from its frozen system prompt and tools[] pin, over its stored transcript."""
    from agent import system_prompt
    from agent.context_breakdown import compute_session_context_breakdown
    from gateway.session_tui_reads import _stored_id

    sid = (await owner.call('session.create', request_id='idle', source='tui', cwd=str(owner.workdir)))['session_id']
    db = owner.authority.db
    stored = _stored_id(owner.authority, SimpleNamespace(session_id=sid))
    prompt = 'You are Hermes. ' * 50 + '\n<available_skills>\n  - plan: write plans\n</available_skills>'
    tools = [{'type': 'function', 'function': {'name': 'terminal', 'parameters': {'type': 'object'}}},
             {'type': 'function', 'function': {'name': 'mcp_docs_search', 'description': 'search ' * 20}}]
    db.update_system_prompt(stored, prompt)
    db.update_session_tool_names(stored, {'version': 'v', 'tools': tools})
    db.append_message(stored, 'user', 'hello ' * 100)
    db.append_message(stored, 'assistant', 'hi')
    assert owner.authority.agent(SimpleNamespace(session_id=sid)) is None

    monkeypatch.setattr(system_prompt, 'build_system_prompt_parts', lambda a: {'stable': prompt})
    expected = compute_session_context_breakdown(SimpleNamespace(tools=tools), db.get_resume_conversations(stored)[0])
    actual = await owner.call('session.context_breakdown', session_id=sid)
    assert actual['categories'] == expected['categories'], actual
    assert {c['id'] for c in actual['categories']} == {'system_prompt', 'tool_definitions', 'skills', 'mcp',
                                                       'conversation'}
    assert actual['estimated_total'] == expected['estimated_total']


@pytest.mark.asyncio
async def test_context_breakdown_measures_only_the_compression_tips_model_history(owner, monkeypatch):
    """The runtime id still names the compressed parent; neither its superseded segment nor the tip's
    session bookkeeping is model context, in the popover or the TUI's /context table."""
    from agent import system_prompt
    from agent.context_breakdown import compute_session_context_breakdown, render_context_breakdown_lines
    from gateway.session_tui_reads import _stored_id

    sid = (await owner.call('session.create', request_id='lineage', source='tui', cwd=str(owner.workdir)))['session_id']
    db = owner.authority.db
    root = _stored_id(owner.authority, SimpleNamespace(session_id=sid))
    for i in range(4):
        db.append_message(root, 'user' if i % 2 == 0 else 'assistant', f'old segment {i} ' * 1000)
    db.end_session(root, 'compression')
    db.create_session('tip', source='tui', parent_session_id=root)
    db.append_message('tip', 'session_meta', 'not model input ' * 1000)
    db.append_message('tip', 'user', 'summary of prior turns')
    db.append_message('tip', 'assistant', 'current answer')
    agent = SimpleNamespace(model='fake-model', provider='custom', tools=[])
    owner.authority.runner._cached_agent_for = lambda route: agent
    monkeypatch.setattr(system_prompt, 'build_system_prompt_parts', lambda a: {'stable': 'frozen identity'})

    model_history = db.get_resume_conversations('tip')[0]
    assert [m['content'] for m in model_history] == ['summary of prior turns', 'current answer']
    expected = compute_session_context_breakdown(agent, model_history)
    actual = await owner.call('session.context_breakdown', session_id=sid)
    assert actual['categories'] == expected['categories'], actual
    assert actual['estimated_total'] == expected['estimated_total']
    assert '\n'.join(render_context_breakdown_lines(expected, grid=False)) in await reads._exec(owner, sid, 'context')
