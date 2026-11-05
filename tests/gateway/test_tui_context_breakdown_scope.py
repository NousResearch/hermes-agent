"""``session.context_breakdown`` (Desktop context popover) and TUI ``/context`` measure the compression
tip's model history, and the popover's categories survive a session with no resident agent."""
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.gateway import test_tui_session_reads as reads
from tests.fixtures.storage_holder import raw_holder
from tests.fixtures.storage_seed import seed_messages, seed_session

owner = reads.owner


@pytest.mark.asyncio
async def test_context_breakdown_without_a_resident_agent_estimates_the_stored_freeze(owner, monkeypatch):
    """A session not run since the gateway started (or reaped) still gets the agent path's categories,
    from its frozen system prompt and tools[] pin, over its stored transcript."""
    from agent import system_prompt
    from agent.context_breakdown import compute_session_context_breakdown
    from gateway.session_tui_reads import _stored_id

    sid = (await owner.call('session.create', request_id='idle', source='tui', cwd=str(owner.workdir)))['session_id']
    db = owner.authority.persistence
    stored = _stored_id(owner.authority, SimpleNamespace(session_id=sid))
    prompt = 'You are Hermes. ' * 50 + '\n<available_skills>\n  - plan: write plans\n</available_skills>'
    tools = [{'type': 'function', 'function': {'name': 'terminal', 'parameters': {'type': 'object'}}},
             {'type': 'function', 'function': {'name': 'mcp_docs_search', 'description': 'search ' * 20}}]
    db.sessions.update_system_prompt(stored, prompt)
    db.sessions.update_tool_pin(stored, {'version': 'v', 'tools': tools})
    seed_messages(db, stored, {'role': 'user', 'content': 'hello ' * 100}, {'role': 'assistant', 'content': 'hi'})
    assert owner.authority.agent(SimpleNamespace(session_id=sid)) is None

    monkeypatch.setattr(system_prompt, 'build_system_prompt_parts', lambda a: {'stable': prompt})
    history = [m.as_openai_message() for m in db.messages.read_resume_conversations(stored).model]
    expected = compute_session_context_breakdown(SimpleNamespace(tools=tools), history)
    actual = await owner.call('session.context_breakdown', session_id=sid)
    assert actual['categories'] == expected['categories'], actual
    assert {c['id'] for c in actual['categories']} == {'system_prompt', 'tool_definitions', 'skills', 'mcp',
                                                       'conversation'}
    assert actual['estimated_total'] == expected['estimated_total']


@pytest.mark.asyncio
async def test_agentless_breakdown_refreshes_when_the_stored_prompt_or_tools_change(owner):
    """A compute worker can freeze the prompt or tools pin without touching the transcript; the cached
    agentless breakdown must not outlive either write."""
    from gateway.session_tui_reads import _stored_id

    sid = (await owner.call('session.create', request_id='refreeze', source='tui', cwd=str(owner.workdir)))['session_id']
    db = owner.authority.persistence
    stored = _stored_id(owner.authority, SimpleNamespace(session_id=sid))
    seed_messages(db, stored, {'role': 'user', 'content': 'hello'})
    before = await owner.call('session.context_breakdown', session_id=sid)
    db.sessions.update_system_prompt(stored, 'You are Hermes. ' * 50)
    prompted = await owner.call('session.context_breakdown', session_id=sid)
    assert prompted['estimated_total'] > before['estimated_total'], prompted
    db.sessions.update_tool_pin(stored, {'version': 'v', 'tools': [
        {'type': 'function', 'function': {'name': 'terminal', 'parameters': {'type': 'object'}}}]})
    tooled = await owner.call('session.context_breakdown', session_id=sid)
    assert 'tool_definitions' in {c['id'] for c in tooled['categories']}, tooled


async def _agentless_session(owner, request_id, pin):
    from gateway.session_tui_reads import _stored_id

    sid = (await owner.call('session.create', request_id=request_id, source='tui', cwd=str(owner.workdir)))['session_id']
    stored = _stored_id(owner.authority, SimpleNamespace(session_id=sid))
    owner.authority.persistence.sessions.update_system_prompt(stored, 'You are Hermes. ' * 50)
    owner.authority.persistence.sessions.update_tool_pin(stored, pin)
    seed_messages(owner.authority.persistence, stored, {'role': 'user', 'content': 'hello'})
    return sid, stored


@pytest.mark.asyncio
async def test_agentless_breakdown_measures_a_legacy_name_only_pin(owner):
    """A legacy name-only pin measures each tool's registry schema, as the next agent restores it."""
    import tools.terminal_tool  # registers ``terminal``

    sid, _ = await _agentless_session(owner, 'legacy', ['terminal'])
    actual = await owner.call('session.context_breakdown', session_id=sid)
    assert 'tool_definitions' in {c['id'] for c in actual['categories']}, actual


def test_a_legacy_pin_resolves_before_any_agent_ran_tool_discovery():
    """The gateway reads an agentless breakdown before its warm-up imports the tools (or with it off);
    only a fresh interpreter has that cold registry."""
    code = ("from tools.mcp_tool_agent import pinned_tool_defs; "
            "print(sorted(d['function']['name'] for d in pinned_tool_defs(['terminal'])))")
    out = subprocess.run([sys.executable, '-c', code], cwd=Path(__file__).resolve().parents[2],
                         capture_output=True, text=True, check=True, timeout=120).stdout
    assert out.strip().splitlines()[-1] == "['terminal']", out


@pytest.mark.asyncio
async def test_agentless_breakdown_survives_a_swept_tools_pin(owner):
    """A pin hash whose ``system_prompts`` row was swept resolves to the bare hash: no tools, no error."""
    sid, stored = await _agentless_session(owner, 'swept', {'version': 'v', 'tools': [
        {'type': 'function', 'function': {'name': 'terminal', 'parameters': {'type': 'object'}}}]})
    raw_holder(owner.authority.persistence)._execute_write(lambda conn: conn.execute(
        'DELETE FROM system_prompts WHERE hash = (SELECT tool_names FROM sessions WHERE id = ?)', (stored,)))
    actual = await owner.call('session.context_breakdown', session_id=sid)
    assert {c['id'] for c in actual['categories']} == {'system_prompt', 'conversation'}, actual


@pytest.mark.asyncio
async def test_context_breakdown_measures_only_the_compression_tips_model_history(owner, monkeypatch):
    """The runtime id still names the compressed parent; neither its superseded segment nor the tip's
    session bookkeeping is model context, in the popover or the TUI's /context table."""
    from agent import system_prompt
    from agent.context_breakdown import compute_session_context_breakdown, render_context_breakdown_lines
    from gateway.session_tui_reads import _stored_id

    sid = (await owner.call('session.create', request_id='lineage', source='tui', cwd=str(owner.workdir)))['session_id']
    db = owner.authority.persistence
    root = _stored_id(owner.authority, SimpleNamespace(session_id=sid))
    for i in range(4):
        seed_messages(db, root, {'role': 'user' if i % 2 == 0 else 'assistant', 'content': f'old segment {i} ' * 1000})
    db.sessions.end(root, 'compression')
    seed_session(db, 'tip', 'tui', parent_session_id=root)
    seed_messages(db, 'tip', {'role': 'session_meta', 'content': 'not model input ' * 1000},
                  {'role': 'user', 'content': 'summary of prior turns'}, {'role': 'assistant', 'content': 'current answer'})
    agent = SimpleNamespace(model='fake-model', provider='custom', tools=[])
    owner.authority.runner._cached_agent_for = lambda route: agent
    monkeypatch.setattr(system_prompt, 'build_system_prompt_parts', lambda a: {'stable': 'frozen identity'})

    model_history = [m.as_openai_message() for m in db.messages.read_resume_conversations('tip').model]
    assert [m['content'] for m in model_history] == ['summary of prior turns', 'current answer']
    expected = compute_session_context_breakdown(agent, model_history)
    actual = await owner.call('session.context_breakdown', session_id=sid)
    assert actual['categories'] == expected['categories'], actual
    assert actual['estimated_total'] == expected['estimated_total']
    assert '\n'.join(render_context_breakdown_lines(expected, grid=False)) in await reads._exec(owner, sid, 'context')
