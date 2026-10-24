"""Cache-preserving effort updates must survive route and session boundaries."""
import time
from pathlib import Path
from types import SimpleNamespace

from agent.effort_updates import effort_update, record_effort_switch
from agent.transports.codex import ResponsesApiTransport
from agent.turn_context import build_api_messages
from hermes_state import SessionDB


def _wire(history):
    # Exercise the real durable-row -> API-copy projection, including metadata stripping.
    agent = SimpleNamespace(
        _current_turn_timestamp=time.time(), ephemeral_system_prompt=None,
        _copy_reasoning_content_for_api=lambda msg, row: None,
        _should_sanitize_tool_calls=lambda: False,
    )
    rows, _ = build_api_messages(
        agent, history, current_turn_user_idx=len(history) - 1,
        ext_prefetch_cache=None, plugin_user_context=None, moa_config=None,
        active_system_prompt=None,
    )
    return rows


def test_sol_effort_updates_preserve_baseline_through_sqlite_resume(tmp_path):
    db = SessionDB(db_path=Path(tmp_path) / 'state.db')
    db.create_session('effort-thread', 'cli', model='gpt-6.1-sol', model_config={})
    agent = SimpleNamespace(
        _session_db=db, session_id='effort-thread', _session_init_model_config={},
        reasoning_config={'enabled': True, 'effort': 'low'}, provider='openai-codex',
        model='gpt-6.1-sol', base_url='https://chatgpt.com/backend-api/codex',
    )
    history = []
    assert not record_effort_switch(agent, history)
    history += [{'role': 'user', 'content': 'first'}, {'role': 'assistant', 'content': 'OK'}]
    agent.reasoning_config = {'enabled': True, 'effort': 'medium'}
    assert record_effort_switch(agent, history)
    history.append({'role': 'user', 'content': 'second'})
    params = dict(model=agent.model, base_url=agent.base_url, provider=agent.provider,
                  is_codex_backend=True, session_id=agent.session_id)
    transport = ResponsesApiTransport()
    first = transport.preflight_kwargs(transport.build_kwargs(
        messages=_wire(history), reasoning_config=agent.reasoning_config, **params))
    assert first['reasoning']['effort'] == 'low'
    assert first['input'][-2]['type'] == 'configuration_update'
    assert first['input'][-2]['reasoning']['effort'] == 'medium'
    history.append({'role': 'assistant', 'content': 'OK'})
    for message in history:
        db.append_message(
            'effort-thread', message['role'], content=message.get('content'),
            display_kind=message.get('display_kind'), display_metadata=message.get('display_metadata'),
        )
    history = db.get_messages_as_conversation('effort-thread', repair_alternation=True)
    assert any(effort_update(message) is not None for message in history)
    # A fresh agent obtains the baseline and marker from the real database, not process state.
    resumed = SimpleNamespace(**{**vars(agent), '_session_init_model_config': {},
                                'reasoning_config': {'enabled': True, 'effort': 'low'}})
    assert record_effort_switch(resumed, history)
    history.append({'role': 'user', 'content': 'third'})
    second = transport.preflight_kwargs(transport.build_kwargs(
        messages=_wire(history), reasoning_config=resumed.reasoning_config, **params))
    assert second['reasoning']['effort'] == 'low'
    assert second['input'][:len(first['input'])] == first['input']
    assert second['input'][-2] == {'type': 'configuration_update', 'reasoning': {'effort': 'low'}}
    db.close()


def test_explicit_effort_override_wins_without_an_in_band_switch():
    # No marker in history: request_overrides must keep the last word on reasoning.effort.
    kwargs = ResponsesApiTransport().build_kwargs(
        model='gpt-6-luna', messages=[{'role': 'user', 'content': 'hi'}], provider='openai',
        base_url='https://api.openai.com/v1', reasoning_config={'enabled': True, 'effort': 'low'},
        request_overrides={'reasoning': {'effort': 'high'}},
    )
    assert kwargs['reasoning']['effort'] == 'high'


def test_claude_lineage_filtered_markers_keep_the_beta_and_effort_mapping():
    from agent.anthropic_adapter import build_anthropic_kwargs
    from agent.effort_updates import ANTHROPIC_MID_CONVERSATION_EFFORT_BETA, EFFORT_UPDATE_KEY, make_effort_update_message

    def wire(msg):
        api = {k: v for k, v in msg.items() if k not in ('display_kind', 'display_metadata')}
        if (u := effort_update(msg)) is not None:
            api[EFFORT_UPDATE_KEY] = u
        return api
    history = [
        {'role': 'user', 'content': 'a'}, {'role': 'assistant', 'content': 'b'},
        make_effort_update_message('low', 'high', lineage='old'),  # previous model's lineage
        {'role': 'user', 'content': 'c'}, {'role': 'assistant', 'content': 'd'},
        make_effort_update_message('medium', 'medium', lineage='new', reset=True),
        {'role': 'user', 'content': 'e'}, {'role': 'assistant', 'content': 'f'},
        make_effort_update_message('ultra', 'medium', lineage='new'),
        {'role': 'user', 'content': 'g'},
    ]
    kwargs = build_anthropic_kwargs('claude-fable-5-1', [wire(m) for m in history], None, None,
                                    {'enabled': True, 'effort': 'ultra'})
    in_band = [m for m in kwargs['messages'] if m['role'] == 'system']
    assert in_band == [{'role': 'system', 'content': [], 'output_config': {'effort': 'max'}}]
    assert ANTHROPIC_MID_CONVERSATION_EFFORT_BETA in kwargs['extra_headers']['anthropic-beta']
