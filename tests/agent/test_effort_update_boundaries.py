"""Cache-preserving effort updates must survive route and session boundaries."""
from pathlib import Path
from types import SimpleNamespace

from agent.effort_updates import EFFORT_UPDATE_KEY, effort_update, record_effort_switch
from agent.transports.codex import ResponsesApiTransport
from hermes_state import SessionDB


def _wire(history):
    rows = []
    for message in history:
        row = {k: v for k, v in message.items() if k not in ('display_kind', 'display_metadata')}
        if (update := effort_update(message)) is not None:
            row[EFFORT_UPDATE_KEY] = dict(update)
        rows.append(row)
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
    # A fresh agent obtains the baseline from the real database, not process state.
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
