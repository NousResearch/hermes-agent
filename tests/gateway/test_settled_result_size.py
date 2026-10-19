"""A settled turn result stores this turn's output and accounting, never per-session copies."""
from gateway.session_results import admission_result, finish_result
from hermes_state_runtime import admit_session_input, claim_session_input


def test_stored_result_is_bounded_and_redaction_skips_earlier_turns(owner, monkeypatch):
    import agent.redact
    seen = []
    redact = agent.redact.redact_sensitive_text
    monkeypatch.setattr(agent.redact, 'redact_sensitive_text', lambda text, **kw: seen.append(text) or redact(text, **kw))
    owner.db.create_session('s', source='api_server')
    tools = [{'type': 'function', 'function': {'name': f'tool{i}', 'description': 'd' * 500}} for i in range(40)]
    history = [{'role': 'user', 'content': 'history-sentinel ' + 'h' * 2000}] * 50
    sizes = []
    for index in range(2):
        admit_session_input(owner.db, epoch=owner.epoch, principal_id='api', session_id='s',
                            request_id=str(index), payload={'text': f'ask {index}'})
        row = claim_session_input(owner.db, epoch=owner.epoch, session_id='s')
        history = history + [{'role': 'user', 'content': f'ask {index}'}, {'role': 'assistant', 'content': 'done'}]
        finish_result(owner.db, epoch=owner.epoch, row=row, response='done', outcome='completed',
                      result={'result': {'final_response': 'done', 'messages': list(history), 'tools': tools,
                                         'completed': True, 'model': 'm', 'turn_exit_reason': 'text_response',
                                         'estimated_cost_usd': .25},
                              'usage': {'input_tokens': 3}})
        saved = admission_result(owner.db, row['admission_id'])
        # Every stored-result reader keeps its fields: final text, -z ledger keys, usage.
        assert saved['usage'] == {'input_tokens': 3}
        assert {k: saved['result'][k] for k in ('final_response', 'completed', 'model', 'turn_exit_reason',
                'estimated_cost_usd')} == {'final_response': 'done', 'completed': True, 'model': 'm',
                'turn_exit_reason': 'text_response', 'estimated_cost_usd': .25}
        assert saved['result']['messages'] == [{'role': 'assistant', 'content': 'done'}]
        assert 'tools' not in saved['result']
        with owner.db._read_ctx() as conn:
            sizes.append(conn.execute('SELECT length(value) FROM state_meta WHERE key=?',
                                      ('gateway.admission.result.v1.' + row['admission_id'],)).fetchone()[0])
    assert sizes[0] == sizes[1] < 1024, sizes
    assert not any('history-sentinel' in text or 'd' * 500 in text for text in seen), 'redacted discarded copies'
