"""Current Files projection vs ordinary sidecar/replay, without provider I/O."""
import asyncio
import copy
import json
from pathlib import Path

import pytest

from hermes_state import SessionDB
from agent.session_persistence import files_user_message_persistence
from tests.agent.files_persistence_fixtures import inert_agent


@pytest.mark.parametrize('exit_kind', ['success', 'exception', 'interrupt', 'cancel'])
@pytest.mark.parametrize('native', [False, True])
def test_current_projection_resets_without_rewriting_prefix(tmp_path, monkeypatch, exit_kind, native, record_property):
    db = SessionDB(tmp_path / 'state.db')
    try:
        agent, sent, at_provider = inert_agent(monkeypatch, db, 's')
        db.create_session(session_id='s', source='api_server')
        agent._session_db_created = True
        # Historical private-looking strings MUST stay untouched: no global purge.
        old_wire = 'old ordinary /private/history.txt\n\nSIDE-CAR'
        db.append_message('s', 'user', content='old ordinary', api_content=old_wire)
        db.append_message('s', 'assistant', content='old answer')
        history = db.get_messages_as_conversation('s')
        before = copy.deepcopy(history)
        rows_before = db.get_messages('s')
        live = 'new document /private/working-document.txt'
        if native:
            live = [{'type': 'text', 'text': live},
                    {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AA=='}}]
        safe = 'accepted prompt\n\n[Attached file: "document.txt"]'
        original_provider = agent.client.chat.completions.create.side_effect
        def provider(**kwargs):
            value = original_provider(**kwargs)
            # Force a repeated request assembly on the actual live context. It must
            # neither erase the model's files early nor rewrite older prefixes.
            from agent.turn_context import build_api_messages
            rebuilt, _ = build_api_messages(agent, agent._session_messages,
                current_turn_user_idx=agent._persist_user_message_idx, ext_prefetch_cache='',
                plugin_user_context='', moa_config=None, active_system_prompt=agent._cached_system_prompt)
            assert rebuilt[-1]['content'] == kwargs['messages'][-1]['content']
            assert kwargs['messages'][1]['content'] == old_wire
            if exit_kind == 'exception':
                raise RuntimeError('inert provider exception')
            if exit_kind == 'cancel':
                raise asyncio.CancelledError('inert cancellation')
            if exit_kind == 'interrupt':
                agent.interrupt()
            return value
        agent.client.chat.completions.create.side_effect = provider
        # Exception/cancellation exercise the actual prologue and crash flush before
        # raising, not a recorder that never staged/persisted a user row.
        with files_user_message_persistence(agent, safe) as projection:
            if exit_kind in ('exception', 'cancel'):
                # Raise outside retry policy, after actual build/persist but before
                # the SDK request. This keeps the test bounded (zero provider retries).
                from agent import conversation_loop
                real_assemble = conversation_loop.assemble_api_request
                from functools import wraps
                @wraps(real_assemble)
                def failing_assemble(*a, **kw):
                    real_assemble(*a, **kw)
                    exc = RuntimeError if exit_kind == 'exception' else asyncio.CancelledError
                    raise exc('inert turn exception')
                with monkeypatch.context() as m:
                    m.setattr(conversation_loop, 'assemble_api_request', failing_assemble)
                    with pytest.raises((RuntimeError, asyncio.CancelledError)):
                        agent.run_conversation(live, conversation_history=history, persist_user_message=projection)
            else:
                result = agent.run_conversation(live, conversation_history=history, persist_user_message=projection)
        assert agent._persist_user_message_override is None
        assert history == before
        assert db.get_messages('s')[:2] == rows_before
        user = [r for r in db.get_messages('s') if r['role'] == 'user'][-1]
        assert user['content'] == safe and user['api_content'] is None
        assert projection.message['content'] == safe
        assert 'api_content' not in projection.message
        assert '/private/working-document.txt' not in json.dumps(agent._session_messages)
        agent._persist_session(agent._session_messages)
        # The next ordinary turn must use ordinary sidecar semantics, not inherit
        # the Files privacy policy; saved history retains the historical sidecar.
        agent.clear_interrupt()
        agent.client.chat.completions.create.side_effect = original_provider
        replay = db.get_messages_as_conversation('s')
        agent.run_conversation('ordinary /private/followup.txt', conversation_history=replay,
                               persist_user_message='ordinary caption')
        final = [r for r in db.get_messages('s') if r['role'] == 'user'][-1]
        assert final['content'] == 'ordinary caption'
        assert final['api_content'] == 'ordinary /private/followup.txt'
        assert sent[-1][0]['content'] == 'Frozen system prefix'
        assert [m for m in sent[-1] if m['role'] == 'user'][0]['content'] == old_wire
        record_property('files_sql', json.dumps(user, default=lambda b: b.hex()))
        record_property('ordinary_followup_sql', json.dumps(final, default=lambda b: b.hex()))
    finally:
        db.close()


@pytest.mark.parametrize('native', [False, True])
def test_ordinary_multimodal_and_sidecar_saved_replay_unchanged(tmp_path, monkeypatch, native):
    db = SessionDB(tmp_path / 'ordinary.db')
    try:
        agent, sent, _ = inert_agent(monkeypatch, db, 'ordinary')
        content = 'ordinary /private/reference.txt'
        if native:
            content = [{'type': 'text', 'text': content},
                       {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AA=='}}]
        result = agent.run_conversation(content, persist_user_message='caption')
        row = db.get_messages('ordinary')[0]
        assert sent[0][-1]['content'] == content
        if native:
            assert row['content'] == 'ordinary /private/reference.txt\n[screenshot]'
            assert row['api_content'] is None
            assert result['messages'][0]['content'] == content
        else:
            assert row['content'] == 'caption'
            assert row['api_content'] == content
        saved = db.get_messages_as_conversation('ordinary')
        agent.run_conversation('next', conversation_history=saved)
        assert [m for m in sent[-1] if m['role'] == 'user'][0]['content'] == (row['content'] if native else content)
    finally:
        db.close()


def test_trajectory_projection_is_current_only_and_actual_jsonl(tmp_path, monkeypatch):
    db = SessionDB(tmp_path / 'trajectory.db')
    monkeypatch.chdir(tmp_path)
    try:
        agent, sent, _ = inert_agent(monkeypatch, db, 'trajectory')
        agent.save_trajectories = True
        safe = 'accepted prompt\n\n[Attached file: "a.txt"]'
        with files_user_message_persistence(agent, safe) as projection:
            result = agent.run_conversation('/private/current.txt', persist_user_message=projection)
        body = (tmp_path / 'trajectory_samples.jsonl').read_text()
        assert '/private/current.txt' not in body
        assert 'accepted prompt' in body
        assert json.loads(body)['completed'] is True
        assert result['messages'][0]['content'] == safe
        assert sent[0][-1]['content'] == '/private/current.txt'
    finally:
        db.close()
