"""Safe canonical history and request-only Files disclosure (inert SDK, real SQL)."""
import copy
import json
from types import SimpleNamespace

import pytest

from agent.session_persistence import files_user_message_persistence
from hermes_state import SessionDB
from tests.agent.files_persistence_fixtures import inert_agent

SAFE = 'accepted prompt\n\n[Attached file: "a.txt"]'
PRIVATE = '/private/only-provider-document.txt'


@pytest.fixture
def live(tmp_path, monkeypatch):
    db = SessionDB(tmp_path / 'state.db')
    agent, sent, sql = inert_agent(monkeypatch, db, 'live')
    yield agent, sent, sql, db
    db.close()


@pytest.mark.parametrize('native', [False, True])
def test_provider_copy_and_result_are_independent(live, monkeypatch, native):
    agent, sent, sql, db = live
    payload = ([{'type': 'text', 'text': PRIVATE},
                {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AA=='}}]
               if native else PRIVATE)
    during = []
    original = agent.client.chat.completions.create.side_effect
    def provider(**kwargs):
        during.append(copy.deepcopy(agent._session_messages))
        response = original(**kwargs)
        kwargs['messages'][-1]['content'] = 'SDK MUTATION'
        if native:
            payload[0]['text'] = 'CALLER MUTATION'
        return response
    agent.client.chat.completions.create.side_effect = provider
    with files_user_message_persistence(agent, SAFE) as transcript:
        result = agent.run_conversation(payload, persist_user_message=transcript)
    assert PRIVATE in json.dumps(sent[0])
    assert PRIVATE not in json.dumps(during)
    assert 'data:image/' not in json.dumps(during)
    result['messages'][0]['content'] = 'RESULT MUTATION'
    assert agent._session_messages[0]['content'] == SAFE
    agent.run_conversation('followup', conversation_history=agent._session_messages)
    assert sent[1][:len(sent[0])] == sent[0]
    assert db.get_messages('live')[0]['api_content'] is None


def test_debug_and_hook_exports_omit_request_before_copy(live, monkeypatch, tmp_path):
    agent, sent, sql, db = live
    from agent.turn_api_request import _fire_pre_api_request_hook
    calls, dumps = [], []
    monkeypatch.setattr('hermes_cli.lifecycle.has_hook', lambda name: name == 'pre_api_request')
    def hook(name, **kwargs):
        if name == 'pre_api_request':
            calls.append(kwargs)
        return []
    monkeypatch.setattr('hermes_cli.lifecycle.invoke_hook', hook)
    agent.logs_dir = tmp_path
    original = agent.client.chat.completions.create.side_effect
    def provider(**kwargs):
        dumps.append(agent._dump_api_request_debug(kwargs, reason='error', error=ValueError(PRIVATE)))
        return original(**kwargs)
    agent.client.chat.completions.create.side_effect = provider
    with files_user_message_persistence(agent, SAFE) as transcript:
        agent.run_conversation(PRIVATE, persist_user_message=transcript)
    assert calls
    assert calls[0]['request'] == {}
    assert calls[0]['request_messages'] == []
    assert calls[0]['middleware_trace'] == []
    assert calls[0]['files_payload_omitted'] is True
    assert PRIVATE not in json.dumps(calls)
    assert dumps[0] is not None
    dump = json.loads(dumps[0].read_text())
    assert dump['files_payload_omitted'] is True
    assert PRIVATE not in json.dumps(dump)


@pytest.mark.parametrize('replace_row', [False, True])
def test_current_compaction_preserves_only_exact_binding(live, monkeypatch, replace_row):
    agent, sent, sql, db = live
    seen = []
    from agent.turn_context_compaction import CompactionOutcome
    def compact(_agent, **kwargs):
        messages = kwargs['messages']
        seen.append(copy.deepcopy(messages))
        return CompactionOutcome(copy.deepcopy(messages) if replace_row else messages,
            kwargs['active_system_prompt'], kwargs['conversation_history'],
            kwargs['current_turn_user_idx'], compressed=replace_row)
    monkeypatch.setattr('agent.turn_context_compaction.run_turn_start_compaction', compact)
    with files_user_message_persistence(agent, SAFE) as transcript:
        if replace_row:
            with pytest.raises(RuntimeError, match='Files context unavailable'):
                agent.run_conversation(PRIVATE, persist_user_message=transcript)
        else:
            agent.run_conversation(PRIVATE, persist_user_message=transcript)
    assert PRIVATE not in json.dumps(seen)
    assert len(sent) == (0 if replace_row else 1)


def test_preflight_prices_private_content_without_exposing_canonical(live, monkeypatch):
    agent, sent, sql, db = live
    from agent.turn_context import _preflight_request_tokens
    from agent.model_metadata import estimate_request_tokens_rough
    observations = []
    from agent.turn_context_compaction import CompactionOutcome
    payload = PRIVATE + ' many tokens ' * 1000
    def compact(_agent, **kwargs):
        messages = kwargs['messages']
        priced = _preflight_request_tokens(agent, messages, '')
        expected = estimate_request_tokens_rough([{'role': 'user', 'content': payload}], system_prompt='', tools=agent.tools)
        observations.append((priced, expected, copy.deepcopy(messages)))
        return CompactionOutcome(messages, kwargs['active_system_prompt'],
            kwargs['conversation_history'], kwargs['current_turn_user_idx'])
    monkeypatch.setattr('agent.turn_context_compaction.run_turn_start_compaction', compact)
    with files_user_message_persistence(agent, SAFE) as transcript:
        agent.run_conversation(payload, persist_user_message=transcript)
    assert observations[0][0] >= observations[0][1]
    assert PRIVATE not in json.dumps(observations[0][2])


def test_alternate_raw_result_is_refused_before_serialization(live):
    agent, sent, sql, db = live
    from agent.files_live_context import safe_files_result
    with files_user_message_persistence(agent, SAFE) as transcript:
        agent.run_conversation(PRIVATE, persist_user_message=transcript)
        result = safe_files_result(agent, {'messages': sent[0], 'completed': True})
    assert result['failed'] is True
    assert result['messages'] == []
    assert PRIVATE not in json.dumps(result)


def test_result_boundary_survives_repeated_safe_projection(live):
    agent, sent, sql, db = live
    from agent.files_live_context import safe_files_result
    with files_user_message_persistence(agent, SAFE) as transcript:
        result = agent.run_conversation(PRIVATE, persist_user_message=transcript)
        assert result['current_turn_user_idx'] == 0
        again = safe_files_result(agent, result)
        assert again['current_turn_user_idx'] == 0
        assert again['turn_id'] == result['turn_id']


def test_small_safe_caption_cannot_skip_private_pressure_gate(live, monkeypatch):
    agent, sent, sql, db = live
    agent.compression_enabled = True
    agent.context_compressor.threshold_tokens = 100
    monkeypatch.setattr(agent.context_compressor, 'should_compress', lambda n: n >= 100)
    monkeypatch.setattr(agent.context_compressor, 'should_defer_preflight_to_real_usage', lambda n: False)
    monkeypatch.setattr(agent.context_compressor, 'get_active_compression_failure_cooldown', lambda: None)
    seen = []
    def compress(messages, system, **kwargs):
        seen.append(copy.deepcopy(messages))
        return copy.deepcopy(messages), system
    agent._compress_context = compress
    with files_user_message_persistence(agent, SAFE) as transcript:
        with pytest.raises(RuntimeError, match='Files context unavailable'):
            agent.run_conversation(PRIVATE + ' word ' * 1000, persist_user_message=transcript)
    assert seen and not sent
    assert PRIVATE not in json.dumps(seen)


def test_copied_owner_cannot_retire_original_payload(live):
    agent, sent, sql, db = live
    from agent.files_live_context import prune_files_context, files_provider_content
    with files_user_message_persistence(agent, SAFE) as transcript:
        agent.run_conversation(PRIVATE, persist_user_message=transcript)
    fork = copy.copy(agent)
    prune_files_context(fork, [])
    assert files_provider_content(agent, agent._session_messages[0]) == (True, PRIVATE)


@pytest.mark.parametrize('native', [False, True])
def test_frozen_notes_and_prefetch_are_sealed_once(live, monkeypatch, native):
    agent, sent, sql, db = live
    monkeypatch.setattr('agent.turn_context._collect_pre_llm_call_context', lambda *a, **k: 'PLUGIN ORIGINAL')
    monkeypatch.setattr('agent.turn_context._memory_turn_start_and_prefetch', lambda *a, **k: 'MEMORY ORIGINAL')
    agent._gateway_turn_context_notes = 'GATEWAY ORIGINAL'
    payload = [{'type': 'text', 'text': PRIVATE}, {'type': 'image_url',
        'image_url': {'url': 'data:image/png;base64,AA=='}}] if native else PRIVATE
    with files_user_message_persistence(agent, SAFE) as transcript:
        result = agent.run_conversation(payload, persist_user_message=transcript,
            persist_user_display_metadata={'nested': {'x': ['original']}})
    first = copy.deepcopy(sent[0])
    result['messages'][0]['display_metadata']['nested']['x'][0] = 'CHANGED'
    assert agent._session_messages[0]['display_metadata']['nested']['x'][0] == 'original'
    if native:
        assert first[-1]['content'][-1] == {'type': 'text', 'text': 'GATEWAY ORIGINAL'}
        assert 'MEMORY ORIGINAL' not in json.dumps(first)
    else:
        assert 'PLUGIN ORIGINAL' in first[-1]['content']
        assert 'MEMORY ORIGINAL' in first[-1]['content']
    monkeypatch.setattr('agent.turn_context._collect_pre_llm_call_context', lambda *a, **k: 'NEW PLUGIN')
    agent.run_conversation('next', conversation_history=agent._session_messages)
    assert sent[-1][:len(first)] == first


@pytest.mark.parametrize('lifecycle', ['soft', 'close', 'session', 'db', 'history'])
def test_private_lifetime_is_bounded_by_owner_and_selected_history(live, monkeypatch, lifecycle, tmp_path):
    agent, sent, sql, db = live
    with files_user_message_persistence(agent, SAFE) as transcript:
        agent.run_conversation(PRIVATE, persist_user_message=transcript)
    entry = agent._files_live_entries[0]
    from agent.files_live_context import prune_files_context
    if lifecycle == 'soft':
        from gateway.run import GatewayRunner
        monkeypatch.setattr(agent, 'release_clients', lambda: None)
        GatewayRunner._release_evicted_agent_soft(object.__new__(GatewayRunner), agent)
    elif lifecycle == 'close':
        # Test the real close boundary, replacing only resource/native teardown.
        for name in ('shutdown_memory_provider', '_close_task_resources', '_close_active_children',
                     '_drop_shared_client', '_close_request_clients', '_close_codex_session',
                     '_trim_process_memory', '_finalize_owned_session_row'):
            monkeypatch.setattr(agent, name, lambda *a, **k: None)
        agent.close()
    else:
        if lifecycle == 'session':
            agent.session_id = 'another-session'
        elif lifecycle == 'db':
            agent._session_db = object()
        prune_files_context(agent, [] if lifecycle == 'history' else agent._session_messages)
    assert not agent._files_live_entries
    assert entry.prepared is None and entry.sealed is None and entry.row is None


def test_real_compression_worker_receives_safe_clone_and_noop_retains_identity(live):
    import threading
    from agent.compression_facade import _run_under_progress_timeout
    from agent.conversation_compression import CompressionCommitFence
    from agent.files_live_context import files_provider_content
    agent, sent, sql, db = live
    with files_user_message_persistence(agent, SAFE) as transcript:
        agent.run_conversation(PRIVATE, persist_user_message=transcript)
    original = agent._session_messages
    seen = []
    def worker(fence, *, target_messages, **kwargs):
        assert target_messages is not original
        assert target_messages[0] is not original[0]
        seen.append(copy.deepcopy(target_messages))
        return target_messages, 'system'
    messages, _ = _run_under_progress_timeout(agent, worker, original, 'system',
        active_fence=CompressionCommitFence(), fence_registration_lock=threading.Lock(),
        idle_timeout=5, total_ceiling=10)
    assert messages is original
    assert PRIVATE not in json.dumps(seen)
    assert files_provider_content(agent, messages[0]) == (True, PRIVATE)


def test_dump_refuses_to_copy_private_body_and_error(live, tmp_path):
    agent, sent, sql, db = live
    with files_user_message_persistence(agent, SAFE) as transcript:
        agent.run_conversation(PRIVATE, persist_user_message=transcript)
    class Uncopyable(dict):
        def __deepcopy__(self, memo):
            pytest.fail('raw provider dump was copied')
    class UnprintableError(Exception):
        def __str__(self):
            pytest.fail('private provider error was formatted')
    agent.logs_dir = tmp_path
    path = agent._dump_api_request_debug(Uncopyable(messages=sent[0]), reason=PRIVATE, error=UnprintableError())
    assert path and PRIVATE not in path.read_text()


@pytest.mark.parametrize('exit_kind', ['normal', 'partial', 'alternate_raw'])
def test_public_facade_projects_before_task_observation(live, monkeypatch, exit_kind):
    from types import MethodType
    from run_agent import AIAgent
    agent, sent, sql, db = live
    observed = []
    coordinator = SimpleNamespace(
        acquire_conversation=lambda **k: object(),
        begin_turn=lambda *a, **k: SimpleNamespace(relay_enabled=True),
        finish_logical_calls=lambda *a, **k: None,
        end_turn=lambda *a, **k: None,
        release_conversation=lambda *a, **k: None)
    monkeypatch.setattr('agent.relay_runtime.SESSION_COORDINATOR', coordinator)
    monkeypatch.setattr('hermes_cli.observability.relay_shared_metrics.start_task_run', lambda **k: None)
    monkeypatch.setattr('hermes_cli.observability.relay_shared_metrics.finish_task_run',
                        lambda **k: observed.append(copy.deepcopy(k)))
    agent.run_conversation = MethodType(AIAgent.run_conversation, agent)
    if exit_kind != 'normal':
        from agent import conversation_loop
        real_loop = conversation_loop.run_conversation
        def alternate(*args, **kwargs):
            result = real_loop(*args, **kwargs)
            if exit_kind == 'partial':
                return {**result, 'partial': True, 'completed': False}
            return {'completed': True, 'messages': sent[-1]}
        monkeypatch.setattr(conversation_loop, 'run_conversation', alternate)
    with files_user_message_persistence(agent, SAFE) as transcript:
        result = agent.run_conversation(PRIVATE, persist_user_message=transcript)
    assert observed and PRIVATE not in json.dumps(observed)
    assert PRIVATE not in json.dumps(result)
    if exit_kind == 'alternate_raw':
        assert result['failed'] and result['messages'] == []
    else:
        result['messages'][0]['content'] = 'external mutation'
        assert agent._session_messages[0]['content'] == SAFE
