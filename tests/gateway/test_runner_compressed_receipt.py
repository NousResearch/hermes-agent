"""A TurnRunner result that compressed the transcript is stored whole, never cut to a turn suffix."""
import pytest

from tests.gateway.test_compression_failure_session_sync import (
    _CompressionThenFailureAgent, _install_compression_failure_agent, _run_compression_failure_turn, _runner,
    _SessionStore)
from gateway.config import Platform
from gateway.session import SessionSource


class _CompressingAgent(_CompressionThenFailureAgent):
    rotate = True

    def run_conversation(self, user_message, conversation_history=None, task_id=None, **_kwargs):
        if self.rotate:
            self.session_id = 'session-after-compression'
        else:
            self._last_compaction_in_place = True
        return {'final_response': 'compressed answer', 'completed': True, 'api_calls': 1, 'messages': [
            {'role': 'user', 'content': '[CONTEXT COMPACTION] summary', '_compressed_summary': True},
            {'role': 'user', 'content': user_message}, {'role': 'assistant', 'content': 'compressed answer'}]}


@pytest.mark.parametrize('rotate', [True, False], ids=['rotated', 'in_place'])
def test_runner_compression_keeps_the_replacement_transcript_in_the_receipt(monkeypatch, rotate):
    from hermes_state_terminal import compact_result
    monkeypatch.setattr(_CompressingAgent, 'rotate', rotate)
    _install_compression_failure_agent(monkeypatch, _CompressingAgent)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id='12345', chat_type='dm', user_id='user-1')
    result = _run_compression_failure_turn(_runner(_SessionStore()), source)
    stored = compact_result({'result': result}, user_message='continue')['result']
    assert stored['messages'] == result['messages'] and len(stored['messages']) == 3, (
        'compression replacement cut to the turn suffix: the next chained turn re-sends uncompressed history')


class _NudgedImageAgent(_CompressionThenFailureAgent):
    def run_conversation(self, user_message, conversation_history=None, task_id=None, **_kwargs):
        # The stored row differs from the payload text (image hint appended), then a verify-on-stop nudge.
        messages = [*(conversation_history or []), {'role': 'user', 'content': user_message + '\n[Image attached at: /x.png]'},
                    {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 't1', 'type': 'function',
                     'function': {'name': 'read_file', 'arguments': '{}'}}]},
                    {'role': 'tool', 'tool_call_id': 't1', 'content': 'tool-output-sentinel'},
                    {'role': 'user', 'content': 'Before finishing, verify.', '_verification_stop_synthetic': True},
                    {'role': 'assistant', 'content': 'answer'}]
        return {'final_response': 'answer', 'completed': True, 'api_calls': 2, 'messages': messages,
                'current_turn_user_idx': len(conversation_history or [])}


def test_runner_passes_the_proven_turn_boundary_so_a_nudge_cannot_erase_tool_output(monkeypatch):
    from hermes_state_terminal import compact_result
    _install_compression_failure_agent(monkeypatch, _NudgedImageAgent)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id='12345', chat_type='dm', user_id='user-1')
    result = _run_compression_failure_turn(_runner(_SessionStore()), source)
    stored = compact_result({'result': result}, user_message=[{'type': 'text', 'text': 'continue'}])['result']
    assert 'tool-output-sentinel' in str(stored['messages']), 'boundary re-guessed onto the nudge'
    assert stored['messages'][-1] == {'role': 'assistant', 'content': 'answer'} and len(stored['messages']) == 4
