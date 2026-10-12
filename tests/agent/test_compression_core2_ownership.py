"""CORE2 : remplacement véritable via dispatch natif sur le même compresseur."""
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.auxiliary_client import AuxiliaryExplicitCancellation
from agent.context_compressor import ContextCompressor, SUMMARY_PREFIX
from agent.conversation_compression import _claim_compressor_attempt, _run_summary_dispatch


def _reply(text):
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = f'## Goal\n{text}'
    response.choices[0].finish_reason = 'stop'
    return response


def test_superseding_dispatch_never_reads_partial_summary_and_late_worker_cannot_commit():
    c = ContextCompressor(model='test/model', config_context_length=100_000, quiet_mode=True)
    c._SUMMARY_INPUT_MAX_CHARS = 12_000
    c._previous_summary = 'old checkpoint'
    agent = SimpleNamespace(context_compressor=c, session_id='s1')
    messages = [
        {'role': 'user' if i % 2 == 0 else 'assistant', 'content': f'turn {i}'}
        for i in range(24)
    ]
    messages[0] = {'role': 'system', 'content': 'test instructions'}
    messages[1] = {'role': 'user', 'content': f'{SUMMARY_PREFIX}\nold checkpoint'}
    messages[6] = {'role': 'tool', 'tool_call_id': 'large', 'content': 'note ' + 'x' * 8_000}
    kwargs = {'current_tokens': 999_999, 'force': True}
    stalled, resume, finished = threading.Event(), threading.Event(), threading.Event()
    primary = [None]
    seen_primary = []
    observed_by_fallback = []
    outcomes = []

    def response(**kw):
        if threading.current_thread() is primary[0]:
            seen_primary.append(c._previous_summary)
            if len(seen_primary) == 2:
                stalled.set()
                assert resume.wait(5), 'fallback must finish while primary is stalled'
            return _reply(f'primary part {len(seen_primary)}')
        observed_by_fallback.append(c._previous_summary)
        return _reply('fallback complete')

    def run_primary():
        try:
            _run_summary_dispatch(
                agent, messages, c.compress, kwargs,
                commit_fence=None, attempt_generation=1, hard_cancel_event=None,
            )
            outcomes.append('committed')
        except AuxiliaryExplicitCancellation:
            outcomes.append('cancelled')
        finally:
            finished.set()

    with patch('agent.context_compressor.call_llm', side_effect=response):
        assert _claim_compressor_attempt(c) == 1
        t = threading.Thread(target=run_primary, daemon=True)
        primary[0] = t
        t.start()
        assert stalled.wait(5), 'expected at least two genuine segment calls'
        try:
            assert c._previous_summary == 'old checkpoint'
            assert _claim_compressor_attempt(c) == 2
            _run_summary_dispatch(
                agent, messages, c.compress, kwargs,
                commit_fence=None, attempt_generation=2, hard_cancel_event=None,
            )
        finally:
            resume.set()
            assert finished.wait(5)
            t.join(5)
    assert seen_primary[:2] == ['old checkpoint', 'old checkpoint']
    assert observed_by_fallback and observed_by_fallback[0] == 'old checkpoint'
    assert outcomes == ['cancelled']
    assert 'fallback complete' in (c._previous_summary or '')
