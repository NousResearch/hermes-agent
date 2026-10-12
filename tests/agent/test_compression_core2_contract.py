"""Contrats CORE2/CORE3 indépendants : état privé et borne de transport avant tout envoi."""
from unittest.mock import MagicMock, patch

import pytest

from agent.auxiliary_client import AuxiliaryExplicitCancellation
from agent.context_compressor import ContextCompressor, pin_summary_route


def _answer(text='## Historical Task Snapshot\nNone. This session contains no user-authored turns.\n\n## Goal\nS0 checkpoint'):
    r = MagicMock()
    r.choices = [MagicMock()]
    r.choices[0].message.content = text
    r.choices[0].finish_reason = 'stop'
    return r


def _compressor():
    c = ContextCompressor(model='test/model', provider='test', config_context_length=100_000, quiet_mode=True)
    c._previous_summary = 'old checkpoint'
    return c


def test_intermediate_chunks_do_not_publish_previous_summary():
    c = _compressor()
    c._SUMMARY_INPUT_MAX_CHARS = 8_000
    seen = []

    def answer(**kw):
        seen.append(c._previous_summary)
        return _answer(f'## Historical Task Snapshot\nNone. This session contains no user-authored turns.\n\n## Goal\npart-{len(seen)}')

    with patch('agent.context_compressor.call_llm', side_effect=answer):
        result = c._generate_summary([{'role': 'tool', 'tool_call_id': 't', 'content': 'x' * 10_000}])
    assert len(seen) > 1
    assert seen == ['old checkpoint'] * len(seen)
    assert result and c._previous_summary == c._strip_summary_prefix(result)


def test_intermediate_failure_and_cancel_never_publish():
    for failure in (RuntimeError('synthetic later segment'), AuxiliaryExplicitCancellation()):
        c = _compressor()
        c._SUMMARY_INPUT_MAX_CHARS = 8_000
        seen = []

        def answer(**kw):
            seen.append(c._previous_summary)
            if len(seen) == 2:
                raise failure
            return _answer('## Historical Task Snapshot\nNone. This session contains no user-authored turns.\n\n## Goal\npartial')

        with patch('agent.context_compressor.call_llm', side_effect=answer):
            if isinstance(failure, AuxiliaryExplicitCancellation):
                with pytest.raises(AuxiliaryExplicitCancellation):
                    c._generate_summary([{'role': 'tool', 'content': 'x' * 10_000}])
            else:
                assert c._generate_summary([{'role': 'tool', 'content': 'x' * 10_000}]) is None
        assert seen == ['old checkpoint'] * len(seen)
        assert c._previous_summary == 'old checkpoint'


def test_one_call_when_the_whole_prompt_fits_under_the_transport_ceiling():
    c = _compressor()
    source = 'S0:' + 'x' * 120_000 + ':end'
    def resolved(**kwargs):
        kwargs['route_info'].update(provider='test', model='test/model')
        return _answer()
    with pin_summary_route({'provider': 'test', 'model': 'test/model'}), \
         patch('agent.context_compressor.call_llm', side_effect=resolved) as call:
        assert c._generate_summary([{'role': 'tool', 'content': source}])
    assert call.call_count == 1
    assert source in call.call_args.kwargs['messages'][0]['content']


def test_advertised_long_window_never_widens_the_transport_ceiling():
    c = _compressor()
    source = 'S0:' + 'x' * 190_000 + ':end'
    def resolved(**kwargs):
        kwargs['route_info'].update(provider='test', model='test/model')
        return _answer()
    with patch('agent.context_compressor.get_model_context_length', return_value=1_000_000), \
         pin_summary_route({'provider': 'test', 'model': 'test/model'}), \
         patch('agent.context_compressor.call_llm', side_effect=resolved) as call:
        assert c._generate_summary([{'role': 'tool', 'content': source}])
    prompts = [k.kwargs['messages'][0]['content'] for k in call.call_args_list]
    assert len(prompts) == 2
    assert max(map(len, prompts)) <= c._SUMMARY_INPUT_MAX_CHARS
    assert all(k.kwargs.get('provider') == 'test' and k.kwargs.get('model') == 'test/model'
               for k in call.call_args_list)


def test_route_change_inside_the_ladder_only_ever_sees_bounded_prompts():
    c = _compressor()
    source = 'x' * 190_000
    seen = []
    def changed(**kwargs):
        seen.append(c._previous_summary)
        kwargs['route_info'].update(provider='other', model='other/model')
        return _answer(f'## Historical Task Snapshot\nNone. This session contains no user-authored turns.\n\n## Goal\npart-{len(seen)}')
    with pin_summary_route({'provider': 'test', 'model': 'test/model'}), \
         patch('agent.context_compressor.call_llm', side_effect=changed) as call:
        result = c._generate_summary([{'role': 'tool', 'content': source}])
    assert call.call_count == 2
    assert max(len(k.kwargs['messages'][0]['content']) for k in call.call_args_list) <= c._SUMMARY_INPUT_MAX_CHARS
    assert seen == ['old checkpoint'] * 2
    assert result and c._previous_summary == c._strip_summary_prefix(result)
