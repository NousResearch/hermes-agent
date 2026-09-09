"""Completed partial generations retain the normal usage-rearm contract."""
import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock

from agent import turn_response_check as check


def test_length_response_clears_rearmed_compression_latch(monkeypatch):
    agent = MagicMock(quiet_mode=True, verbose_logging=False)
    monkeypatch.setattr('agent.turn_recovery.validate_response_shape', lambda *_: (False, None))
    monkeypatch.setattr(check, '_derive_finish_reason', lambda *_: 'length')
    monkeypatch.setattr('agent.turn_response_intake._fire_post_api_request_hook', lambda *a, **k: None)
    monkeypatch.setattr('agent.turn_truncation.normalize_response_for_agent', lambda *_: SimpleNamespace(content='part'))
    usage = MagicMock(return_value=SimpleNamespace(compression_attempts=0, rearmed=True))
    monkeypatch.setattr(check, 'record_response_usage', usage)

    def restart(*args, **kwargs):
        return SimpleNamespace(
            action='break', result=None, messages=kwargs['messages'],
            length_continue_retries=1, truncated_response_parts=['part'],
            truncated_tool_call_retries=0, retry_count=0,
            compression_attempts=kwargs['compression_attempts'],
        )

    monkeypatch.setattr(check, 'recover_from_truncation', restart)
    kwargs = {name: None for name in inspect.signature(check.check_api_response).parameters if name != 'agent'}
    kwargs.update(
        response=SimpleNamespace(), messages=[], api_start_time=0,
        compression_attempts=2, max_compression_attempts=3,
        _preflight_compression_blocked=True, _last_preflight_pressure=900,
    )
    result = check.check_api_response(agent, **kwargs)
    assert result.action == 'break'
    assert result.compression_attempts == 0
    assert result._preflight_compression_blocked is False
    assert result._last_preflight_pressure is None
    usage.assert_called_once()
