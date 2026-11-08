"""The post-tool preflight "compacting context" line must ride the lifecycle status rail.

It used to be a bare ``agent._safe_print``: a stdout write that never reached
``_prepare_gateway_status_message``, so a chat surface delivered the raw internal line while every
sibling compaction status was suppressed. It is emitted through ``_emit_status`` now, and its text is
part of the routine-compression sample set the gateway noise filter is pinned against.
"""

from types import SimpleNamespace

import pytest

from agent.conversation_compression import PREFLIGHT_COMPACTING_STATUS, ROUTINE_COMPRESSION_STATUS_SAMPLES
from agent.turn_preflight import compress_after_tool_results


class _CompressionReached(Exception):
    """Raised by the double once the caller has invoked ``_compress_context``."""


def _messages() -> list:
    return [
        {"role": "user", "content": "x" * 8_000},
        {"role": "assistant", "content": "y" * 8_000},
        {"role": "user", "content": "current ask"},
    ]


def _agent(statuses: list, prints: list) -> SimpleNamespace:
    def _record_and_stop(_messages, _system, **kwargs):
        raise _CompressionReached

    compressor = SimpleNamespace(
        threshold_tokens=1_000, context_length=200_000, summary_target_ratio=0.2,
        protect_first_n=1, protect_last_n=1, last_prompt_tokens=5_000,
        last_compression_rough_tokens=0, awaiting_real_usage_after_compression=False,
        should_compress=lambda _tokens: True,
    )
    return SimpleNamespace(
        context_compressor=compressor, compression_enabled=True, model="m", session_id="s1",
        max_compression_attempts=3, _compression_feasibility_checked=True,
        _emit_status=statuses.append, _safe_print=lambda *_a, **_kw: prints.append(_a),
        _compress_context=_record_and_stop,
    )


def _drive_post_tool(agent: SimpleNamespace) -> None:
    messages = _messages()
    compress_after_tool_results(
        agent, messages=messages, system_message="sys", user_message="current ask",
        active_system_prompt="sys", conversation_history=None, compression_attempts=0,
        max_compression_attempts=3, effective_task_id="task", final_response=None,
        turn_exit_reason=None, current_turn_user_idx=len(messages) - 1,
    )


def test_post_tool_compaction_reaches_the_status_rail_and_not_stdout():
    """The line a chat surface filters must arrive as a status, not as an unfilterable print."""
    statuses: list = []
    prints: list = []

    with pytest.raises(_CompressionReached):
        _drive_post_tool(_agent(statuses, prints))

    assert statuses == [PREFLIGHT_COMPACTING_STATUS]
    assert prints == []


def test_the_preflight_line_is_in_the_routine_compression_sample_set():
    """Rewording the emit site without the gateway noise regex fails the noise-filter suite."""
    assert PREFLIGHT_COMPACTING_STATUS in ROUTINE_COMPRESSION_STATUS_SAMPLES


@pytest.mark.parametrize("platform", ["telegram", "slack", "feishu"])
def test_chat_surfaces_suppress_the_preflight_line(platform):
    from gateway.run import _prepare_gateway_status_message

    assert _prepare_gateway_status_message(platform, "lifecycle", PREFLIGHT_COMPACTING_STATUS) is None


def test_progress_notices_opt_in_keeps_the_preflight_line(monkeypatch):
    """`compression.progress_notices: true` shows routine compression progress, this line included."""
    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_gateway_compression_progress_notices_enabled", lambda: True)

    assert gateway_run._prepare_gateway_status_message(
        "telegram", "lifecycle", PREFLIGHT_COMPACTING_STATUS,
    ) == PREFLIGHT_COMPACTING_STATUS
