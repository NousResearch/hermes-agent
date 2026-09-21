"""Cutover regression guard for source contracts dropped by merge convergence.

Each contract below has an existing behavioural suite.  These smoke tests keep
collection failures local and explicit so a future source/test split cannot hide
behind unrelated optional-dependency errors.
"""

from __future__ import annotations


def test_memory_context_filter_contract_is_importable() -> None:
    # Was GatewayStreamConsumer._MEMORY_LEAK_RE; the memory-context filter moved to
    # agent.memory_manager during the 2026-09 refactor chain. Same contract: memory
    # context never reaches channel output.
    from agent.memory_manager import StreamingContextScrubber, sanitize_context

    assert sanitize_context("<memory-context>secret</memory-context>visible") == "visible"
    assert StreamingContextScrubber().feed("plain") == "plain"


def test_runtime_tool_scope_fence_contract_is_importable() -> None:
    from agent.tool_executor import (
        _SCOPE_FENCE_ESCAPE_HATCHES,
        _allowed_tool_names_for_agent,
        _tool_scope_decision,
    )

    assert _SCOPE_FENCE_ESCAPE_HATCHES
    assert callable(_allowed_tool_names_for_agent)
    assert callable(_tool_scope_decision)


def test_auto_tts_output_contract_is_importable() -> None:
    from gateway.platforms.base import build_auto_tts_output_path

    assert build_auto_tts_output_path("telegram").endswith(".ogg")


def test_hygiene_cooldown_ladder_contract_is_importable() -> None:
    from gateway.run import (
        _HYGIENE_COOLDOWN_LADDER_MULTIPLIERS,
        _hygiene_cooldown_for_failure,
        _reset_hygiene_failure_streak,
        hygiene_compaction_recovered,
    )

    assert _HYGIENE_COOLDOWN_LADDER_MULTIPLIERS
    assert callable(_hygiene_cooldown_for_failure)
    assert callable(_reset_hygiene_failure_streak)
    assert callable(hygiene_compaction_recovered)


def test_goal_quality_gate_contract_is_importable() -> None:
    from hermes_cli.goals import DEFAULT_GATE_MAX_RETRIES, GoalGate, run_gate

    assert DEFAULT_GATE_MAX_RETRIES > 0
    assert GoalGate(command="true").command == "true"
    assert callable(run_gate)


def test_setup_telemetry_contract_is_importable() -> None:
    from hermes_cli.setup import setup_telemetry

    assert callable(setup_telemetry)


def test_voice_provider_secret_contract_is_importable() -> None:
    from tools.tool_backend_helpers import resolve_provider_secret

    assert callable(resolve_provider_secret)
