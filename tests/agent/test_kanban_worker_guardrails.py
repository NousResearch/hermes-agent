import os

from agent.tool_guardrails import (
    ToolCallGuardrailConfig,
    ToolCallGuardrailController,
    is_stall_guard_repeatable,
)


def test_kanban_worker_enables_hard_stop_even_on_cli(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_test")

    cfg = ToolCallGuardrailConfig.from_mapping({}, platform="cli")

    assert cfg.hard_stop_enabled is True


def test_kanban_worker_can_explicitly_disable_noninteractive_hard_stop(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_test")

    cfg = ToolCallGuardrailConfig.from_mapping(
        {"non_interactive_hard_stop_enabled": False},
        platform="cli",
    )

    assert cfg.hard_stop_enabled is False


def test_kanban_show_is_not_repeatable_in_worker(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_test")

    assert is_stall_guard_repeatable("kanban_show") is False


def test_repeated_kanban_show_halts_worker(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_test")

    cfg = ToolCallGuardrailConfig.from_mapping({}, platform="cli")
    controller = ToolCallGuardrailController(cfg)

    for i in range(cfg.no_progress_block_after):
        controller.observe_call(
            "kanban_show",
            {},
            '{"task":"same"}',
            tool_call_id=f"test-{i}",
        )

    assert controller._halt_decision is not None
