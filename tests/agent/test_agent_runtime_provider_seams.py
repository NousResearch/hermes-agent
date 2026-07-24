"""Generic agent-runtime provider seam tests."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import patch


def _make_agent(**overrides):
    base = dict(
        load_soul_identity=False,
        skip_context_files=True,
        valid_tool_names=[],
        _task_completion_guidance=False,
        _tool_use_enforcement=False,
        _environment_probe=False,
        _kanban_worker_guidance="",
        _memory_store=None,
        _memory_manager=None,
        _memory_enabled=False,
        model="",
        provider="",
        platform="",
        pass_session_id=False,
        session_id="",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_system_prompt_part_provider_contributes_ordered_tiers():
    from agent.system_prompt import build_system_prompt_parts
    from agent.system_prompt_part_providers import (
        clear_system_prompt_part_providers_for_tests,
        register_system_prompt_part_provider,
    )

    clear_system_prompt_part_providers_for_tests()

    def provider(agent, *, system_message=None):
        assert agent.platform == "cli"
        assert system_message == "caller context"
        return {
            "stable": ["provider stable A", "provider stable B"],
            "context": "provider context",
            "volatile": ["provider volatile"],
        }

    register_system_prompt_part_provider("test-provider", provider)
    try:
        fake_run_agent = SimpleNamespace(
            load_soul_md=lambda _ctx_len=None: "",
            build_nous_subscription_prompt=lambda _tools: "",
            build_environment_hints=lambda: "",
            build_context_files_prompt=lambda **_kwargs: "",
        )
        with patch.dict(sys.modules, {"run_agent": fake_run_agent}):
            parts = build_system_prompt_parts(
                _make_agent(platform="cli"), system_message="caller context"
            )
    finally:
        clear_system_prompt_part_providers_for_tests()

    assert "provider stable A\n\nprovider stable B" in parts["stable"]
    assert "caller context" in parts["context"]
    assert "provider context" in parts["context"]
    assert "provider volatile" in parts["volatile"]
