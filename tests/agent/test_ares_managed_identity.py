"""Ares-managed fallback identity must be explicit and SOUL.md must win."""

from types import SimpleNamespace
from unittest.mock import patch

from agent.codex_responses_adapter import _preflight_codex_api_kwargs
from agent.prompt_builder import (
    ARES_DEFAULT_AGENT_IDENTITY,
    DEFAULT_AGENT_IDENTITY,
    select_default_agent_identity,
)
from agent.system_prompt import build_system_prompt_parts


def _agent() -> SimpleNamespace:
    return SimpleNamespace(
        load_soul_identity=False,
        skip_context_files=False,
        valid_tool_names=[],
        _task_completion_guidance=False,
        _tool_use_enforcement=False,
        _environment_probe=False,
        _kanban_worker_guidance="",
        _memory_store=None,
        _memory_manager=None,
        model="",
        provider="",
        platform="",
        pass_session_id=False,
        session_id="",
    )


def _stable_with_soul(soul: str) -> str:
    with (
        patch("run_agent.load_soul_md", return_value=soul),
        patch("run_agent.build_environment_hints", return_value=""),
        patch("run_agent.build_context_files_prompt", return_value=""),
    ):
        return build_system_prompt_parts(_agent())["stable"]


def test_explicit_managed_runtime_selects_ares_fallback(monkeypatch):
    monkeypatch.setenv("ARES_MANAGED_RUNTIME", "1")
    assert select_default_agent_identity() == ARES_DEFAULT_AGENT_IDENTITY
    stable = _stable_with_soul("")
    assert ARES_DEFAULT_AGENT_IDENTITY in stable
    assert DEFAULT_AGENT_IDENTITY not in stable


def test_normal_hermes_runtime_preserves_its_fallback(monkeypatch):
    monkeypatch.delenv("ARES_MANAGED_RUNTIME", raising=False)
    assert select_default_agent_identity() == DEFAULT_AGENT_IDENTITY
    assert DEFAULT_AGENT_IDENTITY in _stable_with_soul("")


def test_custom_soul_precedes_managed_ares_fallback(monkeypatch):
    monkeypatch.setenv("ARES_MANAGED_RUNTIME", "1")
    stable = _stable_with_soul("CUSTOM PROFILE SOUL")
    assert "CUSTOM PROFILE SOUL" in stable
    assert ARES_DEFAULT_AGENT_IDENTITY not in stable


def test_responses_preflight_uses_the_same_managed_fallback(monkeypatch):
    monkeypatch.setenv("ARES_MANAGED_RUNTIME", "1")
    normalized = _preflight_codex_api_kwargs(
        {
            "model": "gpt-5-codex",
            "instructions": "",
            "input": [{"role": "user", "content": "hello"}],
            "tools": [],
        }
    )
    assert normalized["instructions"] == ARES_DEFAULT_AGENT_IDENTITY
