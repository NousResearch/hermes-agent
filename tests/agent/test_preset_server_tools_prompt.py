"""Preset server-side tools are discoverable.

An OpenRouter preset can switch on server-side tools that execute inside the router. They never
appear in the client tool list and leave no tool-call trace, so the capability is undiscoverable
from the model's side unless the prompt says it may exist. One factual paragraph is rendered for
preset routes — in the STABLE tier (``_guidance_parts``), so it cannot churn the cache prefix
mid-session — and for nothing else.

Both preset forms are valid (`@preset/<slug>` and `<model>@preset/<slug>`, see
hermes_cli/models_validate.py), so the test is a substring, not a prefix. No preset is fetched
and no client tool is declared: these tests assert the rendered text only.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.prompt_builder import PRESET_SERVER_TOOLS_NOTE
from agent.system_prompt import _guidance_parts, _is_openrouter_preset, build_system_prompt_parts


def _agent(model, **overrides):
    base = dict(
        load_soul_identity=False, skip_context_files=True, valid_tool_names={"read_file"},
        _task_completion_guidance=False, _parallel_tool_call_guidance=False,
        _tool_use_enforcement=False, _execution_guidance=False, _environment_probe=False,
        _kanban_worker_guidance="", _memory_store=None, _memory_manager=None,
        _plugin_system_prompt_sections=None, _platform_hint_overrides=None,
        _auto_load_skills=None, model=model, provider="openrouter", platform="cli",
        pass_session_id=False, session_id="", _bot_mode_protocol=False, _bot_chat_timeless_prompt=False,
        _api_key=None, _is_anthropic_oauth=False, openrouter_min_coding_score=None,
        _emit_status=lambda *_a, **_k: None, _emit_diagnostic_status=lambda *_a, **_k: None,
        context_compressor=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.mark.parametrize("model", ["@preset/test-preset", "openrouter/auto@preset/test-preset"])
def test_preset_guidance_renders_for_both_forms(model):
    assert _is_openrouter_preset(model)
    assert PRESET_SERVER_TOOLS_NOTE in _guidance_parts(_agent(model))


@pytest.mark.parametrize("model", ["openrouter/auto", "anthropic/claude-opus-4.7", "", None,
                                   "openrouter/preset-x"])
def test_preset_guidance_absent_for_a_plain_model(model):
    assert not _is_openrouter_preset(model)
    assert PRESET_SERVER_TOOLS_NOTE not in _guidance_parts(_agent(model))


@pytest.mark.parametrize("model", ["@preset/test-preset", "openrouter/auto@preset/test-preset"])
def test_paragraph_lands_in_the_stable_tier(model):
    """Stable tier = the reusable cache prefix; a per-turn rebuild must not shift it."""
    agent = _agent(model)
    with patch("agent.prompt_builder.build_environment_hints", return_value=""), \
         patch("agent.prompt_builder.build_context_files_prompt", return_value=""), \
         patch("agent.prompt_builder.load_soul_md", return_value=""):
        parts = build_system_prompt_parts(agent)
    assert PRESET_SERVER_TOOLS_NOTE in parts["stable"]
    assert PRESET_SERVER_TOOLS_NOTE not in parts["context"]
    assert PRESET_SERVER_TOOLS_NOTE not in parts["volatile"]


def test_paragraph_names_the_tools_and_the_no_trace_rule():
    """Factual content: names the router-side tools, and forbids claiming an advisor answer as
    verified evidence."""
    for name in ("openrouter:advisor", "openrouter:fusion", "openrouter:datetime",
                 "openrouter:web_search", "openrouter:web_fetch", "openrouter:subagent"):
        assert name in PRESET_SERVER_TOOLS_NOTE
    assert "never" in PRESET_SERVER_TOOLS_NOTE and "trace" in PRESET_SERVER_TOOLS_NOTE
    # Nothing fabricated as a client-side capability, and no invented lookup.
    assert "you can call" not in PRESET_SERVER_TOOLS_NOTE.lower()
    assert "fetch the preset" not in PRESET_SERVER_TOOLS_NOTE.lower()


def test_rendered_once_not_per_tool_block():
    agent = _agent("@preset/test-preset")
    assert _guidance_parts(agent).count(PRESET_SERVER_TOOLS_NOTE) == 1
