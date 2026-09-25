"""Unit tests for the declarative system-prompt fragment override engine
(agent/prompt_overrides.py)."""

from agent.prompt_overrides import (
    FRAGMENT_KEYS,
    VALID_MODES,
    apply_fragment_override,
    normalize_overrides,
)
from unittest.mock import patch
from types import SimpleNamespace

import pytest

from agent.coding_context import CODING_AGENT_GUIDANCE, WORKSPACE_BLOCK_HEADER
from agent.prompt_builder import PARALLEL_TOOL_CALL_GUIDANCE, STEER_CHANNEL_NOTE, TASK_COMPLETION_GUIDANCE
from run_agent import AIAgent


def _new_test_agent(platform="cli"):
    tool_defs = [{"type": "function", "function": {
        "name": "terminal", "description": "Run a command",
        "parameters": {"type": "object", "properties": {}},
    }}]
    with (
        patch("model_tools.get_tool_definitions", return_value=tool_defs),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            model="anthropic/claude-opus-4.8", api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1", quiet_mode=True,
            skip_context_files=True, skip_memory=True, platform=platform,
            session_id="20260101_120000_prompt_override",
        )
        return agent


def _prompt_from_config(tmp_path, monkeypatch, override_yaml, *, return_agent=False, platform="cli", coding=False):
    home = tmp_path / "hermes-home"
    home.mkdir()
    if coding:
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'coding-test'\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
    coding_config = "  coding_context: on\n  coding_instructions: base-coding-instructions\n" if coding else ""
    (home / "config.yaml").write_text(
        "agent:\n" + coding_config + "  prompt_overrides:\n" + override_yaml,
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    agent = _new_test_agent(platform)
    return agent if return_agent else agent._build_system_prompt()


def test_configured_append_reaches_full_prompt(tmp_path, monkeypatch):
    prompt = _prompt_from_config(tmp_path, monkeypatch,
        "    task_completion:\n      mode: append\n      text: custom-completion-note\n")
    assert TASK_COMPLETION_GUIDANCE in prompt
    assert "custom-completion-note" in prompt


def test_configured_remove_reaches_full_prompt(tmp_path, monkeypatch):
    prompt = _prompt_from_config(tmp_path, monkeypatch,
        "    task_completion:\n      mode: remove\n")
    assert TASK_COMPLETION_GUIDANCE not in prompt


def test_configured_parallel_guidance_replace_reaches_full_prompt(tmp_path, monkeypatch):
    prompt = _prompt_from_config(tmp_path, monkeypatch,
        "    parallel_tool_call_guidance:\n      mode: replace\n      text: custom-parallel-note\n")
    assert PARALLEL_TOOL_CALL_GUIDANCE not in prompt
    assert "custom-parallel-note" in prompt


def test_configured_steer_channel_replace_reaches_full_prompt(tmp_path, monkeypatch):
    prompt = _prompt_from_config(tmp_path, monkeypatch,
        "    steer_channel: custom-steer-note\n")
    assert STEER_CHANNEL_NOTE not in prompt
    assert "custom-steer-note" in prompt


def test_runtime_cwd_anchor_cannot_be_overridden(tmp_path, monkeypatch):
    from agent.surface_switch import runtime_host_value

    prompt = _prompt_from_config(tmp_path, monkeypatch,
        "    environment_hints:\n      mode: remove\n")
    assert runtime_host_value(prompt, "Current working directory")


def test_empty_override_map_keeps_assembled_prompt_bytes(tmp_path, monkeypatch):
    from agent.system_prompt import build_system_prompt_parts

    agent = _prompt_from_config(tmp_path, monkeypatch, "    {}\n", return_agent=True, coding=True)
    with_empty = build_system_prompt_parts(agent)
    assert CODING_AGENT_GUIDANCE.splitlines()[0] in with_empty["stable"]
    assert WORKSPACE_BLOCK_HEADER in with_empty["context"]
    assert "base-coding-instructions" in with_empty["context"]
    del agent._prompt_overrides
    without_attribute = build_system_prompt_parts(agent)
    assert with_empty == without_attribute


def test_absent_platform_hint_is_not_created_by_replace(tmp_path, monkeypatch):
    prompt = _prompt_from_config(tmp_path, monkeypatch,
        "    platform_hints: inserted-platform-hint\n", platform="unrecognized-platform")
    assert "inserted-platform-hint" not in prompt


@pytest.mark.parametrize("mode", ("replace", "append", "prepend", "remove"))
def test_coding_brief_override_modes_reach_full_prompt(tmp_path, monkeypatch, mode):
    spec = f"    coding_brief:\n      mode: {mode}\n"
    if mode != "remove":
        spec += "      text: coding-override-marker\n"
    prompt = _prompt_from_config(tmp_path, monkeypatch, spec, coding=True)
    brief_start = CODING_AGENT_GUIDANCE.splitlines()[0]
    assert (brief_start in prompt) is (mode in {"append", "prepend"})
    assert ("coding-override-marker" in prompt) is (mode != "remove")
    if mode == "prepend":
        assert prompt.index("coding-override-marker") < prompt.index(brief_start)
    elif mode == "append":
        assert prompt.index(brief_start) < prompt.index("coding-override-marker")


def test_coding_workspace_and_instructions_are_overridable(tmp_path, monkeypatch):
    prompt = _prompt_from_config(tmp_path, monkeypatch,
        "    coding_workspace: coding-workspace-marker\n"
        "    coding_instructions: {mode: remove}\n", coding=True)
    assert "coding-workspace-marker" in prompt
    assert WORKSPACE_BLOCK_HEADER not in prompt
    assert "base-coding-instructions" not in prompt


def test_removing_coding_workspace_keeps_trailing_block_after_context(tmp_path, monkeypatch):
    from agent.system_prompt import build_system_prompt_parts

    agent = _prompt_from_config(tmp_path, monkeypatch,
        "    coding_workspace: {mode: remove}\n", return_agent=True, coding=True)
    parts = build_system_prompt_parts(agent)
    assert WORKSPACE_BLOCK_HEADER not in parts["context"]
    assert "base-coding-instructions" in parts["context"]
    assert "base-coding-instructions" not in parts["stable"]


@pytest.mark.parametrize("mode", ("replace", "append", "prepend", "remove"))
def test_overridden_workspace_survives_fresh_agent_resume(tmp_path, monkeypatch, mode):
    spec = f"    coding_workspace:\n      mode: {mode}\n"
    if mode != "remove":
        spec += "      text: coding-workspace-marker\n"
    first = _prompt_from_config(tmp_path, monkeypatch, spec, return_agent=True, coding=True)
    stored = first._build_system_prompt()
    (tmp_path / "pyproject.toml").unlink()

    resumed = _new_test_agent()
    resumed._session_db = SimpleNamespace(get_session=lambda _sid: {"system_prompt": stored})
    assert resumed._build_system_prompt() == stored


class TestNormalizeOverrides:
    def test_empty_and_none(self):
        assert normalize_overrides(None) == {}
        assert normalize_overrides({}) == {}

    def test_non_mapping_ignored(self):
        assert normalize_overrides("nope") == {}
        assert normalize_overrides(["a", "b"]) == {}

    def test_bare_string_is_replace_shorthand(self):
        out = normalize_overrides({"steer_channel": "short note"})
        assert out == {"steer_channel": {"mode": "replace", "text": "short note"}}

    def test_full_spec(self):
        out = normalize_overrides(
            {"task_completion": {"mode": "append", "text": "more"}}
        )
        assert out == {"task_completion": {"mode": "append", "text": "more"}}

    def test_remove_drops_text(self):
        out = normalize_overrides({"google_operational": {"mode": "remove"}})
        assert out == {"google_operational": {"mode": "remove", "text": ""}}

    def test_unknown_key_dropped(self):
        assert normalize_overrides({"bogus": "x"}) == {}

    def test_invalid_mode_dropped(self):
        assert normalize_overrides({"identity": {"mode": "sideways", "text": "x"}}) == {}

    def test_mode_case_insensitive(self):
        out = normalize_overrides({"identity": {"mode": "REPLACE", "text": "x"}})
        assert out["identity"]["mode"] == "replace"

    def test_missing_mode_defaults_replace(self):
        out = normalize_overrides({"identity": {"text": "x"}})
        assert out["identity"]["mode"] == "replace"

    def test_append_empty_text_is_noop_dropped(self):
        assert normalize_overrides({"identity": {"mode": "append", "text": "  "}}) == {}

    def test_replace_empty_text_allowed(self):
        # replace with "" is a legitimate way to blank a fragment via text
        out = normalize_overrides({"identity": {"mode": "replace", "text": ""}})
        assert out == {"identity": {"mode": "replace", "text": ""}}

    def test_non_string_text_dropped(self):
        assert normalize_overrides({"identity": {"mode": "replace", "text": 5}}) == {}

    def test_non_dict_non_str_spec_dropped(self):
        assert normalize_overrides({"identity": ["a"]}) == {}

    def test_mixed_valid_and_invalid(self):
        out = normalize_overrides(
            {
                "task_completion": "keep this",
                "bogus": "drop this",
                "google_operational": {"mode": "remove"},
            }
        )
        assert set(out) == {"task_completion", "google_operational"}


class TestApplyFragmentOverride:
    def test_no_overrides_passthrough(self):
        assert apply_fragment_override(None, "identity", "ID") == "ID"
        assert apply_fragment_override({}, "identity", "ID") == "ID"

    def test_unmatched_key_passthrough(self):
        ov = normalize_overrides({"identity": "X"})
        assert apply_fragment_override(ov, "task_completion", "ORIG") == "ORIG"

    def test_replace(self):
        ov = normalize_overrides({"identity": {"mode": "replace", "text": "NEW"}})
        assert apply_fragment_override(ov, "identity", "OLD") == "NEW"

    def test_remove_returns_none(self):
        ov = normalize_overrides({"identity": {"mode": "remove"}})
        assert apply_fragment_override(ov, "identity", "OLD") is None

    def test_append(self):
        ov = normalize_overrides({"identity": {"mode": "append", "text": "B"}})
        assert apply_fragment_override(ov, "identity", "A") == "A\n\nB"

    def test_prepend(self):
        ov = normalize_overrides({"identity": {"mode": "prepend", "text": "B"}})
        assert apply_fragment_override(ov, "identity", "A") == "B\n\nA"

    def test_append_to_empty_default_is_noop(self):
        ov = normalize_overrides({"identity": {"mode": "append", "text": "B"}})
        assert apply_fragment_override(ov, "identity", "") == ""
        assert apply_fragment_override(ov, "identity", None) is None

    def test_prepend_to_empty_default_is_noop(self):
        ov = normalize_overrides({"identity": {"mode": "prepend", "text": "B"}})
        assert apply_fragment_override(ov, "identity", "   ") == "   "


class TestFragmentKeyRegistry:
    def test_keys_present_and_documented(self):
        for key, desc in FRAGMENT_KEYS.items():
            assert isinstance(key, str) and key
            assert isinstance(desc, str) and desc.strip()
            spec = normalize_overrides({key: "replacement"})
            assert apply_fragment_override(spec, key, "original") == "replacement"

    def test_core_pushy_blocks_addressable(self):
        # The fragments that motivated this feature must be overridable.
        for key in (
            "task_completion",
            "tool_use_enforcement",
            "execution_discipline",
            "google_operational",
        ):
            assert key in FRAGMENT_KEYS

    def test_valid_modes(self):
        assert VALID_MODES == ("replace", "append", "prepend", "remove")
