"""Tests for agent/system_prompt.py — context-file cwd wiring."""

from types import SimpleNamespace
from unittest.mock import patch

from agent.system_prompt import build_system_prompt_parts


def _make_agent(**overrides):
    base = dict(
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
    base.update(overrides)
    return SimpleNamespace(**base)


def _captured_context_cwd(agent):
    """The cwd build_system_prompt_parts hands to build_context_files_prompt."""
    captured = {}

    def fake_context_files(cwd=None, skip_soul=False):
        captured["cwd"] = cwd
        return ""

    with (
        patch("run_agent.load_soul_md", return_value=""),
        patch("run_agent.build_nous_subscription_prompt", return_value=""),
        patch("run_agent.build_environment_hints", return_value=""),
        patch("run_agent.build_context_files_prompt", side_effect=fake_context_files),
    ):
        build_system_prompt_parts(agent)
    return captured["cwd"]


def _stable_text(agent):
    """Joined 'stable' tier of the system prompt, with the run_agent helpers
    stubbed so the build is hermetic."""
    with (
        patch("run_agent.load_soul_md", return_value=""),
        patch("run_agent.build_nous_subscription_prompt", return_value=""),
        patch("run_agent.build_environment_hints", return_value=""),
        patch("run_agent.build_context_files_prompt", return_value=""),
    ):
        return build_system_prompt_parts(agent)["stable"]


class TestQwenCoderToolcallGuidance:
    """M-fix (QwenLM/Qwen3-Coder#475): qwen-coder models omit the opening
    <tool_call> tag; inject an anti-omission reminder, gated on the model name."""

    MARKER = "Do NOT omit the opening <tool_call>"

    def test_injected_for_qwen3_coder_with_tools(self):
        txt = _stable_text(_make_agent(model="qwen3-coder:30b", valid_tool_names=["write_file"]))
        assert self.MARKER in txt

    def test_injected_for_qwen25_coder(self):
        txt = _stable_text(_make_agent(model="qwen2.5-coder:14b", valid_tool_names=["write_file"]))
        assert self.MARKER in txt

    def test_not_injected_for_non_qwen_model(self):
        txt = _stable_text(_make_agent(model="claude-opus-4-8", valid_tool_names=["write_file"]))
        assert self.MARKER not in txt

    def test_not_injected_for_plain_qwen_non_coder(self):
        # Plain qwen chat models use native structured tool_calls — don't pollute.
        txt = _stable_text(_make_agent(model="qwen3:8b", valid_tool_names=["write_file"]))
        assert self.MARKER not in txt

    def test_not_injected_without_tools(self):
        txt = _stable_text(_make_agent(model="qwen3-coder:30b", valid_tool_names=[]))
        assert self.MARKER not in txt


class TestContextFileCwd:
    def test_none_when_terminal_cwd_unset(self, monkeypatch):
        # Unset → None, so discovery falls back to the launch dir inside
        # build_context_files_prompt (the local-CLI #19242 contract).
        monkeypatch.delenv("TERMINAL_CWD", raising=False)
        assert _captured_context_cwd(_make_agent()) is None

    def test_configured_dir_when_terminal_cwd_set(self, monkeypatch, tmp_path):
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        assert _captured_context_cwd(_make_agent()) == tmp_path
