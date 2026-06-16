"""Tests for agent/system_prompt.py — context-file cwd wiring."""

from types import SimpleNamespace
from unittest.mock import patch

from agent.system_prompt import (
    build_system_prompt,
    build_system_prompt_parts,
    restore_system_prompt_cache_parts,
)


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


class TestContextFileCwd:
    def test_none_when_terminal_cwd_unset(self, monkeypatch):
        # Unset → None, so discovery falls back to the launch dir inside
        # build_context_files_prompt (the local-CLI #19242 contract).
        monkeypatch.delenv("TERMINAL_CWD", raising=False)
        assert _captured_context_cwd(_make_agent()) is None

    def test_configured_dir_when_terminal_cwd_set(self, monkeypatch, tmp_path):
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        assert _captured_context_cwd(_make_agent()) == tmp_path


def _stable_prompt(agent):
    with (
        patch("run_agent.load_soul_md", return_value=""),
        patch("run_agent.build_nous_subscription_prompt", return_value=""),
        patch("run_agent.build_environment_hints", return_value=""),
        patch("run_agent.build_context_files_prompt", return_value=""),
    ):
        return build_system_prompt_parts(agent)["stable"]


class TestCodingContextBlock:
    def test_injected_when_active(self, monkeypatch, tmp_path):
        import subprocess

        subprocess.run(["git", "-C", str(tmp_path), "init", "-q"], check=True)
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        agent = _make_agent(valid_tool_names=["read_file"], platform="cli")
        stable = _stable_prompt(agent)
        assert "coding agent" in stable
        assert "Workspace" in stable

    def test_absent_when_off(self, monkeypatch, tmp_path):
        import subprocess

        subprocess.run(["git", "-C", str(tmp_path), "init", "-q"], check=True)
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        agent = _make_agent(valid_tool_names=["read_file"], platform="cli")
        # Drive the real path: force the resolved mode to "off" via config.
        with patch("agent.coding_context._coding_mode", return_value="off"):
            stable = _stable_prompt(agent)
        assert "coding agent" not in stable

    def test_absent_without_tools(self, monkeypatch, tmp_path):
        import subprocess

        subprocess.run(["git", "-C", str(tmp_path), "init", "-q"], check=True)
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        agent = _make_agent(valid_tool_names=[], platform="cli")
        assert "coding agent" not in _stable_prompt(agent)


class TestSystemPromptCacheParts:
    def test_build_records_cacheable_prefix_and_volatile_tail(self):
        agent = _make_agent()
        with (
            patch("run_agent.load_soul_md", return_value=""),
            patch("run_agent.build_nous_subscription_prompt", return_value=""),
            patch("run_agent.build_environment_hints", return_value=""),
            patch("run_agent.build_context_files_prompt", return_value="Context file text"),
        ):
            prompt = build_system_prompt(agent, system_message="Custom instruction")

        assert agent._cached_system_prompt_cacheable
        assert "Custom instruction" in agent._cached_system_prompt_cacheable
        assert "Context file text" in agent._cached_system_prompt_cacheable
        assert "Conversation started:" not in agent._cached_system_prompt_cacheable
        assert "Conversation started:" in agent._cached_system_prompt_volatile
        assert prompt == "\n\n".join(
            p
            for p in (
                agent._cached_system_prompt_cacheable,
                agent._cached_system_prompt_volatile,
            )
            if p
        )

    def test_restore_recovers_split_from_stored_prompt_without_changing_prompt(self):
        agent = _make_agent()
        with (
            patch("run_agent.load_soul_md", return_value=""),
            patch("run_agent.build_nous_subscription_prompt", return_value=""),
            patch("run_agent.build_environment_hints", return_value=""),
            patch("run_agent.build_context_files_prompt", return_value="Context file text"),
        ):
            stored = build_system_prompt(agent, system_message="Custom instruction")

        restored = _make_agent()
        with (
            patch("run_agent.load_soul_md", return_value=""),
            patch("run_agent.build_nous_subscription_prompt", return_value=""),
            patch("run_agent.build_environment_hints", return_value=""),
            patch("run_agent.build_context_files_prompt", return_value="Context file text"),
        ):
            ok = restore_system_prompt_cache_parts(
                restored,
                stored,
                system_message="Custom instruction",
            )

        assert ok is True
        assert restored._cached_system_prompt_cacheable
        assert stored == "\n\n".join(
            p
            for p in (
                restored._cached_system_prompt_cacheable,
                restored._cached_system_prompt_volatile,
            )
            if p
        )
