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
        _memory_enabled=True,
        _user_profile_enabled=True,
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


class TestMemoryPromptSkip:
    def test_skip_memory_prompt_suppresses_builtin_and_external_memory_blocks(self):
        class FakeMemoryStore:
            def format_for_system_prompt(self, target):
                return f"BUILTIN-{target}-BLOCK"

        class FakeMemoryManager:
            def build_system_prompt(self):
                return "EXTERNAL-MEMORY-BLOCK"

        agent = _make_agent(
            _memory_store=FakeMemoryStore(),
            _memory_manager=FakeMemoryManager(),
            skip_memory_prompt=True,
        )

        with (
            patch("run_agent.load_soul_md", return_value=""),
            patch("run_agent.build_nous_subscription_prompt", return_value=""),
            patch("run_agent.build_environment_hints", return_value=""),
            patch("run_agent.build_context_files_prompt", return_value=""),
        ):
            parts = build_system_prompt_parts(agent)

        assert "BUILTIN-memory-BLOCK" not in parts["volatile"]
        assert "BUILTIN-user-BLOCK" not in parts["volatile"]
        assert "EXTERNAL-MEMORY-BLOCK" not in parts["volatile"]

    def test_memory_prompt_includes_memory_blocks_by_default(self):
        class FakeMemoryStore:
            def format_for_system_prompt(self, target):
                return f"BUILTIN-{target}-BLOCK"

        class FakeMemoryManager:
            def build_system_prompt(self):
                return "EXTERNAL-MEMORY-BLOCK"

        agent = _make_agent(
            _memory_store=FakeMemoryStore(),
            _memory_manager=FakeMemoryManager(),
        )

        with (
            patch("run_agent.load_soul_md", return_value=""),
            patch("run_agent.build_nous_subscription_prompt", return_value=""),
            patch("run_agent.build_environment_hints", return_value=""),
            patch("run_agent.build_context_files_prompt", return_value=""),
        ):
            parts = build_system_prompt_parts(agent)

        assert "BUILTIN-memory-BLOCK" in parts["volatile"]
        assert "BUILTIN-user-BLOCK" in parts["volatile"]
        assert "EXTERNAL-MEMORY-BLOCK" in parts["volatile"]
