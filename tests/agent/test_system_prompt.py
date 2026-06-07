"""Tests for agent/system_prompt.py — context-file cwd wiring."""

from types import SimpleNamespace
from unittest.mock import patch

from agent.system_prompt import build_system_prompt_parts
from tools.memory_tool import MemoryStore


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
        _memory_prompt_initialized=False,
        _memory_enabled=False,
        _user_profile_enabled=False,
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


class TestMemoryPromptInitialization:
    def test_first_prompt_uses_frozen_snapshot(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        mem_dir = tmp_path / "memories"
        mem_dir.mkdir(parents=True)
        (mem_dir / "MEMORY.md").write_text("loaded at start")

        store = MemoryStore()
        store.load_from_disk()
        store._last_prompt_snapshot["memory"] = ""
        agent = _make_agent(
            _memory_store=store,
            _memory_enabled=True,
        )

        with (
            patch("run_agent.load_soul_md", return_value=""),
            patch("run_agent.build_nous_subscription_prompt", return_value=""),
            patch("run_agent.build_environment_hints", return_value=""),
            patch("run_agent.build_context_files_prompt", return_value=""),
        ):
            parts = build_system_prompt_parts(agent)

        assert "loaded at start" in parts["volatile"]
        assert agent._memory_prompt_initialized is True

    def test_rebuild_after_write_uses_diff_only_memory_block(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        mem_dir = tmp_path / "memories"
        mem_dir.mkdir(parents=True)
        (mem_dir / "MEMORY.md").write_text("loaded at start")

        store = MemoryStore()
        store.load_from_disk()
        agent = _make_agent(
            _memory_store=store,
            _memory_enabled=True,
            _memory_prompt_initialized=True,
        )
        store._last_prompt_snapshot["memory"] = store._render_live_snapshot("memory")
        store.add("memory", "added later")

        with (
            patch("run_agent.load_soul_md", return_value=""),
            patch("run_agent.build_nous_subscription_prompt", return_value=""),
            patch("run_agent.build_environment_hints", return_value=""),
            patch("run_agent.build_context_files_prompt", return_value=""),
        ):
            parts = build_system_prompt_parts(agent)

        assert "MEMORY UPDATE" in parts["volatile"]
        assert "+ added later" in parts["volatile"]
        assert "loaded at start" not in parts["volatile"]
