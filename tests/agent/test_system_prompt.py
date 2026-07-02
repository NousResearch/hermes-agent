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

    def fake_context_files(cwd=None, skip_soul=False, context_length=None):
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


def _init_code_repo(path):
    """A git repo that actually holds code — the coding posture requires a source
    file (or manifest), not a bare ``.git`` (a prose/notes repo stays general)."""
    import subprocess

    subprocess.run(["git", "-C", str(path), "init", "-q"], check=True)
    (path / "main.py").write_text("print('hi')\n")


class TestCodingContextBlock:
    def test_injected_when_active(self, monkeypatch, tmp_path):
        _init_code_repo(tmp_path)
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        agent = _make_agent(valid_tool_names=["read_file"], platform="cli")
        stable = _stable_prompt(agent)
        assert "coding agent" in stable
        assert "Workspace" in stable

    def test_absent_when_off(self, monkeypatch, tmp_path):
        _init_code_repo(tmp_path)
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        agent = _make_agent(valid_tool_names=["read_file"], platform="cli")
        # Drive the real path: force the resolved mode to "off" via config.
        with patch("agent.coding_context._coding_mode", return_value="off"):
            stable = _stable_prompt(agent)
        assert "coding agent" not in stable

    def test_absent_without_tools(self, monkeypatch, tmp_path):
        _init_code_repo(tmp_path)
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        agent = _make_agent(valid_tool_names=[], platform="cli")
        assert "coding agent" not in _stable_prompt(agent)


class TestProfilePathDisplay:
    """The stable prompt must show the correct profile path per platform."""

    def _stable_with_profile(self, profile_name: str) -> str:
        from unittest.mock import patch as mock_patch

        with (
            mock_patch("run_agent.load_soul_md", return_value=""),
            mock_patch("run_agent.build_nous_subscription_prompt", return_value=""),
            mock_patch("run_agent.build_environment_hints", return_value=""),
            mock_patch("run_agent.build_context_files_prompt", return_value=""),
            mock_patch(
                "agent.file_safety._resolve_active_profile_name",
                return_value=profile_name,
            ),
        ):
            return build_system_prompt_parts(_make_agent())["stable"]

    def test_windows_profile_includes_localappdata(self, monkeypatch):
        """On native Windows the prompt should reference LOCALAPPDATA."""
        monkeypatch.setattr("agent._platform_paths.os.name", "nt")
        stable = self._stable_with_profile("hades")
        assert "%LOCALAPPDATA%" in stable
        assert r"\hermes\profiles\hades" in stable
        assert "~/.hermes" not in stable

    def test_windows_default_profile_includes_localappdata(self, monkeypatch):
        """Default profile hint should also be Windows-aware."""
        monkeypatch.setattr("agent._platform_paths.os.name", "nt")
        stable = self._stable_with_profile("default")
        assert "%LOCALAPPDATA%" in stable

    def test_posix_profile_uses_tilde(self, monkeypatch):
        """On POSIX (WSL / Linux / macOS) the old ~/.hermes path is correct."""
        monkeypatch.setattr("agent._platform_paths.os.name", "posix")
        stable = self._stable_with_profile("hades")
        assert "~/.hermes" in stable
        assert "%LOCALAPPDATA%" not in stable
