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


class TestOpenAIExecutionGuidanceInjection:
    """Regression tests for the tool_use enforcement / OPENAI execution
    discipline injection block in ``build_system_prompt_parts``.

    Background — 2026-06-27 Telegram stall: GLM-5.2 replied with a
    plain-text ``[TOOL_CALL]...[/TOOL_CALL]`` marker instead of a
    structured ``tool_calls`` JSON block, so the runtime saw
    ``tool_calls=None`` and finished with ``finish_reason=stop`` after
    one assistant turn. Root cause: the system prompt injector only
    added ``OPENAI_MODEL_EXECUTION_GUIDANCE`` for ``gpt/codex/grok``
    substrings, so non-Google TOOL_USE_ENFORCEMENT_MODELS families
    (``glm``, ``qwen``, ``deepseek``) got the lighter ``TOOL_USE_
    ENFORCEMENT_GUIDANCE`` block but not the execution-discipline
    block (tool persistence, anti-fabrication, mandatory_tool_use).
    """

    def _prompt(self, model, *, valid_tool_names=("terminal", "read_file"),
                tool_use_enforcement=True):
        agent = _make_agent(
            valid_tool_names=list(valid_tool_names),
            model=model,
            _tool_use_enforcement=tool_use_enforcement,
        )
        return _stable_prompt(agent)

    def test_glm_5_2_receives_openai_execution_guidance(self):
        # The exact model from the 2026-06-27 Telegram stall reproduction.
        stable = self._prompt("z-ai/glm-5.2")
        assert "Execution discipline" in stable
        assert "<tool_persistence>" in stable
        assert "<mandatory_tool_use>" in stable

    def test_deepseek_receives_openai_execution_guidance(self):
        stable = self._prompt("deepseek/deepseek-v4-pro")
        assert "Execution discipline" in stable

    def test_qwen_receives_openai_execution_guidance(self):
        stable = self._prompt("qwen/qwen-3-max")
        assert "Execution discipline" in stable

    def test_gpt_still_receives_openai_execution_guidance(self):
        # No regression on the original coverage.
        stable = self._prompt("openai/gpt-5.5")
        assert "Execution discipline" in stable

    def test_grok_still_receives_openai_execution_guidance(self):
        stable = self._prompt("xai/grok-4")
        assert "Execution discipline" in stable

    def test_anthropic_opus_does_not_receive_openai_execution_guidance(self):
        # Claude has its own anthropic-transport guidance; OPENAI_ block
        # is body-agnostic but conceptually targeted at the families that
        # share GPT/Grok/GLM failure modes. Don't double-inject.
        stable = self._prompt("anthropic/claude-opus-4.8")
        assert "Execution discipline" not in stable

    def test_google_gemini_does_not_receive_openai_execution_guidance(self):
        # Google gets the more specific GOOGLE_MODEL_OPERATIONAL block
        # instead. Adding OPENAI_ would duplicate the parallel-call steer.
        stable = self._prompt("google/gemini-2.5-pro")
        assert "Execution discipline" not in stable
        assert "Google model operational directives" in stable

    def test_disabled_when_enforcement_off(self):
        stable = self._prompt("z-ai/glm-5.2", tool_use_enforcement=False)
        # When the whole block is off neither guidance should land.
        assert "Execution discipline" not in stable
        assert "Tool-use enforcement" not in stable
