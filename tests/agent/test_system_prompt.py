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


class TestToolUseEnforcementInjectionWithoutTools:
    """Regression tests for the tool_use-enforcement guidance block when
    the session starts with ``valid_tool_names=[]``.

    Background — 2026-06-28 Telegram stall (session
    ``20260628_003118_249c66d5``): the system prompt was 20,401 chars
    and contained none of ``Tool-use enforcement`` /
    ``Parallel tool calls`` markers. The GLM-5.2 model emitted its tool
    intent as a literal ``bash`` markdown code block (different syntax
    than the earlier ``[TOOL_CALL]`` markers but identical underlying
    failure mode), the runtime saw ``tool_calls=None``, and the
    conversation loop ended with ``finish_reason=stop`` after one turn.

    Root cause: the tool-use-enforcement block was wrapped in
    ``if agent.valid_tool_names:`` so the entire guidance — including
    the ``TOOL_USE_ENFORCEMENT_GUIDANCE`` text the model needs to emit
    structured ``tool_calls`` instead of markdown — was skipped when
    the session happened to have no tools loaded. Telegram/Discord
    gateway entry points often start without tools, so this affected
    every GLM/Qwen/DeepSeek session that arrived through those
    channels.

    The fix decouples the guidance injection from ``valid_tool_names``:
    the model-name gate drives whether the block lands, the
    ``valid_tool_names`` check is preserved only for the
    Google-operational sub-block (whose content is about how to use
    file/edit tools, irrelevant without tools) and for
    ``parallel_tool_call_guidance`` (no point steering a no-tool session
    to batch calls).
    """

    def _prompt(self, model, *, valid_tool_names=(), tool_use_enforcement="auto"):
        # type: ignore[assignment]
        agent = _make_agent(
            valid_tool_names=list(valid_tool_names),
            model=model,
            _tool_use_enforcement=tool_use_enforcement,
        )
        return _stable_prompt(agent)

    def test_glm_5_2_receives_enforcement_guidance_without_tools(self):
        # The exact reproduction conditions for the 2026-06-28 stall.
        stable = self._prompt("z-ai/glm-5.2")
        assert "Tool-use enforcement" in stable

    def test_qwen_receives_enforcement_guidance_without_tools(self):
        stable = self._prompt("qwen/qwen-3-max")
        assert "Tool-use enforcement" in stable

    def test_deepseek_receives_enforcement_guidance_without_tools(self):
        stable = self._prompt("deepseek/deepseek-v4-pro")
        assert "Tool-use enforcement" in stable

    def test_anthropic_opus_does_not_receive_enforcement_guidance(self):
        # Opus is not in TOOL_USE_ENFORCEMENT_MODELS — the no-tools
        # path must NOT accidentally inject guidance for models that
        # don't need it.
        stable = self._prompt("anthropic/claude-opus-4.8")
        assert "Tool-use enforcement" not in stable

    def test_parallel_tool_call_guidance_still_tools_gated(self):
        # Parallel-call guidance is about batching tool calls — without
        # tools there's nothing to batch, so the gate stays. This is
        # intentional and prevents prompt-bloat for read-only sessions.
        stable = self._prompt("z-ai/glm-5.2", valid_tool_names=())
        assert "Parallel tool calls" not in stable

    def test_parallel_tool_call_guidance_present_with_tools(self):
        # Sanity check: with tools, both blocks land as before.
        stable = self._prompt(
            "z-ai/glm-5.2",
            valid_tool_names=("terminal", "read_file"),
        )
        assert "Tool-use enforcement" in stable
        assert "Parallel tool calls" in stable

    def test_enforcement_off_still_disables_without_tools(self):
        # The config knob still works on the no-tools path — turning
        # off enforcement means no guidance, period.
        stable = self._prompt(
            "z-ai/glm-5.2",
            valid_tool_names=(),
            tool_use_enforcement=False,
        )
        assert "Tool-use enforcement" not in stable

    def test_enforcement_explicit_on_injects_for_all_models(self):
        # The "true" override should land guidance even for models not
        # in TOOL_USE_ENFORCEMENT_MODELS — this is the escape hatch.
        stable = self._prompt(
            "anthropic/claude-opus-4.8",
            valid_tool_names=(),
            tool_use_enforcement=True,
        )
        assert "Tool-use enforcement" in stable

    def test_google_operational_block_still_tools_gated(self):
        # The Gemini/Gemma-specific block is about file paths and edit
        # commands — without tools the content is irrelevant noise.
        stable = self._prompt("google/gemini-2.5-pro")
        assert "Google model operational directives" not in stable
