"""Tests for agent/system_prompt.py — context-file cwd wiring."""

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agent.system_prompt import build_system_prompt, build_system_prompt_parts


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
        # build_system_prompt drains pending truncation warnings and
        # forwards each to this; a warning left in the ContextVar by an
        # earlier test file (they share one thread's context under plain
        # pytest) must not make this stub AttributeError.
        _emit_status=lambda *_args, **_kwargs: None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _captured_context_cwd(agent):
    """The cwd build_system_prompt_parts hands to build_context_files_prompt."""
    captured = {}

    def fake_context_files(
        cwd=None, skip_soul=False, context_length=None,
        allow_install_tree_fallback=False, home_override=None,
    ):
        captured["cwd"] = cwd
        return ""

    with (
        patch("run_agent.load_soul_md", return_value=""),
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
        patch("run_agent.build_environment_hints", return_value=""),
        patch("run_agent.build_context_files_prompt", return_value=""),
    ):
        return build_system_prompt_parts(agent)["stable"]


def _prompt_parts(agent):
    with (
        patch("run_agent.load_soul_md", return_value=""),
        patch("run_agent.build_environment_hints", return_value=""),
        patch("run_agent.build_context_files_prompt", return_value=""),
    ):
        return build_system_prompt_parts(agent)


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
        parts = _prompt_parts(agent)
        assert "coding agent" in parts["stable"]
        assert "Workspace" in parts["context"]

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


class TestExecutionGuidanceInjection:
    """Injection gate for OPENAI_MODEL_EXECUTION_GUIDANCE via
    ``agent.execution_guidance`` (auto/true/false/list).

    Background — Composio agentic-eval traces (2026-08): the block was
    historically fenced to gpt/codex/grok AND nested inside the
    tool-use-enforcement branch, so DeepSeek/Kimi/Qwen-class models
    received no execution discipline at all. The gate is now independent
    of tool_use_enforcement and defaults to a broader family list.
    """

    def _prompt(self, model, execution_guidance="auto", *,
                tool_use_enforcement=False,
                valid_tool_names=("terminal", "read_file")):
        agent = _make_agent(
            valid_tool_names=list(valid_tool_names),
            model=model,
            _tool_use_enforcement=tool_use_enforcement,
            _execution_guidance=execution_guidance,
        )
        return _stable_prompt(agent)

    def test_deepseek_gets_guidance_by_default(self):
        stable = self._prompt("deepseek/deepseek-v4-pro")
        assert "Execution discipline" in stable
        assert "<external_state_verification>" in stable

    def test_kimi_gets_guidance_by_default(self):
        assert "Execution discipline" in self._prompt("moonshotai/kimi-k3")

    def test_qwen_glm_minimax_mimo_mistral_get_guidance_by_default(self):
        for model in ("qwen/qwen-3-max", "z-ai/glm-5.2",
                      "minimax/minimax-m2", "xiaomi/mimo-v2",
                      "mistralai/mistral-large-3"):
            assert "Execution discipline" in self._prompt(model), model

    def test_gpt_still_gets_guidance(self):
        assert "Execution discipline" in self._prompt("openai/gpt-5.5")

    def test_grok_still_gets_guidance(self):
        assert "Execution discipline" in self._prompt("xai/grok-4")

    def test_independent_of_tool_use_enforcement(self):
        # The gate must not require tool-use enforcement to be on.
        stable = self._prompt("deepseek/deepseek-v4-flash",
                              tool_use_enforcement=False)
        assert "Execution discipline" in stable
        assert "Tool-use enforcement" not in stable

    def test_claude_does_not_get_guidance_by_default(self):
        assert "Execution discipline" not in self._prompt(
            "anthropic/claude-opus-4.8")

    def test_gemini_does_not_get_guidance_by_default(self):
        assert "Execution discipline" not in self._prompt(
            "google/gemini-2.5-pro")

    def test_config_false_suppresses(self):
        assert "Execution discipline" not in self._prompt(
            "openai/gpt-5.5", execution_guidance=False)
        assert "Execution discipline" not in self._prompt(
            "deepseek/deepseek-v4-pro", execution_guidance="off")

    def test_config_true_forces_for_any_model(self):
        assert "Execution discipline" in self._prompt(
            "anthropic/claude-opus-4.8", execution_guidance=True)

    def test_config_list_matches_substring(self):
        stable = self._prompt("mycorp/custom-llm-7b",
                              execution_guidance=["custom-llm", "gpt"])
        assert "Execution discipline" in stable

    def test_config_list_non_match_suppresses(self):
        assert "Execution discipline" not in self._prompt(
            "openai/gpt-5.5", execution_guidance=["deepseek"])

    def test_no_tools_no_guidance(self):
        assert "Execution discipline" not in self._prompt(
            "deepseek/deepseek-v4-pro", valid_tool_names=())


class TestNamedProfileHintIntegration:
    """The same defect through the REAL resolution chain (#72894).

    ``TestNamedProfileHint`` mocks ``get_hermes_home``,
    ``get_default_hermes_root`` and ``_resolve_active_profile_name``, so it
    validates template rendering but not the relationship that causes the bug:
    ``_resolve_active_profile_name`` returns a named profile *only* when the
    active home is already ``<root>/profiles/<name>``, which is exactly why
    appending that suffix again doubled it. Drive it with a real
    ``HERMES_HOME`` and no resolver mocks.
    """

    def test_real_hermes_home_under_profiles_renders_correct_paths(
        self, tmp_path, monkeypatch
    ):
        root = tmp_path / ".hermes"
        profile_home = root / "profiles" / "coder"
        profile_home.mkdir(parents=True)

        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(profile_home))
        monkeypatch.delenv("TERMINAL_CWD", raising=False)

        # Sanity-check the real chain before asserting on the prompt.
        from agent.file_safety import _resolve_active_profile_name
        from hermes_constants import get_default_hermes_root, get_hermes_home

        assert _resolve_active_profile_name() == "coder"
        assert get_hermes_home() == profile_home
        assert get_default_hermes_root() == root

        agent = _make_agent(valid_tool_names=["read_file"])
        with patch("agent.coding_context._coding_mode", return_value="off"):
            prompt = "\n\n".join(_prompt_parts(agent).values())

        assert "Active Hermes profile: coder." in prompt
        assert f"reads and writes {profile_home}/." in prompt
        # The doubled form must not appear anywhere.
        assert f"{profile_home}/profiles/coder" not in prompt
        # Default-profile pointers belong at the root, not inside the profile.
        assert f"The default profile's data lives at {root}/skills/" in prompt
        assert f"{profile_home}/skills/" not in prompt

    def test_real_default_home_renders_default_branch(self, tmp_path, monkeypatch):
        """HERMES_HOME at the root resolves to the default profile, unchanged."""
        root = tmp_path / ".hermes"
        root.mkdir(parents=True)

        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(root))
        monkeypatch.delenv("TERMINAL_CWD", raising=False)

        from agent.file_safety import _resolve_active_profile_name

        assert _resolve_active_profile_name() == "default"

        agent = _make_agent(valid_tool_names=["read_file"])
        with patch("agent.coding_context._coding_mode", return_value="off"):
            prompt = "\n\n".join(_prompt_parts(agent).values())

        assert "Active Hermes profile: default." in prompt
        assert f"under {root}/profiles/<name>/." in prompt


def test_build_system_prompt_records_stable_prefix():
    agent = _make_agent()
    with (
        patch("run_agent.load_soul_md", return_value=""),
        patch("run_agent.build_environment_hints", return_value=""),
        patch("run_agent.build_context_files_prompt", return_value="context"),
    ):
        prompt = build_system_prompt(agent)

    assert prompt.startswith(agent._cached_system_prompt_static)
    assert prompt[len(agent._cached_system_prompt_static):].startswith("\n\ncontext")


def test_coding_prompt_preserves_legacy_workspace_order(monkeypatch):
    """The cache split must not reorder the stored coding prompt."""
    import agent.system_prompt as system_prompt

    agent = _make_agent(
        valid_tool_names=["read_file"],
        _parallel_tool_call_guidance=False,
    )
    monkeypatch.setattr(system_prompt, "DEFAULT_AGENT_IDENTITY", "IDENTITY")
    monkeypatch.setattr(system_prompt, "HERMES_AGENT_HELP_GUIDANCE", "HELP")
    monkeypatch.setattr(system_prompt, "HERMES_AGENT_HELP_GUIDANCE_NO_SKILLS", "HELP")
    monkeypatch.setattr(system_prompt, "STEER_CHANNEL_NOTE", "STEER")
    monkeypatch.setattr(system_prompt, "get_hermes_home", lambda: Path("/hermes"))

    expected_profile = (
        "Active Hermes profile: default. Other profiles (if any) live "
        "under /hermes/profiles/<name>/. Each profile has its own skills/, "
        "plugins/, cron/, and memories/ that affect a different session than "
        "this one. Do not modify another profile's skills/plugins/cron/memories "
        "unless the user explicitly directs you to."
    )
    expected = "\n\n".join((
        "IDENTITY",
        "HELP",
        "STEER",
        "CODING_STABLE",
        "WORKSPACE",
        "Operator instructions (from config):\nOPERATOR",
        expected_profile,
        "SYSTEM_MESSAGE",
        "CONTEXT_FILES",
        "Conversation started: Friday, January 02, 2026",
    ))

    with (
        patch("run_agent.load_soul_md", return_value=""),
        patch("run_agent.build_environment_hints", return_value=""),
        patch("run_agent.build_context_files_prompt", return_value="CONTEXT_FILES"),
        patch(
            "agent.coding_context.coding_system_prompt_parts",
            return_value=(
                ["CODING_STABLE"],
                ["WORKSPACE"],
                ["Operator instructions (from config):\nOPERATOR"],
            ),
        ),
        patch("agent.file_safety._resolve_active_profile_name", return_value="default"),
        patch("hermes_time.now", return_value=datetime(2026, 1, 2)),
    ):
        prompt = build_system_prompt(agent, system_message="SYSTEM_MESSAGE")

    assert prompt == expected
    assert agent._cached_system_prompt_static == "\n\n".join(expected.split("\n\n")[:4])


class TestTelegramRichMessagesHint:
    """Verify that TELEGRAM_RICH_MESSAGES_HINT is conditionally included."""

    def test_base_hint_without_rich_messages(self, monkeypatch):
        """When rich_messages is False, only the base hint is used."""
        agent = _make_agent(platform="telegram")
        with patch("hermes_cli.config.load_config_readonly") as mock_cfg:
            mock_cfg.return_value = {
                "gateway": {"platforms": {"telegram": {"extra": {"rich_messages": False}}}}
            }
            stable = _stable_prompt(agent)
        assert "Standard Markdown auto-converts" in stable
        assert "lean into it" not in stable
        assert "task lists" not in stable

    def test_rich_hint_with_rich_messages_enabled(self, monkeypatch):
        """When rich_messages is True in gateway.platforms, the extension
        is appended (the canonical/primary location)."""
        agent = _make_agent(platform="telegram")
        with patch("hermes_cli.config.load_config_readonly") as mock_cfg:
            mock_cfg.return_value = {
                "gateway": {"platforms": {"telegram": {"extra": {"rich_messages": True}}}}
            }
            stable = _stable_prompt(agent)
        assert "lean into it" in stable
        assert "task lists" in stable
        assert "math/formulas" in stable

    def test_rich_hint_from_top_level_platforms(self):
        """Top-level ``platforms.telegram.extra.rich_messages`` is merged
        alongside gateway.platforms, so it works on its own."""
        agent = _make_agent(platform="telegram")
        with patch("hermes_cli.config.load_config_readonly") as mock_cfg:
            mock_cfg.return_value = {
                "platforms": {"telegram": {"extra": {"rich_messages": True}}}
            }
            stable = _stable_prompt(agent)
        assert "lean into it" in stable
        assert "task lists" in stable

    def test_top_level_overrides_gateway_rich_messages(self):
        """Top-level ``platforms.telegram.extra`` wins over gateway.platforms
        at the leaf, matching the adapter's merge precedence."""
        agent = _make_agent(platform="telegram")
        with patch("hermes_cli.config.load_config_readonly") as mock_cfg:
            mock_cfg.return_value = {
                "gateway": {"platforms": {"telegram": {"extra": {"rich_messages": False}}}},
                "platforms": {"telegram": {"extra": {"rich_messages": True}}},
            }
            stable = _stable_prompt(agent)
        assert "lean into it" in stable

    def test_gateway_extra_other_keys_does_not_block_top_level_rich_messages(self):
        """When gateway.platforms.telegram.extra has other keys but not
        rich_messages, the top-level rich_messages still activates."""
        agent = _make_agent(platform="telegram")
        with patch("hermes_cli.config.load_config_readonly") as mock_cfg:
            mock_cfg.return_value = {
                "gateway": {"platforms": {"telegram": {"extra": {"disable_link_previews": True}}}},
                "platforms": {"telegram": {"extra": {"rich_messages": True}}},
            }
            stable = _stable_prompt(agent)
        assert "lean into it" in stable

    def test_base_hint_without_config(self, monkeypatch):
        """When config has no telegram section, only base hint is used."""
        agent = _make_agent(platform="telegram")
        with patch("hermes_cli.config.load_config_readonly") as mock_cfg:
            mock_cfg.return_value = {}
            stable = _stable_prompt(agent)
        assert "Standard Markdown auto-converts" in stable
        assert "lean into it" not in stable


    def test_gateway_rich_messages_integration_via_real_config(self, tmp_path, monkeypatch):
        """End-to-end through the real config-resolution chain: a config.yaml
        under HERMES_HOME with ``gateway.platforms.telegram.extra.rich_messages``
        must activate the rich hint. ``load_config_readonly`` is NOT mocked here,
        so this guards against the exact path-mismatch bug this PR fixes.
        """
        config_yaml = (
            "gateway:\n"
            "  platforms:\n"
            "    telegram:\n"
            "      extra:\n"
            "        rich_messages: true\n"
        )
        home = tmp_path / "hermes_home"
        home.mkdir()
        (home / "config.yaml").write_text(config_yaml)

        monkeypatch.setenv("HERMES_HOME", str(home))
        # Point config resolution at the temp file without mocking the loader:
        # mirror the pattern used in test_config_env_expansion.py.
        from hermes_cli import config as _cfgmod
        monkeypatch.setattr(_cfgmod, "get_config_path", lambda: home / "config.yaml")

        agent = _make_agent(platform="telegram")
        stable = _stable_prompt(agent)
        assert "lean into it" in stable
        assert "task lists" in stable

    def test_malformed_extra_value_falls_back_to_base_hint(self, tmp_path, monkeypatch):
        """A truthy non-mapping ``extra`` must not crash prompt construction —
        it should fail open to the base hint (Tek's fail-open concern).
        """
        agent = _make_agent(platform="telegram")
        with patch("hermes_cli.config.load_config_readonly") as mock_cfg:
            mock_cfg.return_value = {
                "gateway": {"platforms": {"telegram": {"extra": "not-a-map"}}}
            }
            stable = _stable_prompt(agent)
        assert "Standard Markdown auto-converts" in stable
        assert "lean into it" not in stable


_SKILLS = "SKILLS_INDEX_SENTINEL"
_CONTEXT = "CONTEXT_FILES_SENTINEL"


def _build(builder, **overrides):
    """Run a build_* function with skills + context files present."""
    agent = _make_agent(valid_tool_names=["skills_list"], **overrides)
    with (
        patch("run_agent.load_soul_md", return_value=""),
        patch("run_agent.build_environment_hints", return_value=""),
        patch("run_agent.build_context_files_prompt", return_value=_CONTEXT),
        patch("run_agent.get_toolset_for_tool", return_value=None),
        patch("run_agent.build_skills_system_prompt", return_value=_SKILLS),
    ):
        return builder(agent)


class TestSkillsInVolatileBand:
    """The skills index is runtime-mutable, so it lives in the volatile band,
    not the stable band, to keep the cached stable prefix reusable when a
    rebuild picks up a skill change."""

    def test_skills_not_in_stable_band(self):
        parts = _build(build_system_prompt_parts)
        assert _SKILLS not in parts["stable"]

    def test_skills_lead_the_volatile_band(self):
        parts = _build(build_system_prompt_parts)
        assert parts["volatile"].startswith(_SKILLS)

    def test_full_order_is_stable_context_then_skills(self):
        # build_system_prompt joins stable + context + volatile, so the skills
        # index renders after the context files and before the per-turn
        # memory/timestamp tail.
        full = _build(build_system_prompt)
        assert full.index(_CONTEXT) < full.index(_SKILLS)
        assert full.index(_SKILLS) < full.index("Conversation started:")


class TestMemoryProviderSystemPromptGating:
    """Issue #81014: the provider's ``system_prompt_block()`` must be gated
    on the same ``memory_provider_tools_enabled`` check as tool injection.
    Otherwise the agent receives instructions for tools that don't exist in
    its tool surface.
    """

    @staticmethod
    def _make_fake_manager(prompt_block: str):
        """Build a MemoryManager-like object exposing only what
        ``build_system_prompt_parts`` touches."""
        from unittest.mock import MagicMock
        mgr = MagicMock()
        mgr.build_system_prompt.return_value = prompt_block
        return mgr

    def _agent(self, *, enabled_toolsets, disabled_toolsets, prompt_block):
        return _make_agent(
            valid_tool_names=["skills_list"],
            enabled_toolsets=enabled_toolsets,
            disabled_toolsets=disabled_toolsets,
            _memory_manager=self._make_fake_manager(prompt_block),
        )

    def test_block_injected_when_memory_toolset_enabled(self):
        block = "PROVIDER_BLOCK_SENTINEL"
        agent = self._agent(
            enabled_toolsets=["memory"],
            disabled_toolsets=None,
            prompt_block=block,
        )
        full = _build(build_system_prompt, _memory_manager=agent._memory_manager,
                      enabled_toolsets=["memory"], disabled_toolsets=None)
        assert block in full

    def test_block_dropped_when_memory_toolset_disabled(self):
        block = "PROVIDER_BLOCK_SENTINEL"
        agent = self._agent(
            enabled_toolsets=None,
            disabled_toolsets=["memory"],
            prompt_block=block,
        )
        full = _build(build_system_prompt, _memory_manager=agent._memory_manager,
                      enabled_toolsets=None, disabled_toolsets=["memory"])
        assert block not in full

    def test_block_dropped_when_memory_not_in_enabled_toolsets(self):
        block = "PROVIDER_BLOCK_SENTINEL"
        agent = self._agent(
            enabled_toolsets=["web_search"],
            disabled_toolsets=None,
            prompt_block=block,
        )
        full = _build(build_system_prompt, _memory_manager=agent._memory_manager,
                      enabled_toolsets=["web_search"], disabled_toolsets=None)
        assert block not in full
