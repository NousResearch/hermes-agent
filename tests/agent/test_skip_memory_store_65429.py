"""Regression test for issue #65429.

An agent built with ``skip_memory=True`` AND ``enabled_toolsets=["memory"]``
used to get a memory tool wired to ``store=None`` (the built-in
``MemoryStore`` was skipped along with the external provider), so every
memory call failed silently and the main auto-capture path was dead.

The fix creates the built-in store whenever memory is enabled in config OR the
memory toolset is explicitly enabled, while the external-provider block stays
gated on ``skip_memory``.
"""

import pytest

from run_agent import AIAgent


class _FakeOpenAI:
    def __init__(self, **kw):
        self.api_key = kw.get("api_key", "test")
        self.base_url = kw.get("base_url", "http://test")

    def close(self):
        pass


def _make_agent(
    monkeypatch,
    enabled_toolsets=None,
    skip_memory=True,
    memory_enabled=None,
    user_profile_enabled=None,
):
    monkeypatch.setattr("run_agent.get_tool_definitions", lambda **kw: [])
    monkeypatch.setattr("run_agent.check_toolset_requirements", lambda: {})
    monkeypatch.setattr("run_agent.OpenAI", _FakeOpenAI)
    # Deterministic memory config for the agent-under-test. The CLI now passes
    # skip_memory=False under --ignore-rules, so memory enablement comes purely
    # from config (memory.memory_enabled / user_profile_enabled). Leave the
    # real config untouched when neither flag is given (repo default enables
    # memory), so pre-existing tests keep their original behavior.
    if memory_enabled is not None or user_profile_enabled is not None:
        monkeypatch.setattr(
            "hermes_cli.config.load_config_readonly",
            lambda: {
                "memory": {
                    "memory_enabled": bool(memory_enabled),
                    "user_profile_enabled": bool(user_profile_enabled),
                }
            },
        )
    return AIAgent(
        api_key="test-key",
        base_url="http://test",
        provider="openrouter",
        api_mode="chat_completions",
        max_iterations=1,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=skip_memory,
        enabled_toolsets=enabled_toolsets,
    )


def test_skip_memory_with_memory_toolset_creates_store(monkeypatch, tmp_path):
    """Memory toolset enabled despite skip_memory=True must still build the built-in store."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hm"))
    agent = _make_agent(monkeypatch, enabled_toolsets=["memory"], skip_memory=True)
    assert agent._memory_store is not None, (
        "memory toolset enabled despite skip_memory=True must still build "
        "the built-in store (#65429)"
    )






def test_skip_memory_memory_tool_handler_works_and_provider_skipped(
    monkeypatch, tmp_path
):
    """End-to-end behavioral check for #65429.

    The memory tool handler must actually WORK (not return the
    "Memory is not available" store=None error) on a skip_memory=True agent
    with the memory toolset enabled, while the external memory provider
    sync/prefetch stays skipped (no MemoryManager is created).
    """
    import json

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hm"))
    agent = _make_agent(monkeypatch, enabled_toolsets=["memory"], skip_memory=True)

    # Provider sync/prefetch must remain skipped: skip_memory still gates the
    # external memory provider block.
    assert agent._memory_manager is None, (
        "skip_memory=True must still skip the external memory provider"
    )

    # Dispatch through the same entry point the tool executor uses
    # (agent/tool_executor.py wires store=agent._memory_store).
    from tools.memory_tool import memory_tool

    raw = memory_tool(
        action="add",
        target="memory",
        content="User prefers concise answers.",
        store=agent._memory_store,
    )
    result = json.loads(raw)
    assert result.get("success") is True, (
        f"memory tool handler must work with skip_memory=True + memory "
        f"toolset (#65429), got: {raw}"
    )
    assert "Memory is not available" not in raw

    # The write must actually persist to the profile-scoped memories dir.
    memory_md = tmp_path / "hm" / "memories" / "MEMORY.md"
    assert memory_md.exists()
    assert "User prefers concise answers." in memory_md.read_text()


def test_ignore_rules_keeps_builtin_store_when_memory_enabled(monkeypatch, tmp_path):
    """--ignore-rules must NOT disable the built-in memory store.

    Regression: ``cli_agent_setup_mixin.py`` passed ``skip_memory=self.ignore_rules``,
    so ``hermes chat --ignore-rules`` silently dropped MEMORY.md/USER.md — the
    user's durable state — even though built-in memory is not an injected rule
    file (AGENTS.md/SOUL.md/.cursorrules are the rules that --ignore-rules
    skips, via skip_context_files). The CLI now passes skip_memory=False.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hm"))
    agent = _make_agent(
        monkeypatch,
        skip_memory=False,
        memory_enabled=True,
        user_profile_enabled=True,
    )
    assert agent._memory_store is not None, (
        "--ignore-rules must keep the built-in memory store when "
        "memory_enabled/user_profile_enabled are set in config"
    )


def test_ignore_rules_memory_disabled_leaves_no_store(monkeypatch, tmp_path):
    """Behavior preservation: with memory disabled in config the store stays
    None regardless of skip_memory — --ignore-rules must not *enable* memory
    that the user explicitly disabled."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hm"))
    agent = _make_agent(
        monkeypatch,
        skip_memory=False,
        memory_enabled=False,
        user_profile_enabled=False,
    )
    assert agent._memory_store is None


def test_cli_agent_setup_mixin_ignore_rules_passes_skip_context_files_only(monkeypatch):
    """CLIAgentSetupMixin._init_agent with ignore_rules=True passes
    skip_context_files=True and skip_memory=False to AIAgent constructor."""
    from unittest.mock import MagicMock
    from hermes_cli.cli_agent_setup_mixin import CLIAgentSetupMixin

    class DummyCLI(CLIAgentSetupMixin):
        def __init__(self):
            self.agent = None
            self.ignore_rules = True
            self.verbose = False
            self.system_prompt = None
            self.prefill_messages = None
            self.reasoning_config = None
            self.service_tier = None
            self._providers_only = None
            self._providers_ignore = None
            self._providers_order = None
            self._provider_sort = None
            self._provider_require_params = False
            self._provider_data_collection = False
            self._openrouter_min_coding_score = None
            self.session_id = "test-session"
            self._session_db = None
            self._clarify_callback = None
            self._fallback_model = None
            self._on_thinking = None
            self.checkpoints_enabled = False
            self.checkpoint_max_snapshots = 1
            self.checkpoint_max_total_size_mb = 10
            self.checkpoint_max_file_size_mb = 5
            self.pass_session_id = False
            self._on_tool_progress = None
            self._inline_diffs_enabled = False
            self.streaming_enabled = False
            self._on_notice = None
            self._on_notice_clear = None
            self._on_reaction = None
            self._resumed = False
            self.conversation_history = []
            self.model = "test-model"
            self.provider = "openrouter"
            self.base_url = "http://test"
            self.api_key = "test-key"
            self.api_mode = "chat_completions"
            self.max_tokens = 100
            self.temperature = 0.7
            self.max_turns = 1
            self.exec_timeout = 30
            self.enabled_toolsets = []
            self.disabled_toolsets = []
            self.acp_command = None
            self.acp_args = []
            self.agent_kwargs = {}
            self._pending_title = None

        def finalize_preloaded_skills(self):
            pass

        def _install_tool_callbacks(self):
            pass

        def _ensure_tirith_security(self):
            pass

        def _ensure_runtime_credentials(self):
            return True

        def _current_reasoning_callback(self):
            return None

    dummy = DummyCLI()
    captured_kwargs = {}

    def fake_ai_agent(**kwargs):
        captured_kwargs.update(kwargs)
        return MagicMock()

    monkeypatch.setattr("cli.AIAgent", fake_ai_agent)
    monkeypatch.setattr("cli._prepare_deferred_agent_startup", lambda: None)
    monkeypatch.setattr(
        "hermes_cli.mcp_startup.ensure_mcp_discovery_before_agent_build",
        lambda **kw: None,
    )

    success = dummy._init_agent()
    assert success is True
    assert captured_kwargs.get("skip_context_files") is True
    assert captured_kwargs.get("skip_memory") is False
