"""Tests that the background review agent restricts tools at runtime, not at schema time.

Regression coverage for issue #15204 (the background skill-review agent must
not perform non-skill side effects like terminal, send_message, delegate_task)
combined with issue #25322 / PR #17276 (the review fork must hit the parent's
Anthropic/OpenRouter prefix cache).

Reconciling the two: the fork now inherits the parent's full ``tools`` schema
so the cache-key matches, and enforces the memory+skills restriction at
runtime via a thread-local whitelist on the existing
``get_pre_tool_call_block_message`` gate. Safety is preserved mechanically
(any non-whitelisted dispatch is blocked) without the schema-level narrowing
that caused the prefix-cache miss.
"""

from unittest.mock import patch


def _make_agent_stub(agent_cls):
    """Create a minimal AIAgent-like object with just enough state for _spawn_background_review."""
    agent = object.__new__(agent_cls)
    agent.model = "test-model"
    agent.platform = "test"
    agent.provider = "openai"
    agent.session_id = "sess-123"
    agent.quiet_mode = True
    agent._memory_store = None
    agent._memory_enabled = True
    agent._user_profile_enabled = False
    agent._memory_nudge_interval = 5
    agent._skill_nudge_interval = 5
    agent.background_review_callback = None
    agent.status_callback = None
    agent._cached_system_prompt = None
    import datetime as _dt
    agent.session_start = _dt.datetime(2026, 1, 1, 12, 0, 0)
    agent._MEMORY_REVIEW_PROMPT = "review memory"
    agent._SKILL_REVIEW_PROMPT = "review skills"
    agent._COMBINED_REVIEW_PROMPT = "review both"
    # Non-None so the test catches a missing-kwarg regression.
    agent.enabled_toolsets = ["memory", "skills", "terminal"]
    agent.disabled_toolsets = ["spotify", "feishu_doc"]
    return agent


class _SyncThread:
    """Drop-in replacement for threading.Thread that runs the target inline."""

    def __init__(self, *, target=None, daemon=None, name=None):
        self._target = target

    def start(self):
        if self._target:
            self._target()


def test_background_review_matches_parent_toolset_config():
    """Fork must receive parent's toolset config so ``tools[]`` cache key matches."""
    import run_agent

    agent = _make_agent_stub(run_agent.AIAgent)
    captured = {}

    def _capture_init(self, *args, **kwargs):
        captured["enabled_toolsets"] = kwargs.get("enabled_toolsets", "UNSET")
        captured["disabled_toolsets"] = kwargs.get("disabled_toolsets", "UNSET")
        raise RuntimeError("stop after capturing init args")

    with patch.object(run_agent.AIAgent, "__init__", _capture_init), \
         patch("threading.Thread", _SyncThread):
        agent._spawn_background_review(
            messages_snapshot=[],
            review_memory=True,
            review_skills=False,
        )

    assert "enabled_toolsets" in captured, "AIAgent.__init__ was not called"
    assert captured["enabled_toolsets"] == agent.enabled_toolsets, (
        f"enabled_toolsets mismatch: {captured['enabled_toolsets']!r} "
        f"vs expected {agent.enabled_toolsets!r}"
    )
    assert captured["disabled_toolsets"] == agent.disabled_toolsets, (
        f"disabled_toolsets mismatch: {captured['disabled_toolsets']!r} "
        f"vs expected {agent.disabled_toolsets!r}"
    )


def test_background_review_installs_thread_local_whitelist():
    """The review fork must install a memory/skills-only thread-local whitelist.

    The schema-level toolset narrowing was lifted (for prefix-cache parity),
    so #15204's safety contract now relies on the runtime whitelist gate to
    deny terminal/send_message/delegate_task at dispatch time. Verify the
    whitelist is set with exactly the memory+skills tool names.
    """
    import run_agent
    from hermes_cli import plugins as _plugins

    captured = {}

    def _capture_whitelist(whitelist, deny_msg_fmt=None):
        captured["whitelist"] = set(whitelist)
        captured["deny_msg_fmt"] = deny_msg_fmt
        # Stop here — we just want to see what gets installed.
        raise RuntimeError("stop after capturing whitelist")

    agent = _make_agent_stub(run_agent.AIAgent)

    def _no_init(self, *args, **kwargs):
        # Don't crash AIAgent.__init__; let execution flow reach
        # set_thread_tool_whitelist.
        return None

    with patch.object(run_agent.AIAgent, "__init__", _no_init), \
         patch.object(_plugins, "set_thread_tool_whitelist", _capture_whitelist), \
         patch("threading.Thread", _SyncThread):
        agent._spawn_background_review(
            messages_snapshot=[],
            review_memory=True,
            review_skills=False,
        )

    assert "whitelist" in captured, "set_thread_tool_whitelist was not called"
    whitelist = captured["whitelist"]
    # memory + skills tools must be allowed
    assert "memory" in whitelist
    assert "skill_manage" in whitelist
    assert "skill_view" in whitelist
    assert "skills_list" in whitelist
    # dangerous tools must NOT be in the whitelist
    assert "terminal" not in whitelist
    assert "send_message" not in whitelist
    assert "delegate_task" not in whitelist
    assert "web_search" not in whitelist
    assert "execute_code" not in whitelist


def test_background_review_agent_tools_are_limited():
    """Verify the resolved memory+skills toolsets only contain memory and skill tools.

    Sanity check on the source of truth for what the runtime whitelist is
    derived from — if a future PR adds e.g. `terminal` to the `memory`
    toolset, the review-fork safety contract silently breaks.
    """
    from toolsets import resolve_multiple_toolsets

    expected_tools = set(resolve_multiple_toolsets(["memory", "skills"]))

    assert "memory" in expected_tools
    assert "skill_manage" in expected_tools
    assert "skill_view" in expected_tools
    assert "skills_list" in expected_tools

    assert "terminal" not in expected_tools
    assert "send_message" not in expected_tools
    assert "delegate_task" not in expected_tools
    assert "web_search" not in expected_tools
    assert "execute_code" not in expected_tools


def _run_with_flag(monkeypatch, flag_value):
    """Spawn the review fork with the mem0-write flag set; capture the whitelist."""
    import run_agent
    from hermes_cli import plugins as _plugins
    import hermes_cli.config as _config

    captured = {}

    def _capture_whitelist(whitelist, deny_msg_fmt=None):
        captured["whitelist"] = set(whitelist)
        raise RuntimeError("stop after capturing whitelist")

    # Make the flag read deterministic regardless of the real config.yaml.
    monkeypatch.setattr(_config, "load_config_readonly", lambda: {
        "memory": {"background_review_mem0_write": flag_value}})

    agent = _make_agent_stub(run_agent.AIAgent)

    def _no_init(self, *args, **kwargs):
        return None

    with patch.object(run_agent.AIAgent, "__init__", _no_init), \
         patch.object(_plugins, "set_thread_tool_whitelist", _capture_whitelist), \
         patch("threading.Thread", _SyncThread):
        agent._spawn_background_review(
            messages_snapshot=[],
            review_memory=True,
            review_skills=False,
        )
    return captured.get("whitelist", set())


def test_mem0_write_flag_off_excludes_mem0_remember(monkeypatch):
    """Default-OFF: mem0_remember must NOT be in the fork whitelist (denied-not-absent)."""
    whitelist = _run_with_flag(monkeypatch, False)
    assert "mem0_remember" not in whitelist
    # sanity: the normal memory/skill tools are still there
    assert "memory" in whitelist
    assert "skill_manage" in whitelist


def test_mem0_write_flag_on_includes_mem0_remember(monkeypatch):
    """Flag ON: mem0_remember is added to the fork whitelist (dispatch allowed)."""
    whitelist = _run_with_flag(monkeypatch, True)
    assert "mem0_remember" in whitelist
    # dangerous tools still excluded
    assert "terminal" not in whitelist
    assert "delegate_task" not in whitelist


def test_memory_prompt_mem0_clause_gated_by_flag(monkeypatch):
    """The mem0_remember save clause is appended to the memory-review prompt ONLY
    when the flag is on (and only for memory passes)."""
    import run_agent
    from agent import background_review as br
    import hermes_cli.config as _config

    agent = _make_agent_stub(run_agent.AIAgent)

    # Flag OFF -> clause absent
    monkeypatch.setattr(_config, "load_config_readonly", lambda: {
        "memory": {"background_review_mem0_write": False}})
    _, prompt_off = br.spawn_background_review_thread(
        agent, [], review_memory=True, review_skills=False)
    assert "mem0_remember" not in prompt_off

    # Flag ON -> clause present, names the tool + the durable-fact rubric
    monkeypatch.setattr(_config, "load_config_readonly", lambda: {
        "memory": {"background_review_mem0_write": True}})
    _, prompt_on = br.spawn_background_review_thread(
        agent, [], review_memory=True, review_skills=False)
    assert "mem0_remember" in prompt_on
    assert "durable" in prompt_on.lower()
    assert "never the conversation" in prompt_on.lower()

    # Flag ON but a SKILL-only pass -> no mem0 clause (it's a memory-pass feature)
    _, prompt_skill = br.spawn_background_review_thread(
        agent, [], review_memory=False, review_skills=True)
    assert "mem0_remember" not in prompt_skill


def test_tool_reminder_includes_mem0_remember_when_flag_on():
    """When background_review_mem0_write is ON, the runtime tool-restriction
    reminder appended to the review prompt must NOT say 'only memory and skill
    tools' (which contradicts the mem0 clause and suppresses the call) — it must
    explicitly allow mem0_remember. Regression for the 'feature dark / model never
    calls mem0_remember' bug (2026-06-30)."""
    import run_agent
    from hermes_cli import plugins as _plugins
    import hermes_cli.config as _cfgmod

    captured = {}

    def _capture_run_conversation(self, *, user_message=None, conversation_history=None, **kw):
        captured["user_message"] = user_message
        raise RuntimeError("stop after capturing the review prompt")

    agent = _make_agent_stub(run_agent.AIAgent)

    def _no_init(self, *args, **kwargs):
        return None

    # Force the mem0 flag ON regardless of the host config. cfg_get is imported
    # locally from hermes_cli.config inside the function, so patch it there.
    def _flag_on(cfg, *keys, default=None):
        if keys and keys[-1] == "background_review_mem0_write":
            return True
        return default

    with patch.object(run_agent.AIAgent, "__init__", _no_init), \
         patch.object(run_agent.AIAgent, "run_conversation", _capture_run_conversation), \
         patch.object(_cfgmod, "cfg_get", _flag_on), \
         patch.object(_plugins, "set_thread_tool_whitelist", lambda *a, **k: None), \
         patch.object(_plugins, "clear_thread_tool_whitelist", lambda *a, **k: None), \
         patch("threading.Thread", _SyncThread):
        try:
            agent._spawn_background_review(
                messages_snapshot=[], review_memory=True, review_skills=False,
            )
        except RuntimeError:
            pass

    msg = captured.get("user_message", "")
    assert "mem0_remember" in msg, (
        "flag-on review reminder must allow mem0_remember, got: " + repr(msg[-300:]))
    # And it must NOT carry the blanket 'only memory and skill' restriction.
    assert "only call memory and skill" not in msg.lower()


def test_tool_reminder_is_memory_skill_only_when_flag_off():
    """Flag OFF: the reminder keeps the original 'only memory and skill tools'
    restriction (mem0_remember is denied at dispatch and must not be advertised)."""
    import run_agent
    from hermes_cli import plugins as _plugins
    import hermes_cli.config as _cfgmod

    captured = {}

    def _capture_run_conversation(self, *, user_message=None, conversation_history=None, **kw):
        captured["user_message"] = user_message
        raise RuntimeError("stop after capturing the review prompt")

    agent = _make_agent_stub(run_agent.AIAgent)

    def _no_init(self, *args, **kwargs):
        return None

    def _flag_off(cfg, *keys, default=None):
        return default  # never returns True for the flag

    with patch.object(run_agent.AIAgent, "__init__", _no_init), \
         patch.object(run_agent.AIAgent, "run_conversation", _capture_run_conversation), \
         patch.object(_cfgmod, "cfg_get", _flag_off), \
         patch.object(_plugins, "set_thread_tool_whitelist", lambda *a, **k: None), \
         patch.object(_plugins, "clear_thread_tool_whitelist", lambda *a, **k: None), \
         patch("threading.Thread", _SyncThread):
        try:
            agent._spawn_background_review(
                messages_snapshot=[], review_memory=True, review_skills=False,
            )
        except RuntimeError:
            pass

    msg = captured.get("user_message", "")
    assert "only call memory and skill" in msg.lower()
    assert "mem0_remember" not in msg


def test_background_review_excludes_memory_when_disabled():
    """A memory-disabled profile must NOT get the memory tool in the review fork.

    Regression for #54937 layer 2: the whitelist hardcoded ["memory", "skills"],
    so a skill-review fork on a profile with memory_enabled=false still granted
    the LLM the MEMORY.md read/write tool, contaminating a profile that opted
    out of built-in memory. The whitelist must gate "memory" on the flag.
    """
    import run_agent
    from hermes_cli import plugins as _plugins

    captured = {}

    def _capture_whitelist(whitelist, deny_msg_fmt=None):
        captured["whitelist"] = set(whitelist)
        raise RuntimeError("stop after capturing whitelist")

    agent = _make_agent_stub(run_agent.AIAgent)
    agent._memory_enabled = False
    agent._user_profile_enabled = False

    def _no_init(self, *args, **kwargs):
        return None

    with patch.object(run_agent.AIAgent, "__init__", _no_init), \
         patch.object(_plugins, "set_thread_tool_whitelist", _capture_whitelist), \
         patch("threading.Thread", _SyncThread):
        agent._spawn_background_review(
            messages_snapshot=[],
            review_memory=False,
            review_skills=True,
        )

    whitelist = captured["whitelist"]
    # Skill tools still allowed...
    assert "skill_manage" in whitelist
    assert "skill_view" in whitelist
    # ...but the built-in memory tool must be gated out.
    assert "memory" not in whitelist


def test_background_review_includes_memory_when_user_profile_enabled():
    """user_profile_enabled alone (USER.md) still needs the memory tool."""
    import run_agent
    from hermes_cli import plugins as _plugins

    captured = {}

    def _capture_whitelist(whitelist, deny_msg_fmt=None):
        captured["whitelist"] = set(whitelist)
        raise RuntimeError("stop after capturing whitelist")

    agent = _make_agent_stub(run_agent.AIAgent)
    agent._memory_enabled = False
    agent._user_profile_enabled = True

    def _no_init(self, *args, **kwargs):
        return None

    with patch.object(run_agent.AIAgent, "__init__", _no_init), \
         patch.object(_plugins, "set_thread_tool_whitelist", _capture_whitelist), \
         patch("threading.Thread", _SyncThread):
        agent._spawn_background_review(
            messages_snapshot=[],
            review_memory=True,
            review_skills=False,
        )

    assert "memory" in captured["whitelist"]
