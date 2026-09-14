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
    # read-only file tools are allowed too (#61521): the model reaches for
    # read_file to inspect a skill before patching; denying it caused a
    # per-review denial storm that starved the self-improvement loop.
    assert "read_file" in whitelist
    assert "search_files" in whitelist
    # write/dangerous tools must NOT be in the whitelist
    assert "write_file" not in whitelist
    assert "patch" not in whitelist
    assert "terminal" not in whitelist
    assert "send_message" not in whitelist
    assert "delegate_task" not in whitelist
    assert "web_search" not in whitelist
    assert "execute_code" not in whitelist
    # The deny message must name the correct substitutes so a single denial
    # redirects the model instead of a 142-denial storm (#61521).
    deny = captured.get("deny_msg_fmt") or ""
    assert "skill_manage" in deny
    assert "skill_view" in deny


def test_read_file_registers_background_review_read_mark(tmp_path):
    """read_file inside a review fork must satisfy the read-before-write guard.

    The whitelist now allows read_file; without this mark, the model would
    read SKILL.md via read_file and still get "content has not been loaded
    in this review turn" on the follow-up skill_manage patch (#61521).
    """
    from tools.file_tools import read_file_tool
    from tools.skill_manager_guards import (
        _background_review_has_read,
        _reset_background_review_read_marks,
    )
    from tools.skill_provenance import (
        BACKGROUND_REVIEW,
        reset_current_write_origin,
        set_current_write_origin,
    )

    target = tmp_path / "SKILL.md"
    target.write_text("---\nname: t\n---\nbody\n")

    token = set_current_write_origin(BACKGROUND_REVIEW)
    try:
        _reset_background_review_read_marks()
        assert not _background_review_has_read(target)
        out = read_file_tool(str(target), task_id="bg-review-test")
        assert "body" in out
        assert _background_review_has_read(target), (
            "full read_file inside a review fork must register with the "
            "read-before-write guard"
        )

        # A partial read must NOT satisfy the guard.
        _reset_background_review_read_marks()
        read_file_tool(str(target), offset=2, task_id="bg-review-test2")
        assert not _background_review_has_read(target)
    finally:
        reset_current_write_origin(token)


def test_read_file_outside_review_does_not_mark(tmp_path):
    """Foreground reads must not populate the review-fork read set."""
    from tools.file_tools import read_file_tool
    from tools.skill_manager_guards import (
        _background_review_has_read,
        _reset_background_review_read_marks,
    )

    target = tmp_path / "SKILL.md"
    target.write_text("content\n")
    _reset_background_review_read_marks()
    read_file_tool(str(target), task_id="fg-test")
    assert not _background_review_has_read(target)


def test_background_review_whitelist_includes_configured_extra_tools(
    tmp_path, monkeypatch
):
    """A profile may opt a specific proposal tool into background review.

    The review fork inherits the parent's full tool schema for cache parity,
    but runtime dispatch remains denied unless the tool is also present in the
    thread-local whitelist.  This config hook lets profiles grant a narrowly
    scoped, human-gated proposal tool without enabling unrelated side effects.
    """
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "auxiliary:\n"
        "  background_review:\n"
        "    extra_tools:\n"
        "      - propose_shared_memory\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    import run_agent
    from hermes_cli import config as config_module
    from hermes_cli import plugins as _plugins

    config_module._LOAD_CONFIG_CACHE.clear()
    config_module._RAW_CONFIG_CACHE.clear()

    captured = {}

    def _capture_whitelist(whitelist, deny_msg_fmt=None):
        captured["whitelist"] = set(whitelist)

    def _capture_run_conversation(self, *, user_message, **kwargs):
        captured["review_prompt"] = user_message
        return {"final_response": "Nothing to save."}

    agent = _make_agent_stub(run_agent.AIAgent)

    def _no_init(self, *args, **kwargs):
        return None

    with patch.object(run_agent.AIAgent, "__init__", _no_init), \
         patch.object(
             run_agent.AIAgent,
             "run_conversation",
             _capture_run_conversation,
         ), \
         patch.object(run_agent.AIAgent, "shutdown_memory_provider", lambda self: None), \
         patch.object(run_agent.AIAgent, "close", lambda self: None), \
         patch.object(_plugins, "set_thread_tool_whitelist", _capture_whitelist), \
         patch("threading.Thread", _SyncThread):
        agent._spawn_background_review(
            messages_snapshot=[],
            review_memory=True,
            review_skills=False,
        )

    assert "propose_shared_memory" in captured["whitelist"]
    assert "terminal" not in captured["whitelist"]
    assert "propose_shared_memory" in captured["review_prompt"]


def test_background_review_extra_tools_lead_with_allowance(tmp_path, monkeypatch):
    """When extra_tools are configured, the fork's task message must LEAD with the
    allowance, not trail it after the blanket deny.

    Reason (issue found in production, 2026-09-14): the fork's user message ended
    '...Other tools will be denied at runtime — do not attempt them. Exception —
    these configured tools are also allowed: X, Y.' Small/fast models anchor on
    the deny sentence and never exercise the exception — an insight-sidecar fork
    with insight_save whitelisted, advertised in-schema, and named in the prompt
    still made zero attempts. Allowance-first framing keeps the deny guarantee
    (dispatch is still whitelisted) while putting the granted tools where the
    model's attention actually is: before the deny, not after it.
    """
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "auxiliary:\n"
        "  background_review:\n"
        "    extra_tools:\n"
        "      - insight_save\n"
        "      - insight_recall\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    import run_agent
    from hermes_cli import config as config_module

    config_module._LOAD_CONFIG_CACHE.clear()
    config_module._RAW_CONFIG_CACHE.clear()

    captured = {}

    def _capture_run_conversation(self, *, user_message, **kwargs):
        captured["review_prompt"] = user_message
        return {"final_response": "Nothing to save."}

    agent = _make_agent_stub(run_agent.AIAgent)

    with patch.object(run_agent.AIAgent, "__init__", lambda self, *a, **k: None), \
         patch.object(
             run_agent.AIAgent,
             "run_conversation",
             _capture_run_conversation,
         ), \
         patch.object(run_agent.AIAgent, "shutdown_memory_provider", lambda self: None), \
         patch.object(run_agent.AIAgent, "close", lambda self: None), \
         patch("threading.Thread", _SyncThread):
        agent._spawn_background_review(
            messages_snapshot=[],
            review_memory=False,
            review_skills=True,
        )

    msg = captured["review_prompt"]
    assert "insight_save" in msg and "insight_recall" in msg
    # Allowance must LEAD: it appears BEFORE the deny sentence, not appended after it.
    allow_pos = msg.find("You can call skill management tools plus:")
    deny_pos = msg.find("denied at runtime")
    assert allow_pos != -1, f"allowance sentence missing from: {msg!r}"
    assert deny_pos != -1, f"deny sentence missing from: {msg!r}"
    assert allow_pos < deny_pos, (
        "allowance must precede the deny sentence so models anchor on the grant, not the deny"
    )
    # No legacy trailing-exception framing anywhere in the message.
    assert "Exception — these configured tools are also allowed" not in msg


def test_background_review_no_extra_tools_keeps_deny_sentence(tmp_path, monkeypatch):
    """Without extra_tools the framing must stay exactly as before — allowance-first
    wording is reserved for the configured-exception case."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text("logging:\n  level: INFO\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    import run_agent
    from hermes_cli import config as config_module

    config_module._LOAD_CONFIG_CACHE.clear()
    config_module._RAW_CONFIG_CACHE.clear()

    captured = {}

    def _capture_run_conversation(self, *, user_message, **kwargs):
        captured["review_prompt"] = user_message
        return {"final_response": "Nothing to save."}

    agent = _make_agent_stub(run_agent.AIAgent)

    with patch.object(run_agent.AIAgent, "__init__", lambda self, *a, **k: None), \
         patch.object(
             run_agent.AIAgent,
             "run_conversation",
             _capture_run_conversation,
         ), \
         patch.object(run_agent.AIAgent, "shutdown_memory_provider", lambda self: None), \
         patch.object(run_agent.AIAgent, "close", lambda self: None), \
         patch("threading.Thread", _SyncThread):
        agent._spawn_background_review(
            messages_snapshot=[],
            review_memory=False,
            review_skills=True,
        )

    msg = captured["review_prompt"]
    assert "You can only call skill management tools." in msg
    assert "Other tools will be denied at runtime" in msg
    assert "You can call skill management tools plus:" not in msg


def test_skill_review_prompt_names_thesis_genre_only_via_whitelist(tmp_path, monkeypatch):
    """The thesis-genre block must appear exactly when insight_save is dispatchable.

    Gate on the computed review whitelist (union of toolsets + extra_tools), not on
    the raw config: memory-scoped forks get insight_save via the memory toolset with
    no extra_tools configured, and the block must never be shown to a fork whose
    dispatch would refuse the tool it names.
    """
    # Case 1: skill-only review WITH insight_save in extra_tools -> block present.
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "auxiliary:\n  background_review:\n    extra_tools:\n      - insight_save\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    import run_agent
    from hermes_cli import config as config_module

    config_module._LOAD_CONFIG_CACHE.clear()
    config_module._RAW_CONFIG_CACHE.clear()

    captured = {}

    def _capture_run_conversation(self, *, user_message, **kwargs):
        captured["review_prompt"] = user_message
        return {"final_response": "Nothing to save."}

    agent = _make_agent_stub(run_agent.AIAgent)

    with patch.object(run_agent.AIAgent, "__init__", lambda self, *a, **k: None), \
         patch.object(run_agent.AIAgent, "run_conversation", _capture_run_conversation), \
         patch.object(run_agent.AIAgent, "shutdown_memory_provider", lambda self: None), \
         patch.object(run_agent.AIAgent, "close", lambda self: None), \
         patch("threading.Thread", _SyncThread):
        agent._spawn_background_review(
            messages_snapshot=[], review_memory=False, review_skills=True,
        )

    msg = captured["review_prompt"]
    assert "THESIS" in msg
    assert "insight_save(text, topic)" in msg
    assert "teaches a future session the wrong genre" in msg
    # The block sits INSIDE the allowed region — adjacent to the grant, before nothing else matters.
    assert msg.find("insight_save(text, topic)") > msg.find("management tools plus:")

    # Case 2: skill-only review, NO extra_tools -> block absent (tool would be denied).
    hermes_home2 = tmp_path / ".hermes2"
    hermes_home2.mkdir()
    (hermes_home2 / "config.yaml").write_text("logging:\n  level: INFO\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home2))
    config_module._LOAD_CONFIG_CACHE.clear()
    config_module._RAW_CONFIG_CACHE.clear()

    captured2 = {}

    def _capture_run_conversation2(self, *, user_message, **kwargs):
        captured2["review_prompt"] = user_message
        return {"final_response": "Nothing to save."}

    agent2 = _make_agent_stub(run_agent.AIAgent)

    with patch.object(run_agent.AIAgent, "__init__", lambda self, *a, **k: None), \
         patch.object(run_agent.AIAgent, "run_conversation", _capture_run_conversation2), \
         patch.object(run_agent.AIAgent, "shutdown_memory_provider", lambda self: None), \
         patch.object(run_agent.AIAgent, "close", lambda self: None), \
         patch("threading.Thread", _SyncThread):
        agent2._spawn_background_review(
            messages_snapshot=[], review_memory=False, review_skills=True,
        )

    msg2 = captured2["review_prompt"]
    assert "insight_save(text, topic)" not in msg2
    assert "THESIS" not in msg2


def test_skill_review_prompt_carries_refusal_recovery_and_no_dead_end():
    """Prompts must teach single-refusal recovery, and must not instruct the fork to
    report into its discarded reply (the 'overlap -> mention it' dead end)."""
    import run_agent

    for prompt_attr in ("_SKILL_REVIEW_PROMPT", "_COMBINED_REVIEW_PROMPT"):
        prompt = getattr(run_agent.AIAgent, prompt_attr)
        assert "fix that specific cause, retry once" in prompt, prompt_attr
        assert "a second identical failure means stop" in prompt, prompt_attr
        # Dead-end instruction: the fork's reply is discarded (result=skill is a
        # category); nothing reads its prose. This must not tell it to communicate.
        assert "background curator handles consolidation" not in prompt, prompt_attr
        assert "mention it" not in prompt, prompt_attr


def test_background_review_extra_tools_allowance_keeps_memory_phrase(tmp_path, monkeypatch):
    """Combined review (memory+skills) with extra_tools: the allowance must still
    mention memory — routing it through the same memory_phrase as the deny-only
    branch is what prevents 'memory-less review is told memory exists' in either
    direction (a memory review must not lose its mention either)."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "auxiliary:\n"
        "  background_review:\n"
        "    extra_tools:\n"
        "      - insight_save\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    import run_agent
    from hermes_cli import config as config_module

    config_module._LOAD_CONFIG_CACHE.clear()
    config_module._RAW_CONFIG_CACHE.clear()

    captured = {}

    def _capture_run_conversation(self, *, user_message, **kwargs):
        captured["review_prompt"] = user_message
        return {"final_response": "Nothing to save."}

    agent = _make_agent_stub(run_agent.AIAgent)

    with patch.object(run_agent.AIAgent, "__init__", lambda self, *a, **k: None), \
         patch.object(
             run_agent.AIAgent,
             "run_conversation",
             _capture_run_conversation,
         ), \
         patch.object(run_agent.AIAgent, "shutdown_memory_provider", lambda self: None), \
         patch.object(run_agent.AIAgent, "close", lambda self: None), \
         patch("threading.Thread", _SyncThread):
        agent._spawn_background_review(
            messages_snapshot=[],
            review_memory=True,
            review_skills=True,
        )

    msg = captured["review_prompt"]
    allow_pos = msg.find("You can call ")
    deny_pos = msg.find("denied at runtime")
    assert allow_pos != -1 and deny_pos != -1 and allow_pos < deny_pos
    # The allowance branch names memory when the review is memory-scoped.
    assert "memory and skill " in msg[allow_pos:deny_pos]


