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


def _capture_init_kwargs(captured):
    def _init(self, *args, **kwargs):
        captured["enabled_toolsets"] = kwargs.get("enabled_toolsets", "UNSET")
        captured["disabled_toolsets"] = kwargs.get("disabled_toolsets", "UNSET")
        captured["base_url"] = kwargs.get("base_url", "UNSET")
        raise RuntimeError("stop after capturing init args")

    return _init


def test_background_review_narrows_toolset_for_local_endpoint():
    """Local endpoints have no prefix cache to preserve, so the fork advertises
    only memory/skills instead of the parent's full schema — a weak local model
    otherwise imitates the snapshot history and thrashes against the dispatch
    deny-wall.
    """
    import run_agent

    agent = _make_agent_stub(run_agent.AIAgent)
    agent._current_main_runtime = lambda: {
        "model": "Qwen3-Coder-Next-4bit",
        "provider": "custom",
        "base_url": "http://127.0.0.1:8149/v1",
        "api_key": "k",
        "api_mode": "",
    }
    captured = {}

    with patch.object(run_agent.AIAgent, "__init__", _capture_init_kwargs(captured)), \
         patch("threading.Thread", _SyncThread):
        agent._spawn_background_review(
            messages_snapshot=[],
            review_memory=True,
            review_skills=True,
        )

    assert captured.get("enabled_toolsets") == ["memory", "skills"], captured
    assert captured.get("disabled_toolsets") is None, captured


def test_background_review_keeps_parent_toolset_for_remote_endpoint():
    """Cache-backed (remote) endpoints still mirror the parent's toolsets so
    the ``tools[]`` payload stays byte-identical for the prefix cache.
    """
    import run_agent

    agent = _make_agent_stub(run_agent.AIAgent)
    agent._current_main_runtime = lambda: {
        "model": "claude-sonnet-4-6",
        "provider": "anthropic",
        "base_url": "https://api.anthropic.com/v1",
        "api_key": "k",
        "api_mode": "",
    }
    captured = {}

    with patch.object(run_agent.AIAgent, "__init__", _capture_init_kwargs(captured)), \
         patch("threading.Thread", _SyncThread):
        agent._spawn_background_review(
            messages_snapshot=[],
            review_memory=True,
            review_skills=True,
        )

    assert captured.get("enabled_toolsets") == agent.enabled_toolsets, captured
    assert captured.get("disabled_toolsets") == agent.disabled_toolsets, captured


def test_background_review_local_narrowing_respects_memory_gate():
    """The narrowed schema must honor the memory gate: a memory-disabled
    profile gets only the skills toolset advertised, mirroring the runtime
    whitelist's conditional memory grant (no schema-level re-grant).
    """
    import run_agent

    agent = _make_agent_stub(run_agent.AIAgent)
    agent._memory_enabled = False
    agent._user_profile_enabled = False
    agent._current_main_runtime = lambda: {
        "model": "Qwen3-Coder-Next-4bit",
        "provider": "custom",
        "base_url": "http://127.0.0.1:8149/v1",
        "api_key": "k",
        "api_mode": "",
    }
    captured = {}

    with patch.object(run_agent.AIAgent, "__init__", _capture_init_kwargs(captured)), \
         patch("threading.Thread", _SyncThread):
        agent._spawn_background_review(
            messages_snapshot=[],
            review_memory=False,
            review_skills=True,
        )

    assert captured.get("enabled_toolsets") == ["skills"], captured
    assert captured.get("disabled_toolsets") is None, captured


def test_background_review_routed_endpoint_classified_by_review_runtime():
    """An auxiliary.background_review route must be classified by the RESOLVED
    review runtime's endpoint, not the parent's: remote parent + local aux
    route still narrows, and the fork is constructed on the routed base_url.
    """
    import run_agent

    agent = _make_agent_stub(run_agent.AIAgent)
    agent.provider = "anthropic"
    agent.model = "claude-sonnet-4-6"
    agent._current_main_runtime = lambda: {
        "model": "claude-sonnet-4-6",
        "provider": "anthropic",
        "base_url": "https://api.anthropic.com/v1",
        "api_key": "k",
        "api_mode": "",
    }
    captured = {}
    routed_url = "http://127.0.0.1:8150/v1"

    def _fake_load_config():
        return {
            "auxiliary": {
                "background_review": {"provider": "custom", "model": "qwen-local"}
            }
        }

    def _fake_resolve_runtime_provider(**kwargs):
        return {
            "provider": "custom",
            "model": "qwen-local",
            "api_key": "k2",
            "base_url": routed_url,
            "api_mode": "",
            "credential_pool": None,
            "request_overrides": {},
            "max_output_tokens": None,
            "command": None,
            "args": [],
        }

    with patch.object(run_agent.AIAgent, "__init__", _capture_init_kwargs(captured)), \
         patch("hermes_cli.config.load_config", _fake_load_config), \
         patch(
             "hermes_cli.runtime_provider.resolve_runtime_provider",
             _fake_resolve_runtime_provider,
         ), \
         patch("threading.Thread", _SyncThread):
        agent._spawn_background_review(
            messages_snapshot=[],
            review_memory=True,
            review_skills=True,
        )

    assert captured.get("base_url") == routed_url, captured
    assert captured.get("enabled_toolsets") == ["memory", "skills"], captured
    assert captured.get("disabled_toolsets") is None, captured
