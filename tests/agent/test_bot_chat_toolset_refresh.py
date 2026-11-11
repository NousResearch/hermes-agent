"""#124211: a Bot Chat capability refresh must rebuild tools[] through the surface's own
builder (what a fresh desktop/TUI session gets), not just the system prompt."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from agent.conversation_loop import _restore_or_build_system_prompt

STORED_PROMPT = (
    "SYSTEM PROMPT BODY\n\nConversation started: Monday, January 05, 2026\n"
    "Model: test-model\nProvider: openrouter\nPlatform: desktop"
)


def _stale_bot_chat_agent(platform: str):
    db = MagicMock()
    db.get_session.return_value = {"system_prompt": STORED_PROMPT, "tool_names": None}
    agent = MagicMock()
    agent._cached_system_prompt = None
    agent.session_id = "test-session-id"
    agent.model = "test-model"
    agent.provider = "openrouter"
    agent.platform = platform
    agent._session_db = db
    agent._use_prompt_caching = False
    agent._build_system_prompt = MagicMock(return_value="NEW_PROMPT")
    agent.enabled_toolsets = ["web"]
    agent.disabled_toolsets = None
    agent.tools = [{"type": "function", "function": {"name": "web_search", "parameters": {}}}]
    agent.valid_tool_names = {"web_search"}
    agent._bot_mode_protocol = True
    agent._session_title_hint = "Bot Chat"
    agent._surface_switch_note = ""
    agent._gateway_turn_context_notes = ""
    return agent, db


def _run_stale_refresh(agent):
    builder = MagicMock(return_value=["terminal", "file", "desktop_ui"])
    with (
        patch("tools.bot_mode_probe.stored_prompt_capability_stale", return_value=True),
        patch("tui_gateway.server._load_enabled_toolsets", builder),
        patch("tui_gateway.server._load_disabled_toolsets", return_value=["computer_use"]),
        patch("tools.mcp_tool_agent.refresh_agent_mcp_tools", return_value=set()) as refresh,
    ):
        _restore_or_build_system_prompt(agent, None, [{"role": "user", "content": "hi"}])
    assert agent._cached_system_prompt == "NEW_PROMPT"
    return builder, refresh


def test_stale_epoch_rebuilds_tools_through_the_surface_builder():
    """The selection handed to the rebuild is the desktop builder's answer for THIS surface
    (never a `platform_toolsets.<surface>` lookup, which resolves to no tools), a disabled
    toolset is not carried forward, and the result is re-pinned for the next turn."""
    agent, db = _stale_bot_chat_agent("desktop")

    builder, refresh = _run_stale_refresh(agent)

    builder.assert_called_once_with("desktop")
    kwargs = refresh.call_args.kwargs
    assert kwargs["enabled_override"] == ["terminal", "file", "desktop_ui"]
    assert kwargs["disabled_override"] == ["computer_use"]
    assert not kwargs.get("preserve_prefix")
    assert db.update_session_tool_names.called


def test_stale_epoch_leaves_a_per_process_surface_alone():
    """A CLI resume builds a fresh agent from config every process; the desktop builder's
    fold-in (client-surface toolsets) must not be applied to it."""
    agent, _db = _stale_bot_chat_agent("cli")

    builder, refresh = _run_stale_refresh(agent)

    builder.assert_not_called()
    refresh.assert_not_called()


def test_frozen_session_policy_wins_over_live_surface_config():
    """A gateway-owned session runs under its frozen launch policy: the refresh re-pins the
    policy's toolsets and never consults the live-config surface builder."""
    from gateway.session_policy import _ACTIVE_POLICY
    agent, _db = _stale_bot_chat_agent("desktop")
    token = _ACTIVE_POLICY.set(MagicMock(toolsets=("file", "web")))
    try:
        builder, refresh = _run_stale_refresh(agent)
    finally:
        _ACTIVE_POLICY.reset(token)

    builder.assert_not_called()
    assert refresh.call_args.kwargs["enabled_override"] == ["file", "web"]
    assert refresh.call_args.kwargs.get("disabled_override") is None
