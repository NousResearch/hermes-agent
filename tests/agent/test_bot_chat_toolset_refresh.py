"""RED tests for #124211: toolset changes must reach the canonical Bot Chat.

1. The capability epoch must flip when ``platform_toolsets`` (the key
   ``hermes tools enable/disable`` actually writes) changes. The epoch used to
   watch only ``tools.enabled_toolsets``, a key no surface writes, so the
   staleness detector was blind to real toolset edits.
2. The Bot Chat capability-refresh branch must rebuild ``agent.tools`` (and
   re-pin the session row), not just the system prompt. It used to return
   early after rebuilding the prompt, leaving tools[] a fossil of session
   creation forever.
"""

from __future__ import annotations

import textwrap
from unittest.mock import MagicMock, patch

from tools import bot_mode_probe


def _make_bot_profile(root, name, *, managed=True):
    d = root / "profiles" / name
    d.mkdir(parents=True, exist_ok=True)
    if managed:
        (d / "profile.yaml").write_text(
            textwrap.dedent(
                """\
                ui_meta:
                  hermes-bots:
                    shape: cloud
                    color: '#8b5cf6'
                """
            ),
            encoding="utf-8",
        )
    return d


def test_capability_epoch_follows_platform_toolset_edits(tmp_path):
    """``hermes tools enable computer_use`` writes platform_toolsets.<platform>;
    the Bot Chat epoch must flip so the next turn rebuilds."""
    bot_mode_probe._reset_cache_for_tests()
    try:
        home = tmp_path / ".hermes"
        home.mkdir()
        _make_bot_profile(home, "researcher", managed=True)
        (home / "config.yaml").write_text(
            "platform_toolsets:\n  desktop: [hermes-desktop]\n", encoding="utf-8"
        )
        base = bot_mode_probe.capability_fingerprint(home)
        (home / "config.yaml").write_text(
            "platform_toolsets:\n  desktop: [hermes-desktop, computer_use]\n",
            encoding="utf-8",
        )
        assert bot_mode_probe.capability_fingerprint(home) != base
    finally:
        bot_mode_probe._reset_cache_for_tests()


def test_capability_epoch_follows_global_toolset_suppression(tmp_path):
    """``agent.disabled_toolsets`` suppresses tools in every session; flipping it
    must flip the Bot Chat epoch too."""
    bot_mode_probe._reset_cache_for_tests()
    try:
        home = tmp_path / ".hermes"
        home.mkdir()
        _make_bot_profile(home, "researcher", managed=True)
        (home / "config.yaml").write_text("agent:\n  disabled_toolsets: []\n", encoding="utf-8")
        base = bot_mode_probe.capability_fingerprint(home)
        (home / "config.yaml").write_text(
            "agent:\n  disabled_toolsets: [computer_use]\n", encoding="utf-8"
        )
        assert bot_mode_probe.capability_fingerprint(home) != base
    finally:
        bot_mode_probe._reset_cache_for_tests()


def _stored_prompt() -> str:
    return (
        "SYSTEM PROMPT BODY\n\nConversation started: Monday, January 05, 2026\n"
        "Model: test-model\nProvider: openrouter\nPlatform: desktop"
    )


def _make_agent(db):
    from agent.conversation_loop import _restore_or_build_system_prompt  # noqa: F401

    agent = MagicMock()
    agent._cached_system_prompt = None
    agent.session_id = "test-session-id"
    agent.model = "test-model"
    agent.provider = "openrouter"
    agent.platform = "desktop"
    agent._session_db = db
    agent._use_prompt_caching = False
    agent._build_system_prompt = MagicMock(return_value="NEW_PROMPT")
    agent.enabled_toolsets = ["web"]
    agent.disabled_toolsets = None
    agent.tools = [{"type": "function", "function": {"name": "web_search", "parameters": {}}}]
    agent.valid_tool_names = {"web_search"}
    agent._bot_mode_protocol = True
    agent._session_title_hint = "Bot Chat"
    agent._platform_hint_overrides = None
    agent._surface_switch_note = ""
    agent._gateway_turn_context_notes = ""
    return agent


def test_bot_chat_capability_refresh_rebuilds_tools():
    """When the Bot Chat epoch is stale, the refresh must rebuild tools[] (not
    just the prompt) and re-pin the session row."""
    from agent.conversation_loop import _restore_or_build_system_prompt

    db = MagicMock()
    db.get_session.return_value = {"system_prompt": _stored_prompt(), "tool_names": None}
    agent = _make_agent(db)
    history = [{"role": "user", "content": "hi"}]
    with (
        patch("tools.bot_mode_probe.stored_prompt_capability_stale", return_value=True),
        patch("tools.mcp_tool_agent.refresh_agent_mcp_tools", return_value={"new_tool"}) as refresh,
    ):
        _restore_or_build_system_prompt(agent, None, history)
    assert agent._cached_system_prompt == "NEW_PROMPT"
    refresh.assert_called_once()
    # The rebuilt tool snapshot must be re-pinned so the next turn restores it.
    assert db.update_session_tool_names.called
