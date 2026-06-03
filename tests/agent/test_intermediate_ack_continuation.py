"""Tests for the intermediate-ack continuation guard (issue #38192).

The guard previously only ran when ``agent.api_mode == "codex_responses"``.
For tool-capable ``chat_completions`` deployments the same model behavior
(replying "Let me diagnose this..." then ending the turn with zero tool
calls) was not caught.  These tests cover the broadened action-marker set
and confirm the helper recognizes the diagnostic-acknowledgement pattern
regardless of ``api_mode``.
"""

from __future__ import annotations

from types import SimpleNamespace

from agent.agent_runtime_helpers import looks_like_codex_intermediate_ack


def _fake_agent() -> SimpleNamespace:
    return SimpleNamespace(_strip_think_blocks=lambda s: s)


def test_let_me_diagnose_search_online_triggers_continuation() -> None:
    agent = _fake_agent()
    user = "Diagnose why my repo telegram path fails and search the codebase."
    assistant = "Let me diagnose this..."
    assert looks_like_codex_intermediate_ack(agent, user, assistant, []) is True


def test_let_me_check_repo_triggers_continuation() -> None:
    agent = _fake_agent()
    user = "Please check the repository for the failing path."
    assistant = "Let me check the project files..."
    assert looks_like_codex_intermediate_ack(agent, user, assistant, []) is True


def test_new_action_markers_recognized() -> None:
    agent = _fake_agent()
    user = "Investigate the failure in the codebase."
    for ack in (
        "I'll investigate the codebase shortly.",
        "Let me troubleshoot the project now.",
        "I will research the repository.",
        "Let me verify the codebase state.",
        "I'll benchmark the project hot path.",
    ):
        assert looks_like_codex_intermediate_ack(agent, user, ack, []) is True, ack


def test_no_future_ack_returns_false() -> None:
    agent = _fake_agent()
    assert (
        looks_like_codex_intermediate_ack(
            agent,
            "investigate the codebase",
            "Here is the final answer.",
            [],
        )
        is False
    )


def test_existing_tool_message_disables_guard() -> None:
    agent = _fake_agent()
    msgs = [{"role": "tool", "name": "x", "content": "ok"}]
    assert (
        looks_like_codex_intermediate_ack(
            agent,
            "diagnose the codebase",
            "Let me diagnose this...",
            msgs,
        )
        is False
    )
