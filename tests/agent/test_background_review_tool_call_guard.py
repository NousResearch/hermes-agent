"""The background-review fork must not spawn when it could only no-op.

The review fork's whole job is to emit ``memory`` / ``skill_manage`` tool calls,
and by default it inherits the parent's live runtime. When the parent provider IS
an autonomous agent reached through a client shim that cannot carry Hermes tool
calls back, that fork is a guaranteed no-op — one that still pays for a full
agent spawn (a whole CLI process, sometimes a JVM) on every review cadence.

So: a client declaring ``SUPPORTS_HERMES_TOOL_CALLS = False`` skips the fork with
a log line pointing at the ``auxiliary.background_review`` override. A client
that can emit tool calls is unaffected, as are ordinary providers whose clients
say nothing at all.
"""

from __future__ import annotations

import logging
import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import agent.background_review as bg  # noqa: E402


def _fake_parent(client, *, runtime=None) -> SimpleNamespace:
    """The minimal parent-agent surface _run_review_in_thread touches pre-fork."""
    return SimpleNamespace(
        provider="acp-agent",
        model="acp-agent",
        client=client,
        session_id="s1",
        platform="cli",
        request_overrides={},
        max_tokens=None,
        acp_command="acp-agent",
        acp_args=["--acp"],
        enabled_toolsets=None,
        disabled_toolsets=None,
        reasoning_config=None,
        _credential_pool=None,
        _current_main_runtime=lambda: runtime or {
            "api_key": "k",
            "base_url": "acp://agent",
            "api_mode": "chat_completions",
        },
        _emit_auxiliary_failure=lambda *_a, **_k: None,
        _safe_print=lambda *_a, **_k: None,
        background_review_callback=None,
    )


def _run(agent, task_cfg=None):
    """Run the worker with AIAgent patched; return the AIAgent mock."""
    with (
        patch("hermes_cli.config.load_config", return_value={}),
        patch("run_agent.AIAgent") as mock_aiagent,
        patch("tools.terminal_tool.set_approval_callback"),
    ):
        bg._run_review_in_thread(
            agent, [{"role": "user", "content": "hi"}], "review please", task_cfg
        )
    return mock_aiagent


def test_fork_is_skipped_when_the_provider_cannot_emit_tool_calls(caplog):
    client = MagicMock()
    client.SUPPORTS_HERMES_TOOL_CALLS = False
    with caplog.at_level(logging.WARNING, logger=bg.logger.name):
        mock_aiagent = _run(_fake_parent(client))
    mock_aiagent.assert_not_called()
    # The user needs to know which knob makes the review work again.
    assert "auxiliary.background_review" in caplog.text


def test_fork_is_spawned_when_the_provider_can_emit_tool_calls():
    client = MagicMock()
    client.SUPPORTS_HERMES_TOOL_CALLS = True
    assert _run(_fake_parent(client)).called


def test_ordinary_providers_are_unaffected():
    # A plain OpenAI-style client says nothing about the capability.
    class _PlainClient:
        pass

    assert _run(_fake_parent(_PlainClient())).called


def test_the_capability_is_read_off_the_class_too():
    """Clients declare it as a class attribute; an instance need not set it."""

    class _IncapableClient:
        SUPPORTS_HERMES_TOOL_CALLS = False

    assert bg._parent_can_emit_tool_calls(_fake_parent(_IncapableClient())) is False
    assert bg._parent_can_emit_tool_calls(_fake_parent(None)) is True


def test_an_incapable_provider_still_reviews_when_the_review_is_routed_away():
    """``auxiliary.background_review.{provider,model}`` sends the fork to a normal
    model, so the parent's shim no longer matters."""

    class _IncapableClient:
        SUPPORTS_HERMES_TOOL_CALLS = False

    routed = {
        "provider": "openai",
        "model": "gpt-5",
        "api_key": "k",
        "base_url": None,
        "api_mode": "chat_completions",
        "credential_pool": None,
        "request_overrides": {},
        "max_tokens": None,
        "command": None,
        "args": [],
        "routed": True,
    }
    with patch.object(bg, "_resolve_review_runtime", return_value=routed):
        assert _run(_fake_parent(_IncapableClient())).called
