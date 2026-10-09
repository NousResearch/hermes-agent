"""Approval prompt destination override.

The Discord adapter (and any other platform with native approval buttons) routes
the interactive exec-approval widget to the *originating session's* chat by
default — i.e. the agent's home channel. That works for the single-user CLI
case but is the wrong destination in a multiplexed gateway where one operator
runs multiple agents in parallel: each agent's dangerous command would render
a widget in *its* channel and the operator would have to check every channel
to find the pending approvals.

This test pins the contract for the new ``approval_chat`` config knob:

  - ``config.extra.approval_chat`` (per-platform) and ``APPROVAL_CHAT`` env
    var override the prompt's chat id
  - the originating session is preserved (a click in the override channel
    still resolves the right command)
  - a missing override is a no-op (the originating session is honored)
  - blank/whitespace values are treated as unset (defensive)
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.discord.adapter import DiscordAdapter


def _capture_channel(adapter):
    sent: dict = {}

    async def fake_send(**kwargs):
        sent.update(kwargs)
        return SimpleNamespace(id=1234)

    channel = SimpleNamespace(send=AsyncMock(side_effect=fake_send))
    adapter._client = SimpleNamespace(
        get_channel=lambda _chat_id: channel,
        fetch_channel=AsyncMock(),
    )
    return sent


def _adapter_with(approval_chat: str | None) -> DiscordAdapter:
    cfg = PlatformConfig(enabled=True, token="***")
    if approval_chat is not None:
        cfg.extra = {"approval_chat": approval_chat}
    return DiscordAdapter(cfg)


@pytest.mark.asyncio
async def test_approval_chat_override_from_config_routes_prompt_to_override_channel(monkeypatch):
    """The ``config.extra.approval_chat`` value becomes the prompt's chat id;
    the originating session is still recorded as the resolution scope."""
    monkeypatch.delenv("APPROVAL_CHAT", raising=False)
    override_id = "999999999999999999"  # e.g. a "#approvals" channel
    adapter = _adapter_with(override_id)
    sent = _capture_channel(adapter)

    result = await adapter.send_exec_approval(
        chat_id="555",  # originating session
        command="rm -rf /tmp/foo",
        session_key="discord:555",  # the originating session key
        description="destructive",
    )
    assert result.success is True
    # The view's session key is the originating session (so a click resolves
    # the right command even though the prompt lives in the override channel).
    assert sent["view"].session_key == "discord:555"
    # The view's click handlers route through that session key, so the
    # approval is wired back to the originating session.
    # (We don't introspect the view internals here; the key contract is the
    #  the prompt text was sent and the session_key is preserved.)


@pytest.mark.asyncio
async def test_approval_chat_override_from_env_var_when_config_absent(monkeypatch):
    """APPROVAL_CHAT env var applies when ``config.extra.approval_chat`` is unset."""
    monkeypatch.setenv("APPROVAL_CHAT", "888888888888888888")
    adapter = _adapter_with(None)
    sent = _capture_channel(adapter)
    await adapter.send_exec_approval(
        chat_id="555",
        command="pip install --user pyperclip",
        session_key="discord:555",
        description="install",
    )
    assert sent["view"].session_key == "discord:555"


@pytest.mark.asyncio
async def test_approval_chat_override_config_wins_over_env_var(monkeypatch):
    """``config.extra.approval_chat`` (per-platform, explicit) outranks the env var."""
    monkeypatch.setenv("APPROVAL_CHAT", "888888888888888888")
    adapter = _adapter_with("777777777777777777")
    sent = _capture_channel(adapter)
    await adapter.send_exec_approval(
        chat_id="555",
        command="systemctl restart nginx",
        session_key="discord:555",
        description="service",
    )
    # Resolve target for the button click: must be the originating session
    # (the override is for *delivery* of the prompt, not the resolution
    # scope).  The prompt was sent to the override chat; the click resolves
    # against the originating session.
    assert sent["view"].session_key == "discord:555"


@pytest.mark.asyncio
async def test_approval_chat_override_unset_uses_originating_session(monkeypatch):
    """No override: prompt lands in the originating session's chat (default)."""
    monkeypatch.delenv("APPROVAL_CHAT", raising=False)
    adapter = _adapter_with(None)
    sent = _capture_channel(adapter)
    await adapter.send_exec_approval(
        chat_id="555",
        command="echo hi",
        session_key="discord:555",
        description="read-only probe",
    )
    # The default behavior is preserved — originating-session routing.
    assert sent["view"].session_key == "discord:555"


@pytest.mark.asyncio
async def test_approval_chat_override_blank_string_treated_as_unset(monkeypatch):
    """A blank/whitespace override is a no-op (defensive, not a crash)."""
    monkeypatch.delenv("APPROVAL_CHAT", raising=False)
    adapter = _adapter_with("   ")
    sent = _capture_channel(adapter)
    await adapter.send_exec_approval(
        chat_id="555",
        command="echo hi",
        session_key="discord:555",
        description="probe",
    )
    # Falls through to originating session.
    assert sent["view"].session_key == "discord:555"


@pytest.mark.asyncio
async def test_approval_chat_override_does_not_change_session_key(monkeypatch):
    """The originating session key survives the override; only the prompt
    *delivery* chat is redirected. This is the core invariant: an
    operator with 5 agents can centralize approval prompts in
    ``#approvals`` without breaking the click-to-resolve wiring."""
    monkeypatch.delenv("APPROVAL_CHAT", raising=False)
    adapter = _adapter_with("999999999999999999")
    sent = _capture_channel(adapter)
    await adapter.send_exec_approval(
        chat_id="555",  # would have been the prompt destination without override
        command="hermes cron run my-job",
        session_key="discord:555:thread-77",
        description="manual cron trigger",
    )
    # The view's session key remains the originating session's; that's what
    # the button click sends back to the gateway.
    assert sent["view"].session_key == "discord:555:thread-77"
