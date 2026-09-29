"""Approval context on the Discord approval card as actually sent.

Drives the real chain -- ``check_all_command_guards`` in a gateway session, the
registered ``TurnRunner._approval_notify_sync`` and the real
``DiscordAdapter.send_exec_approval`` -- into a fake channel, so the card is checked
under the adapter's 2000-char content cap and reason budget rather than at an uncapped
test double.
"""

import asyncio
import re
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

import tools.approval as approval_module
from gateway.config import PlatformConfig
from gateway.run_turn_runner import TurnRunner
from plugins.platforms.discord.adapter import DiscordAdapter
from tools import approval_context
from tools.approval import check_all_command_guards
from tools.approval_context import reset_current_session_key, set_current_session_key

_HEAD = "—— Model-provided context (unverified) ——"
_TAIL = "—— End unverified context ——"
_SESSION = "discord:card-render"
_COMMAND = "rm -rf /tmp/example"
_LONG_CONTEXT = {"purpose": "p" * 900, "effect": "e" * 900, "risk": "r" * 900}
# Three findings put the scanner text past Discord's 300-char reason budget.
_LONG_FINDINGS = [{"rule_id": "long_report", "severity": "HIGH", "title": f"finding {i}",
                   "description": "d" * 120} for i in range(3)]


@pytest.fixture(autouse=True)
def _gateway_session(monkeypatch):
    for key in ("HERMES_INTERACTIVE", "HERMES_EXEC_ASK", "HERMES_YOLO_MODE", "DISCORD_APPROVAL_MENTIONS"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
    monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "manual")
    token = set_current_session_key(_SESSION)
    yield
    reset_current_session_key(token)
    approval_module._gateway_queues.clear()
    approval_module._gateway_notify_cbs.clear()
    approval_module._session_approved.clear()


def _card(approval_context=None, *, command=_COMMAND, findings=(), mentions=False):
    """Run the guard once and return the Discord card's content (the request is denied)."""
    adapter = DiscordAdapter(PlatformConfig(
        enabled=True, token="***", extra={"approval_mentions": True} if mentions else {}))
    adapter._allowed_user_ids = {"111111111111111111", "222222222222222222"}
    sent = {}

    async def channel_send(**kwargs):
        sent.update(kwargs)
        entry = approval_module._gateway_queues[_SESSION][0]
        entry.result = "deny"
        entry.event.set()
        return SimpleNamespace(id=1234)

    channel = SimpleNamespace(send=AsyncMock(side_effect=channel_send))
    adapter._client = SimpleNamespace(get_channel=lambda _chat_id: channel, fetch_channel=AsyncMock())

    def schedule(coro, _label):
        loop = asyncio.new_event_loop()
        try:
            value = loop.run_until_complete(coro)
        finally:
            loop.close()
        return SimpleNamespace(result=lambda timeout=None: value)

    runner = object.__new__(TurnRunner)
    runner._ctx = SimpleNamespace(_status_adapter=adapter, _status_chat_id="555",
                                  _status_thread_metadata=None, session_key=_SESSION)
    runner._close_native_stream_boundary = lambda reason: None
    runner._schedule = schedule
    tirith = {"action": "warn" if findings else "allow", "findings": list(findings), "summary": ""}
    approval_module.register_gateway_notify(_SESSION, runner._approval_notify_sync)
    try:
        with patch("tools.tirith_security.check_command_security", return_value=tirith):
            result = check_all_command_guards(command, "local", approval_context=approval_context)
    finally:
        approval_module.unregister_gateway_notify(_SESSION)
    assert result["approved"] is False
    assert sent["view"] is not None  # still an actionable card
    content = sent["content"]
    assert len(content) <= adapter.MAX_MESSAGE_LENGTH
    return content


def _warnings(command=_COMMAND, findings=()):
    warnings = [approval_module.detect_dangerous_command(command)[2]]
    if findings:
        tirith = {"action": "warn", "findings": list(findings), "summary": ""}
        warnings.insert(0, approval_module._format_tirith_description(tirith))
    return warnings


def _command_preview(content):
    return re.search(r"```bash\n(.*?)\n```", content, re.DOTALL).group(1)


def _context_block(content):
    """The whole annotation, or None; a started annotation must be closed."""
    if _HEAD not in content:
        assert _TAIL not in content
        return None
    start = content.index(_HEAD)
    assert _TAIL in content[start:], "annotation lost its closing delimiter"
    return content[start:content.index(_TAIL, start) + len(_TAIL)]


@pytest.mark.parametrize("approval_context", [{}, {"purpose": "   ", "effect": 123}],
                         ids=["empty", "blank-and-non-string"])
def test_card_without_usable_context_matches_the_plain_card(approval_context):
    plain = _card()
    assert _card(approval_context) == plain
    assert _context_block(plain) is None
    assert _COMMAND in plain
    assert all(w in plain for w in _warnings())


def test_short_context_renders_whole_on_the_card():
    content = _card({"purpose": "clean a temp path", "effect": "removes temporary files",
                     "risk": "deleted files cannot be recovered"})
    block = _context_block(content)
    assert "Purpose: clean a temp path" in block
    assert "Effect: removes temporary files" in block
    assert "Risk: deleted files cannot be recovered" in block
    assert approval_module._ENHANCED_DESC_TRUNC.strip() not in block
    assert _command_preview(content) == _COMMAND
    assert all(w in content.split(_HEAD)[0] for w in _warnings())


def test_long_context_is_shortened_to_the_card_with_its_closing_delimiter():
    content = _card(_LONG_CONTEXT)
    block = _context_block(content)
    assert block is not None and "Purpose: ppp" in block
    assert approval_module._ENHANCED_DESC_TRUNC.strip() in block
    assert _command_preview(content) == _COMMAND
    assert all(w in content.split(_HEAD)[0] for w in _warnings())


def test_context_adds_nothing_when_the_scanner_text_already_fills_the_reason():
    # The adapter's reason budget cuts this scanner text with or without context (the
    # same card base renders); context must not be squeezed in behind it.
    plain = _card(findings=_LONG_FINDINGS)
    assert len("; ".join(_warnings(findings=_LONG_FINDINGS))) > DiscordAdapter._EA_REASON_BUDGET
    assert _card(_LONG_CONTEXT, findings=_LONG_FINDINGS) == plain
    assert _context_block(plain) is None


@pytest.mark.parametrize("mentions", [False, True], ids=["no-mentions", "owner-mentions"])
# 1900 overflows the command budget even without context (and stays under the
# separator-free parser limit, which would block before any prompt).
@pytest.mark.parametrize("tail", [200, 1450, 1650, 1900])
def test_context_never_shortens_the_command_preview(tail, mentions):
    command = "rm -rf /tmp/" + "a" * tail
    plain = _card(command=command, mentions=mentions)
    content = _card(_LONG_CONTEXT, command=command, mentions=mentions)
    assert _command_preview(content) == _command_preview(plain)
    _context_block(content)
    assert all(w in content for w in _warnings(command))
