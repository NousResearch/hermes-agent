"""Approval context on the Discord and WhatsApp Cloud approval cards as actually sent.

Drives the real chain -- ``check_all_command_guards`` in a gateway session, the
registered ``TurnRunner._approval_notify_sync`` and the real adapter's
``send_exec_approval`` -- into a fake channel or fake Graph client, so the card is
checked under the adapter's own limits (Discord: 2000-char content cap and reason
budget; WhatsApp Cloud: 1024-char interactive body) rather than at an uncapped test
double.
"""

import asyncio
import re
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

import tools.approval as approval_module
from gateway.config import PlatformConfig
from gateway.platforms.whatsapp_cloud import WhatsAppCloudAdapter
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


def _deny_pending():
    """The user taps Deny on the card just sent."""
    for entry in approval_module._gateway_queues.get(_SESSION, [])[:1]:
        entry.result = "deny"
        entry.event.set()


def _discord_adapter(mentions):
    """Real DiscordAdapter on a fake channel: (adapter, sends, card(send) -> its checked content)."""
    adapter = DiscordAdapter(PlatformConfig(
        enabled=True, token="***", extra={"approval_mentions": True} if mentions else {}))
    adapter._allowed_user_ids = {"111111111111111111", "222222222222222222"}
    sent = []

    async def channel_send(**kwargs):
        sent.append(kwargs)
        _deny_pending()
        return SimpleNamespace(id=1234)

    channel = SimpleNamespace(send=AsyncMock(side_effect=channel_send))
    adapter._client = SimpleNamespace(get_channel=lambda _chat_id: channel, fetch_channel=AsyncMock())

    def card(kwargs):
        assert kwargs["view"] is not None  # still an actionable card
        assert len(kwargs["content"]) <= adapter.MAX_MESSAGE_LENGTH
        return kwargs["content"]

    return adapter, sent, card


def _whatsapp_adapter(_mentions):
    """Real WhatsAppCloudAdapter on a fake Graph client (no network): same shape as above."""
    adapter = WhatsAppCloudAdapter(PlatformConfig(
        enabled=True, extra={"phone_number_id": "1234567890", "access_token": "***"}))
    sent = []

    async def post(_url, **kwargs):
        sent.append(kwargs["json"])
        _deny_pending()
        return SimpleNamespace(status_code=200, json=lambda: {"messages": [{"id": "wamid.card"}]})

    adapter._http_client = SimpleNamespace(post=post)

    def card(payload):
        interactive = payload["interactive"]
        assert [b["reply"]["id"].rsplit(":", 1)[1] for b in interactive["action"]["buttons"]] == ["approve", "deny"]
        assert len(interactive["body"]["text"]) <= 1024  # interactive.body.text cap
        return interactive["body"]["text"]

    return adapter, sent, card


_ADAPTERS = {"discord": _discord_adapter, "whatsapp": _whatsapp_adapter}


def _run(approval_context=None, *, command=_COMMAND, findings=(), mentions=False, platform="discord",
         before_notify=None):
    """Run the guard once: (result, contents of the approval cards sent)."""
    adapter, sent, card = _ADAPTERS[platform](mentions)

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
    notify = runner._approval_notify_sync
    if before_notify is not None:
        def notify(data, _notify=runner._approval_notify_sync):
            before_notify()
            return _notify(data)
    tirith = {"action": "warn" if findings else "allow", "findings": list(findings), "summary": ""}
    approval_module.register_gateway_notify(_SESSION, notify)
    try:
        with patch("tools.tirith_security.check_command_security", return_value=tirith):
            result = check_all_command_guards(command, "local", approval_context=approval_context)
    finally:
        approval_module.unregister_gateway_notify(_SESSION)
    return result, [card(s) for s in sent]


def _card(approval_context=None, **kwargs):
    """The one approval card the guard sent (the request is denied from it)."""
    result, cards = _run(approval_context, **kwargs)
    assert result["approved"] is False
    assert len(cards) == 1
    return cards[0]


def _warnings(command=_COMMAND, findings=()):
    warnings = [approval_module.detect_dangerous_command(command)[2]]
    if findings:
        tirith = {"action": "warn", "findings": list(findings), "summary": ""}
        warnings.insert(0, approval_module._format_tirith_description(tirith))
    return warnings


def _command_preview(content):
    return re.search(r"```(?:bash)?\n(.*?)\n```", content, re.DOTALL).group(1)


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


# ── WhatsApp Cloud: the adapter cuts the finished card text to its 1024-char body, which
# would take the closing delimiter and the "will NOT run" deadline line first.

def _whatsapp_deadline():
    return _whatsapp_adapter(False)[0]._ea_deadline_line().strip()


@pytest.mark.parametrize("approval_context", [{}, {"purpose": "   ", "effect": 123}],
                         ids=["empty", "blank-and-non-string"])
def test_whatsapp_card_without_usable_context_matches_the_plain_card(approval_context):
    plain = _card(platform="whatsapp")
    assert _card(approval_context, platform="whatsapp") == plain
    assert _context_block(plain) is None
    assert _command_preview(plain) == _COMMAND
    assert plain.endswith(_whatsapp_deadline())


def test_whatsapp_short_context_renders_whole_with_the_deadline():
    content = _card({"purpose": "clean a temp path", "effect": "removes temporary files",
                     "risk": "deleted files cannot be recovered"}, platform="whatsapp")
    block = _context_block(content)
    assert "Purpose: clean a temp path" in block
    assert "Risk: deleted files cannot be recovered" in block
    assert approval_module._ENHANCED_DESC_TRUNC.strip() not in block
    assert _command_preview(content) == _COMMAND
    assert content.endswith(_whatsapp_deadline())


def test_whatsapp_long_context_keeps_its_closing_delimiter_and_the_deadline():
    content = _card(_LONG_CONTEXT, platform="whatsapp")
    block = _context_block(content)
    assert block is not None and "Purpose: ppp" in block
    assert approval_module._ENHANCED_DESC_TRUNC.strip() in block
    assert _command_preview(content) == _COMMAND
    assert all(w in content.split(_HEAD)[0] for w in _warnings())
    assert content.endswith(_whatsapp_deadline())


# 1000 is cut by the adapter's 800-char command budget even without context.
@pytest.mark.parametrize("tail", [200, 600, 760, 1000])
def test_whatsapp_context_never_costs_the_command_preview_or_the_deadline(tail):
    command = "rm -rf /tmp/" + "a" * tail
    plain = _card(command=command, platform="whatsapp")
    content = _card(_LONG_CONTEXT, command=command, platform="whatsapp")
    assert _command_preview(content) == _command_preview(plain)
    assert content.endswith(_whatsapp_deadline())
    _context_block(content)
    assert all(w in content for w in _warnings(command))


def test_whatsapp_context_adds_nothing_when_the_plain_card_is_already_cut():
    # Scanner text plus command already overflow the body without context, so the adapter
    # cuts the plain card (inherited); context must not be squeezed in ahead of that cut.
    findings = [dict(f, description="d" * 300) for f in _LONG_FINDINGS]
    command = "rm -rf /tmp/" + "a" * 600
    plain = _card(command=command, findings=findings, platform="whatsapp")
    assert not plain.endswith(_whatsapp_deadline())
    assert _card(_LONG_CONTEXT, command=command, findings=findings, platform="whatsapp") == plain
    assert _context_block(plain) is None


# ── A request answered or withdrawn between queueing and its prompt is no longer pending, so
# its context cannot be fitted against the queued scanner text: no card is sent, and the gate
# returns the outcome already recorded rather than asking again.

def _withdraw():
    entry = approval_module._gateway_queues[_SESSION][0]
    approval_module.withdraw_gateway_approval(_SESSION, entry.data["request_id"], "turn ended")


@pytest.mark.parametrize("platform", ["discord", "whatsapp"])
@pytest.mark.parametrize("settle", [lambda: approval_module.resolve_gateway_approval(_SESSION, "deny"), _withdraw],
                         ids=["denied-elsewhere", "withdrawn"])
def test_no_card_for_a_request_settled_before_its_prompt(platform, settle):
    result, cards = _run(_LONG_CONTEXT, platform=platform, before_notify=settle)
    assert cards == []
    assert result["approved"] is False
