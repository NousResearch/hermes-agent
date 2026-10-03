"""Approval context on approval prompts as actually sent.

Drives the real chain -- ``check_all_command_guards`` in a gateway session, the
registered ``TurnRunner._approval_notify_sync`` and the real adapter's
``send_exec_approval`` (or its text fallback) -- into a fake channel, bot, Graph client
or relay connector, so the prompt is checked under the adapter's own limits (Discord:
2000-char content cap and reason budget; WhatsApp Cloud: 1024-char interactive body;
Telegram: 4096 UTF-16 units; Slack: 3000-char section) rather than at an uncapped test
double. Relay prompts, whose limits the gateway cannot establish, must be the plain prompt.
"""

import asyncio
import html
import json
import re
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

import tools.approval as approval_module
from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult, utf16_len
from gateway.platforms.base_exec_approval import approval_timeout_seconds, format_approval_deadline_line
from gateway.platforms.whatsapp_cloud import WhatsAppCloudAdapter
from gateway.relay.adapter import RelayAdapter
from gateway.relay.descriptor import CONTRACT_VERSION, CapabilityDescriptor
from gateway.run_turn_runner import TurnRunner
from plugins.platforms.discord.adapter import DiscordAdapter
from plugins.platforms.matrix.adapter import MatrixAdapter
from plugins.platforms.slack.adapter import SlackAdapter
from plugins.platforms.telegram.adapter import TelegramAdapter
from tests.gateway.relay.stub_connector import StubConnector
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


def _telegram_adapter(_mentions):
    """Real TelegramAdapter on a fake bot; the card is HTML, capped at 4096 UTF-16 units."""
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    sent = []

    async def send_message(**kwargs):
        sent.append(kwargs)
        _deny_pending()
        return SimpleNamespace(message_id=42)

    adapter._bot = SimpleNamespace(send_message=send_message)

    def card(kwargs):
        assert kwargs["reply_markup"] is not None
        assert utf16_len(kwargs["text"]) <= adapter.MAX_MESSAGE_LENGTH
        return html.unescape(kwargs["text"])

    return adapter, sent, card


def _slack_adapter(_mentions):
    """Real SlackAdapter on a fake workspace client; the card is one 3000-char section."""
    adapter = SlackAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._app = SimpleNamespace()
    sent = []

    async def chat_post_message(**kwargs):
        sent.append(kwargs)
        _deny_pending()
        return {"ts": "1234.5678"}

    adapter._team_clients = {"T1": SimpleNamespace(chat_postMessage=chat_post_message)}
    adapter._channel_team = {"555": "T1"}

    def card(kwargs):
        section, actions = kwargs["blocks"]
        assert actions["type"] == "actions" and actions["elements"]
        assert len(section["text"]["text"]) <= SlackAdapter._EA_SECTION_CAP
        return section["text"]["text"]

    return adapter, sent, card


def _matrix_adapter(_mentions):
    """Real MatrixAdapter with a recorded send: its reaction card declares no text budget."""
    adapter = MatrixAdapter(PlatformConfig(
        enabled=True, token="***", extra={"homeserver": "https://matrix.example.org"}))
    adapter._client = SimpleNamespace()
    adapter._send_reaction = AsyncMock(return_value="$reaction")
    sent = []

    async def send(_chat_id, content, reply_to=None, metadata=None):
        sent.append(content)
        _deny_pending()
        return SendResult(success=True, message_id="$card")

    adapter.send = send
    return adapter, sent, lambda content: content


# Negotiated per-platform caps and length units; the relay primary (Slack, 39000 chars)
# must not stand in for them.
_RELAY_CAPS = {"discord": (2000, "chars"), "telegram": (4096, "utf16"), "whatsapp": (4096, "chars")}
_RELAY_PRIMARY = ("slack", 39000, "chars")
_RELAY_PROMPT_OPS = ("send", "edit", "typing", "prompt")
_RELAY_LEGACY_OPS = ("send", "edit", "typing")  # a connector without the prompt op


def _relay_descriptor(platform, max_message_length, len_unit, ops):
    return CapabilityDescriptor(
        contract_version=CONTRACT_VERSION, platform=platform, label=platform, max_message_length=max_message_length,
        supports_draft_streaming=False, supports_edit=True, supports_threads=False,
        markdown_dialect="markdown", len_unit=len_unit, supported_ops=ops)


def _negotiated(platform, ops):
    cap = _RELAY_CAPS.get(platform)
    return _relay_descriptor(platform, *cap, ops) if cap else None


class _Connector(StubConnector):
    """The in-memory relay connector (no network). ``lookup(platform, ops)`` serves its
    ``descriptor_for_platform``; None models a transport without one."""

    def __init__(self, ops, lookup=_negotiated):
        super().__init__(_relay_descriptor(*_RELAY_PRIMARY, ops))
        self.prompts = []
        if lookup is not None:
            self.descriptor_for_platform = lambda platform: lookup(platform, ops)

    async def send_outbound(self, action, *, platform=None):
        result = await super().send_outbound(action, platform=platform)
        if action["op"] in ("prompt", "send") and result.get("success"):
            self.prompts.append(action)
            _deny_pending()
        return result


def _relay_adapter(fronts, ops):
    """Real RelayAdapter whose chat "555" fronts ``fronts``: (adapter, wire frames, content)."""
    def make(_mentions):
        connector = _Connector(ops)
        adapter = RelayAdapter(PlatformConfig(), connector._descriptor, transport=connector)
        adapter._platform_by_chat["555"] = fronts
        op = "prompt" if "prompt" in ops else "send"

        def card(action):
            assert action["op"] == op and action["chat_id"] == "555"
            if op == "prompt":
                assert [o["id"] for o in action["options"]] == ["once", "session", "always", "deny"]
            return action["content"]

        return adapter, connector.prompts, card
    return make


_ADAPTERS = {"discord": _discord_adapter, "whatsapp": _whatsapp_adapter,
             "telegram": _telegram_adapter, "slack": _slack_adapter, "matrix": _matrix_adapter}
_ADAPTERS.update({f"relay-{p}": _relay_adapter(p, _RELAY_PROMPT_OPS) for p in _RELAY_CAPS})
_ADAPTERS.update({f"relay-text-{p}": _relay_adapter(p, _RELAY_LEGACY_OPS) for p in _RELAY_CAPS})


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
    match = re.search(r"<pre>(.*?)</pre>|```(?:bash\n|\n)?(.*?)\n?```", content, re.DOTALL)
    return match.group(1) if match.group(1) is not None else match.group(2)


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


# ── One invariant for every prompt that goes out. Where the prompt's complete budget is
# established (a card's declared text budget; a text prompt the adapter splits, or the chat's
# own cap and length unit), the prompt with context is exactly the plain prompt plus one closed
# annotation, within that budget. Where it is not (the relay: the connector renders its cards
# natively and its per-chat text cap may be another platform's; an adapter that declares no
# card budget), the context is left out and the prompt is the plain one.

_FITTED = ["discord", "whatsapp", "telegram", "slack"]
_PLAIN_ONLY = ["relay-discord", "relay-telegram", "relay-whatsapp", "matrix",
               "relay-text-discord", "relay-text-telegram", "relay-text-whatsapp"]
_SHORT_CONTEXT = {"purpose": "clean a temp path", "effect": "removes temporary files",
                  "risk": "deleted files cannot be recovered"}
# One code point, two UTF-16 units each: fits a 4096 cap counted in chars, not in UTF-16.
_WIDE_CONTEXT = {"purpose": "\U0001F642" * 900, "effect": "\U0001F642" * 900, "risk": "\U0001F642" * 900}
_CONTEXTS = {"short": _SHORT_CONTEXT, "long": _LONG_CONTEXT, "wide": _WIDE_CONTEXT}


def _deadline(platform):
    if platform.startswith("relay-text-"):
        return format_approval_deadline_line(approval_timeout_seconds())
    return html.unescape(_ADAPTERS[platform](False)[0]._ea_deadline_line().strip())


def _plain_plus_context(content, plain):
    """The annotation ``content`` adds to ``plain``; asserts it changes nothing else."""
    block = _context_block(content)
    if block is None:
        assert content == plain
    else:
        assert content.replace("\n\n" + block, "", 1) == plain
    return block


@pytest.mark.parametrize("platform", _FITTED + _PLAIN_ONLY)
@pytest.mark.parametrize("approval_context", [{}, {"purpose": "   ", "effect": 123}],
                         ids=["empty", "blank-and-non-string"])
def test_prompt_without_usable_context_is_the_plain_prompt(platform, approval_context):
    plain = _card(platform=platform)
    assert _card(approval_context, platform=platform) == plain
    assert _context_block(plain) is None
    assert _command_preview(plain) == _COMMAND
    assert _deadline(platform) in plain


@pytest.mark.parametrize("platform", _FITTED)
def test_short_context_is_added_whole(platform):
    content = _card(_SHORT_CONTEXT, platform=platform)
    block = _plain_plus_context(content, _card(platform=platform))
    assert "Purpose: clean a temp path" in block
    assert "Effect: removes temporary files" in block
    assert "Risk: deleted files cannot be recovered" in block
    assert approval_module._ENHANCED_DESC_TRUNC.strip() not in block


@pytest.mark.parametrize("platform", _FITTED)
@pytest.mark.parametrize("context", ["long", "wide"])
@pytest.mark.parametrize("tail", [0, 600, 1900])
def test_context_never_costs_the_plain_prompt_or_its_budget(platform, context, tail):
    command = "rm -rf /tmp/" + "a" * tail if tail else _COMMAND
    plain = _card(command=command, platform=platform)
    content = _card(_CONTEXTS[context], command=command, platform=platform)
    block = _plain_plus_context(content, plain)  # same warnings, preview, choices and deadline
    assert _deadline(platform) in content
    if not tail:  # room is left for some of it (whole, or shortened between both delimiters)
        assert block is not None and "Purpose: " in block


@pytest.mark.parametrize("platform", _PLAIN_ONLY)
@pytest.mark.parametrize("context", _CONTEXTS)
@pytest.mark.parametrize("tail", [0, 1900])
def test_context_is_left_out_where_the_prompt_budget_is_not_established(platform, context, tail):
    # The relay connector renders the card natively under a per-platform cap the contract does not
    # negotiate, and its text cap is not established per chat (below); Matrix declares no card budget.
    command = "rm -rf /tmp/" + "a" * tail if tail else _COMMAND
    assert _card(_CONTEXTS[context], command=command, platform=platform) == _card(command=command, platform=platform)


# ── The relay's per-chat text cap (max_message_length_for_chat / message_len_fn_for_chat) is the
# chat's platform descriptor where the transport has one, else the primary's; the handshake maps a
# malformed cap to 4096, and an unknown unit counts chars. None of that establishes the budget of
# the chat the prompt goes to, so every relay text prompt is the plain one, whichever route reaches
# it: exactly one send frame, and a plain prompt already over the chat's real cap goes out as
# upstream sends it. Each state: (the chat's real platform, whether inbound recorded it, lookup).

_CHAT_CAPS = dict(_RELAY_CAPS, slack=_RELAY_PRIMARY[1:])  # what each chat really takes


def _lookup_raises(platform, ops):
    raise RuntimeError("descriptor lookup failed")


def _handshake(**frame):
    """``descriptor_for_platform`` as the transport builds it: the connector's descriptor frame
    (with ``frame`` overrides) read through ``CapabilityDescriptor.from_json``."""
    def lookup(platform, ops):
        cap, unit = _CHAT_CAPS[platform]
        sent = json.loads(_relay_descriptor(platform, cap, unit, ops).to_json())
        return CapabilityDescriptor.from_json(json.dumps({**sent, **frame}))
    return lookup


_RELAY_TEXT_STATES = {
    "primary-chat": ("slack", True, _handshake()),
    "secondary-exact": ("discord", True, _handshake()),
    "secondary-exact-utf16": ("telegram", True, _handshake()),
    "platform-unrecorded": ("discord", False, _handshake()),
    "no-lookup": ("discord", True, None),
    "lookup-none": ("discord", True, lambda platform, ops: None),
    "lookup-raises": ("discord", True, _lookup_raises),
    "malformed-cap": ("discord", True, _handshake(max_message_length=0)),  # read as 4096
    "unknown-unit": ("telegram", True, _handshake(len_unit="utf-16")),  # counted in chars
}
# Scanner text past Discord's 2000 chars: that plain prompt already overflows a Discord chat.
_LONG_SCANNER = [dict(f, description="d" * 700) for f in _LONG_FINDINGS]


@pytest.mark.parametrize("route", ["legacy", "card-failed"])
@pytest.mark.parametrize("case", ["long", "wide", "long-scanner"])
@pytest.mark.parametrize("state", _RELAY_TEXT_STATES)
def test_relay_text_prompt_is_the_plain_prompt_in_every_descriptor_state(monkeypatch, state, case, route):
    platform, recorded, lookup = _RELAY_TEXT_STATES[state]
    ops = _RELAY_LEGACY_OPS if route == "legacy" else _RELAY_PROMPT_OPS
    frames = []

    def make(_mentions):
        connector = _Connector(ops, lookup)
        connector.next_prompt_result = {"success": False, "error": "card rejected"}
        adapter = RelayAdapter(PlatformConfig(), connector._descriptor, transport=connector)
        if recorded:
            adapter._platform_by_chat["555"] = platform
        frames.append(connector.sent)

        def text(action):
            assert action["op"] == "send" and action["chat_id"] == "555"
            return action["content"]
        return adapter, connector.prompts, text

    monkeypatch.setitem(_ADAPTERS, "relay-state", make)
    findings = _LONG_SCANNER if case == "long-scanner" else ()
    plain = _card(findings=findings, platform="relay-state")
    content = _card(_CONTEXTS["wide" if case == "wide" else "long"], findings=findings, platform="relay-state")
    cap, unit = _CHAT_CAPS[platform]
    size = utf16_len if unit == "utf16" else len
    if size(plain) <= cap:  # else the plain prompt already overflows the chat, as upstream sends it
        assert size(content) <= cap, f"context took the prompt to {size(content)} past the chat's {cap}"
    assert content == plain
    assert _context_block(content) is None and _command_preview(content) == _COMMAND
    assert _deadline("relay-text-discord") in content
    wire = ["send"] if route == "legacy" else ["prompt", "send"]
    assert [[action["op"] for action in sent] for sent in frames] == [wire, wire]


@pytest.mark.parametrize("platform", ["telegram", "slack", "matrix", "relay-discord", "relay-text-discord"])
@pytest.mark.parametrize("settle", [lambda: approval_module.resolve_gateway_approval(_SESSION, "deny"), _withdraw],
                         ids=["denied-elsewhere", "withdrawn"])
def test_no_prompt_for_a_request_settled_before_it_on_more_surfaces(platform, settle):
    result, cards = _run(_LONG_CONTEXT, platform=platform, before_notify=settle)
    assert cards == []
    assert result["approved"] is False


def test_no_text_prompt_for_a_request_settled_while_its_card_failed(monkeypatch):
    # The card goes out while the request is pending, the connector rejects it, and the request is
    # answered elsewhere meanwhile: the text fallback re-reads the queue and sends nothing.
    frames = []

    def failing_card(mentions):
        adapter, _prompts, card = _ADAPTERS["relay-discord"](mentions)

        async def send_outbound(action, *, platform=None):
            frames.append(action["op"])
            approval_module.resolve_gateway_approval(_SESSION, "deny")
            return {"success": False, "error": "card rejected"}

        adapter._transport.send_outbound = send_outbound
        return adapter, [], card

    monkeypatch.setitem(_ADAPTERS, "relay-failing", failing_card)
    result, _cards = _run(_LONG_CONTEXT, platform="relay-failing")
    assert frames == ["prompt"]
    assert result["approved"] is False
