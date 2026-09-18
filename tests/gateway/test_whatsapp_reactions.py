"""Tests for the WhatsApp processing-lifecycle reaction ack (👍 → ✅/❌).

The WhatsApp adapter opts in to the shared reaction-ack flow in
``gateway/platforms/base.py`` by setting ``_ACK_EMOJI``/``_OK_EMOJI``/
``_FAIL_EMOJI`` and defining ``_add_reaction``/``_remove_reaction``
(HTTP calls to the bridge's ``/react`` endpoint). These tests pin that
wiring so a future build/import cannot silently drop it again (the
2026-09-01 regression after the gateway restart to a fresh build).
"""

from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType, ProcessingOutcome
from gateway.session import SessionSource


class _FakeResp:
    def __init__(self, status: int = 200):
        self.status = status


class _FakeRespCtx:
    def __init__(self, status: int = 200):
        self._resp = _FakeResp(status)

    async def __aenter__(self):
        return self._resp

    async def __aexit__(self, *exc_info):
        return False


class _FakeHttpSession:
    def __init__(self, status: int = 200):
        self._status = status
        self.calls = []

    def post(self, url, json=None, timeout=None):
        self.calls.append((url, json))
        return _FakeRespCtx(self._status)


def _make_adapter(status: int = 200):
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    adapter = object.__new__(WhatsAppAdapter)
    adapter.platform = Platform.WHATSAPP
    adapter.config = PlatformConfig(enabled=True)
    adapter._running = True
    adapter._bridge_port = 3000
    adapter._bridge_process = None
    adapter._http_session = _FakeHttpSession(status=status)
    adapter._auto_react_index = 0
    return adapter


def _make_event(
    chat_id: str = "256018856587355@lid",
    message_id: str = "ABC123",
    text: str = "hello",
    chat_type: str = "private",
    metadata=None,
) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.WHATSAPP,
            chat_id=chat_id,
            chat_type=chat_type,
            user_id="42",
            user_name="TestUser",
        ),
        message_id=message_id,
        metadata=metadata or {},
    )


def _calls(adapter):
    return adapter._http_session.calls


# ── opt-in emoji set ────────────────────────────────────────────────


def test_emoji_opt_in_attributes_are_set():
    """The adapter must set the shared-flow emoji class attributes."""
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    assert WhatsAppAdapter._ACK_EMOJI == "\U0001f44d"  # 👍
    assert WhatsAppAdapter._OK_EMOJI == "\u2705"  # ✅
    assert WhatsAppAdapter._FAIL_EMOJI == "\u274c"  # ❌


# ── _reactions_enabled ──────────────────────────────────────────────


def test_reactions_enabled_by_default(monkeypatch):
    """WhatsApp reactions default ON (restores the pre-restart behaviour)."""
    monkeypatch.delenv("WHATSAPP_REACTIONS", raising=False)
    adapter = _make_adapter()
    assert adapter._reactions_enabled() is True


def test_reactions_disabled_when_env_false(monkeypatch):
    monkeypatch.setenv("WHATSAPP_REACTIONS", "false")
    adapter = _make_adapter()
    assert adapter._reactions_enabled() is False


# ── _add_reaction / _remove_reaction ────────────────────────────────


@pytest.mark.asyncio
async def test_add_reaction_posts_to_bridge_react_endpoint(monkeypatch):
    from gateway.whatsapp_identity import to_whatsapp_jid

    monkeypatch.setenv("WHATSAPP_REACTIONS", "true")
    adapter = _make_adapter()

    result = await adapter._add_reaction("256018856587355@lid", "ABC123", "\U0001f44d")

    assert result is True
    assert len(_calls(adapter)) == 1
    url, payload = _calls(adapter)[0]
    assert url == "http://127.0.0.1:3000/react"
    assert payload == {
        "chatId": to_whatsapp_jid("256018856587355@lid"),
        "messageId": "ABC123",
        "emoji": "\U0001f44d",
    }


@pytest.mark.asyncio
async def test_add_reaction_soft_fails_when_not_running(monkeypatch):
    monkeypatch.setenv("WHATSAPP_REACTIONS", "true")
    adapter = _make_adapter()
    adapter._running = False

    result = await adapter._add_reaction("chat", "msg", "👍")

    assert result is False
    assert _calls(adapter) == []


@pytest.mark.asyncio
async def test_add_reaction_soft_fails_on_http_error(monkeypatch):
    monkeypatch.setenv("WHATSAPP_REACTIONS", "true")
    adapter = _make_adapter(status=500)

    result = await adapter._add_reaction("chat", "msg", "👍")

    assert result is False


@pytest.mark.asyncio
async def test_remove_reaction_sends_empty_emoji(monkeypatch):
    monkeypatch.setenv("WHATSAPP_REACTIONS", "true")
    adapter = _make_adapter()

    result = await adapter._remove_reaction("chat", "msg")

    assert result is True
    _, payload = _calls(adapter)[0]
    assert payload["emoji"] == ""


# ── on_processing_start ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_on_processing_start_themed_emoji(monkeypatch):
    monkeypatch.setenv("WHATSAPP_REACTIONS", "true")
    adapter = _make_adapter()

    await adapter.on_processing_start(_make_event(text="book a flight to LA tomorrow"))

    assert len(_calls(adapter)) == 1
    _, payload = _calls(adapter)[0]
    assert payload["emoji"] == "✈️"


@pytest.mark.asyncio
async def test_on_processing_start_aside_acks_thumbs_up(monkeypatch):
    monkeypatch.setenv("WHATSAPP_REACTIONS", "true")
    adapter = _make_adapter()

    await adapter.on_processing_start(_make_event(text="by the way here is extra context"))

    assert len(_calls(adapter)) == 1
    _, payload = _calls(adapter)[0]
    assert payload["emoji"] == "👍"


@pytest.mark.asyncio
async def test_on_processing_start_rotation_fallback(monkeypatch):
    monkeypatch.setenv("WHATSAPP_REACTIONS", "true")
    adapter = _make_adapter()

    await adapter.on_processing_start(_make_event(text="just a plain hello"))

    assert len(_calls(adapter)) == 1
    _, payload = _calls(adapter)[0]
    assert payload["emoji"] == "🔥"  # first rotation emoji when no theme matches


@pytest.mark.asyncio
async def test_on_processing_start_skips_group_messages(monkeypatch):
    monkeypatch.setenv("WHATSAPP_REACTIONS", "true")
    adapter = _make_adapter()

    await adapter.on_processing_start(_make_event(chat_type="group", text="book a flight"))

    assert _calls(adapter) == []


@pytest.mark.asyncio
async def test_on_processing_start_skips_reaction_messages(monkeypatch):
    monkeypatch.setenv("WHATSAPP_REACTIONS", "true")
    adapter = _make_adapter()

    await adapter.on_processing_start(
        _make_event(text="book a flight", metadata={"whatsapp_native_type": "reactionMessage"})
    )

    assert _calls(adapter) == []


@pytest.mark.asyncio
async def test_on_processing_start_skips_when_disabled(monkeypatch):
    monkeypatch.setenv("WHATSAPP_REACTIONS", "false")
    adapter = _make_adapter()

    await adapter.on_processing_start(_make_event())

    assert _calls(adapter) == []


@pytest.mark.asyncio
async def test_on_processing_start_handles_missing_ids(monkeypatch):
    monkeypatch.setenv("WHATSAPP_REACTIONS", "true")
    adapter = _make_adapter()

    await adapter.on_processing_start(
        MessageEvent(
            text="hello",
            message_type=MessageType.TEXT,
            source=SessionSource(
                platform=Platform.WHATSAPP,
                chat_id="",
                chat_type="private",
                user_id="42",
                user_name="TestUser",
            ),
            message_id=None,
        )
    )

    assert _calls(adapter) == []


# ── on_processing_complete (shared base flow) ───────────────────────


@pytest.mark.asyncio
async def test_on_processing_complete_swaps_ack_for_ok(monkeypatch):
    monkeypatch.setenv("WHATSAPP_REACTIONS", "true")
    adapter = _make_adapter()

    await adapter.on_processing_complete(_make_event(), ProcessingOutcome.SUCCESS)

    assert [c[1]["emoji"] for c in _calls(adapter)] == ["", "\u2705"]


@pytest.mark.asyncio
async def test_on_processing_complete_marks_failure(monkeypatch):
    monkeypatch.setenv("WHATSAPP_REACTIONS", "true")
    adapter = _make_adapter()

    await adapter.on_processing_complete(_make_event(), ProcessingOutcome.FAILURE)

    assert [c[1]["emoji"] for c in _calls(adapter)] == ["", "\u274c"]


@pytest.mark.asyncio
async def test_on_processing_complete_cancelled_leaves_unreacted(monkeypatch):
    monkeypatch.setenv("WHATSAPP_REACTIONS", "true")
    adapter = _make_adapter()

    await adapter.on_processing_complete(_make_event(), ProcessingOutcome.CANCELLED)

    assert [c[1]["emoji"] for c in _calls(adapter)] == [""]


@pytest.mark.asyncio
async def test_on_processing_complete_noop_when_disabled(monkeypatch):
    monkeypatch.setenv("WHATSAPP_REACTIONS", "false")
    adapter = _make_adapter()

    await adapter.on_processing_complete(_make_event(), ProcessingOutcome.SUCCESS)

    assert _calls(adapter) == []


# ── themed ack picker (_pick_context_react_emoji) ─────────────────────


def _pick(text):
    from plugins.platforms.whatsapp.adapter import _pick_context_react_emoji

    return _pick_context_react_emoji(text)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("book a flight to LA tomorrow", "✈️"),
        ("search hotels near the venue", "🏨"),
        ("drop the new demo", "🎵"),
        ("check the kanban board", "🎫"),
        ("any open tickets this week", "🎫"),
        ("draft an email to the team", "✉️"),
        ("push the new branch", "💻"),
        ("what's the weather today", "🌦️"),
    ],
)
def test_thematic_emoji_matches(text, expected):
    assert _pick(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "by the way here is some extra context",
        "btw I need this too",
        "cue task: handle the follow up for me",
        "for context, here is what happened",
        "for reference, that was last week",
        "also, can you include the numbers",
        "one more thing before you go",
        "just so you know, this changed",
        "extra detail for you",
        "note that this is due Friday",
    ],
)
def test_aside_detail_cues_ack_thumbs_up(text):
    assert _pick(text) == "👍"


@pytest.mark.parametrize(
    "text",
    [
        "hello there",
        "just a quick hello",
        "ok sounds good",
        "",
    ],
)
def test_plain_messages_fall_through_to_rotation(text):
    assert _pick(text) is None


def test_domain_beats_aside_rule():
    # "hotel" is a specific domain ahead of the generic aside rule; the
    # aside keyword "by the way" must not override it.
    assert _pick("search hotels by the way") == "🏨"


@pytest.mark.parametrize(
    "text,expected",
    [
        # whole-word, case-insensitive matching for the aside keywords
        ("BTW also send the file", "👍"),
        ("By The Way, here is more", "👍"),
        # "btw" must not fire inside a larger word
        ("the abtw abbreviation is wrong", None),
        # "mail" must not fire inside "email" (email still matches, whole-word)
        ("forward this email", "✉️"),
        # "fly" must not fire inside "butterfly"
        ("a butterfly landed", None),
    ],
)
def test_whole_word_boundaries(text, expected):
    assert _pick(text) == expected


def test_cue_task_is_thumbs_up_not_ticket():
    # Regression: "task" used to live in the 🎫 rule and shadowed the "cue task"
    # phrase. "cue task" must ack 👍, never 🎫.
    assert _pick("cue task: send the recap") == "👍"


def test_rotation_set_unchanged():
    from plugins.platforms.whatsapp.adapter import _AUTO_REACT_EMOJIS

    assert _AUTO_REACT_EMOJIS == ("🔥", "⚡️", "🎯", "🙂", "👍")
