"""Signal processing-lifecycle reactions (👀 then ✅/❌) and the scope that gates them.

The hooks fire BEFORE run.py's auth gate, so their gate has to be the scope that would admit the
message: the adapter's own group intake policy for groups, the DM allowlist for DMs. A group-scoped
install (``SIGNAL_ALLOWED_USERS`` empty because DMs are deliberately ignored,
``SIGNAL_GROUP_ALLOWED_USERS`` listing one group) used to get no reaction anywhere — not even in its
allowlisted group — while the anti-leak intent only ever concerned unauthorized DMs.
"""

from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType, ProcessingOutcome
from gateway.session import SessionSource

GROUP_ID = "abcdefghijklmnopqrstuvwxyz0123456789+/=="
GROUP_CHAT = f"group:{GROUP_ID}"
OTHER_GROUP_CHAT = "group:someothergroupid=="
OWNER = "+15550000001"
STRANGER = "+15559999999"
TS_MS = 1_700_000_000_000

_ABSENT = object()  # distinguish "env var not set" (Signal's own defaults) from an explicit ""


def _make_adapter(monkeypatch, *, dm_allowed=_ABSENT, group_allowed=_ABSENT, **extra):
    """SignalAdapter built from explicit allowlist env values."""
    for var, value in (("SIGNAL_ALLOWED_USERS", dm_allowed), ("SIGNAL_GROUP_ALLOWED_USERS", group_allowed)):
        if value is _ABSENT:
            monkeypatch.delenv(var, raising=False)
        else:
            monkeypatch.setenv(var, str(value))
    monkeypatch.delenv("SIGNAL_REACTIONS", raising=False)

    from gateway.platforms.signal import SignalAdapter

    config = PlatformConfig(enabled=True)
    config.extra = {"http_url": "http://localhost:8080", "account": OWNER, **extra}
    return SignalAdapter(config)


def _event(chat_id: str, user_id: str = OWNER) -> MessageEvent:
    return MessageEvent(
        text="hello",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.SIGNAL,
            chat_id=chat_id,
            chat_type="group" if chat_id.startswith("group:") else "dm",
            user_id=user_id,
            user_name="Tester",
        ),
        raw_message={"sender": user_id, "timestamp_ms": TS_MS},
    )


# ── _reactions_enabled: which scope decides ──────────────────────────────────


def test_group_scoped_install_reacts_in_its_allowlisted_group(monkeypatch):
    """DMs ignored (empty DM allowlist) must not silence the group the install IS scoped to."""
    adapter = _make_adapter(monkeypatch, dm_allowed="", group_allowed=GROUP_ID)
    assert adapter._reactions_enabled(_event(GROUP_CHAT)) is True


def test_group_allowlist_wildcard_reacts_in_any_group(monkeypatch):
    adapter = _make_adapter(monkeypatch, dm_allowed="", group_allowed="*")
    assert adapter._reactions_enabled(_event(OTHER_GROUP_CHAT)) is True


def test_unlisted_group_never_reacts(monkeypatch):
    """Intake drops the group anyway; the reaction must not outrun that verdict."""
    adapter = _make_adapter(monkeypatch, dm_allowed="*", group_allowed=GROUP_ID)
    assert adapter._reactions_enabled(_event(OTHER_GROUP_CHAT)) is False


def test_groups_disabled_never_react(monkeypatch):
    """No SIGNAL_GROUP_ALLOWED_USERS → groups are off, so their messages are not processed."""
    adapter = _make_adapter(monkeypatch, dm_allowed="*", group_allowed="")
    assert adapter._reactions_enabled(_event(GROUP_CHAT)) is False


def test_unlisted_dm_sender_does_not_react(monkeypatch):
    """The anti-leak case the sender gate exists for: no 👀 from a bot that will ignore them."""
    adapter = _make_adapter(monkeypatch, dm_allowed=OWNER, group_allowed=GROUP_ID)
    assert adapter._reactions_enabled(_event(STRANGER, user_id=STRANGER)) is False


def test_listed_dm_sender_reacts(monkeypatch):
    adapter = _make_adapter(monkeypatch, dm_allowed=OWNER, group_allowed=GROUP_ID)
    assert adapter._reactions_enabled(_event(OWNER, user_id=OWNER)) is True


def test_open_dm_allowlist_reacts_to_any_sender(monkeypatch):
    adapter = _make_adapter(monkeypatch, dm_allowed="*", group_allowed="")
    assert adapter._reactions_enabled(_event(STRANGER, user_id=STRANGER)) is True


def test_absent_dm_allowlist_keeps_its_open_default(monkeypatch):
    """Unset SIGNAL_ALLOWED_USERS means "*" (open), unlike an explicitly empty value."""
    adapter = _make_adapter(monkeypatch, group_allowed=GROUP_ID)
    assert adapter.dm_allow_from == {"*"}
    assert adapter._reactions_enabled(_event(STRANGER, user_id=STRANGER)) is True


def test_reactions_env_switch_wins_over_an_admitted_group(monkeypatch):
    adapter = _make_adapter(monkeypatch, dm_allowed="", group_allowed=GROUP_ID)
    monkeypatch.setenv("SIGNAL_REACTIONS", "false")
    assert adapter._reactions_enabled(_event(GROUP_CHAT)) is False


def test_no_event_stays_enabled_for_callers_without_one(monkeypatch):
    """base.py's generic reaction path probes the gate with no event; keep it a plain env check."""
    adapter = _make_adapter(monkeypatch, dm_allowed="", group_allowed="")
    assert adapter._reactions_enabled() is True


# ── the hooks themselves ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_processing_start_reacts_with_eyes_in_a_group_scoped_install(monkeypatch):
    adapter = _make_adapter(monkeypatch, dm_allowed="", group_allowed=GROUP_ID)
    adapter.send_reaction = AsyncMock()

    await adapter.on_processing_start(_event(GROUP_CHAT))

    adapter.send_reaction.assert_awaited_once_with(GROUP_CHAT, "👀", OWNER, TS_MS)


@pytest.mark.asyncio
async def test_processing_start_stays_silent_for_an_unlisted_dm(monkeypatch):
    adapter = _make_adapter(monkeypatch, dm_allowed=OWNER, group_allowed=GROUP_ID)
    adapter.send_reaction = AsyncMock()

    await adapter.on_processing_start(_event(STRANGER, user_id=STRANGER))

    adapter.send_reaction.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome,emoji",
    [(ProcessingOutcome.SUCCESS, "✅"), (ProcessingOutcome.FAILURE, "❌")],
)
async def test_processing_complete_swaps_eyes_for_the_outcome(monkeypatch, outcome, emoji):
    adapter = _make_adapter(monkeypatch, dm_allowed="", group_allowed=GROUP_ID)
    adapter.send_reaction = AsyncMock()
    adapter.remove_reaction = AsyncMock()

    await adapter.on_processing_complete(_event(GROUP_CHAT), outcome)

    adapter.remove_reaction.assert_awaited_once_with(GROUP_CHAT, OWNER, TS_MS)
    adapter.send_reaction.assert_awaited_once_with(GROUP_CHAT, emoji, OWNER, TS_MS)


@pytest.mark.asyncio
async def test_processing_complete_leaves_the_eyes_on_cancelled(monkeypatch):
    """CANCELLED has no outcome to report; the in-progress reaction stays (matches Telegram)."""
    adapter = _make_adapter(monkeypatch, dm_allowed="", group_allowed=GROUP_ID)
    adapter.send_reaction = AsyncMock()
    adapter.remove_reaction = AsyncMock()

    await adapter.on_processing_complete(_event(GROUP_CHAT), ProcessingOutcome.CANCELLED)

    adapter.remove_reaction.assert_not_awaited()
    adapter.send_reaction.assert_not_awaited()


@pytest.mark.asyncio
async def test_processing_complete_stays_silent_for_an_unlisted_dm(monkeypatch):
    adapter = _make_adapter(monkeypatch, dm_allowed=OWNER, group_allowed=GROUP_ID)
    adapter.send_reaction = AsyncMock()
    adapter.remove_reaction = AsyncMock()

    await adapter.on_processing_complete(_event(STRANGER, user_id=STRANGER), ProcessingOutcome.SUCCESS)

    adapter.remove_reaction.assert_not_awaited()
    adapter.send_reaction.assert_not_awaited()
