"""RelayAdapter capability-advertisement tests (relay Phase 1, Task 1.1)."""

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.relay.adapter import RelayAdapter
from gateway.relay.descriptor import CONTRACT_VERSION, CapabilityDescriptor


def make_desc(**kw) -> CapabilityDescriptor:
    base = dict(
        contract_version=CONTRACT_VERSION,
        platform="telegram",
        label="Telegram",
        max_message_length=4096,
        supports_draft_streaming=False,
        supports_edit=True,
        supports_threads=True,
        markdown_dialect="markdown_v2",
        len_unit="utf16",
        emoji="\u2708\ufe0f",
        platform_hint="",
        pii_safe=False,
    )
    base.update(kw)
    return CapabilityDescriptor(**base)


def _adapter(**desc_kw) -> RelayAdapter:
    return RelayAdapter(PlatformConfig(), make_desc(**desc_kw))


def test_relay_platform_member_exists():
    assert Platform("relay") is Platform.RELAY


def test_advertises_descriptor_max_length():
    a = _adapter(max_message_length=2000)
    assert a.MAX_MESSAGE_LENGTH == 2000


def test_supports_draft_streaming_follows_descriptor():
    assert _adapter(supports_draft_streaming=False).supports_draft_streaming() is False
    assert _adapter(supports_draft_streaming=True).supports_draft_streaming() is True


def test_len_fn_utf16_counts_code_units():
    a = _adapter(len_unit="utf16")
    # An astral-plane emoji is two UTF-16 code units.
    assert a.message_len_fn("\U0001f600") == 2


def test_len_fn_chars_uses_builtin_len():
    a = _adapter(len_unit="chars")
    assert a.message_len_fn("\U0001f600") == 1


def test_is_a_base_platform_adapter():
    # stream_consumer's isinstance(adapter, BasePlatformAdapter) guard must pass.
    from gateway.platforms.base import BasePlatformAdapter

    assert isinstance(_adapter(), BasePlatformAdapter)


@pytest.mark.asyncio
async def test_connect_without_transport_raises():
    a = _adapter()
    with pytest.raises(RuntimeError, match="no transport"):
        await a.connect()


@pytest.mark.asyncio
async def test_connect_accepts_is_reconnect_kwarg():
    """Regression: RelayAdapter.connect must accept the BasePlatformAdapter
    contract's ``is_reconnect`` kwarg. The gateway reconnect watcher recovers a
    platform after a fatal adapter error by calling ``connect(is_reconnect=True)``
    (gateway/run.py); before the fix, RelayAdapter.connect was bare ``connect()``
    and that recovery path raised ``TypeError: connect() got an unexpected
    keyword argument 'is_reconnect'`` (observed live: relay never reconnected,
    no DMs). It must reach the SAME transport-less RuntimeError as connect() —
    i.e. accept the kwarg, never TypeError on it."""
    a = _adapter()
    with pytest.raises(RuntimeError, match="no transport"):
        await a.connect(is_reconnect=True)


def test_connect_signature_matches_base_contract():
    """The is_reconnect parameter must be keyword-accepting and default False,
    matching BasePlatformAdapter.connect, so the reconnect watcher's
    ``connect(is_reconnect=...)`` call is valid for relay as for every other
    adapter."""
    import inspect

    from gateway.platforms.base import BasePlatformAdapter

    sig = inspect.signature(RelayAdapter.connect)
    base_sig = inspect.signature(BasePlatformAdapter.connect)
    assert "is_reconnect" in sig.parameters
    param = sig.parameters["is_reconnect"]
    base_param = base_sig.parameters["is_reconnect"]
    # Keyword-acceptable (KEYWORD_ONLY here, matching the base) with a False default.
    assert param.kind is base_param.kind
    assert param.default is False


@pytest.mark.asyncio
async def test_send_without_transport_returns_failure():
    a = _adapter()
    result = await a.send("chat1", "hello")
    assert result.success is False
    assert result.error == "no transport"


class _CaptureTransport:
    """Minimal RelayTransport stand-in that records the outbound action."""

    def __init__(self):
        self.sent = None
        self.sent_platform = None
        # No concrete fronted identities ⇒ _platform_is_fronted is a no-op here.
        self._identities = []

    def set_inbound_handler(self, h):  # noqa: D401
        self._h = h

    async def send_outbound(self, action, *, platform=None):
        self.sent = action
        self.sent_platform = platform
        return {"success": True, "message_id": "m1"}


def _make_event(chat_id="chan-1", scope_id="scope-9"):
    from gateway.platforms.base import MessageEvent, MessageType
    from gateway.session import SessionSource

    src = SessionSource(
        platform=Platform.RELAY,
        chat_id=chat_id,
        chat_type="channel",
        scope_id=scope_id,
    )
    return MessageEvent(text="hi", source=src, message_type=MessageType.TEXT)


def _make_dm_event(chat_id="dm-1", user_id="user-42"):
    """An inbound DM: no scope_id, carries the authentic author user_id."""
    from gateway.platforms.base import MessageEvent, MessageType
    from gateway.session import SessionSource

    src = SessionSource(
        platform=Platform.RELAY,
        chat_id=chat_id,
        chat_type="dm",
        scope_id=None,
        user_id=user_id,
    )
    return MessageEvent(text="hi", source=src, message_type=MessageType.TEXT)


@pytest.mark.asyncio
async def test_send_reattaches_scope_id_from_inbound_scope():
    """The connector's egress guard resolves the owning tenant from
    metadata.scope_id; the gateway's generic delivery path drops it, so the
    relay adapter must re-attach the scope learned from the inbound event.
    Regression for live 'egress declined: target not routed to an
    onboarded tenant'."""
    t = _CaptureTransport()
    a = RelayAdapter(PlatformConfig(), make_desc(platform="discord"), transport=t)
    # Simulate the connector delivering an inbound message in scope-9 / chan-1,
    # but don't run the full handle_message pipeline — just the scope capture.
    a._capture_scope(_make_event(chat_id="chan-1", scope_id="scope-9"))

    await a.send("chan-1", "the reply")

    assert t.sent["metadata"].get("scope_id") == "scope-9"


@pytest.mark.asyncio
async def test_send_without_known_scope_omits_scope_id():
    """A chat we never saw inbound (e.g. a DM) gets no scope_id — no-op, never
    invents a scope."""
    t = _CaptureTransport()
    a = RelayAdapter(PlatformConfig(), make_desc(platform="discord"), transport=t)
    await a.send("unknown-chat", "hi")
    assert "scope_id" not in t.sent["metadata"]


@pytest.mark.asyncio
async def test_send_preserves_explicit_scope_id():
    """An explicitly-provided metadata.scope_id is never overwritten."""
    t = _CaptureTransport()
    a = RelayAdapter(PlatformConfig(), make_desc(platform="discord"), transport=t)
    a._capture_scope(_make_event(chat_id="chan-1", scope_id="scope-9"))
    await a.send("chan-1", "hi", metadata={"scope_id": "explicit-1"})
    assert t.sent["metadata"]["scope_id"] == "explicit-1"


@pytest.mark.asyncio
async def test_send_reattaches_dm_user_id_from_inbound_scope():
    """A DM reply has no scope_id, so the connector resolves the tenant from the
    recipient's author binding — it needs metadata.user_id. The adapter must
    re-attach the authentic author id learned from the inbound DM. Regression for
    live 'discord egress declined: target not routed to an onboarded tenant' on
    DM replies (the connector-side fix is gateway-gateway #67)."""
    t = _CaptureTransport()
    a = RelayAdapter(PlatformConfig(), make_desc(platform="discord"), transport=t)
    a._capture_scope(_make_dm_event(chat_id="dm-1", user_id="user-42"))

    await a.send("dm-1", "the reply")

    assert t.sent["metadata"].get("user_id") == "user-42"
    # A DM carries no scope_id — only the author discriminator.
    assert "scope_id" not in t.sent["metadata"]


@pytest.mark.asyncio
async def test_send_dm_does_not_invent_user_id_for_unknown_chat():
    """A chat we never saw inbound gets neither discriminator — no-op."""
    t = _CaptureTransport()
    a = RelayAdapter(PlatformConfig(), make_desc(platform="discord"), transport=t)
    await a.send("unknown-dm", "hi")
    assert "user_id" not in t.sent["metadata"]
    assert "scope_id" not in t.sent["metadata"]


@pytest.mark.asyncio
async def test_send_preserves_explicit_user_id():
    """An explicitly-provided metadata.user_id is never overwritten."""
    t = _CaptureTransport()
    a = RelayAdapter(PlatformConfig(), make_desc(platform="discord"), transport=t)
    a._capture_scope(_make_dm_event(chat_id="dm-1", user_id="user-42"))
    await a.send("dm-1", "hi", metadata={"user_id": "explicit-user"})
    assert t.sent["metadata"]["user_id"] == "explicit-user"


@pytest.mark.asyncio
async def test_scoped_reply_does_not_carry_user_id():
    """A scoped reply resolves by scope_id and must NOT carry a DM user_id even if
    the same chat_id was somehow seen — scope capture wins and user_id stays out
    (scope_id is the discriminator; user_id is the DM-only fallback)."""
    t = _CaptureTransport()
    a = RelayAdapter(PlatformConfig(), make_desc(platform="discord"), transport=t)
    a._capture_scope(_make_event(chat_id="chan-1", scope_id="scope-9"))
    await a.send("chan-1", "hi")
    assert t.sent["metadata"].get("scope_id") == "scope-9"
    assert "user_id" not in t.sent["metadata"]


# ── Phase 7 Unit 7d-B: terminal auth revocation → clean "relay disabled" ─────


class _RevokedTransport:
    """Transport stand-in that reports a terminal auth revocation (the
    production WebSocketRelayTransport latches this after a 4401 close that
    follows a successful handshake)."""

    def __init__(self):
        self.auth_revoked = True

    def set_inbound_handler(self, h):  # noqa: D401
        self._h = h


@pytest.mark.asyncio
async def test_revocation_marks_relay_disabled_non_retryable():
    """When the transport reports auth_revoked, the adapter surfaces a clean,
    NON-retryable `relay_disabled` fatal and fires the fatal-error handler."""
    a = RelayAdapter(PlatformConfig(), make_desc(platform="discord"), transport=_RevokedTransport())
    notified = []
    a.set_fatal_error_handler(lambda adapter: notified.append(adapter))

    # Drive the monitor body directly (poll loop breaks immediately on the
    # already-revoked transport).
    await a._watch_for_revocation(poll_interval_s=0.01)

    assert a.has_fatal_error is True
    assert a.fatal_error_code == "relay_disabled"
    assert a.fatal_error_retryable is False
    assert "disabled" in (a.fatal_error_message or "").lower()
    assert notified == [a]


@pytest.mark.asyncio
async def test_no_revocation_no_fatal():
    """A transport that has NOT been revoked never trips the disabled fatal."""

    class _LiveTransport:
        auth_revoked = False

        def set_inbound_handler(self, h):  # noqa: D401
            self._h = h

    a = RelayAdapter(PlatformConfig(), make_desc(platform="discord"), transport=_LiveTransport())
    # Run the monitor with a tiny window then cancel — it should never fire.
    import asyncio

    task = asyncio.create_task(a._watch_for_revocation(poll_interval_s=0.01))
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    assert a.has_fatal_error is False
