"""RelayAdapter capability-advertisement tests (relay Phase 1, Task 1.1)."""

import asyncio

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
    # NS-658: the flag alone no longer advertises drafts — before the
    # "draft" op existed, flag=True was a latent lie (send_draft inherited
    # NotImplementedError). Advertisement now requires flag AND op.
    assert _adapter(supports_draft_streaming=False).supports_draft_streaming() is False
    assert (
        _adapter(supports_draft_streaming=False, supported_ops=("send", "draft"))
        .supports_draft_streaming()
        is False
    ), "op without flag must not advertise"
    assert (
        _adapter(supports_draft_streaming=True, supported_ops=("send", "draft"))
        .supports_draft_streaming()
        is True
    )


def test_len_fn_utf16_counts_code_units():
    a = _adapter(len_unit="utf16")
    # An astral-plane emoji is two UTF-16 code units.
    assert a.message_len_fn("\U0001f600") == 2


def test_is_a_base_platform_adapter():
    # stream_consumer's isinstance(adapter, BasePlatformAdapter) guard must pass.
    from gateway.platforms.base import BasePlatformAdapter

    assert isinstance(_adapter(), BasePlatformAdapter)


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


class _AmbiguousEditTransport(_CaptureTransport):
    async def send_outbound(self, action, *, platform=None):
        self.sent = action
        self.sent_platform = platform
        return {
            "success": False,
            "message_id": action.get("message_id"),
            "error": "relay outbound timed out",
            "ambiguous": True,
            "retryable": True,
            "error_kind": "transient",
        }


@pytest.mark.asyncio
async def test_edit_preserves_typed_relay_ack_ambiguity():
    transport = _AmbiguousEditTransport()
    adapter = RelayAdapter(
        PlatformConfig(), make_desc(platform="slack"), transport=transport
    )

    result = await adapter.edit_message("C1", "171.01", "updated")

    assert result.success is False
    assert result.message_id == "171.01"
    assert result.ambiguous is True
    assert result.retryable is True
    assert result.error_kind == "transient"


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


def _make_scoped_event_with_author(
    chat_id="chan-1", scope_id="scope-9", user_id="user-42"
):
    """An inbound scoped (guild/channel) message that ALSO carries the authentic
    author user_id — the real shape of a Discord guild message (it has both a
    guild scope_id and an author). Used to prove the adapter re-attaches BOTH
    discriminators so the connector can fall back author-first when the guild
    has no route row (managed agents join guilds dynamically)."""
    from gateway.platforms.base import MessageEvent, MessageType
    from gateway.session import SessionSource

    src = SessionSource(
        platform=Platform.RELAY,
        chat_id=chat_id,
        chat_type="channel",
        scope_id=scope_id,
        user_id=user_id,
    )
    return MessageEvent(text="hi", source=src, message_type=MessageType.TEXT)


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
async def test_scoped_reply_reattaches_both_scope_id_and_user_id():
    """A scoped (guild) reply now re-attaches BOTH scope_id AND the authentic
    author user_id. scope_id is the connector's primary discriminator; user_id
    is the author-first FALLBACK the connector uses when the guild has no route
    row (a managed agent joins guilds dynamically, so a provision-time guild
    route is not guaranteed). Regression for live 'discord egress declined:
    target not routed to an onboarded tenant' on GUILD replies (paired with
    gateway-gateway makeDiscordTenantOf guild-route-miss fallback)."""
    t = _CaptureTransport()
    a = RelayAdapter(PlatformConfig(), make_desc(platform="discord"), transport=t)
    a._capture_scope(
        _make_scoped_event_with_author(
            chat_id="chan-1", scope_id="scope-9", user_id="user-42"
        )
    )
    await a.send("chan-1", "hi")
    assert t.sent["metadata"].get("scope_id") == "scope-9"
    assert t.sent["metadata"].get("user_id") == "user-42"


@pytest.mark.asyncio
async def test_stop_typing_forwards_explicit_clear_with_routing_context():
    t = _CaptureTransport()
    a = RelayAdapter(PlatformConfig(), make_desc(platform="slack"), transport=t)
    event = _make_event(chat_id="channel-1", scope_id="workspace-1")
    event.source.platform = Platform.SLACK
    a._capture_scope(event)

    await a.stop_typing("channel-1", metadata={"thread_id": "thread-1"})

    assert t.sent == {
        "op": "typing",
        "chat_id": "channel-1",
        "content": "",
        "metadata": {
            "thread_id": "thread-1",
            "scope_id": "workspace-1",
        },
    }
    assert t.sent_platform == "slack"


# ── typing indicator over the relay (op="typing") ────────────────────────────


@pytest.mark.asyncio
async def test_send_typing_tags_egress_platform():
    """Phase 1.5: a multi-platform gateway must egress typing through the
    platform the chat lives on, exactly like send() — the underlying platform
    learned from the inbound event tags the frame."""
    from gateway.platforms.base import MessageEvent, MessageType
    from gateway.session import SessionSource

    t = _CaptureTransport()
    a = RelayAdapter(PlatformConfig(), make_desc(platform="discord"), transport=t)
    src = SessionSource(
        platform=Platform.DISCORD,
        chat_id="chan-2",
        chat_type="channel",
        scope_id="scope-1",
    )
    a._capture_scope(MessageEvent(text="hi", source=src, message_type=MessageType.TEXT))

    await a.send_typing("chan-2")

    assert t.sent_platform == "discord"


# ── Phase 7 Unit 7d-B: terminal auth revocation → clean "relay disabled" ─────


class _RevokedTransport:
    """Transport stand-in that reports a terminal auth revocation (the
    production WebSocketRelayTransport latches this after a 4401 close that
    follows a successful handshake)."""

    def __init__(self):
        self.auth_revoked = True

    def set_inbound_handler(self, h):  # noqa: D401
        self._h = h


# ─────────────── get_chat_info gated on supported_ops (Phase 1) ───────────────


class _ChatInfoTransport:
    """Transport stub that records whether get_chat_info was proxied."""

    def __init__(self):
        self.calls = []

    def set_inbound_handler(self, h):  # noqa: D401
        self._h = h

    async def get_chat_info(self, chat_id):
        self.calls.append(chat_id)
        return {"name": "general", "type": "channel"}


@pytest.mark.asyncio
async def test_get_chat_info_local_fallback_when_not_advertised():
    """A connector that advertises ops but OMITS get_chat_info is authoritative."""
    t = _ChatInfoTransport()
    a = RelayAdapter(
        PlatformConfig(),
        make_desc(supported_ops=("send", "edit", "typing")),
        transport=t,
    )
    info = await a.get_chat_info("chan-1")
    assert info == {"name": "chan-1", "type": "dm"}
    assert t.calls == []


class _HangOnIdleTransport:
    """Transport that hangs in go_idle so outer disconnect cancellation can race it."""

    def __init__(self):
        self.go_idle_started = asyncio.Event()
        self.go_idle_timeouts: list[float] = []
        self.disconnect_calls = 0

    def set_inbound_handler(self, h):  # noqa: D401
        self._h = h

    async def go_idle(self, timeout_s: float = 10.0):
        self.go_idle_timeouts.append(timeout_s)
        self.go_idle_started.set()
        await asyncio.sleep(3600)
        return False

    async def disconnect(self):
        self.disconnect_calls += 1


@pytest.mark.asyncio
async def test_disconnect_tears_down_transport_when_go_idle_is_cancelled():
    """Runner disconnect budgets can cancel adapter.disconnect mid go_idle.

    The gateway runner's default adapter disconnect budget is 5s, while
    transport.go_idle defaults to 10s. If cancellation lands during the idle
    handshake, transport.disconnect must still run so the websocket/supervisor
    cannot outlive the adapter.
    """
    transport = _HangOnIdleTransport()
    adapter = RelayAdapter(PlatformConfig(), make_desc(platform="discord"), transport=transport)

    task = asyncio.create_task(adapter.disconnect())
    await asyncio.wait_for(transport.go_idle_started.wait(), timeout=1.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert transport.disconnect_calls == 1
    assert transport.go_idle_timeouts
    assert transport.go_idle_timeouts[0] < 5.0
