"""Relay-side Slack unfurl suppression: gateway-directed metadata stamping.

The gateway resolves ``platforms.relay.extra.slack.unfurl_links`` /
``unfurl_media`` and stamps them onto the outbound frame metadata; the
connector just forwards whatever the gateway resolved (no connector config).
Contract under test:
- Slack chats: explicit booleans are stamped; omitted keys are absent.
- Non-Slack chats: never stamped (metadata not polluted cross-platform).
- Non-boolean values (hostile/hand-edited config) are dropped.
- The scheduled/cron lane (send_for_platform) stamps too.
"""

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.relay.adapter import RelayAdapter
from gateway.relay.descriptor import CONTRACT_VERSION, CapabilityDescriptor


def make_desc(**kw) -> CapabilityDescriptor:
    base = dict(
        contract_version=CONTRACT_VERSION,
        platform="slack",
        label="Slack",
        max_message_length=39000,
        supports_draft_streaming=False,
        supports_edit=True,
        supports_threads=True,
        markdown_dialect="slack",
        len_unit="char",
        emoji="\U0001f4bc",
        platform_hint="",
        pii_safe=False,
    )
    base.update(kw)
    return CapabilityDescriptor(**base)


class _CaptureTransport:
    def __init__(self):
        self.sent = None
        self.sent_platform = None
        # Advertise a slack identity so fronts_platform() passes for the
        # send_for_platform (cron) lane.
        self._identities = [("slack", None)]

    def set_inbound_handler(self, h):  # noqa: D401
        self._h = h

    async def send_outbound(self, action, *, platform=None):
        self.sent = action
        self.sent_platform = platform
        return {"success": True, "message_id": "m1"}


def _slack_adapter(extra):
    a = RelayAdapter(
        PlatformConfig(extra=extra), make_desc(platform="slack"), transport=_CaptureTransport()
    )
    return a


def _mark_slack_chat(a, chat_id="chan-1"):
    from gateway.platforms.base import MessageEvent, MessageType
    from gateway.session import SessionSource

    src = SessionSource(
        platform=Platform.SLACK, chat_id=chat_id, chat_type="channel", scope_id="w-1"
    )
    ev = MessageEvent(text="hi", source=src, message_type=MessageType.TEXT)
    a._capture_scope(ev)


class TestUnfurlHints:
    def test_non_slack_returns_none(self):
        a = _slack_adapter({"slack": {"unfurl_links": False}})
        assert a._slack_unfurl_hints("discord") is None
        assert a._slack_unfurl_hints(None) is None

    def test_slack_explicit_bools_returned(self):
        a = _slack_adapter({"slack": {"unfurl_links": False, "unfurl_media": False}})
        assert a._slack_unfurl_hints("slack") == {
            "unfurl_links": False,
            "unfurl_media": False,
        }

    def test_omitted_keys_return_none(self):
        a = _slack_adapter({"slack": {}})
        assert a._slack_unfurl_hints("slack") is None

    def test_string_bools_from_config_set_are_coerced(self):
        # Railway knobs / `hermes config set` persist YAML strings.
        a = _slack_adapter({"slack": {"unfurl_links": "true", "unfurl_media": "false"}})
        assert a._slack_unfurl_hints("slack") == {
            "unfurl_links": True,
            "unfurl_media": False,
        }

    def test_junk_values_dropped(self):
        a = _slack_adapter({"slack": {"unfurl_links": "maybe", "unfurl_media": 0}})
        assert a._slack_unfurl_hints("slack") is None

    def test_flat_legacy_key_fallback(self):
        # _relay_slack_extra falls back to the flat extra when no "slack"
        # object exists (legacy staging configs).
        a = _slack_adapter({"unfurl_links": False})
        assert a._slack_unfurl_hints("slack") == {"unfurl_links": False}


class TestSendStampsUnfurl:
    @pytest.mark.asyncio
    async def test_send_stamps_explicit_bools(self):
        a = _slack_adapter({"slack": {"unfurl_links": False, "unfurl_media": False}})
        _mark_slack_chat(a)
        await a.send("chan-1", "see https://example.com")
        assert a._transport.sent["metadata"]["unfurl_links"] is False
        assert a._transport.sent["metadata"]["unfurl_media"] is False

    @pytest.mark.asyncio
    async def test_send_omits_when_unconfigured(self):
        a = _slack_adapter({"slack": {}})
        _mark_slack_chat(a)
        await a.send("chan-1", "plain text")
        assert "unfurl_links" not in a._transport.sent["metadata"]
        assert "unfurl_media" not in a._transport.sent["metadata"]

    @pytest.mark.asyncio
    async def test_non_slack_chat_never_stamped(self):
        a = _slack_adapter({"slack": {"unfurl_links": False}})
        # A chat mapped to discord, not slack, must not carry the hint.
        from gateway.platforms.base import MessageEvent, MessageType
        from gateway.session import SessionSource

        src = SessionSource(
            platform=Platform.DISCORD,
            chat_id="chan-1",
            chat_type="channel",
            scope_id="w-1",
        )
        a._capture_scope(
            MessageEvent(text="hi", source=src, message_type=MessageType.TEXT)
        )
        await a.send("chan-1", "see https://example.com")
        assert "unfurl_links" not in a._transport.sent["metadata"]

    @pytest.mark.asyncio
    async def test_send_falls_back_to_descriptor_platform(self):
        """No inbound frame yet (e.g. gateway restart): _platform_by_chat is
        empty, so the platform must resolve from the negotiated descriptor —
        the same fallback the streaming gate and delivery resolver use."""
        a = _slack_adapter({"slack": {"unfurl_links": False}})
        assert not a._platform_by_chat
        await a.send("chan-1", "see https://example.com")
        assert a._transport.sent["metadata"]["unfurl_links"] is False


class TestMediaLaneStampsUnfurl:
    """The send_media lane egresses through the connector's Slack sender too,
    so it must stamp the same unfurl hints as the text lane."""

    def _media_adapter(self, extra):
        a = RelayAdapter(
            PlatformConfig(extra=extra),
            make_desc(platform="slack", supported_ops=("send", "send_media")),
            transport=_CaptureTransport(),
        )
        return a

    @pytest.mark.asyncio
    async def test_media_lane_stamps_explicit_bools(self):
        a = self._media_adapter({"slack": {"unfurl_links": False, "unfurl_media": False}})
        _mark_slack_chat(a)
        res = await a.send_image("chan-1", "https://img.example/x.png", caption="cap")
        assert res.success is True
        assert a._transport.sent["op"] == "send_media"
        assert a._transport.sent["metadata"]["unfurl_links"] is False
        assert a._transport.sent["metadata"]["unfurl_media"] is False

    @pytest.mark.asyncio
    async def test_media_lane_falls_back_to_descriptor_platform(self):
        """Regression: _send_media resolved platform only from
        _platform_by_chat; after a gateway restart a proactive media send to a
        Slack chat missed the stamp. Must fall back to descriptor.platform."""
        a = self._media_adapter({"slack": {"unfurl_links": False}})
        assert not a._platform_by_chat
        res = await a.send_image("chan-1", "https://img.example/x.png")
        assert res.success is True
        assert a._transport.sent["op"] == "send_media"
        assert a._transport.sent["metadata"]["unfurl_links"] is False

    @pytest.mark.asyncio
    async def test_media_lane_omits_when_unconfigured(self):
        a = self._media_adapter({"slack": {}})
        _mark_slack_chat(a)
        await a.send_image("chan-1", "https://img.example/x.png")
        assert a._transport.sent["op"] == "send_media"
        assert "unfurl_links" not in a._transport.sent["metadata"]
        assert "unfurl_media" not in a._transport.sent["metadata"]


class TestSendForPlatformStampsUnfurl:
    @pytest.mark.asyncio
    async def test_cron_lane_stamps_explicit_bools(self):
        a = _slack_adapter({"slack": {"unfurl_links": False, "unfurl_media": False}})
        from gateway.config import Platform as P

        res = await a.send_for_platform(P.SLACK, "C123", "brief https://x.dev")
        assert res.success is True
        assert a._transport.sent["metadata"]["unfurl_links"] is False
        assert a._transport.sent["metadata"]["unfurl_media"] is False

    @pytest.mark.asyncio
    async def test_cron_lane_omits_when_unconfigured(self):
        a = _slack_adapter({"slack": {}})
        from gateway.config import Platform as P

        await a.send_for_platform(P.SLACK, "C123", "brief")
        assert "unfurl_links" not in a._transport.sent["metadata"]
        assert "unfurl_media" not in a._transport.sent["metadata"]

class TestUnfurlDisablesDraftStreaming:
    def test_explicit_unfurl_disables_slack_draft_stream(self):
        a = RelayAdapter(
            PlatformConfig(extra={"slack": {"unfurl_links": True}}),
            make_desc(
                platform="slack",
                supports_draft_streaming=True,
                supported_ops=("send", "draft"),
            ),
            transport=_CaptureTransport(),
        )
        assert a.supports_draft_streaming() is False

    def test_omitted_unfurl_keeps_slack_draft_stream(self):
        a = RelayAdapter(
            PlatformConfig(extra={"slack": {}}),
            make_desc(
                platform="slack",
                supports_draft_streaming=True,
                supported_ops=("send", "draft"),
            ),
            transport=_CaptureTransport(),
        )
        assert a.supports_draft_streaming() is True

    def test_string_true_also_disables_stream(self):
        a = RelayAdapter(
            PlatformConfig(extra={"slack": {"unfurl_links": "true"}}),
            make_desc(
                platform="slack",
                supports_draft_streaming=True,
                supported_ops=("send", "draft"),
            ),
            transport=_CaptureTransport(),
        )
        assert a.supports_draft_streaming() is False
