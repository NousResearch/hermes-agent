"""Relay lane parity: block formatting hints on outbound frames (Coatue F2).

Field report 2026-08-18, finding 2: identical agent output renders as native
rich_text lists / Block Kit tables / highlighted code on native Slack, but
literal `-` bullets and code-fence tables on the relay lane. Native reads
platforms.slack.extra.rich_blocks / markdown_blocks; relay frames carry no
formatting signal at all, and the connector has no way to know the operator
wants block rendering.

Contract (additive, v1): the connector advertises
``supports_block_formatting`` in its capability descriptor; when the operator
enables the knobs (relay shape: platforms.relay.extra.slack.rich_blocks /
markdown_blocks — same sub-block as the other relay Slack knobs), the gateway
stamps ``format_hints`` into outbound send metadata. Old connectors never
advertise, so no hint is ever sent (no dead metadata); old gateways never
stamp, so connectors keep rendering plain text.
"""

import json
from types import SimpleNamespace

import pytest

from gateway.config import PlatformConfig
from gateway.relay.adapter import RelayAdapter
from gateway.relay.descriptor import CapabilityDescriptor


def _descriptor(**overrides):
    base = dict(
        contract_version=1,
        platform="slack",
        label="Slack",
        max_message_length=4000,
        supports_draft_streaming=False,
        supports_edit=True,
        supports_threads=True,
        markdown_dialect="mrkdwn",
        len_unit="chars",
    )
    base.update(overrides)
    return CapabilityDescriptor(**base)


class FakeTransport:
    def __init__(self, descriptors_by_platform=None, identities=None):
        self.frames = []
        # Phase 1.5 multi-platform: per-platform negotiated descriptors and
        # the handshaked identity set (fronts_platform reads _identities).
        self._descriptors_by_platform = descriptors_by_platform or {}
        self._identities = identities or [("slack", "hermes")]

    def descriptor_for_platform(self, platform):
        return self._descriptors_by_platform.get(platform)

    async def send_outbound(self, frame, platform=None):
        self.frames.append((frame, platform))
        return {"success": True, "message_id": "1.2"}


def _adapter(extra=None, descriptor=None, transport=None):
    config = PlatformConfig(enabled=True, extra=extra or {})
    a = RelayAdapter(
        config, descriptor or _descriptor(), transport=transport or FakeTransport()
    )
    return a


class TestDescriptorBit:
    def test_default_false(self):
        assert _descriptor().supports_block_formatting is False

    def test_from_json_reads_flag(self):
        payload = dict(
            contract_version=1, platform="slack", label="Slack",
            max_message_length=4000, supports_draft_streaming=False,
            supports_edit=True, supports_threads=True,
            markdown_dialect="mrkdwn", len_unit="chars",
            supports_block_formatting=True,
        )
        assert CapabilityDescriptor.from_json(
            json.dumps(payload)
        ).supports_block_formatting is True


class TestFormatHintsStamping:
    @pytest.mark.asyncio
    async def test_hints_stamped_when_capable_and_enabled(self):
        a = _adapter(
            extra={"slack": {"rich_blocks": True, "markdown_blocks": True}},
            descriptor=_descriptor(supports_block_formatting=True),
        )
        await a.send("D01", "# Report\n\n| a | b |\n|---|---|\n| 1 | 2 |")
        frame, _ = a._transport.frames[-1]
        hints = (frame.get("metadata") or {}).get("format_hints")
        assert hints == {"rich_blocks": True, "markdown_blocks": True}

    @pytest.mark.asyncio
    async def test_no_hints_when_connector_lacks_capability(self):
        """Old connector: knob on, capability absent -> no dead metadata."""
        a = _adapter(
            extra={"slack": {"rich_blocks": True}},
            descriptor=_descriptor(),
        )
        await a.send("D01", "text")
        frame, _ = a._transport.frames[-1]
        assert "format_hints" not in (frame.get("metadata") or {})

    @pytest.mark.asyncio
    async def test_no_hints_when_knobs_off(self):
        """Capable connector, operator never opted in -> no hint (native
        parity: rich_blocks/markdown_blocks are opt-in on native too)."""
        a = _adapter(
            extra={},
            descriptor=_descriptor(supports_block_formatting=True),
        )
        await a.send("D01", "text")
        frame, _ = a._transport.frames[-1]
        assert "format_hints" not in (frame.get("metadata") or {})

    @pytest.mark.asyncio
    async def test_quoted_false_knob_stays_off(self):
        """YAML-quoted 'false' must coerce off — same _coerce_flag semantics
        as the other relay Slack knobs."""
        a = _adapter(
            extra={"slack": {"rich_blocks": "false", "markdown_blocks": "false"}},
            descriptor=_descriptor(supports_block_formatting=True),
        )
        await a.send("D01", "text")
        frame, _ = a._transport.frames[-1]
        assert "format_hints" not in (frame.get("metadata") or {})

    @pytest.mark.asyncio
    async def test_partial_knobs_stamp_only_enabled(self):
        a = _adapter(
            extra={"slack": {"markdown_blocks": True}},
            descriptor=_descriptor(supports_block_formatting=True),
        )
        await a.send("D01", "text")
        frame, _ = a._transport.frames[-1]
        hints = (frame.get("metadata") or {}).get("format_hints")
        assert hints == {"markdown_blocks": True}

    @pytest.mark.asyncio
    async def test_edit_lane_carries_hints_too(self):
        """Boundary rule: every text egress lane crossing the frame contract
        gets the hint — send AND edit (streaming final edits render blocks
        on native)."""
        a = _adapter(
            extra={"slack": {"rich_blocks": True}},
            descriptor=_descriptor(supports_block_formatting=True),
        )
        edit = getattr(a, "edit_message", None)
        if edit is None:
            pytest.skip("relay adapter has no edit lane")
        await edit("D01", "1.2", "updated **content**")
        frame, _ = a._transport.frames[-1]
        hints = (frame.get("metadata") or {}).get("format_hints")
        assert hints == {"rich_blocks": True}

    @pytest.mark.asyncio
    async def test_draft_interim_and_seal_frames_carry_hints(self):
        """Observed in live relay testing: a STREAMED bash
        snippet sealed as a plain code block while send/edit rendered
        blocks. The draft lane (interim frame AND the final=true seal
        frame) is a text egress lane crossing the same frame contract —
        the connector's seal reconcile can only attach the markdown block
        if the seal frame carries the hint. Boundary rule: send, edit,
        send_for_platform, AND draft/seal all stamp format_hints."""
        a = _adapter(
            extra={"slack": {"markdown_blocks": True}},
            descriptor=_descriptor(
                supports_block_formatting=True,
                supports_draft_streaming=True,
                supported_ops=("send", "edit", "draft"),
            ),
        )
        code = "```bash\necho hi\n```"
        await a.send_draft("D01", 7, "```bash\necho")
        interim, _ = a._transport.frames[-1]
        assert interim["op"] == "draft" and interim["final"] is False
        assert (interim.get("metadata") or {}).get("format_hints") == {
            "markdown_blocks": True
        }
        seal = getattr(a, "seal_draft", None) or getattr(
            a, "send_draft_final", None
        )
        if seal is not None:
            await seal("D01", 7, code)
        else:
            # The seal frame is built by _seal_open_draft (send()
            # interception converts an armed final into draft final=true).
            await a._seal_open_draft("D01", code, None)
        final_frame, _ = a._transport.frames[-1]
        assert final_frame["op"] == "draft" and final_frame["final"] is True
        assert (final_frame.get("metadata") or {}).get("format_hints") == {
            "markdown_blocks": True
        }

    @pytest.mark.asyncio
    async def test_draft_frames_no_hints_when_knobs_off(self):
        """Regression control: knob off -> draft frames byte-identical to
        today (no format_hints key)."""
        a = _adapter(
            extra={},
            descriptor=_descriptor(
                supports_block_formatting=True,
                supports_draft_streaming=True,
                supported_ops=("send", "edit", "draft"),
            ),
        )
        await a.send_draft("D01", 8, "plain text")
        interim, _ = a._transport.frames[-1]
        assert "format_hints" not in (interim.get("metadata") or {})


class TestMultiPlatformResolution:
    """One RelayAdapter fronts N platforms: capability must resolve from the
    DESTINATION platform's negotiated descriptor, never the primary identity's
    scalar — a Slack-primary adapter must not leak Slack hints onto Discord,
    and a Discord-primary adapter must not suppress hints for Slack."""

    def _two_platform_transport(self, slack_capable=True, discord_capable=False):
        return FakeTransport(
            descriptors_by_platform={
                "slack": _descriptor(supports_block_formatting=slack_capable),
                "discord": _descriptor(
                    platform="discord", label="Discord",
                    markdown_dialect="markdown", max_message_length=2000,
                    supports_block_formatting=discord_capable,
                ),
            },
            identities=[("slack", "hermes"), ("discord", "hermes")],
        )

    @pytest.mark.asyncio
    async def test_slack_primary_does_not_leak_hints_onto_discord_chat(self):
        """REGRESSION: _format_hints read self.descriptor (Slack primary,
        capable) and the Slack config sub-block for EVERY chat — a known
        Discord chat got Slack's format_hints stamped."""
        transport = self._two_platform_transport()
        a = _adapter(
            extra={"slack": {"rich_blocks": True}},
            descriptor=_descriptor(supports_block_formatting=True),
            transport=transport,
        )
        # The adapter learned this chat is Discord from inbound traffic.
        a._platform_by_chat["777"] = "discord"
        await a.send("777", "text")
        frame, _ = transport.frames[-1]
        assert "format_hints" not in (frame.get("metadata") or {}), (
            "Slack-primary adapter stamped Slack format hints on a Discord chat"
        )

    @pytest.mark.asyncio
    async def test_discord_primary_still_stamps_hints_for_slack_chat(self):
        """The inverse: a non-Slack primary must not suppress hints for a
        chat whose own (Slack) descriptor advertises the capability."""
        transport = self._two_platform_transport()
        a = _adapter(
            extra={"slack": {"rich_blocks": True}},
            descriptor=_descriptor(
                platform="discord", label="Discord",
                markdown_dialect="markdown", max_message_length=2000,
            ),
            transport=transport,
        )
        a._platform_by_chat["D01"] = "slack"
        await a.send("D01", "text")
        frame, _ = transport.frames[-1]
        hints = (frame.get("metadata") or {}).get("format_hints")
        assert hints == {"rich_blocks": True}, (
            "Discord-primary adapter suppressed hints for a capable Slack chat"
        )

    @pytest.mark.asyncio
    async def test_send_for_platform_stamps_hints(self):
        """REGRESSION: the scheduled/persisted-home lane (gateway/delivery.py
        → send_for_platform) never stamped hints at all — yet it is the cron
        delivery path, the flagship consumer of block formatting."""
        transport = self._two_platform_transport()
        a = _adapter(
            extra={"slack": {"rich_blocks": True, "markdown_blocks": True}},
            descriptor=_descriptor(supports_block_formatting=True),
            transport=transport,
        )
        # No inbound ever seen for this chat — the explicit platform routes it.
        await a.send_for_platform("slack", "C123", "| a |\n|---|")
        frame, _ = transport.frames[-1]
        hints = (frame.get("metadata") or {}).get("format_hints")
        assert hints == {"rich_blocks": True, "markdown_blocks": True}

    @pytest.mark.asyncio
    async def test_send_for_platform_no_hints_for_incapable_platform(self):
        """Explicit-platform sends to a platform whose descriptor does not
        advertise the bit stay clean, whatever the primary identity says."""
        transport = self._two_platform_transport()
        a = _adapter(
            extra={
                "slack": {"rich_blocks": True},
                "discord": {"rich_blocks": True},
            },
            descriptor=_descriptor(supports_block_formatting=True),
            transport=transport,
        )
        await a.send_for_platform("discord", "777", "text")
        frame, _ = transport.frames[-1]
        assert "format_hints" not in (frame.get("metadata") or {})
