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
    def __init__(self):
        self.frames = []

    async def send_outbound(self, frame, platform=None):
        self.frames.append((frame, platform))
        return {"success": True, "message_id": "1.2"}


def _adapter(extra=None, descriptor=None):
    config = PlatformConfig(enabled=True, extra=extra or {})
    a = RelayAdapter(config, descriptor or _descriptor(), transport=FakeTransport())
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
