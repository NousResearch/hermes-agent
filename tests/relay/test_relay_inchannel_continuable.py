"""Relay lane parity: flat in_channel continuable cron surface (Coatue F1).

Field report 2026-08-18: on relay-fronted Slack, cron briefs ALWAYS deliver
into a dedicated thread — `cron_continuable_surface: in_channel` (the flat-DM
continuable surface, native Slack's documented shape) is inert on the relay
lane. Three gaps, each pinned here:

1. CapabilityDescriptor has no in_channel capability bit, so RelayAdapter
   inherits supports_inchannel_continuable=False from the base class and the
   scheduler fails safe to thread mode (D6 gate).
2. The scheduler reads the surface knob flat off pconfig.extra; on the relay
   lane pconfig is platforms.relay, whose documented Slack knobs live under
   extra.slack.* (reply_in_thread precedent) — so the operator has no working
   place to put the knob.
3. Descriptor renegotiation (_apply_descriptor) must preserve the new
   capability bit, same as supports_code_blocks.
"""

from types import SimpleNamespace

import pytest

from gateway.relay.descriptor import CapabilityDescriptor
from cron.scheduler import _resolve_cron_surface_mode


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


class TestDescriptorCapabilityBit:
    def test_defaults_false(self):
        d = _descriptor()
        assert d.supports_inchannel_continuable is False

    def test_from_json_reads_flag(self):
        import json

        payload = dict(
            contract_version=1, platform="slack", label="Slack",
            max_message_length=4000, supports_draft_streaming=False,
            supports_edit=True, supports_threads=True,
            markdown_dialect="mrkdwn", len_unit="chars",
            supports_inchannel_continuable=True,
        )
        d = CapabilityDescriptor.from_json(json.dumps(payload))
        assert d.supports_inchannel_continuable is True

    def test_from_json_missing_flag_defaults_false(self):
        """Older connector that never sends the field — legacy-safe."""
        import json

        payload = dict(
            contract_version=1, platform="slack", label="Slack",
            max_message_length=4000, supports_draft_streaming=False,
            supports_edit=True, supports_threads=True,
            markdown_dialect="mrkdwn", len_unit="chars",
        )
        d = CapabilityDescriptor.from_json(json.dumps(payload))
        assert d.supports_inchannel_continuable is False


class TestRelayAdapterCapabilityMapping:
    def _adapter(self, descriptor):
        from gateway.config import PlatformConfig
        from gateway.relay.adapter import RelayAdapter

        config = PlatformConfig(enabled=True, extra={})
        return RelayAdapter(config, descriptor)

    def test_adapter_maps_descriptor_flag_true(self):
        adapter = self._adapter(_descriptor(supports_inchannel_continuable=True))
        assert getattr(adapter, "supports_inchannel_continuable", False) is True

    def test_adapter_maps_descriptor_flag_false(self):
        adapter = self._adapter(_descriptor())
        assert getattr(adapter, "supports_inchannel_continuable", True) is False

    def test_renegotiation_updates_flag(self):
        """_apply_descriptor must carry the bit, like supports_code_blocks."""
        adapter = self._adapter(_descriptor())
        adapter._apply_descriptor(_descriptor(supports_inchannel_continuable=True))
        assert adapter.supports_inchannel_continuable is True


class TestPerPlatformCapability:
    """One RelayAdapter fronts N platforms; the D6 gate must read the
    DESTINATION platform's negotiated descriptor, not the primary identity's
    scalar — the connector advertises the bit per platform at handshake."""

    class _Transport:
        def __init__(self, by_platform):
            self._by_platform = by_platform
            self._identities = [(p, "hermes") for p in by_platform]

        def descriptor_for_platform(self, platform):
            return self._by_platform.get(platform)

    def _adapter(self, primary, by_platform):
        from gateway.config import PlatformConfig
        from gateway.relay.adapter import RelayAdapter

        config = PlatformConfig(enabled=True, extra={})
        return RelayAdapter(config, primary, transport=self._Transport(by_platform))

    def test_primary_true_does_not_leak_onto_other_platform(self):
        """Slack-primary (capable) + Discord fronted (not advertised):
        Discord must NOT inherit Slack's bit through the scalar."""
        slack = _descriptor(supports_inchannel_continuable=True)
        discord = _descriptor(platform="discord", label="Discord",
                              markdown_dialect="markdown")
        adapter = self._adapter(slack, {"slack": slack, "discord": discord})
        assert adapter.supports_inchannel_continuable_for_platform("slack") is True
        assert adapter.supports_inchannel_continuable_for_platform("discord") is False

    def test_nonprimary_advertised_bit_is_honored(self):
        """Discord-primary (not capable) + Slack fronted (advertised):
        Slack's own descriptor must win over the primary scalar False."""
        discord = _descriptor(platform="discord", label="Discord",
                              markdown_dialect="markdown")
        slack = _descriptor(supports_inchannel_continuable=True)
        adapter = self._adapter(discord, {"slack": slack, "discord": discord})
        assert adapter.supports_inchannel_continuable_for_platform("slack") is True
        assert adapter.supports_inchannel_continuable_for_platform("discord") is False

    def test_unknown_platform_falls_back_to_scalar(self):
        """A platform the transport has no descriptor for (or a transport
        predating descriptor_for_platform) keeps the scalar behavior."""
        slack = _descriptor(supports_inchannel_continuable=True)
        adapter = self._adapter(slack, {"slack": slack})
        assert adapter.supports_inchannel_continuable_for_platform("matrix") is True


class TestSurfaceKnobResolution:
    """_resolve_cron_surface_mode reads the knob from BOTH config shapes."""

    def test_native_flat_key(self):
        pconfig = SimpleNamespace(extra={"cron_continuable_surface": "in_channel"})
        assert _resolve_cron_surface_mode(pconfig, "slack") == "in_channel"

    def test_relay_slack_subblock(self):
        """The relay lane's documented shape: platforms.relay.extra.slack.*
        (same seam as reply_in_thread / dm_top_level_threads_as_sessions)."""
        pconfig = SimpleNamespace(
            extra={"slack": {"cron_continuable_surface": "in_channel"}}
        )
        assert _resolve_cron_surface_mode(pconfig, "slack") == "in_channel"

    def test_subblock_is_per_logical_platform(self):
        """A slack sub-block must not leak onto another fronted platform."""
        pconfig = SimpleNamespace(
            extra={"slack": {"cron_continuable_surface": "in_channel"}}
        )
        assert _resolve_cron_surface_mode(pconfig, "discord") == "thread"

    def test_default_thread(self):
        pconfig = SimpleNamespace(extra={})
        assert _resolve_cron_surface_mode(pconfig, "slack") == "thread"

    def test_subblock_wins_over_flat_key(self):
        """Sub-block precedence: the per-logical-platform sub-block is the
        documented relay shape and wins when both are present — matches
        _relay_slack_extra (sub-dict preferred over flat extra)."""
        pconfig = SimpleNamespace(
            extra={
                "cron_continuable_surface": "thread",
                "slack": {"cron_continuable_surface": "in_channel"},
            }
        )
        # Sub-block is the documented relay shape and wins when present —
        # matches _relay_slack_extra (sub-dict preferred over flat extra).
        assert _resolve_cron_surface_mode(pconfig, "slack") == "in_channel"

    def test_none_pconfig_defaults_thread(self):
        assert _resolve_cron_surface_mode(None, "slack") == "thread"
