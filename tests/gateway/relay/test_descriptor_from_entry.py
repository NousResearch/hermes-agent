"""Descriptor <- PlatformEntry projection (relay Phase 0, Task 0.3).

Proves the CapabilityDescriptor is a projection of the existing PlatformEntry,
not a parallel concept: the entry's label/limit/emoji/hint/pii fields carry
straight through.
"""

from gateway.platform_registry import PlatformEntry
from gateway.relay.descriptor import CONTRACT_VERSION, CapabilityDescriptor


def _entry(**overrides) -> PlatformEntry:
    base = dict(
        name="telegram",
        label="Telegram",
        adapter_factory=lambda cfg: None,
        check_fn=lambda: True,
        max_message_length=4096,
        pii_safe=False,
        emoji="\u2708\ufe0f",
        platform_hint="You are on Telegram.",
    )
    base.update(overrides)
    return PlatformEntry(**base)


def test_projection_carries_platform_entry_fields():
    d = CapabilityDescriptor.from_platform_entry(_entry(), len_unit="utf16")
    assert d.contract_version == CONTRACT_VERSION
    assert d.platform == "telegram"
    assert d.label == "Telegram"
    assert d.max_message_length == 4096
    assert d.emoji == "\u2708\ufe0f"
    assert d.platform_hint == "You are on Telegram."
    assert d.pii_safe is False
    assert d.len_unit == "utf16"


def test_zero_max_length_maps_to_4096_default():
    """PlatformEntry.max_message_length == 0 means 'no limit'; the descriptor
    carries a concrete bound matching the stream_consumer default."""
    d = CapabilityDescriptor.from_platform_entry(_entry(max_message_length=0))
    assert d.max_message_length == 4096


def test_runtime_capabilities_supplied_by_caller():
    """PlatformEntry doesn't encode draft/edit/thread/markdown behavior — those
    come from the caller (the connector, reading the live adapter)."""
    d = CapabilityDescriptor.from_platform_entry(
        _entry(),
        supports_draft_streaming=True,
        supports_edit=False,
        supports_threads=True,
        markdown_dialect="discord",
    )
    assert d.supports_draft_streaming is True
    assert d.supports_edit is False
    assert d.supports_threads is True
    assert d.markdown_dialect == "discord"


def test_projection_roundtrips_through_json():
    d = CapabilityDescriptor.from_platform_entry(_entry(), len_unit="utf16")
    assert CapabilityDescriptor.from_json(d.to_json()) == d
