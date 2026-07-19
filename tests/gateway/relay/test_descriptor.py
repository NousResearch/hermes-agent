"""Tests for the experimental CapabilityDescriptor (relay Phase 0, Task 0.2)."""

from gateway.relay.descriptor import CONTRACT_VERSION, CapabilityDescriptor


def _telegram_descriptor(**overrides) -> CapabilityDescriptor:
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
        platform_hint="You are on Telegram.",
        pii_safe=False,
    )
    base.update(overrides)
    return CapabilityDescriptor(**base)


def test_descriptor_roundtrips_json():
    d = _telegram_descriptor()
    assert CapabilityDescriptor.from_json(d.to_json()) == d


def test_descriptor_is_frozen():
    d = _telegram_descriptor()
    try:
        d.max_message_length = 1  # type: ignore[misc]
    except Exception as exc:  # FrozenInstanceError
        assert "cannot assign" in str(exc) or "frozen" in str(exc).lower()
    else:  # pragma: no cover
        raise AssertionError("descriptor should be immutable (frozen)")


def test_from_json_ignores_unknown_keys():
    """A newer connector may send fields this gateway doesn't know — those are
    dropped, not fatal (forward-compat during the experimental phase)."""
    d = _telegram_descriptor()
    raw = d.to_json()[:-1] + ', "future_field": "ignored"}'
    restored = CapabilityDescriptor.from_json(raw)
    assert restored == d


def test_from_json_normalizes_zero_max_message_length_to_default():
    """A connector may advertise max_message_length 0 ("no limit"). from_json
    must normalize it to the documented 4096 default so the receiver never
    carries a degenerate chunking bound into truncate_message()."""
    raw = (
        '{"contract_version": 1, "platform": "x", "label": "X", '
        '"max_message_length": 0, "supports_draft_streaming": false, '
        '"supports_edit": false, "supports_threads": false, '
        '"markdown_dialect": "plain", "len_unit": "chars"}'
    )
    d = CapabilityDescriptor.from_json(raw)
    assert d.max_message_length == 4096


def test_from_json_normalizes_negative_max_message_length_to_default():
    """A buggy/hostile connector sending a negative bound is normalized too."""
    raw = (
        '{"contract_version": 1, "platform": "x", "label": "X", '
        '"max_message_length": -5, "supports_draft_streaming": false, '
        '"supports_edit": false, "supports_threads": false, '
        '"markdown_dialect": "plain", "len_unit": "chars"}'
    )
    d = CapabilityDescriptor.from_json(raw)
    assert d.max_message_length == 4096


def test_from_json_keeps_a_real_positive_bound():
    """A normal positive bound is passed through unchanged."""
    d = CapabilityDescriptor.from_json(_telegram_descriptor(max_message_length=2000).to_json())
    assert d.max_message_length == 2000


def test_from_json_fills_optional_defaults():
    """Optional fields (emoji/platform_hint/pii_safe) fall back to defaults."""
    minimal = (
        '{"contract_version": 1, "platform": "x", "label": "X", '
        '"max_message_length": 2000, "supports_draft_streaming": false, '
        '"supports_edit": false, "supports_threads": false, '
        '"markdown_dialect": "plain", "len_unit": "chars"}'
    )
    d = CapabilityDescriptor.from_json(minimal)
    assert d.pii_safe is False
    assert d.platform_hint == ""
    assert d.emoji == "\U0001f50c"


def test_module_is_marked_experimental():
    import gateway.relay.descriptor as m

    assert "EXPERIMENTAL" in (m.__doc__ or "")
