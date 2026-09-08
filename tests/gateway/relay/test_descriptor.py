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


def test_descriptor_is_frozen():
    d = _telegram_descriptor()
    try:
        d.max_message_length = 1  # type: ignore[misc]
    except Exception as exc:  # FrozenInstanceError
        assert "cannot assign" in str(exc) or "frozen" in str(exc).lower()
    else:  # pragma: no cover
        raise AssertionError("descriptor should be immutable (frozen)")


def test_module_is_marked_experimental():
    import gateway.relay.descriptor as m

    assert "EXPERIMENTAL" in (m.__doc__ or "")


# ─────────────── supported_ops (op-level capability discovery, Phase 1) ───────────────


def test_supports_op_legacy_connector_assumes_legacy_set():
    """An empty supported_ops means the connector predates op discovery: the
    legacy four ops are assumed supported (old connectors keep working), while
    NEW ops are not (discovery semantics — never probe by trying)."""
    d = _telegram_descriptor()  # no supported_ops
    for op in ("send", "edit", "typing", "follow_up"):
        assert d.supports_op(op) is True, op
    assert d.supports_op("get_chat_info") is False


def test_from_json_normalizes_malformed_supported_ops():
    """Non-list shapes and non-string members degrade to the legacy fallback
    (empty tuple), never raise — malformed input can't break the handshake."""
    base = (
        '{"contract_version": 1, "platform": "x", "label": "X", '
        '"max_message_length": 2000, "supports_draft_streaming": false, '
        '"supports_edit": true, "supports_threads": false, '
        '"markdown_dialect": "plain", "len_unit": "chars", '
    )
    d = CapabilityDescriptor.from_json(base + '"supported_ops": "send"}')
    assert d.supported_ops == ()
    d = CapabilityDescriptor.from_json(base + '"supported_ops": ["send", 7, null, ""]}')
    assert d.supported_ops == ("send",)
