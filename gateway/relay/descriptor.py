"""CapabilityDescriptor — the relay handshake payload. EXPERIMENTAL.

The connector hands a ``CapabilityDescriptor`` to the gateway's ``RelayAdapter``
at handshake time; it tells the adapter which platform it is fronting and which
capabilities to advertise to the ``GatewayStreamConsumer`` (char limit,
draft-streaming, edit/threading support, markdown dialect, length unit). It is
the linchpin of the generalization: one gateway adapter serves Discord,
Telegram, Matrix, Signal, ... without per-platform branching.

EXPERIMENTAL: this schema MAY CHANGE without a deprecation cycle until at least
two real Class-1 platforms have validated it. Evolution during the experimental
phase is additive-only, gated by ``contract_version`` (see
docs/relay-connector-contract.md).

Field origins (most are a wire-serializable projection of ``PlatformEntry`` plus
the per-instance capability methods on ``BasePlatformAdapter``):

- ``max_message_length`` -> ``PlatformEntry.max_message_length`` / adapter
  ``MAX_MESSAGE_LENGTH`` attribute (read by stream_consumer).
- ``len_unit``           -> selects which ``message_len_fn`` the adapter installs
  ("chars" = builtin len; "utf16" = Telegram-style UTF-16 code-unit counting).
- ``supports_draft_streaming`` -> adapter ``supports_draft_streaming()`` probe.
- ``supports_edit``      -> whether edit-based streaming is possible (Discord/
  Telegram yes; Signal/SMS no -> consumer degrades to one-message-per-segment).
- ``supports_threads``   -> ``create_handoff_thread`` capability flag.
- ``markdown_dialect``   -> presentation hint (e.g. "markdown_v2", "discord").
- ``emoji`` / ``platform_hint`` / ``pii_safe`` -> ``PlatformEntry`` fields of the
  same name.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

# Bump additively (never reinterpret an existing field) during the experimental
# phase; a breaking change requires updating both repos in lockstep.
CONTRACT_VERSION = 1


@dataclass(frozen=True)
class CapabilityDescriptor:
    """Immutable capability descriptor negotiated at relay handshake.

    Frozen so a descriptor cannot be mutated after handshake — the adapter
    advertises a fixed capability profile for the life of the connection.
    """

    contract_version: int
    platform: str
    label: str
    max_message_length: int
    supports_draft_streaming: bool
    supports_edit: bool
    supports_threads: bool
    markdown_dialect: str
    len_unit: str  # "chars" | "utf16"
    emoji: str = "\U0001f50c"  # 🔌 default (matches PlatformEntry default)
    platform_hint: str = ""
    pii_safe: bool = False
    # Whether the connector can supply surrounding channel/group CONTEXT for an
    # addressed turn (Model A pull / Model B buffer, per platform). Optional +
    # defaults False so an older connector that never sends it is treated as
    # "no context" — additive within contract_version 1. from_json filters
    # unknown keys, so a connector sending this to an older gateway is safe too.
    supports_context: bool = False

    def to_json(self) -> str:
        """Serialize to a compact, stable JSON string for the handshake frame."""
        return json.dumps(asdict(self), sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_json(cls, data: str) -> "CapabilityDescriptor":
        """Deserialize from a handshake JSON string.

        Unknown keys are ignored (forward-compat: a newer connector may send
        fields this gateway does not know yet); missing optional keys fall back
        to dataclass defaults.
        """
        raw = json.loads(data)
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in raw.items() if k in known}
        return cls(**filtered)

    @classmethod
    def from_platform_entry(
        cls,
        entry,
        *,
        len_unit: str = "chars",
        supports_draft_streaming: bool = False,
        supports_edit: bool = True,
        supports_threads: bool = False,
        markdown_dialect: str = "plain",
    ) -> "CapabilityDescriptor":
        """Project a ``gateway.platform_registry.PlatformEntry`` into a descriptor.

        Demonstrates the descriptor is a *subset/projection* of what
        ``PlatformEntry`` already encodes, not a parallel concept: ``label``,
        ``max_message_length``, ``emoji``, ``platform_hint``, ``pii_safe`` and
        the platform name come straight off the entry. The runtime capability
        bits that ``PlatformEntry`` does NOT encode (length unit, draft/edit/
        thread/markdown behavior) are supplied by the caller — in production
        the connector fills these from the live adapter's capability methods.

        ``max_message_length`` of 0 on a ``PlatformEntry`` means "no limit";
        we map that to the stream_consumer default of 4096 so the descriptor
        always carries a concrete chunking bound.
        """
        max_len = getattr(entry, "max_message_length", 0) or 4096
        return cls(
            contract_version=CONTRACT_VERSION,
            platform=entry.name,
            label=entry.label,
            max_message_length=max_len,
            supports_draft_streaming=supports_draft_streaming,
            supports_edit=supports_edit,
            supports_threads=supports_threads,
            markdown_dialect=markdown_dialect,
            len_unit=len_unit,
            emoji=getattr(entry, "emoji", "\U0001f50c"),
            platform_hint=getattr(entry, "platform_hint", ""),
            pii_safe=getattr(entry, "pii_safe", False),
        )
