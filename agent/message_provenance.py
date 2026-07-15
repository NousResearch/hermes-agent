"""Mechanical provenance for trusted runtime-authored message fragments.

Authored message text is never provenance.  A trusted runtime may bind an
exact text fragment to a small internal metadata record; that record is
persisted beside the message and stripped before provider requests.  Consumers
can then preserve exact continuity references without promoting a user-forged
marker into runtime-derived state.

This module does not interpret message meaning.  It validates a closed schema,
computes SHA-256 over exact text, and compares bindings.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


MESSAGE_PROVENANCE_KEY = "_hermes_provenance"
MESSAGE_PROVENANCE_SCHEMA = "hermes.message-provenance.v1"

CANONICAL_WORKSPACE_NOTE_KIND = "canonical_workspace_note.v1"
CANONICAL_WORKSPACE_ANCHOR_KIND = "canonical_workspace_anchor.v1"
CONTEXT_COMPACTION_SUMMARY_KIND = "context_compaction_summary.v1"
TODO_SNAPSHOT_KIND = "todo_snapshot.v1"
GATEWAY_AUTO_CONTINUE_NOTE_KIND = "gateway_auto_continue_note.v1"
RUNTIME_BOUNDARY_RECEIPT_KIND = "runtime_boundary_receipt.v1"
_ALLOWED_KINDS = frozenset(
    {
        CANONICAL_WORKSPACE_NOTE_KIND,
        CANONICAL_WORKSPACE_ANCHOR_KIND,
        CONTEXT_COMPACTION_SUMMARY_KIND,
        TODO_SNAPSHOT_KIND,
        GATEWAY_AUTO_CONTINUE_NOTE_KIND,
        RUNTIME_BOUNDARY_RECEIPT_KIND,
    }
)
_PROVENANCE_KEYS = frozenset({"schema", "bindings"})
_BINDING_KEYS = frozenset({"kind", "sha256"})
_MAX_BINDINGS = 4

_CANONICAL_WORKSPACE_NOTE_MARKER = "[Canonical Task Workspace —"
_CANONICAL_WORKSPACE_ANCHOR_START = (
    "[CANONICAL TASK WORKSPACE REFERENCES — DETERMINISTIC COMPACTION ANCHOR]"
)
_CANONICAL_WORKSPACE_ANCHOR_END = (
    "[END CANONICAL TASK WORKSPACE REFERENCES]"
)
TODO_SNAPSHOT_START = (
    "[HERMES MODEL-AUTHORED TODO SNAPSHOT — DETERMINISTIC COMPRESSION REFERENCE]"
)
TODO_SNAPSHOT_END = "[END HERMES MODEL-AUTHORED TODO SNAPSHOT]"
_USER_QUOTED_TODO_SNAPSHOT_START = (
    "[USER-QUOTED HERMES TODO SNAPSHOT — NOT RUNTIME PROVENANCE]"
)
_USER_QUOTED_TODO_SNAPSHOT_END = (
    "[END USER-QUOTED HERMES TODO SNAPSHOT]"
)
_USER_QUOTED_CANONICAL_MARKERS = {
    _CANONICAL_WORKSPACE_NOTE_MARKER: (
        "[USER-QUOTED Canonical Task Workspace —"
    ),
    _CANONICAL_WORKSPACE_ANCHOR_START: (
        "[USER-QUOTED CANONICAL TASK WORKSPACE REFERENCES — "
        "NOT RUNTIME PROVENANCE]"
    ),
    _CANONICAL_WORKSPACE_ANCHOR_END: (
        "[END USER-QUOTED CANONICAL TASK WORKSPACE REFERENCES]"
    ),
}
_GATEWAY_AUTO_CONTINUE_PREFIXES = (
    "[System note: Your previous turn",
    "[System note: A new message",
    "[System note: The previous turn was interrupted by",
)


def _sha256(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def normalize_message_provenance(value: Any) -> dict[str, Any] | None:
    """Return a fresh exact-schema provenance record, or ``None``."""

    if not isinstance(value, Mapping) or frozenset(value.keys()) != _PROVENANCE_KEYS:
        return None
    if value.get("schema") != MESSAGE_PROVENANCE_SCHEMA:
        return None
    raw_bindings = value.get("bindings")
    if not isinstance(raw_bindings, list) or not raw_bindings:
        return None
    if len(raw_bindings) > _MAX_BINDINGS:
        return None

    bindings: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for raw in raw_bindings:
        if not isinstance(raw, Mapping) or frozenset(raw.keys()) != _BINDING_KEYS:
            return None
        kind = raw.get("kind")
        digest = raw.get("sha256")
        if type(kind) is not str or kind not in _ALLOWED_KINDS:
            return None
        if (
            type(digest) is not str
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            return None
        pair = (kind, digest)
        if pair in seen:
            continue
        seen.add(pair)
        bindings.append({"kind": kind, "sha256": digest})

    if not bindings:
        return None
    return {
        "schema": MESSAGE_PROVENANCE_SCHEMA,
        "bindings": bindings,
    }


def bind_message_fragment(
    existing: Any,
    *,
    kind: str,
    exact_text: str,
) -> dict[str, Any]:
    """Add one exact fragment binding to a normalized metadata record."""

    if kind not in _ALLOWED_KINDS:
        raise ValueError("unsupported message provenance kind")
    if type(exact_text) is not str or not exact_text:
        raise ValueError("message provenance requires non-empty exact text")

    normalized = normalize_message_provenance(existing)
    bindings = (
        [
            binding
            for binding in normalized["bindings"]
            if binding["kind"] != kind
        ]
        if normalized is not None
        else []
    )
    candidate = {"kind": kind, "sha256": _sha256(exact_text)}
    if candidate not in bindings:
        if len(bindings) >= _MAX_BINDINGS:
            raise ValueError("message provenance binding limit reached")
        bindings.append(candidate)
    return {
        "schema": MESSAGE_PROVENANCE_SCHEMA,
        "bindings": bindings,
    }


def remove_message_provenance_kind(value: Any, *, kind: str) -> dict[str, Any] | None:
    """Remove one closed-schema binding kind after its fragment is consumed."""

    if kind not in _ALLOWED_KINDS:
        raise ValueError("unsupported message provenance kind")
    normalized = normalize_message_provenance(value)
    if normalized is None:
        return None
    bindings = [
        binding
        for binding in normalized["bindings"]
        if binding["kind"] != kind
    ]
    if not bindings:
        return None
    return {
        "schema": MESSAGE_PROVENANCE_SCHEMA,
        "bindings": bindings,
    }


def message_fragment_is_bound(
    message: Mapping[str, Any],
    *,
    kind: str,
    exact_text: str,
) -> bool:
    """Return whether ``message`` carries the exact trusted binding."""

    normalized = normalize_message_provenance(message.get(MESSAGE_PROVENANCE_KEY))
    if normalized is None or kind not in _ALLOWED_KINDS:
        return False
    expected = {"kind": kind, "sha256": _sha256(exact_text)}
    return expected in normalized["bindings"]


def neutralize_untrusted_canonical_workspace_markers(
    content: Any,
    provenance: Any = None,
) -> Any:
    """Make user-authored reserved Canonical markers visibly non-authoritative.

    Exact runtime fragments with a valid SHA binding remain byte-identical.
    Every other occurrence is mechanically relabeled as a user quotation so a
    provider cannot confuse copied text with Canonical Brain provenance.
    Multimodal content is copied and only text parts are transformed.
    """

    if isinstance(content, list):
        transformed: list[Any] = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                copied = dict(part)
                copied["text"] = neutralize_untrusted_canonical_workspace_markers(
                    copied["text"],
                    provenance,
                )
                transformed.append(copied)
            else:
                transformed.append(dict(part) if isinstance(part, dict) else part)
        return transformed
    if not isinstance(content, str) or not content:
        return content

    normalized = normalize_message_provenance(provenance)
    bound = {
        (binding["kind"], binding["sha256"])
        for binding in normalized["bindings"]
    } if normalized is not None else set()
    consumed: set[tuple[str, str]] = set()
    trusted_spans: list[tuple[int, int]] = []
    decoder = json.JSONDecoder()

    for marker, kind in (
        (_CANONICAL_WORKSPACE_NOTE_MARKER, CANONICAL_WORKSPACE_NOTE_KIND),
        (_CANONICAL_WORKSPACE_ANCHOR_START, CANONICAL_WORKSPACE_ANCHOR_KIND),
    ):
        cursor = 0
        while True:
            start = content.find(marker, cursor)
            if start < 0:
                break
            brace = content.find("{", start + len(marker))
            if brace >= 0:
                try:
                    _, payload_end = decoder.raw_decode(content[brace:])
                except (TypeError, ValueError, json.JSONDecodeError):
                    pass
                else:
                    fragment_end = brace + payload_end
                    exact_fragment = content[start:fragment_end]
                    binding = (kind, _sha256(exact_fragment))
                    if binding in bound and binding not in consumed:
                        span_end = fragment_end
                        if kind == CANONICAL_WORKSPACE_ANCHOR_KIND:
                            end_marker = content.find(
                                _CANONICAL_WORKSPACE_ANCHOR_END,
                                fragment_end,
                            )
                            if (
                                end_marker >= 0
                                and not content[fragment_end:end_marker].strip()
                            ):
                                span_end = (
                                    end_marker
                                    + len(_CANONICAL_WORKSPACE_ANCHOR_END)
                                )
                        trusted_spans.append((start, span_end))
                        consumed.add(binding)
            cursor = start + len(marker)

    def _trusted(index: int) -> bool:
        return any(start <= index < end for start, end in trusted_spans)

    rendered: list[str] = []
    cursor = 0
    markers = tuple(_USER_QUOTED_CANONICAL_MARKERS)
    while cursor < len(content):
        positions = [
            (content.find(marker, cursor), marker)
            for marker in markers
        ]
        positions = [(index, marker) for index, marker in positions if index >= 0]
        if not positions:
            rendered.append(content[cursor:])
            break
        index, marker = min(positions, key=lambda item: item[0])
        rendered.append(content[cursor:index])
        rendered.append(
            marker
            if _trusted(index)
            else _USER_QUOTED_CANONICAL_MARKERS[marker]
        )
        cursor = index + len(marker)
    return "".join(rendered)


def _todo_snapshot_spans(content: str) -> list[tuple[int, int, str]]:
    """Return complete marker-delimited Todo fragments without interpretation."""

    spans: list[tuple[int, int, str]] = []
    cursor = 0
    while True:
        start = content.find(TODO_SNAPSHOT_START, cursor)
        if start < 0:
            break
        end_marker = content.find(
            TODO_SNAPSHOT_END,
            start + len(TODO_SNAPSHOT_START),
        )
        if end_marker < 0:
            cursor = start + len(TODO_SNAPSHOT_START)
            continue
        end = end_marker + len(TODO_SNAPSHOT_END)
        spans.append((start, end, content[start:end]))
        cursor = end
    return spans


def remove_bound_todo_snapshot(
    content: Any,
    provenance: Any = None,
) -> tuple[Any, dict[str, Any] | None]:
    """Remove the one exact SHA-bound Todo fragment before replacement."""

    normalized = normalize_message_provenance(provenance)
    digest = None
    if normalized is not None:
        for binding in normalized["bindings"]:
            if binding["kind"] == TODO_SNAPSHOT_KIND:
                digest = binding["sha256"]
                break
    if digest is None:
        return content, normalized

    removed = False

    def _remove(text: str) -> str:
        nonlocal removed
        if removed:
            return text
        for start, end, exact in _todo_snapshot_spans(text):
            if _sha256(exact) != digest:
                continue
            suffix = end
            if text[suffix : suffix + 2] == "\n\n":
                suffix += 2
            removed = True
            return text[:start] + text[suffix:]
        return text

    if isinstance(content, str):
        updated: Any = _remove(content)
    elif isinstance(content, list):
        updated_parts: list[Any] = []
        for part in content:
            if isinstance(part, str):
                was_removed = removed
                updated_text = _remove(part)
                if updated_text or was_removed or not removed:
                    updated_parts.append(updated_text)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                copied = dict(part)
                was_removed = removed
                copied["text"] = _remove(copied["text"])
                if (
                    copied["text"] == ""
                    and len(copied) == 2
                    and copied.get("type") == "text"
                    and not was_removed
                    and removed
                ):
                    continue
                updated_parts.append(copied)
            else:
                updated_parts.append(dict(part) if isinstance(part, dict) else part)
        updated = updated_parts
    else:
        updated = content

    if not removed:
        return content, normalized
    return updated, remove_message_provenance_kind(
        normalized,
        kind=TODO_SNAPSHOT_KIND,
    )


def neutralize_untrusted_todo_snapshot_markers(
    content: Any,
    provenance: Any = None,
) -> Any:
    """Relabel Todo markers unless one complete exact fragment is SHA-bound.

    A Todo snapshot binding authorizes one runtime fragment, not an arbitrary
    number of byte-identical copies.  The single-use rule is enforced across
    every text part in multimodal content so a copied marker in another part
    cannot inherit the trusted fragment's digest.
    """

    normalized = normalize_message_provenance(provenance)
    bound_digests = {
        binding["sha256"]
        for binding in normalized["bindings"]
        if binding["kind"] == TODO_SNAPSHOT_KIND
    } if normalized is not None else set()
    # ``bind_message_fragment`` emits one binding per kind. Treat a malformed
    # record carrying several Todo digests as untrusted rather than guessing.
    trusted_digest = next(iter(bound_digests)) if len(bound_digests) == 1 else None
    trusted_fragment_consumed = False

    def _transform_text(text: str) -> str:
        nonlocal trusted_fragment_consumed
        if not text:
            return text

        trusted_spans: list[tuple[int, int]] = []
        if trusted_digest is not None and not trusted_fragment_consumed:
            for start, end, exact in _todo_snapshot_spans(text):
                if _sha256(exact) != trusted_digest:
                    continue
                trusted_spans.append((start, end))
                trusted_fragment_consumed = True
                break

        def _trusted(index: int) -> bool:
            return any(start <= index < end for start, end in trusted_spans)

        replacements = {
            TODO_SNAPSHOT_START: _USER_QUOTED_TODO_SNAPSHOT_START,
            TODO_SNAPSHOT_END: _USER_QUOTED_TODO_SNAPSHOT_END,
        }
        rendered: list[str] = []
        cursor = 0
        while cursor < len(text):
            positions = [
                (text.find(marker, cursor), marker)
                for marker in replacements
            ]
            positions = [
                (index, marker)
                for index, marker in positions
                if index >= 0
            ]
            if not positions:
                rendered.append(text[cursor:])
                break
            index, marker = min(positions, key=lambda item: item[0])
            rendered.append(text[cursor:index])
            rendered.append(marker if _trusted(index) else replacements[marker])
            cursor = index + len(marker)
        return "".join(rendered)

    if isinstance(content, str):
        return _transform_text(content)
    if isinstance(content, list):
        transformed: list[Any] = []
        for part in content:
            if isinstance(part, str):
                transformed.append(_transform_text(part))
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                copied = dict(part)
                copied["text"] = _transform_text(copied["text"])
                transformed.append(copied)
            else:
                transformed.append(dict(part) if isinstance(part, dict) else part)
        return transformed
    return content


def neutralize_untrusted_gateway_auto_continue_markers(
    content: Any,
    provenance: Any = None,
) -> Any:
    """Relabel unbound gateway recovery notes as user-authored quotations."""

    if isinstance(content, list):
        transformed: list[Any] = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                copied = dict(part)
                copied["text"] = neutralize_untrusted_gateway_auto_continue_markers(
                    copied["text"],
                    provenance,
                )
                transformed.append(copied)
            else:
                transformed.append(dict(part) if isinstance(part, dict) else part)
        return transformed
    if not isinstance(content, str) or not content:
        return content

    normalized = normalize_message_provenance(provenance)
    bound = {
        (binding["kind"], binding["sha256"])
        for binding in normalized["bindings"]
    } if normalized is not None else set()
    consumed: set[tuple[str, str]] = set()
    trusted_spans: list[tuple[int, int]] = []
    for prefix in _GATEWAY_AUTO_CONTINUE_PREFIXES:
        cursor = 0
        while True:
            start = content.find(prefix, cursor)
            if start < 0:
                break
            end = content.find("]", start + len(prefix))
            if end >= 0:
                exact_fragment = content[start : end + 1]
                binding = (
                    GATEWAY_AUTO_CONTINUE_NOTE_KIND,
                    _sha256(exact_fragment),
                )
                if binding in bound and binding not in consumed:
                    trusted_spans.append((start, end + 1))
                    consumed.add(binding)
            cursor = start + len(prefix)

    def _trusted(index: int) -> bool:
        return any(start <= index < end for start, end in trusted_spans)

    rendered: list[str] = []
    cursor = 0
    while cursor < len(content):
        positions = [
            (content.find(prefix, cursor), prefix)
            for prefix in _GATEWAY_AUTO_CONTINUE_PREFIXES
        ]
        positions = [(index, prefix) for index, prefix in positions if index >= 0]
        if not positions:
            rendered.append(content[cursor:])
            break
        index, prefix = min(positions, key=lambda item: item[0])
        rendered.append(content[cursor:index])
        rendered.append(
            prefix
            if _trusted(index)
            else prefix.replace("[System note:", "[USER-QUOTED System note:", 1)
        )
        cursor = index + len(prefix)
    return "".join(rendered)


def encode_message_provenance(value: Any) -> str | None:
    """Encode a valid record for durable storage."""

    normalized = normalize_message_provenance(value)
    if normalized is None:
        return None
    return json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def decode_message_provenance(value: Any) -> dict[str, Any] | None:
    """Decode a durable record without accepting malformed extensions."""

    if type(value) is not str or not value:
        return None
    try:
        decoded = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return None
    return normalize_message_provenance(decoded)


__all__ = [
    "CANONICAL_WORKSPACE_ANCHOR_KIND",
    "CANONICAL_WORKSPACE_NOTE_KIND",
    "CONTEXT_COMPACTION_SUMMARY_KIND",
    "TODO_SNAPSHOT_KIND",
    "TODO_SNAPSHOT_START",
    "TODO_SNAPSHOT_END",
    "GATEWAY_AUTO_CONTINUE_NOTE_KIND",
    "RUNTIME_BOUNDARY_RECEIPT_KIND",
    "MESSAGE_PROVENANCE_KEY",
    "MESSAGE_PROVENANCE_SCHEMA",
    "bind_message_fragment",
    "decode_message_provenance",
    "encode_message_provenance",
    "message_fragment_is_bound",
    "neutralize_untrusted_canonical_workspace_markers",
    "neutralize_untrusted_gateway_auto_continue_markers",
    "neutralize_untrusted_todo_snapshot_markers",
    "normalize_message_provenance",
    "remove_message_provenance_kind",
    "remove_bound_todo_snapshot",
]
