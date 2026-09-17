"""Validation, operation claims, and commit support for transcript sanitation."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import inspect
import json
import logging
import re
from typing import Any, Callable, Optional

from agent.message_sanitization import _sanitize_surrogates
from agent.model_metadata import (
    estimate_messages_tokens_rough,
    estimate_request_tokens_rough,
)
from agent.usage_anchor import set_usage_anchor

# Preserve the established compression lifecycle log stream after extraction.
logger = logging.getLogger("agent.conversation_compression")

_SANITATION_PLACEHOLDER_RE = re.compile(
    r"\[LCM sensitive redaction: name=([A-Za-z_][A-Za-z0-9_.-]{0,63}); "
    r"chars=([0-9]{1,9}); bytes=([0-9]{1,9})"
    r"(?:; sha256=([0-9a-f]{16}))?\]"
)
_LCM_EXTERNALIZED_TOOL_OUTPUT_RE = re.compile(
    r"\[Externalized tool output: "
    r"tool_call_id=([A-Za-z0-9_.:/?-]{1,120}); "
    r"chars=([0-9]{1,9}); bytes=([0-9]{1,9}); "
    r"ref=([A-Za-z0-9_.-]{1,255}\.json)\]"
)
_LCM_EXTERNALIZED_PAYLOAD_RE = re.compile(
    r"\[Externalized payload: kind=raw_payload; "
    r"role=(system|user|assistant|tool); "
    r"chars=([0-9]{1,9}); bytes=([0-9]{1,9}); "
    r"ref=([A-Za-z0-9_.-]{1,255}\.json)\]"
)
_LCM_REF_DIGEST_RE = re.compile(r"_([0-9a-f]{12})_[A-Za-z0-9]+\.json$")

_MESSAGE_PAYLOAD_FIELDS = frozenset(
    {
        "content",
        "api_content",
        "reasoning",
        "reasoning_content",
    }
)
_REPLAY_ENVELOPE_FIELDS = frozenset(
    {
        "reasoning_details",
        "codex_reasoning_items",
        "codex_message_items",
        "anthropic_content_blocks",
        "bedrock_content_blocks",
    }
)

# Keys a provider reads to interpret a content block structurally (request shape), never as user
# content. Redacting one (e.g. ``detail`` -> a redaction placeholder) passes this module's declared-
# growth validation but the provider then hard-rejects the request. Only content-bearing fields may
# carry placeholders; control keys stay verbatim (or refuse the candidate).
# Envelope FIELD names (``image_url``, ``file``) are request-shape keys whose nested values may be
# EITHER control (``detail``, ``media_type``) or payload (``url``, ``file_data``): routing the whole
# envelope here verbatim would make a secret inside a signed URL or inline attachment permanently
# unsanitizable, so the envelope re-classifies its nested keys per key. (Anthropic-style ``source``
# stays fully structural via ``_CONTENT_PART_TYPE_FIELDS`` — its media-type shell is request shape.)
_PAYLOAD_BEARING_ENVELOPE_FIELDS = frozenset({"image_url", "file"})
_PROVIDER_CONTROL_FIELDS = frozenset(
    {
        "detail",
        "cache_control",
        "id",
        "name",
        "tool_call_id",
        "tool_use_id",
        "call_id",
        "media_type",
    }
)
_CONTENT_PART_TYPE_FIELDS = frozenset({"type", "source", "media_type"})


@dataclasses.dataclass(frozen=True)
class SanitationChanges:
    changed_fields: int = 0
    placeholders: int = 0
    declared_growth_chars: int = 0

    def plus(self, other: "SanitationChanges") -> "SanitationChanges":
        return SanitationChanges(
            self.changed_fields + other.changed_fields,
            self.placeholders + other.placeholders,
            self.declared_growth_chars + other.declared_growth_chars,
        )


@dataclasses.dataclass(frozen=True)
class PreparedCompressionOperation:
    operation: str
    claim: Any
    session_id: str | None
    attempt_generation: int


@dataclasses.dataclass(frozen=True)
class SanitationCommitPlan:
    original: list
    candidate: list
    input_tokens: int
    output_tokens: int
    changes: Optional[SanitationChanges]
    watermark: Optional[int]
    represented_row_ids: tuple[int, ...]
    member_row_ids: Optional[tuple[tuple[int, ...], ...]] = None


@dataclasses.dataclass(frozen=True)
class SanitationRetryCandidate:
    session_id: str
    original: list
    candidate: list


def _strip_persistence_marker(messages: Any) -> Any:
    from agent.context_compressor import _DB_PERSISTED_MARKER

    if not isinstance(messages, list):
        return messages
    # ``_merged_row_ids`` is repair-provenance bookkeeping (``_rows_to_conversation``), never
    # content: an engine that drops it must not fail structural validation, and the retry
    # transcript comparison must not treat it as part of the content shape.
    # ``_row_id`` is the same class of host bookkeeping (Bugbot 4041436129): the production
    # loaders now stamp it on every resume/replay transcript, and an engine candidate that
    # rebuilds message dicts without echoing it must not fail structural validation or lose
    # the retained retry to a metadata-only diff.
    return [
        (
            {
                key: value
                for key, value in message.items()
                if key not in (_DB_PERSISTED_MARKER, "_merged_row_ids", "_row_id")
            }
            if isinstance(message, dict)
            else message
        )
        for message in messages
    ]


def sanitation_rough_tokens(messages: list) -> int:
    """Estimate sanitation size without host-only persistence stamps."""
    return estimate_messages_tokens_rough(_strip_persistence_marker(messages))


def sanitation_snapshot_watermark(messages: Any) -> Optional[int]:
    """Return the highest durable row represented by a sanitation snapshot."""
    row_ids = sanitation_snapshot_row_ids(messages)
    return max(row_ids) if row_ids else None


def sanitation_snapshot_row_ids(messages: Any) -> tuple[int, ...]:
    """Return the durable rows represented by a sanitation snapshot."""
    if not isinstance(messages, list):
        return ()
    return tuple(
        message.get("_row_id")
        for message in messages
        if isinstance(message, dict)
        and isinstance(message.get("_row_id"), int)
        and not isinstance(message.get("_row_id"), bool)
        and message["_row_id"] > 0
    )


def sanitation_snapshot_member_row_ids(messages: Any) -> tuple[tuple[int, ...], ...]:
    """Row ids each message of a snapshot REPRESENTS, positionally.

    Durably REPAIRED snapshots (``get_messages_as_conversation(repair_alternation=True)``) can hold
    one message that merged two source rows (repair_message_sequence), so a single ``_row_id`` per
    position under-represents membership: the omitted source row would be re-cloned byte-exact
    (secret intact, FTS re-indexed) by the absent-from-snapshot branch. Extra ``_row_ids`` live in
    a ``_merged_row_ids`` list; without any, the message represents exactly its own id.
    """
    if not isinstance(messages, list):
        return ()
    return tuple(
        (
            *(row_id for row_id in [message.get("_row_id")] if isinstance(row_id, int) and not isinstance(row_id, bool) and row_id > 0),
            *(extra_id for extra_id in (message.get("_merged_row_ids") or ()) if isinstance(extra_id, int) and not isinstance(extra_id, bool) and extra_id > 0),
        )
        if isinstance(message, dict)
        else ()
        for message in messages
    )


def _normalized_text(value: str) -> str:
    return _sanitize_surrogates(value)


def _externalized_ref_digest(ref: str) -> Optional[str]:
    match = _LCM_REF_DIGEST_RE.search(ref)
    return match.group(1) if match is not None else None


def _ref_matches_observable_content(ref: str, content: str) -> bool:
    digest = _externalized_ref_digest(ref)
    return (
        digest is not None
        and hashlib.sha256(content.encode("utf-8")).hexdigest()[:12] == digest
    )


def _validate_externalized_string(
    original: str,
    candidate: str,
    *,
    role: str,
    tool_call_id: str | None,
) -> Optional[SanitationChanges]:
    tool_match = (
        _LCM_EXTERNALIZED_TOOL_OUTPUT_RE.fullmatch(candidate)
        if role == "tool"
        else None
    )
    payload_match = _LCM_EXTERNALIZED_PAYLOAD_RE.fullmatch(candidate)
    if tool_match is not None:
        marker_tool_id, chars_raw, bytes_raw, ref = tool_match.groups()
        if marker_tool_id != tool_call_id:
            return None
    elif payload_match is not None and payload_match.group(1) == role:
        _, chars_raw, bytes_raw, ref = payload_match.groups()
    else:
        return None

    normalized = _normalized_text(original)
    declared_chars = int(chars_raw)
    declared_bytes = int(bytes_raw)
    original_bytes = len(normalized.encode("utf-8"))
    exact_source = declared_chars == len(normalized) and declared_bytes == original_bytes
    if not (exact_source and _ref_matches_observable_content(ref, normalized)):
        return None
    return SanitationChanges(
        changed_fields=1,
        placeholders=1,
        declared_growth_chars=len(candidate) - len(original),
    )


def _validate_structured_externalization(
    original: Any,
    candidate: str,
    *,
    role: str,
) -> Optional[SanitationChanges]:
    """Opaque structured externalization is unverifiable without its sidecar."""
    return None


def _validate_sanitized_string(
    original: str,
    candidate: str,
    *,
    externalized_role: str | None = None,
    externalized_tool_call_id: str | None = None,
    allow_json_normalization: bool = False,
) -> Optional[SanitationChanges]:
    if candidate == original:
        return SanitationChanges()
    if externalized_role is not None:
        externalized = _validate_externalized_string(
            original,
            candidate,
            role=externalized_role,
            tool_call_id=externalized_tool_call_id,
        )
        if externalized is not None:
            return externalized

    placeholders = list(_SANITATION_PLACEHOLDER_RE.finditer(candidate))
    if not placeholders:
        return None
    if allow_json_normalization:
        try:
            original_json = json.loads(original)
            candidate_json = json.loads(candidate)
        except (TypeError, ValueError):
            pass
        else:
            json_changes = _validate_sanitized_value(
                original_json,
                candidate_json,
                context="payload",
            )
            if json_changes is not None and json_changes.placeholders:
                return dataclasses.replace(
                    json_changes,
                    declared_growth_chars=len(candidate) - len(original),
                )

    pattern_parts: list[str] = []
    cursor = 0
    for index, match in enumerate(placeholders):
        chars = int(match.group(2))
        if chars < 1 or chars > len(original):
            return None
        pattern_parts.append(re.escape(candidate[cursor : match.start()]))
        pattern_parts.append(fr"(?P<s{index}>[\s\S]{{{chars}}})")
        cursor = match.end()
    pattern_parts.append(re.escape(candidate[cursor:]))
    original_match = re.fullmatch("".join(pattern_parts), original)
    if original_match is None:
        return None
    for index, placeholder in enumerate(placeholders):
        secret = _normalized_text(original_match.group(f"s{index}"))
        if len(secret.encode("utf-8")) != int(placeholder.group(3)):
            return None
        digest = placeholder.group(4)
        if digest and hashlib.sha256(secret.encode("utf-8")).hexdigest()[:16] != digest:
            return None
    return SanitationChanges(
        changed_fields=1,
        placeholders=len(placeholders),
        declared_growth_chars=len(candidate) - len(original),
    )


def _dict_context(context: str, original: dict) -> str:
    if context != "payload":
        return context
    return "content_part" if "type" in original else "payload"


def _same_typed_value(original: Any, candidate: Any) -> bool:
    if type(original) is not type(candidate):
        return False
    if isinstance(original, dict):
        if len(original) != len(candidate):
            return False
        for key, value in original.items():
            matches = [
                candidate_key
                for candidate_key in candidate
                if type(candidate_key) is type(key) and candidate_key == key
            ]
            if len(matches) != 1 or not _same_typed_value(
                value, candidate[matches[0]]
            ):
                return False
        return True
    if isinstance(original, (list, tuple)):
        return len(original) == len(candidate) and all(
            _same_typed_value(left, right)
            for left, right in zip(original, candidate)
        )
    return original == candidate


def _child_context(context: str, key: Any) -> tuple[str, bool]:
    if context == "message":
        if key in _MESSAGE_PAYLOAD_FIELDS:
            return ("payload", key == "content")
        if key == "tool_calls":
            return ("tool_calls", False)
        if key in _REPLAY_ENVELOPE_FIELDS:
            return ("replay_envelope", False)
        return ("structural", False)
    if context == "tool_calls":
        return ("function" if key == "function" else "structural", False)
    if context == "function":
        return (
            ("payload", False)
            if key == "arguments"
            else ("structural", False)
        )
    if context == "content_part":
        # Multimodal-block routing (round-6 multimodal finding): provider-control keys ride along
        # inside content lists; redacting one (``detail`` -> placeholder) passes declared-growth
        # validation but the provider then rejects the whole request. ``type``/media-type fields
        # are structural; identity/behavior fields are control (verbatim); everything else is
        # content and may carry placeholders.
        if key in _CONTENT_PART_TYPE_FIELDS:
            return "structural", False
        if key in _PROVIDER_CONTROL_FIELDS:
            return "control", False
        if key in _PAYLOAD_BEARING_ENVELOPE_FIELDS:
            # Round-7 finding: an envelope like ``image_url``/``file`` holds BOTH request shape
            # (``detail``, ``media_type``) and payload (``url``, ``file_data``, ``data``). Route
            # it as content_part so its nested keys re-classify per key: a secret inside the
            # payload stays sanitizable while the nested control keys stay verbatim.
            return "content_part", False
        return "payload", False
    if context in {"payload", "content_part"}:
        return "payload", False
    return "structural", False


def _validate_sanitized_value(
    original: Any,
    candidate: Any,
    *,
    context: str = "structural",
    externalized_role: str | None = None,
    externalized_tool_call_id: str | None = None,
    allow_structured_externalization: bool = False,
    allow_json_normalization: bool = False,
) -> Optional[SanitationChanges]:
    if context == "replay_envelope":
        return SanitationChanges() if _same_typed_value(original, candidate) else None
    if context == "control":
        # Provider-control field (round-6 multimodal finding): the provider reads this verbatim to
        # shape the request. Placeholders here are undetectable at validation time (they redact
        # nothing secret) but break the request at replay — require byte-identical values.
        return SanitationChanges() if _same_typed_value(original, candidate) else None
    if (
        allow_structured_externalization
        and not isinstance(original, str)
        and isinstance(candidate, str)
        and externalized_role is not None
    ):
        return _validate_structured_externalization(
            original,
            candidate,
            role=externalized_role,
        )
    if isinstance(original, str) and isinstance(candidate, str):
        if context in {"structural", "control"}:
            return SanitationChanges() if original == candidate else None
        return _validate_sanitized_string(
            original,
            candidate,
            externalized_role=externalized_role,
            externalized_tool_call_id=externalized_tool_call_id,
            allow_json_normalization=allow_json_normalization,
        )
    if isinstance(original, list) and isinstance(candidate, list):
        if len(original) != len(candidate):
            return None
        changes = SanitationChanges()
        item_context = "message" if context == "messages" else (
            "tool_calls" if context == "tool_calls" else context
        )
        for original_item, candidate_item in zip(original, candidate):
            item_changes = _validate_sanitized_value(
                original_item,
                candidate_item,
                context=item_context,
                externalized_role=externalized_role,
                externalized_tool_call_id=externalized_tool_call_id,
            )
            if item_changes is None:
                return None
            changes = changes.plus(item_changes)
        return changes
    if isinstance(original, dict) and isinstance(candidate, dict):
        if len(original) != len(candidate):
            return None
        changes = SanitationChanges()
        context = _dict_context(context, original)
        if context == "control":
            # Control fields (``cache_control``, ``image_url`` envelopes, tool identity) stay
            # verbatim: no key redaction, no nested placeholder (round-6 multimodal finding).
            return SanitationChanges() if _same_typed_value(original, candidate) else None
        available_candidate_keys = list(candidate)
        matched_keys: list[tuple[Any, Any]] = []
        unmatched_original = []
        for original_key in original:
            exact_matches = [
                (index, candidate_key)
                for index, candidate_key in enumerate(available_candidate_keys)
                if type(candidate_key) is type(original_key)
                and candidate_key == original_key
            ]
            if len(exact_matches) != 1:
                unmatched_original.append(original_key)
                continue
            candidate_index, candidate_key = exact_matches[0]
            matched_keys.append((original_key, candidate_key))
            available_candidate_keys.pop(candidate_index)
        unmatched_candidate = available_candidate_keys
        for original_key, candidate_key in matched_keys:
            child_context, structured_externalization = _child_context(
                context,
                original_key,
            )
            if child_context == "control" and candidate_key != original_key:
                # Round-6 multimodal finding: control keys are request shape. A redacted key
                # (``detail`` -> a placeholder) survives declared-growth validation but the
                # provider then rejects the whole request — refuse instead.
                return None
            role = (
                str(original.get("role"))
                if (
                    context == "message"
                    and original_key == "content"
                    and original.get("role") == candidate.get("role")
                    and original.get("role")
                    in {"system", "user", "assistant", "tool"}
                )
                else externalized_role
            )
            tool_call_id = (
                str(original.get("tool_call_id"))
                if (
                    context == "message"
                    and original_key == "content"
                    and original.get("role") == candidate.get("role") == "tool"
                    and isinstance(original.get("tool_call_id"), str)
                )
                else externalized_tool_call_id
            )
            value_changes = _validate_sanitized_value(
                original[original_key],
                candidate[candidate_key],
                context=child_context,
                externalized_role=role,
                externalized_tool_call_id=tool_call_id,
                allow_structured_externalization=structured_externalization,
                allow_json_normalization=(
                    context == "function" and original_key == "arguments"
                ),
            )
            if value_changes is None:
                return None
            changes = changes.plus(value_changes)
        if len(unmatched_original) != len(unmatched_candidate):
            return None
        if unmatched_original and context not in {"payload", "content_part"}:
            return None
        # Round-6 multimodal finding: in a content part a redacted key is indistinguishable from
        # an added control key ("detail" -> placeholder reads as a NEW control field). Only
        # content-bearing (payload) redactions may rename keys here.
        if unmatched_candidate and context == "content_part":
            return None
        available = list(unmatched_candidate)
        for original_key in unmatched_original:
            if not isinstance(original_key, str):
                return None
            matches: list[tuple[int, SanitationChanges, SanitationChanges]] = []
            for candidate_index, candidate_key in enumerate(available):
                if not isinstance(candidate_key, str):
                    continue
                key_changes = _validate_sanitized_string(original_key, candidate_key)
                if key_changes is None:
                    continue
                value_changes = _validate_sanitized_value(
                    original[original_key],
                    candidate[candidate_key],
                    context="payload",
                    externalized_role=externalized_role,
                    externalized_tool_call_id=externalized_tool_call_id,
                )
                if value_changes is not None:
                    matches.append((candidate_index, key_changes, value_changes))
            if len(matches) != 1:
                return None
            candidate_index, key_changes, value_changes = matches[0]
            changes = changes.plus(key_changes).plus(value_changes)
            available.pop(candidate_index)
        return changes
    return (
        SanitationChanges()
        if type(original) is type(candidate) and original == candidate
        else None
    )


def _drop_stale_api_content(original: Any, candidate: Any) -> None:
    if not isinstance(original, list) or not isinstance(candidate, list):
        return
    for original_message, candidate_message in zip(original, candidate):
        if not isinstance(original_message, dict) or not isinstance(candidate_message, dict):
            continue
        if (
            original_message.get("role") == candidate_message.get("role")
            and original_message.get("content") != candidate_message.get("content")
        ):
            candidate_message.pop("api_content", None)


def _verified_externalized_content(
    marker: str,
    *,
    role: str,
    tool_call_id: str | None,
    loader: Callable[[str], Any],
) -> Any:
    tool_match = (
        _LCM_EXTERNALIZED_TOOL_OUTPUT_RE.fullmatch(marker)
        if role == "tool"
        else None
    )
    payload_match = _LCM_EXTERNALIZED_PAYLOAD_RE.fullmatch(marker)
    if tool_match is not None:
        marker_tool_id, chars_raw, bytes_raw, ref = tool_match.groups()
        if marker_tool_id != tool_call_id:
            return None
        expected_kind = "tool_result"
    elif payload_match is not None and payload_match.group(1) == role:
        _, chars_raw, bytes_raw, ref = payload_match.groups()
        expected_kind = "raw_payload"
    else:
        return None
    try:
        payload = loader(ref)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    content = payload.get("content")
    if not isinstance(content, str):
        return None
    normalized = _normalized_text(content)
    chars = len(normalized)
    byte_count = len(normalized.encode("utf-8"))
    if (
        payload.get("kind", "tool_result") != expected_kind
        or int(chars_raw) != chars
        or int(bytes_raw) != byte_count
        or payload.get("content_chars", chars) != chars
        or payload.get("content_bytes", byte_count) != byte_count
        or not _ref_matches_observable_content(ref, normalized)
    ):
        return None
    if expected_kind == "tool_result":
        if payload.get("tool_call_id", "") != (tool_call_id or ""):
            return None
    elif payload.get("role", "") != role:
        return None
    return normalized


def _is_externalization_marker(candidate: str, *, role: str) -> bool:
    if role == "tool" and _LCM_EXTERNALIZED_TOOL_OUTPUT_RE.fullmatch(candidate):
        return True
    return _LCM_EXTERNALIZED_PAYLOAD_RE.fullmatch(candidate) is not None


def _nested_externalization_markers_verified(
    original_content: Any,
    candidate_content: Any,
    *,
    role: str,
    tool_call_id: str | None,
    loader: Callable[[str], Any] | None,
) -> bool:
    """Verify every externalization marker nested under message content.

    Walks the candidate payload tree (content parts, lists, nested dicts),
    pairing each node with the original node at the same position when the
    shapes line up. Markers are validated through the same sidecar contract as
    top-level markers: a missing, malformed, or identity-mismatched sidecar
    returns False so the sanitation commit fails closed instead of dropping
    the last recoverable copy of the payload. A marker byte-identical to the
    original was verified when it committed and passes through without
    re-verification. Non-marker values pass through for structural validation
    to judge.
    """
    if isinstance(candidate_content, str):
        if not _is_externalization_marker(candidate_content, role=role):
            return True
        if candidate_content == original_content:
            return True
        if loader is None:
            return False
        return (
            _verified_externalized_content(
                candidate_content,
                role=role,
                tool_call_id=tool_call_id,
                loader=loader,
            )
            is not None
        )
    if isinstance(candidate_content, dict):
        original_map = original_content if isinstance(original_content, dict) else {}
        return all(
            _nested_externalization_markers_verified(
                original_map.get(key),
                value,
                role=role,
                tool_call_id=tool_call_id,
                loader=loader,
            )
            for key, value in candidate_content.items()
        )
    if isinstance(candidate_content, list):
        original_items = original_content if isinstance(original_content, list) else []
        return all(
            _nested_externalization_markers_verified(
                original_items[index] if index < len(original_items) else None,
                value,
                role=role,
                tool_call_id=tool_call_id,
                loader=loader,
            )
            for index, value in enumerate(candidate_content)
        )
    return True


def _validation_views(
    original: list,
    candidate: list,
    externalized_payload_loader: Callable[[str], Any] | None = None,
) -> tuple[list, list, int, bool]:
    original_view = copy.deepcopy(_strip_persistence_marker(original))
    candidate_view = copy.deepcopy(_strip_persistence_marker(candidate))
    verified_externalizations = 0
    failed_externalization_verification = False
    if externalized_payload_loader is None:
        for original_message, candidate_message in zip(original_view, candidate_view):
            if not isinstance(candidate_message, dict):
                continue
            role = str(candidate_message.get("role", ""))
            original_content = (
                original_message.get("content")
                if isinstance(original_message, dict)
                else None
            )
            content = candidate_message.get("content")
            if isinstance(content, str) and _is_externalization_marker(
                content,
                role=role,
            ):
                # An unchanged marker was verified when it committed; only a
                # new marker is unverifiable without the sidecar contract.
                if content != original_content:
                    failed_externalization_verification = True
                    break
                continue
            if not _nested_externalization_markers_verified(
                original_content,
                content,
                role=role,
                tool_call_id=None,
                loader=None,
            ):
                failed_externalization_verification = True
                break
    for original_message, candidate_message in zip(original_view, candidate_view):
        if (
            externalized_payload_loader is not None
            and isinstance(original_message, dict)
            and isinstance(candidate_message, dict)
            and isinstance(candidate_message.get("content"), str)
            and original_message.get("role") == candidate_message.get("role")
        ):
            role = str(original_message.get("role", ""))
            verified = _verified_externalized_content(
                candidate_message["content"],
                role=role,
                tool_call_id=(
                    str(original_message.get("tool_call_id"))
                    if isinstance(original_message.get("tool_call_id"), str)
                    else None
                ),
                loader=externalized_payload_loader,
            )
            # An unchanged, previously verified marker passes through: expanding
            # it would rewrite only the candidate view to the sidecar secret
            # while the original view kept the marker, and structural validation
            # would then reject the whole candidate — permanently blocking
            # second-pass sanitation. Re-verification is also not required: the
            # marker was verified when it committed.
            unchanged_marker = (
                candidate_message["content"] == original_message.get("content")
            )
            if (
                verified is None
                and _is_externalization_marker(
                    candidate_message["content"],
                    role=role,
                )
                and not unchanged_marker
            ):
                failed_externalization_verification = True
            if (
                verified is not None
                and not unchanged_marker
                and (
                    not isinstance(original_message.get("content"), str)
                    or verified != _normalized_text(original_message["content"])
                )
            ):
                if not isinstance(original_message.get("content"), str):
                    try:
                        verified = json.loads(verified)
                    except (TypeError, ValueError):
                        verified = None
                if verified is not None:
                    candidate_message["content"] = verified
                    verified_externalizations += 1
        elif (
            externalized_payload_loader is not None
            and isinstance(original_message, dict)
            and isinstance(candidate_message, dict)
            and not isinstance(candidate_message.get("content"), str)
            and original_message.get("role") == candidate_message.get("role")
        ):
            if not _nested_externalization_markers_verified(
                original_message.get("content"),
                candidate_message.get("content"),
                role=str(original_message.get("role", "")),
                tool_call_id=(
                    str(original_message.get("tool_call_id"))
                    if isinstance(original_message.get("tool_call_id"), str)
                    else None
                ),
                loader=externalized_payload_loader,
            ):
                failed_externalization_verification = True
        if (
            isinstance(original_message, dict)
            and isinstance(candidate_message, dict)
            and original_message.get("role") == candidate_message.get("role")
            and original_message.get("content") != candidate_message.get("content")
        ):
            original_message.pop("api_content", None)
            candidate_message.pop("api_content", None)
    return (
        original_view,
        candidate_view,
        verified_externalizations,
        failed_externalization_verification,
    )


def validate_sanitation_candidate(
    messages: list,
    candidate: list,
    *,
    externalized_payload_loader: Callable[[str], Any] | None = None,
) -> Optional[SanitationChanges]:
    (
        original_view,
        candidate_view,
        verified_externalizations,
        failed_externalization_verification,
    ) = _validation_views(
        messages,
        candidate,
        externalized_payload_loader,
    )
    if failed_externalization_verification:
        return None
    changes = _validate_sanitized_value(
        original_view,
        candidate_view,
        context="messages",
    )
    if changes is None or not verified_externalizations:
        return changes
    return changes.plus(
        SanitationChanges(
            changed_fields=verified_externalizations,
            placeholders=verified_externalizations,
        )
    )


def prepare_sanitation_commit(
    original: list,
    candidate: list,
    *,
    watermark_messages: Any,
    externalized_payload_loader: Callable[[str], Any] | None = None,
) -> SanitationCommitPlan:
    """Remove replay-unsafe sidecars and derive one validation/metrics snapshot."""
    _drop_stale_api_content(original, candidate)
    represented_row_ids = sanitation_snapshot_row_ids(watermark_messages)
    member_row_ids = sanitation_snapshot_member_row_ids(watermark_messages)
    plan_original = copy.deepcopy(original)
    plan_candidate = copy.deepcopy(candidate)
    return SanitationCommitPlan(
        original=plan_original,
        candidate=plan_candidate,
        input_tokens=sanitation_rough_tokens(plan_original),
        output_tokens=sanitation_rough_tokens(plan_candidate),
        changes=validate_sanitation_candidate(
            plan_original,
            plan_candidate,
            externalized_payload_loader=externalized_payload_loader,
        ),
        watermark=(max(represented_row_ids) if represented_row_ids else 0),
        represented_row_ids=represented_row_ids,
        member_row_ids=member_row_ids,
    )


def externalized_payload_loader(agent: Any) -> Callable[[str], Any] | None:
    """Context-engine contract for sidecar verification during sanitation."""
    compressor = getattr(agent, "context_compressor", None)
    loader = getattr(compressor, "load_externalized_payload_sidecar", None)
    if not callable(loader):
        return None
    return loader


def remember_sanitation_retry(
    agent: Any,
    plan: SanitationCommitPlan,
) -> None:
    """Retain one bounded, already-validated candidate after a DB rollback."""
    if plan.changes is None or not agent.session_id:
        return
    agent._pending_sanitation_retry = SanitationRetryCandidate(
        session_id=agent.session_id,
        original=copy.deepcopy(plan.original),
        candidate=copy.deepcopy(plan.candidate),
    )


def take_sanitation_retry(agent: Any, messages: list) -> list | None:
    """Consume a retained candidate only when it still validates for this input."""
    retry = getattr(agent, "_pending_sanitation_retry", None)
    if not isinstance(retry, SanitationRetryCandidate):
        return None
    if not has_sanitation_retry(agent, messages):
        agent._pending_sanitation_retry = None
        return None
    agent._pending_sanitation_retry = None
    stripped_messages = _strip_persistence_marker(messages)
    stripped_original = _strip_persistence_marker(retry.original)
    rebased = _rebase_retry_onto_prefix(stripped_messages, stripped_original, retry.candidate)
    if rebased is not None:
        return rebased
    return copy.deepcopy(retry.candidate)


def has_sanitation_retry(agent: Any, messages: list) -> bool:
    """Whether the retained candidate is still safe for this exact transcript."""
    retry = getattr(agent, "_pending_sanitation_retry", None)
    if not isinstance(retry, SanitationRetryCandidate):
        return False
    if retry.session_id != agent.session_id:
        return False
    stripped_messages = _strip_persistence_marker(messages)
    stripped_original = _strip_persistence_marker(retry.original)
    if _same_typed_value(stripped_messages, stripped_original):
        return _candidate_validates_for(agent, messages, retry.candidate)
    # Append-only tail rebase (round-8 finding): after a refusal the raw request
    # continues and appends, so the retry-time transcript is no longer byte-equal
    # and the byte-equality check would drop the validated candidate permanently
    # (the secret stays in SQLite/FTS). When the candidate's original is an exact
    # PREFIX of the current transcript, rebase onto that prefix and preserve the
    # unchanged tail — mirroring the concurrent-tail preservation in
    # sanitize_and_compact. Anything else (edited mid-list content) still drops.
    rebased = _rebase_retry_onto_prefix(stripped_messages, stripped_original, retry.candidate)
    if rebased is None:
        return False
    return _candidate_validates_for(agent, messages, rebased)


def _rebase_retry_onto_prefix(
    stripped_messages: Any,
    stripped_original: Any,
    candidate: list,
) -> Optional[list]:
    """Candidate for a transcript whose head is the retry's exact original.

    Returns the candidate extended with the unchanged append-only tail, or
    ``None`` when the transcript is not an exact original-prefix growth.
    """
    if not isinstance(stripped_messages, list) or not isinstance(stripped_original, list):
        return None
    if len(stripped_messages) <= len(stripped_original):
        return None
    if not _same_typed_value(stripped_messages[: len(stripped_original)], stripped_original):
        return None
    if len(candidate) != len(stripped_original):
        return None
    return [
        *copy.deepcopy(candidate),
        *copy.deepcopy(stripped_messages[len(stripped_original):]),
    ]


def _candidate_validates_for(
    agent: Any,
    messages: list,
    candidate: list,
) -> bool:
    return (
        validate_sanitation_candidate(
            messages,
            candidate,
            externalized_payload_loader=externalized_payload_loader(agent),
        )
        is not None
    )


def supports_operation_claim(compress_fn: Callable[..., Any]) -> bool:
    try:
        parameters = inspect.signature(compress_fn).parameters
    except (TypeError, ValueError):
        return False
    return "operation_claim" in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )


def prepare_automatic_compression_operation(
    agent: Any,
    messages: list,
    *,
    attempt_generation: int,
    force: bool,
    bypass_cooldown: bool,
) -> Optional[PreparedCompressionOperation]:
    """Claim pure sanitation for exactly one supported engine invocation."""
    if force or bypass_cooldown:
        return None
    compressor = agent.context_compressor
    prepare = getattr(compressor, "prepare_compression_operation", None)
    if not callable(prepare) or not supports_operation_claim(compressor.compress):
        return None
    kwargs = {
        "session_id": agent.session_id,
        "attempt_generation": attempt_generation,
    }
    try:
        parameters = inspect.signature(prepare).parameters
        if not any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        ):
            kwargs = {
                name: value for name, value in kwargs.items() if name in parameters
            }
        prepared = prepare(messages, **kwargs)
    except Exception as exc:
        logger.debug(
            "prepare_compression_operation raised %s; treating the invocation "
            "as generic compression",
            type(exc).__name__,
        )
        return None
    if (
        not isinstance(prepared, tuple)
        or len(prepared) != 2
        or prepared[0] != "sanitize"
        or prepared[1] is None
    ):
        return None
    return PreparedCompressionOperation(
        operation="sanitize",
        claim=prepared[1],
        session_id=agent.session_id,
        attempt_generation=attempt_generation,
    )


def accept_prepared_sanitation_result(
    agent: Any,
    prepared: PreparedCompressionOperation,
    result: Any,
    *,
    invocation_messages: list | None = None,
) -> Optional[list]:
    """Unwrap only a result carrying the exact one-shot invocation claim."""
    if (
        not isinstance(result, tuple)
        or len(result) != 2
        or not isinstance(result[0], list)
        or result[1] is not prepared.claim
        or agent.session_id != prepared.session_id
    ):
        return None
    accepted = result[0]
    if invocation_messages is not None and accepted is invocation_messages:
        return copy.deepcopy(accepted)
    return accepted


def finish_sanitation_commit(
    agent: Any,
    compressed: list,
    *,
    system_prompt: str,
    compacted_in_place: bool,
) -> int:
    """Publish only the state required after an in-place sanitation rewrite."""
    agent._last_compression_attempt_in_place = compacted_in_place
    agent._last_compaction_in_place = compacted_in_place
    set_usage_anchor(agent, None)
    return estimate_request_tokens_rough(
        compressed,
        system_prompt=system_prompt or "",
        tools=agent.tools or None,
    )


def log_sanitation_commit(
    plan: SanitationCommitPlan,
    *,
    terminal: str,
    session_id: str,
) -> None:
    changes = plan.changes
    logger.log(
        logging.INFO if terminal == "committed" else logging.WARNING,
        "Sanitation commit finished: operation=sanitize reason=external_engine_status "
        "measurement=rough_message_tokens input_tokens=%d output_tokens=%d growth_delta=%d "
        "growth_bound=structural changed_fields=%d declared_placeholders=%d "
        "declared_growth_chars=%d salvage=false terminal_result=%s session=%s",
        plan.input_tokens,
        plan.output_tokens,
        plan.output_tokens - plan.input_tokens,
        changes.changed_fields if changes else 0,
        changes.placeholders if changes else 0,
        changes.declared_growth_chars if changes else 0,
        terminal,
        session_id or "none",
    )
