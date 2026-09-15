"""Validation, operation claims, and commit support for transcript sanitation."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib
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
        "reasoning_details",
        "codex_reasoning_items",
        "codex_message_items",
        "anthropic_content_blocks",
        "bedrock_content_blocks",
    }
)


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


@dataclasses.dataclass(frozen=True)
class SanitationRetryCandidate:
    session_id: str
    original: list
    candidate: list


def _strip_persistence_marker(messages: Any) -> Any:
    from agent.context_compressor import _DB_PERSISTED_MARKER

    if not isinstance(messages, list):
        return messages
    return [
        (
            {key: value for key, value in message.items() if key != _DB_PERSISTED_MARKER}
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


def _child_context(context: str, key: Any) -> tuple[str, bool]:
    if context == "message":
        return (
            ("payload", key == "content")
            if key in _MESSAGE_PAYLOAD_FIELDS
            else ("tool_calls" if key == "tool_calls" else "structural", False)
        )
    if context == "tool_calls":
        return ("function" if key == "function" else "structural", False)
    if context == "function":
        return (
            ("payload", False)
            if key == "arguments"
            else ("structural", False)
        )
    if context == "content_part" and key == "type":
        return "structural", False
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
        if context == "structural":
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


def _validation_views(
    original: list,
    candidate: list,
    externalized_payload_loader: Callable[[str], Any] | None = None,
) -> tuple[list, list, int]:
    original_view = copy.deepcopy(_strip_persistence_marker(original))
    candidate_view = copy.deepcopy(_strip_persistence_marker(candidate))
    verified_externalizations = 0
    for original_message, candidate_message in zip(original_view, candidate_view):
        if (
            externalized_payload_loader is not None
            and isinstance(original_message, dict)
            and isinstance(candidate_message, dict)
            and isinstance(candidate_message.get("content"), str)
            and original_message.get("role") == candidate_message.get("role")
        ):
            verified = _verified_externalized_content(
                candidate_message["content"],
                role=str(original_message.get("role", "")),
                tool_call_id=(
                    str(original_message.get("tool_call_id"))
                    if isinstance(original_message.get("tool_call_id"), str)
                    else None
                ),
                loader=externalized_payload_loader,
            )
            if verified is not None and (
                not isinstance(original_message.get("content"), str)
                or verified != _normalized_text(original_message["content"])
            ):
                if not isinstance(original_message.get("content"), str):
                    try:
                        verified = json.loads(verified)
                    except (TypeError, ValueError):
                        verified = None
                if verified is not None:
                    candidate_message["content"] = verified
                    verified_externalizations += 1
        if (
            isinstance(original_message, dict)
            and isinstance(candidate_message, dict)
            and original_message.get("role") == candidate_message.get("role")
            and original_message.get("content") != candidate_message.get("content")
        ):
            original_message.pop("api_content", None)
            candidate_message.pop("api_content", None)
    return original_view, candidate_view, verified_externalizations


def validate_sanitation_candidate(
    messages: list,
    candidate: list,
    *,
    externalized_payload_loader: Callable[[str], Any] | None = None,
) -> Optional[SanitationChanges]:
    original_view, candidate_view, verified_externalizations = _validation_views(
        messages,
        candidate,
        externalized_payload_loader,
    )
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
        watermark=max(represented_row_ids) if represented_row_ids else None,
        represented_row_ids=represented_row_ids,
    )


def externalized_payload_loader(agent: Any) -> Callable[[str], Any] | None:
    """Adapt the paired LCM sidecar reader without adding a host protocol."""
    compressor = getattr(agent, "context_compressor", None)
    module_name = type(compressor).__module__
    package_name = module_name.split(".", 1)[0]
    if package_name != "hermes_lcm":
        return None
    try:
        externalize = importlib.import_module(f"{package_name}.externalize")
        config = compressor._config
        storage_dir = externalize.get_large_output_storage_dir(
            config,
            hermes_home=str(getattr(compressor, "_hermes_home", "") or ""),
            create=False,
        )
        safe_read = externalize._read_legacy_externalized_payload_json
    except (AttributeError, ImportError, OSError):
        return None
    return lambda ref: safe_read(
        storage_dir / ref,
        storage_dir=storage_dir,
    )


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
    return copy.deepcopy(retry.candidate)


def has_sanitation_retry(agent: Any, messages: list) -> bool:
    """Whether the retained candidate is still safe for this exact transcript."""
    retry = getattr(agent, "_pending_sanitation_retry", None)
    if (
        not isinstance(retry, SanitationRetryCandidate)
        or retry.session_id != agent.session_id
        or _strip_persistence_marker(messages)
        != _strip_persistence_marker(retry.original)
    ):
        return False
    return (
        validate_sanitation_candidate(
            messages,
            retry.candidate,
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
    return result[0]


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
