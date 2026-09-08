"""Bounded canonical-JSON boundary for hostile SDK permission payloads.

Control-character sanitizers, the exact-builtin bounded JSON freezer, canonical
request serialization and the safe tool presentation/identity/deny-reason
helpers. Extracted from ``claude_agent_sdk_session.py``; ``tools.approval_sdk_gateway``
imports the two public validators through the facade.
"""

from __future__ import annotations

import json
import logging
import math
import unicodedata
from agent.redact import redact_sensitive_text

# Same logger name as the origin module so log records / caplog filters are unchanged.
logger = logging.getLogger("agent.transports.claude_agent_sdk_session")


# SDK permission payloads are hostile Python objects.  Nothing derived from
# them may reach an auto-allow, callback, log, observer, or card until the
# complete request has crossed this exact-builtin, bounded JSON boundary.
_SDK_CANONICAL_MAX_DEPTH = 64


_SDK_CANONICAL_MAX_NODES = 10_000


_SDK_CANONICAL_MAX_UTF8_BYTES = 64 * 1024


_SDK_CANONICAL_MAX_INPUT_UTF8_BYTES = 64 * 1024


_SDK_TOOL_USE_ID_MAX_UTF8_BYTES = 256


_SDK_PRESENTATION_MAX_UTF8_BYTES = 512


_SDK_CALLBACK_CHOICE_MAX_UTF8_BYTES = 16


_SDK_CALLBACK_REASON_MAX_UTF8_BYTES = 512


_SDK_PATH_TOOL_FIELDS = {
    "Read": ("file_path", "path"),
    "Write": ("file_path", "path"),
    "Edit": ("file_path", "path"),
    "MultiEdit": ("file_path", "path"),
    "Glob": ("path",),
    "Grep": ("path",),
    "NotebookEdit": ("notebook_path", "file_path", "path"),
}


_SDK_FIXED_LOG_TOOL_IDENTITIES = frozenset({"Bash", *_SDK_PATH_TOOL_FIELDS})


def _control_sanitized_text(value: str) -> str:
    safe = "".join(
        " " if unicodedata.category(char).startswith("C") else char
        for char in value
    )
    return " ".join(safe.split())


def _bounded_control_sanitized_text(value: str, max_utf8_bytes: int) -> str:
    """Return one bounded display line from an already validated string."""
    safe = _control_sanitized_text(value)
    output: list[str] = []
    used = 0
    for char in safe:
        encoded_size = len(char.encode("utf-8"))
        if used + encoded_size > max_utf8_bytes:
            break
        output.append(char)
        used += encoded_size
    return "".join(output)


def _bounded_control_sanitized_head_tail(value: str, max_utf8_bytes: int) -> str:
    """Bound one line while disclosing both ends and explicit truncation."""
    safe = _control_sanitized_text(value)
    if len(safe.encode("utf-8")) <= max_utf8_bytes:
        return safe
    marker = " … [truncated] … "
    marker_bytes = len(marker.encode("utf-8"))
    if marker_bytes > max_utf8_bytes:
        return _bounded_control_sanitized_text(marker, max_utf8_bytes)
    remaining = max_utf8_bytes - marker_bytes
    head_budget = remaining // 2
    tail_budget = remaining - head_budget
    head = _bounded_control_sanitized_text(safe, head_budget)
    tail_chars: list[str] = []
    used = 0
    for char in reversed(safe):
        encoded_size = len(char.encode("utf-8"))
        if used + encoded_size > tail_budget:
            break
        tail_chars.append(char)
        used += encoded_size
    return f"{head}{marker}{''.join(reversed(tail_chars))}"


def _safe_sdk_tool_identity(value: str) -> str:
    """Preserve ordinary identity, but hide embedded credential patterns."""
    bounded = _bounded_control_sanitized_text(value, 96)
    if not bounded:
        return "unknown"
    redacted = redact_sensitive_text(bounded, force=True)
    if redacted != bounded:
        return "unknown"
    # Generic redaction intentionally requires token boundaries. Unknown SDK
    # tool identities may attach a credential prefix directly to punctuation
    # (for example ``Odd-sk-...``), so expose every bounded suffix behind a
    # boundary in one composite probe. Do not attempt to reconstruct a partly
    # secret identity: any match collapses to one fixed, non-sensitive label.
    suffix_probe = "\n".join(bounded[index:] for index in range(len(bounded)))
    if redact_sensitive_text(suffix_probe, force=True) != suffix_probe:
        return "unknown"
    return bounded


_INVALID_SDK_JSON = object()


def _sdk_json_string_sizes(
    value: str,
    input_budget: int = _SDK_CANONICAL_MAX_INPUT_UTF8_BYTES,
) -> tuple[int, int] | None:
    """Count input/token UTF-8 incrementally and stop at the first cap."""
    input_size = 0
    token_size = 2  # surrounding quotes
    try:
        for char in value:
            encoded_size = len(char.encode("utf-8"))
            input_size += encoded_size
            if input_size > input_budget:
                return None
            codepoint = ord(char)
            if char in ('"', "\\") or char in "\b\f\n\r\t":
                token_size += 2
            elif codepoint < 0x20:
                token_size += 6
            else:
                token_size += encoded_size
            if token_size > _SDK_CANONICAL_MAX_UTF8_BYTES:
                return None
    except UnicodeError:
        return None
    return input_size, token_size


def _sdk_json_string_token_utf8_size(value: str) -> int | None:
    """Return the canonical token size without materializing encoded copies."""
    sizes = _sdk_json_string_sizes(value)
    return None if sizes is None else sizes[1]


def _sdk_json_int_token_fits(value: int) -> bool:
    """Conservatively bound decimal conversion without converting the integer."""
    # 30103 / 100000 is a strict upper approximation of log10(2).
    digits_upper_bound = (value.bit_length() * 30_103) // 100_000 + 1
    token_bytes = digits_upper_bound + (1 if value < 0 else 0)
    return token_bytes <= _SDK_CANONICAL_MAX_UTF8_BYTES


def _freeze_bounded_plain_sdk_json(value: object) -> object:
    """Validate and detach one bounded exact-builtin JSON graph in one pass."""
    active: set[int] = set()
    nodes = 0
    input_utf8_bytes = 0

    def freeze(current: object, depth: int) -> object:
        nonlocal nodes, input_utf8_bytes
        nodes += 1
        if nodes > _SDK_CANONICAL_MAX_NODES or depth > _SDK_CANONICAL_MAX_DEPTH:
            return _INVALID_SDK_JSON

        kind = type(current)
        if kind is str:
            remaining = _SDK_CANONICAL_MAX_INPUT_UTF8_BYTES - input_utf8_bytes
            sizes = _sdk_json_string_sizes(current, remaining)
            if sizes is None:
                return _INVALID_SDK_JSON
            input_utf8_bytes += sizes[0]
            return current
        if current is None or kind is bool:
            return current
        if kind is int:
            return current if _sdk_json_int_token_fits(current) else _INVALID_SDK_JSON
        if kind is float:
            return current if math.isfinite(current) else _INVALID_SDK_JSON
        if kind not in (list, dict):
            return _INVALID_SDK_JSON

        identity = id(current)
        if identity in active:
            return _INVALID_SDK_JSON
        active.add(identity)
        try:
            if kind is list:
                frozen_list = []
                for child in current:
                    frozen_child = freeze(child, depth + 1)
                    if frozen_child is _INVALID_SDK_JSON:
                        return _INVALID_SDK_JSON
                    frozen_list.append(frozen_child)
                return frozen_list

            frozen_dict = {}
            for key, child in current.items():
                if type(key) is not str:
                    return _INVALID_SDK_JSON
                frozen_key = freeze(key, depth + 1)
                if frozen_key is _INVALID_SDK_JSON:
                    return _INVALID_SDK_JSON
                frozen_child = freeze(child, depth + 1)
                if frozen_child is _INVALID_SDK_JSON:
                    return _INVALID_SDK_JSON
                frozen_dict[frozen_key] = frozen_child
            return frozen_dict
        except (RuntimeError, ValueError, TypeError):
            return _INVALID_SDK_JSON
        finally:
            active.discard(identity)

    return freeze(value, 0)


def _is_bounded_plain_sdk_json(value: object) -> bool:
    return _freeze_bounded_plain_sdk_json(value) is not _INVALID_SDK_JSON


def _bounded_canonical_sdk_json(value: object) -> str | None:
    """Encode deterministic JSON incrementally and stop at the byte cap."""
    encoder = json.JSONEncoder(
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    chunks: list[str] = []
    used = 0
    try:
        for chunk in encoder.iterencode(value):
            encoded_size = len(chunk.encode("utf-8"))
            if used + encoded_size > _SDK_CANONICAL_MAX_UTF8_BYTES:
                return None
            chunks.append(chunk)
            used += encoded_size
    except (
        TypeError, ValueError, RuntimeError, RecursionError, OverflowError,
        UnicodeError,
    ):
        return None
    return "".join(chunks)


def _bounded_utf8_size(value: str, max_bytes: int) -> int | None:
    """Count UTF-8 bytes incrementally, stopping when the cap is crossed."""
    if len(value) > max_bytes:
        return None
    used = 0
    for start in range(0, len(value), 4096):
        used += len(value[start:start + 4096].encode("utf-8"))
        if used > max_bytes:
            return None
    return used


def validate_canonical_sdk_request_serialization(
    value: object,
) -> tuple[str, dict] | None:
    """Validate exact deterministic JSON for one bounded SDK tool request."""
    if type(value) is not str:
        return None
    try:
        if _bounded_utf8_size(value, _SDK_CANONICAL_MAX_UTF8_BYTES) is None:
            return None
        request = json.loads(value)
    except (TypeError, ValueError, RecursionError, OverflowError, UnicodeError):
        return None
    frozen = _freeze_bounded_plain_sdk_json(request)
    if (
        frozen is _INVALID_SDK_JSON
        or type(frozen) is not dict
        or set(frozen) != {"tool_name", "tool_input"}
        or type(frozen["tool_name"]) is not str
        or type(frozen["tool_input"]) is not dict
        or (
            frozen["tool_name"] == "Bash"
            and type(frozen["tool_input"].get("command")) is not str
        )
    ):
        return None
    canonical = _bounded_canonical_sdk_json(frozen)
    if canonical is None or value != canonical:
        return None
    return canonical, frozen


def _canonical_sdk_tool_request(tool_name: object, tool_input: object) -> str | None:
    if type(tool_name) is not str or type(tool_input) is not dict:
        return None
    try:
        frozen = _freeze_bounded_plain_sdk_json({
            "tool_name": tool_name,
            "tool_input": tool_input,
        })
        if (
            frozen is _INVALID_SDK_JSON
            or type(frozen) is not dict
            or type(frozen.get("tool_input")) is not dict
            or (
                tool_name == "Bash"
                and type(frozen["tool_input"].get("command")) is not str
            )
        ):
            return None
        canonical = _bounded_canonical_sdk_json(frozen)
    except (
        TypeError, ValueError, RuntimeError, RecursionError, OverflowError,
        UnicodeError,
    ):
        return None
    if canonical is None:
        return None
    checked = validate_canonical_sdk_request_serialization(canonical)
    return checked[0] if checked is not None else None


def _is_bounded_sdk_callback_string(
    value: object,
    max_utf8_bytes: int,
    *,
    allow_space: bool,
) -> bool:
    """Validate callback text incrementally without encoding a hostile copy."""
    if type(value) is not str:
        return False
    used = 0
    for char in value:
        codepoint = ord(char)
        if unicodedata.category(char).startswith("C"):
            return False
        if char.isspace() and (char != " " or not allow_space):
            return False
        if codepoint <= 0x7F:
            used += 1
        elif codepoint <= 0x7FF:
            used += 2
        elif codepoint <= 0xFFFF:
            used += 3
        else:
            used += 4
        if used > max_utf8_bytes:
            return False
    return True


def _safe_sdk_tool_use_id(context: object) -> str:
    try:
        value = getattr(context, "tool_use_id", "")
    except Exception:
        return ""
    if type(value) is not str or not value:
        return ""
    utf8_bytes = 0
    for char in value:
        code_point = ord(char)
        if code_point <= 0x7F:
            utf8_bytes += 1
        elif code_point <= 0x7FF:
            utf8_bytes += 2
        elif 0xD800 <= code_point <= 0xDFFF:
            return ""
        elif code_point <= 0xFFFF:
            utf8_bytes += 3
        else:
            utf8_bytes += 4
        if utf8_bytes > _SDK_TOOL_USE_ID_MAX_UTF8_BYTES:
            return ""
        if not char.isprintable() or char.isspace():
            return ""
    return value


_UNVALIDATED_SDK_REQUEST = object()


def safe_sdk_tool_presentation_from_canonical(
    canonical_request: object,
    *,
    _validated_request: object = _UNVALIDATED_SDK_REQUEST,
) -> tuple[str, str] | None:
    """Build a bounded actionable card summary from validated request fields."""
    checked = _validated_request
    if checked is _UNVALIDATED_SDK_REQUEST:
        checked = validate_canonical_sdk_request_serialization(canonical_request)
    if checked is None:
        return None
    request = checked[1]
    tool_name = request["tool_name"]
    tool_input = request["tool_input"]
    safe_name = _safe_sdk_tool_identity(tool_name)
    if tool_name == "Bash":
        command = redact_sensitive_text(tool_input["command"], force=True)
        preview = _bounded_control_sanitized_head_tail(command, 480)
        return (
            _bounded_control_sanitized_text(
                f"Bash(command={preview})", _SDK_PRESENTATION_MAX_UTF8_BYTES,
            ),
            "Claude requests SDK tool Bash",
        )
    path_fields = _SDK_PATH_TOOL_FIELDS.get(tool_name)
    if path_fields is not None:
        path = next(
            (tool_input[field] for field in path_fields if type(tool_input.get(field)) is str),
            "(unspecified)",
        )
        safe_path = _bounded_control_sanitized_head_tail(
            redact_sensitive_text(path, force=True), 460,
        )
        return (
            _bounded_control_sanitized_text(
                f"{safe_name}(path={safe_path})", _SDK_PRESENTATION_MAX_UTF8_BYTES,
            ),
            f"Claude requests SDK tool {safe_name}",
        )
    return f"SDK tool {safe_name}", "Claude requests an SDK tool"


_SAFE_SDK_DENY_LOG_REASONS = frozenset({
    "approval callback failed",
    "approval timed out — no operator response",
    "approval expired (turn ended)",
    "no approver available (background context)",
    "approval request could not be delivered to the operator (notify failed)",
})


def _safe_sdk_deny_log_reason(reason: object) -> str:
    if type(reason) is str and reason in _SAFE_SDK_DENY_LOG_REASONS:
        return reason
    return "non-operator denial"
