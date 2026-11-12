"""Validated JSON protocol and CLI handler for guarded access resolution."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import struct
import sys
import unicodedata
import uuid
from typing import Any

DIGEST_DOMAIN = "hermes-kanban-access-resolution/v1"
FEATURE_REVISION = "v1"
MAX_INPUT_BYTES = 65_536
_MAX_I64 = 2**63 - 1
_ORDERED_FIELDS = (
    "version", "requestId", "mode", "machine", "cardId", "expectedAssignee",
    "expectedStatus", "expectedBlockKind", "expectedBlockEventId", "expectedBlockedAt",
    "category", "owner", "evidenceRef", "nextAction", "author", "comment",
)
_KEYS = frozenset((*_ORDERED_FIELDS, "payloadDigest"))
_UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\Z")
_DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z")
_MACHINE_RE = re.compile(r"[a-z][a-z0-9-]{0,31}\Z")
_CARD_RE = re.compile(r"t_[0-9a-f]{8}\Z")
_ASSIGNEE_RE = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}\Z")
_MODES = frozenset({"route", "grant"})
_CATEGORIES = frozenset({"credential_or_permission", "missing_tool_or_read_api", "human_receipt"})
# Control, format (bidi overrides, zero-width) and line/paragraph separators would let an
# audit comment or owner read differently from what was digested.
_FORBIDDEN_CHAR_CATEGORIES = frozenset({"Cc", "Cf", "Zl", "Zp"})


class AccessResolutionError(ValueError):
    """Expected fixed-code protocol denial."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _invalid() -> AccessResolutionError:
    return AccessResolutionError("invalid_request")


def _pairs_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _invalid()
        result[key] = value
    return result


def _finite_json_constant(_value: str) -> None:
    raise _invalid()


def _strict_text(value: Any, *, minimum: int, maximum: int, allow_lf: bool = False) -> str:
    if not isinstance(value, str) or not value or unicodedata.normalize("NFC", value) != value:
        raise _invalid()
    try:
        encoded = value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise _invalid() from exc
    if not minimum <= len(encoded) <= maximum:
        raise _invalid()
    for char in value:
        if unicodedata.category(char) in _FORBIDDEN_CHAR_CATEGORIES and not (allow_lf and char == "\n"):
            raise _invalid()
    return value


def _exact_string(value: Any, expected: str) -> None:
    if not isinstance(value, str) or value != expected:
        raise _invalid()


def _bounded_integer(value: Any, minimum: int) -> None:
    if type(value) is not int or not minimum <= value <= _MAX_I64:
        raise _invalid()


def _choice(value: Any, allowed: frozenset[str]) -> None:
    # ``isinstance`` first: an unhashable JSON value (list/object) must not reach the set lookup.
    if not isinstance(value, str) or value not in allowed:
        raise _invalid()


def _matches(value: Any, pattern: re.Pattern[str]) -> None:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise _invalid()


def _validate_request_id(request_id: Any) -> None:
    _matches(request_id, _UUID_RE)
    try:
        canonical = str(uuid.UUID(request_id))
    except ValueError as exc:
        raise _invalid() from exc
    if canonical != request_id:
        raise _invalid()


def _validate_mode_fields(request: dict[str, Any]) -> None:
    """``route`` carries a next action and no evidence; ``grant`` is the inverse."""
    has_evidence = request["evidenceRef"] is not None
    has_next_action = request["nextAction"] is not None
    if request["mode"] == "route" and (has_evidence or not has_next_action):
        raise _invalid()
    if request["mode"] == "grant" and (not has_evidence or has_next_action):
        raise _invalid()


def _validate_request(request: Any) -> dict[str, Any]:
    if not isinstance(request, dict) or set(request) != _KEYS:
        raise _invalid()
    if type(request["version"]) is not int or request["version"] != 1:
        raise _invalid()
    _validate_request_id(request["requestId"])
    _matches(request["payloadDigest"], _DIGEST_RE)
    _choice(request["mode"], _MODES)
    _matches(request["machine"], _MACHINE_RE)
    _matches(request["cardId"], _CARD_RE)
    _matches(request["expectedAssignee"], _ASSIGNEE_RE)
    _exact_string(request["expectedStatus"], "blocked")
    _exact_string(request["expectedBlockKind"], "capability")
    _bounded_integer(request["expectedBlockEventId"], 1)
    _bounded_integer(request["expectedBlockedAt"], 0)
    _choice(request["category"], _CATEGORIES)
    _strict_text(request["owner"], minimum=1, maximum=256)
    for field in ("evidenceRef", "nextAction"):
        if request[field] is not None:
            _strict_text(request[field], minimum=1, maximum=2048)
    _exact_string(request["author"], "seth")
    _strict_text(request["comment"], minimum=1, maximum=8192, allow_lf=True)
    _validate_mode_fields(request)
    return request


def canonical_preimage(request: dict[str, Any]) -> bytes:
    tokens: list[bytes] = []
    for field in _ORDERED_FIELDS:
        value = request[field]
        if value is None:
            tokens.append(b"n")
        elif type(value) is int:
            payload = str(value).encode("ascii")
            tokens.append(b"i" + struct.pack(">I", len(payload)) + payload)
        elif isinstance(value, str):
            payload = value.encode("utf-8", errors="strict")
            tokens.append(b"s" + struct.pack(">I", len(payload)) + payload)
        else:
            raise _invalid()
    return DIGEST_DOMAIN.encode("ascii") + b"\x00" + b"".join(tokens)


def parse_request_bytes(raw: bytes) -> dict[str, Any]:
    if not isinstance(raw, bytes) or len(raw) > MAX_INPUT_BYTES or raw.startswith(b"\xef\xbb\xbf"):
        raise _invalid()
    try:
        text = raw.decode("utf-8", errors="strict")
        request = json.loads(
            text,
            object_pairs_hook=_pairs_no_duplicates,
            parse_constant=_finite_json_constant,
        )
    except AccessResolutionError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise _invalid() from exc
    validated = _validate_request(request)
    computed = hashlib.sha256(canonical_preimage(validated)).hexdigest()
    if not hmac.compare_digest(computed, validated["payloadDigest"]):
        raise AccessResolutionError("digest_mismatch")
    return validated


def configured_machine_identity() -> str:
    from hermes_cli.config import cfg_get, load_config

    value = cfg_get(load_config(), "kanban", "machine_identity")
    if not isinstance(value, str) or not _MACHINE_RE.fullmatch(value):
        raise _invalid()
    return value


def compact_json(payload: dict[str, Any]) -> str:
    """Single-line JSON used for stdout lines and the stored receipt."""
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _json_line(payload: dict[str, Any]) -> None:
    print(compact_json(payload))


def command(args) -> int:
    """Run the public command after board scoping but before automatic DB initialization."""
    if bool(getattr(args, "json_stdin", False)) == bool(getattr(args, "describe", False)):
        _json_line({"error": "invalid_request"})
        return 2
    try:
        machine = configured_machine_identity()
        if args.describe:
            _json_line({
                "version": 1,
                "command": "access-resolution",
                "featureRevision": FEATURE_REVISION,
                "machine": machine,
                "digestDomain": DIGEST_DOMAIN,
                "maxInputBytes": MAX_INPUT_BYTES,
            })
            return 0
        raw = sys.stdin.buffer.read(MAX_INPUT_BYTES + 1)
        request = parse_request_bytes(raw)
        if request["machine"] != machine:
            raise AccessResolutionError("machine_mismatch")
        from hermes_cli import kanban_db as kb
        from hermes_cli import kanban_db_connect as kbc

        kb.init_db()
        with kbc.connect_closing() as conn:
            receipt = kb.resolve_access_request(conn, request, machine=machine)
        _json_line(receipt)
        return 0
    except AccessResolutionError as exc:
        _json_line({"error": exc.code})
        return 1
    except Exception:  # health: allow BLE001 -- fixed public boundary must hide unknown effect details
        print("kanban access-resolution: internal_error", file=sys.stderr)
        return 1
