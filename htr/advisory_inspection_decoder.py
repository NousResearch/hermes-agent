"""Task 29 — bounded control-JSON decoder (R5-04)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

from htr.advisory_inspection_constants import (
    MAX_CONTROL_ARRAY_LENGTH,
    MAX_CONTROL_JSON_BYTES,
    MAX_CONTROL_JSON_DEPTH,
    MAX_CONTROL_OBJECT_MEMBERS,
    MAX_CONTROL_STRING_BYTES,
)

RecordDecodeKind = Literal["manifest", "link"]


@dataclass
class ControlJsonDecodeResult:
    ok: bool
    obj: dict[str, Any] | None
    stage1_findings: list[str] = field(default_factory=list)
    decode_status: str = "ok"
    budget_exceeded: bool = False
    malformed: bool = False


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    seen: set[str] = set()
    obj: dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise ValueError("duplicate_json_keys")
        seen.add(key)
        obj[key] = value
    return obj


def _reject_float(_value: str) -> float:
    raise ValueError("float_rejected")


def _reject_constant(value: str) -> Any:
    raise ValueError("constant_rejected")


def _malformed_token(kind: RecordDecodeKind) -> str:
    return "manifest_json_malformed" if kind == "manifest" else "link_record_json_malformed"


def _utf8_malformed_token(kind: RecordDecodeKind) -> str:
    return "manifest_utf8_malformed" if kind == "manifest" else "link_record_json_malformed"


def _control_budget_token(kind: RecordDecodeKind) -> str:
    return "manifest_control_budget_exceeded" if kind == "manifest" else "link_record_control_budget_exceeded"


def _duplicate_keys_token(kind: RecordDecodeKind) -> str:
    return "manifest_duplicate_json_keys" if kind == "manifest" else "link_record_json_malformed"


def _top_level_token(kind: RecordDecodeKind) -> str:
    return "manifest_top_level_schema_malformed" if kind == "manifest" else "link_record_top_schema_malformed"


def _prepare_body(raw: bytes, kind: RecordDecodeKind) -> tuple[bytes | None, ControlJsonDecodeResult | None]:
    if not raw:
        return None, ControlJsonDecodeResult(
            ok=False,
            obj=None,
            decode_status=_malformed_token(kind),
            malformed=True,
        )

    if raw.startswith(b"\xef\xbb\xbf"):
        return None, ControlJsonDecodeResult(
            ok=False,
            obj=None,
            decode_status=_utf8_malformed_token(kind),
            malformed=True,
        )

    if b"\x0d" in raw:
        return None, ControlJsonDecodeResult(
            ok=False,
            obj=None,
            decode_status=_malformed_token(kind),
            malformed=True,
        )

    if raw.endswith(b"\n\n"):
        return None, ControlJsonDecodeResult(
            ok=False,
            obj=None,
            decode_status=_malformed_token(kind),
            malformed=True,
        )

    if raw.endswith(b"\n"):
        body = raw[:-1]
    else:
        body = raw

    if b"\x0a" in body:
        return None, ControlJsonDecodeResult(
            ok=False,
            obj=None,
            decode_status=_malformed_token(kind),
            malformed=True,
        )

    if len(body) > MAX_CONTROL_JSON_BYTES:
        return None, ControlJsonDecodeResult(
            ok=False,
            obj=None,
            decode_status="budget_control_json_exceeded",
            budget_exceeded=True,
        )

    return body, None


def _walk_stage1(
    obj: Any,
    *,
    depth: int,
    kind: RecordDecodeKind,
    root_artifacts_exempt: bool = False,
) -> ControlJsonDecodeResult | None:
    if depth > MAX_CONTROL_JSON_DEPTH:
        return ControlJsonDecodeResult(
            ok=False,
            obj=None,
            decode_status="budget_control_json_exceeded",
            budget_exceeded=True,
        )

    if isinstance(obj, dict):
        if len(obj) > MAX_CONTROL_OBJECT_MEMBERS:
            return ControlJsonDecodeResult(
                ok=False,
                obj=None,
                decode_status="budget_control_json_exceeded",
                budget_exceeded=True,
            )
        for key, value in obj.items():
            if isinstance(key, str) and len(key.encode("utf-8")) > MAX_CONTROL_STRING_BYTES:
                return ControlJsonDecodeResult(
                    ok=False,
                    obj=None,
                    decode_status="budget_control_json_exceeded",
                    budget_exceeded=True,
                )
            child_exempt = root_artifacts_exempt and key == "artifacts" and depth == 1
            err = _walk_stage1(
                value,
                depth=depth + 1,
                kind=kind,
                root_artifacts_exempt=child_exempt,
            )
            if err is not None:
                return err
    elif isinstance(obj, list):
        if not root_artifacts_exempt and len(obj) > MAX_CONTROL_ARRAY_LENGTH:
            return ControlJsonDecodeResult(
                ok=False,
                obj=None,
                decode_status="budget_control_json_exceeded",
                budget_exceeded=True,
            )
        for item in obj:
            err = _walk_stage1(item, depth=depth + 1, kind=kind, root_artifacts_exempt=False)
            if err is not None:
                return err
    elif isinstance(obj, str):
        if len(obj.encode("utf-8")) > MAX_CONTROL_STRING_BYTES:
            return ControlJsonDecodeResult(
                ok=False,
                obj=None,
                decode_status="budget_control_json_exceeded",
                budget_exceeded=True,
            )
    elif isinstance(obj, bool):
        return None
    elif isinstance(obj, int):
        return None
    elif obj is None:
        return None
    else:
        return ControlJsonDecodeResult(
            ok=False,
            obj=None,
            decode_status=_malformed_token(kind),
            malformed=True,
        )
    return None


def decode_control_json(raw: bytes, *, kind: RecordDecodeKind) -> ControlJsonDecodeResult:
    """Decode control JSON per R5-04 (NOT bounded_action_strict_json)."""
    body, prep_err = _prepare_body(raw, kind)
    if prep_err is not None:
        return prep_err

    assert body is not None
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError:
        return ControlJsonDecodeResult(
            ok=False,
            obj=None,
            decode_status=_utf8_malformed_token(kind),
            malformed=True,
        )

    decoder = json.JSONDecoder(
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_constant,
        parse_float=_reject_float,
    )
    try:
        obj, end_offset = decoder.raw_decode(text, 0)
    except ValueError as exc:
        msg = str(exc)
        if msg == "duplicate_json_keys":
            return ControlJsonDecodeResult(
                ok=False,
                obj=None,
                decode_status=_duplicate_keys_token(kind),
                malformed=True,
            )
        return ControlJsonDecodeResult(
            ok=False,
            obj=None,
            decode_status=_malformed_token(kind),
            malformed=True,
        )

    if end_offset != len(text):
        return ControlJsonDecodeResult(
            ok=False,
            obj=None,
            decode_status=_malformed_token(kind),
            malformed=True,
        )

    if not isinstance(obj, dict):
        return ControlJsonDecodeResult(
            ok=False,
            obj=None,
            decode_status=_top_level_token(kind),
            malformed=True,
        )

    stage1_err = _walk_stage1(obj, depth=1, kind=kind, root_artifacts_exempt=(kind == "manifest"))
    if stage1_err is not None:
        return stage1_err

    return ControlJsonDecodeResult(ok=True, obj=obj, decode_status="ok")


def semantic_digest_bytes(raw: bytes) -> bytes:
    """Canonical semantic bytes exclude optional single trailing file LF (R5-05)."""
    if raw.endswith(b"\n") and not raw.endswith(b"\n\n"):
        return raw[:-1]
    return raw
