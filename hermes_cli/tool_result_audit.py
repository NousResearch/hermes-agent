"""Content-free warnings when plugins change a successful web-search result."""

from __future__ import annotations

import json
import logging
import math
from typing import Any

logger = logging.getLogger(__name__)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(_value: str) -> None:
    raise ValueError("Nonstandard JSON constant")


def _freeze_json(value: Any) -> Any:
    """Keep JSON types distinct and capture mutable containers before plugins see them."""
    kind = type(value)
    if kind is dict:
        if any(type(key) is not str for key in value):
            raise ValueError("Non-string JSON key")
        return (dict, tuple((key, _freeze_json(value[key])) for key in sorted(value)))
    if kind is list:
        return (list, tuple(_freeze_json(item) for item in value))
    if kind is float and not math.isfinite(value):
        raise ValueError("Non-finite JSON number")
    if kind not in (str, bool, int, float, type(None)):
        raise ValueError("Non-JSON value")
    return (kind, value)


def snapshot_web_search_result(result: Any) -> tuple[bool, Any] | None:
    """Capture strict JSON semantics; ambiguous or non-JSON results cannot prove success."""
    try:
        value = (json.loads(result, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
                 if isinstance(result, str) else result)
        frozen = _freeze_json(value)
    except (TypeError, ValueError, RecursionError):
        return None
    return (isinstance(value, dict) and value.get("success") is True, frozen)


def warn_web_search_result_change(original: tuple[bool, Any] | None, result: Any, source: str) -> None:
    """Observe accepted replacements, including redaction, without overriding their output."""
    candidate = snapshot_web_search_result(result)
    original_success = original is not None and original[0]
    candidate_success = candidate is not None and candidate[0]
    changed = original != candidate if original_success else candidate_success
    if changed:
        logger.warning(
            "%s changed a successful web_search result or supplied success without a successful "
            "downstream result; the returned output is preserved and may include intentional redaction",
            source,
        )
