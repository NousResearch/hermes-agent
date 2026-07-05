"""Tiny safe condition evaluator for workflow guards."""

from __future__ import annotations

import operator
import re
from collections.abc import Mapping, Sequence
from typing import Any

_MISSING = object()
_UNRESOLVED = object()

_COMPARISONS = {
    "eq": operator.eq,
    "ne": operator.ne,
    "gt": operator.gt,
    "gte": operator.ge,
    "lt": operator.lt,
    "lte": operator.le,
}


def _missing(path: str, default: Any) -> Any:
    if default is _MISSING:
        raise KeyError(path)
    return default


def _split_part(part: str, full_path: str) -> tuple[str, list[int]]:
    if not part:
        raise ValueError(f"invalid path {full_path!r}")

    start = part.find("[")
    if start == -1:
        return part, []

    name = part[:start]
    rest = part[start:]
    indexes: list[int] = []
    while rest:
        match = re.match(r"\[(\d+)\]", rest)
        if not match:
            raise ValueError(f"invalid path {full_path!r}")
        indexes.append(int(match.group(1)))
        rest = rest[match.end() :]
    return name, indexes


def resolve_path(data: Any, path: str, *, default: Any = _MISSING) -> Any:
    if not isinstance(path, str) or not path.startswith("$."):
        raise ValueError("path must start with '$.'")

    current = data
    for part in path[2:].split("."):
        name, indexes = _split_part(part, path)
        if name:
            if isinstance(current, Mapping) and name in current:
                current = current[name]
            else:
                return _missing(path, default)
        for index in indexes:
            if (
                isinstance(current, Sequence)
                and not isinstance(current, (str, bytes, bytearray))
                and 0 <= index < len(current)
            ):
                current = current[index]
            else:
                return _missing(path, default)
    return current


def _value(spec: Any, data: Any) -> Any:
    if isinstance(spec, Mapping) and set(spec) == {"path"}:
        return resolve_path(data, spec["path"], default=_UNRESOLVED)
    return spec


def _unary_value(cond: Mapping[str, Any], data: Any) -> Any:
    sources = [source for source in ("path", "arg", "left") if source in cond]
    if len(sources) != 1:
        raise ValueError("unary condition requires one operand")
    if sources[0] == "path":
        return resolve_path(data, cond["path"], default=_UNRESOLVED)
    if sources[0] == "arg":
        return _value(cond["arg"], data)
    return _value(cond["left"], data)


def _is_absent(value: Any) -> bool:
    return value is _MISSING or value is _UNRESOLVED


def _condition_args(cond: Mapping[str, Any], op: str) -> Sequence[Any]:
    args = cond.get("args")
    if not isinstance(args, Sequence) or isinstance(args, (str, bytes, bytearray)) or not args:
        raise ValueError(f"{op} requires non-empty args")
    return args


def _binary_values(cond: Mapping[str, Any], data: Any, op: str) -> tuple[Any, Any]:
    if "left" not in cond or "right" not in cond:
        raise ValueError(f"{op} requires left and right")
    return _value(cond["left"], data), _value(cond["right"], data)


def eval_condition(cond: Mapping[str, Any], data: Any) -> bool:
    if not isinstance(cond, Mapping):
        raise ValueError("condition must be a mapping")

    op = cond.get("op")

    if op == "and":
        results = [eval_condition(arg, data) for arg in _condition_args(cond, op)]
        return all(results)
    if op == "or":
        results = [eval_condition(arg, data) for arg in _condition_args(cond, op)]
        return any(results)
    if op == "not":
        if "args" in cond:
            if "arg" in cond:
                raise ValueError("not requires exactly one arg")
            args = _condition_args(cond, op)
            if len(args) != 1:
                raise ValueError("not requires exactly one arg")
            return not eval_condition(args[0], data)
        if "arg" not in cond:
            raise ValueError("not requires arg")
        return not eval_condition(cond["arg"], data)

    if op == "exists":
        return not _is_absent(_unary_value(cond, data))
    if op == "missing":
        return _is_absent(_unary_value(cond, data))

    if op in _COMPARISONS:
        left, right = _binary_values(cond, data, op)
        if _is_absent(left) or _is_absent(right):
            return False
        try:
            return bool(_COMPARISONS[op](left, right))
        except TypeError:
            return False

    if op not in {"contains", "starts_with", "ends_with", "regex"}:
        raise ValueError(f"unsupported condition op: {op}")

    left, right = _binary_values(cond, data, op)
    if _is_absent(left) or _is_absent(right):
        return False

    if op == "contains":
        try:
            return right in left
        except TypeError:
            return False
    if op == "starts_with":
        return isinstance(left, str) and isinstance(right, str) and left.startswith(right)
    if op == "ends_with":
        return isinstance(left, str) and isinstance(right, str) and left.endswith(right)
    if op == "regex":
        if not isinstance(left, str) or not isinstance(right, str):
            return False
        try:
            return re.search(right, left) is not None
        except re.error as exc:
            raise ValueError(f"invalid regex: {exc}") from exc

    raise ValueError(f"unsupported condition op: {op}")

