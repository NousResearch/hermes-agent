"""Shared application parsing and persistence scope for /model commands."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

from hermes_constants import parse_reasoning_effort

ERROR_TEXT = {
    "once_with_global": "/model --once cannot be combined with --global",
    "once_requires_target": "/model --once requires a model or provider.",
    "bad_reasoning": (
        "/model --reasoning takes none, minimal, low, medium, high, xhigh, max or ultra."
    ),
}
_BOOL_FLAGS = {
    "--global": "is_global",
    "--session": "is_session",
    "--refresh": "force_refresh",
    "--once": "is_once",
}
_VALUE_FLAGS = {"--provider": "explicit_provider", "--reasoning": "reasoning_effort"}


@dataclass(frozen=True, slots=True)
class ModelCommandRequest:
    raw: str
    target: str
    explicit_provider: str = ""
    reasoning_effort: str = ""
    is_global: bool = False
    is_session: bool = False
    is_once: bool = False
    force_refresh: bool = False
    scope: str = "default"
    errors: tuple[str, ...] = ()

    def error_messages(self) -> list[str]:
        return [ERROR_TEXT[code] for code in self.errors]


def parse_model_command(raw: str) -> ModelCommandRequest:
    raw = str(raw or "")
    raw = re.sub(
        r"[\u2012\u2013\u2014\u2015](provider|reasoning|global|session|refresh|once)",
        r"--\1",
        raw,
    )
    flags = dict.fromkeys(_BOOL_FLAGS.values(), False)
    values = dict.fromkeys(_VALUE_FLAGS.values(), "")
    target: list[str] = []
    tokens = iter(raw.split())
    for token in tokens:
        if token in _BOOL_FLAGS:
            flags[_BOOL_FLAGS[token]] = True
        elif token in _VALUE_FLAGS and (value := next(tokens, None)) is not None:
            values[_VALUE_FLAGS[token]] = value
        else:
            target.append(token)

    model = " ".join(target).strip()
    errors: list[str] = []
    if flags["is_once"] and flags["is_global"]:
        errors.append("once_with_global")
    if flags["is_once"] and not model and not values["explicit_provider"]:
        errors.append("once_requires_target")
    if values["reasoning_effort"] and parse_reasoning_effort(values["reasoning_effort"]) is None:
        errors.append("bad_reasoning")
    scope = next(
        (
            name
            for name, enabled in (
                ("once", flags["is_once"]),
                ("session", flags["is_session"]),
                ("global", flags["is_global"]),
            )
            if enabled
        ),
        "default",
    )
    return ModelCommandRequest(
        raw=raw,
        target=model,
        scope=scope,
        errors=tuple(errors),
        **values,
        **flags,
    )


def resolve_model_persistence(
    config: Mapping[str, Any],
    request: ModelCommandRequest,
) -> bool:
    if request.is_once or request.is_session:
        return False
    if request.is_global:
        return True
    model_cfg = config.get("model")
    if isinstance(model_cfg, Mapping):
        if not (model_cfg.get("default") or model_cfg.get("provider")):
            return True
        if request.explicit_provider:
            return False
        return bool(model_cfg.get("persist_switch_by_default", False))
    return not model_cfg


__all__ = ["ERROR_TEXT", "ModelCommandRequest", "parse_model_command", "resolve_model_persistence"]
