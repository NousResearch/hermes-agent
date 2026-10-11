"""Per-model quality, friction and context-pressure shared metrics.

Tool-call quality is counted where the agent validates the calls a model emitted (every call,
clean ones as ``issue=none``, so rates have a denominator). Friction and context peaks are
attributed to the model that produced the turn, which the relay runtime tracks per metrics session
in :class:`ModelSessionState`. Provider/model always go through ``model_call_fields`` (the catalog
helpers), so custom endpoints and loopback servers read ``custom``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Iterable

from . import shared_metrics_contract as contract
from . import shared_metrics_fields as fields_

logger = logging.getLogger(__name__)

# A session that ends this soon after a failed turn reads as the user giving up on the model.
QUICK_ABANDON_NS = 60 * 1_000_000_000
# Lower bound inclusive, so 128k (128_000 and 131_072) and 1M windows land in one bucket each.
_WINDOW_THRESHOLDS = (
    (32_000, "lt_32k"), (128_000, "32k_to_128k"), (256_000, "128k_to_256k"), (1_000_000, "256k_to_1m"),
)
# The rejections Hermes answers with a forced (overflow-triggered) compression.
_LIMIT_ERROR_CLASSES = frozenset({"context_overflow", "payload_too_large"})
# Unattended runs have no user to interrupt or abandon them.
_UNATTENDED_ENTRYPOINTS = frozenset({"background", "batch", "delegated", "scheduled_task"})


def model_route(provider: Any, model: Any) -> dict[str, str]:
    return contract.model_call_fields({"provider": provider, "model": model})


def window_bucket(context_length: Any) -> str:
    length = contract._non_negative_number(context_length)
    return contract._bucket(length, _WINDOW_THRESHOLDS, "gte_1m") if length else "unknown"


def friction_fields(signal: str, route: dict[str, str] | None) -> dict[str, str] | None:
    if signal not in contract.FRICTION_SIGNALS:
        return None
    return {**(route or model_route(None, None)), "signal": signal}


def _prompt_tokens(usage: Any) -> float | None:
    if not isinstance(usage, dict):
        return None
    tokens = contract._non_negative_number(usage.get("prompt_tokens"))
    return tokens if tokens is not None else contract._non_negative_number(usage.get("input_tokens"))


@dataclass
class ModelSessionState:
    """Per-session model attribution the runtime keeps between hooks (caller holds session.lock)."""

    last_route: dict[str, str] | None = None
    peak_route: dict[str, str] | None = None
    peak_tokens: float = -1.0
    peak_window: float | None = None
    limit_hit: bool = False
    failed_turn_ns: int = 0
    failed_route: dict[str, str] | None = None

    def observe_call(self, route: dict[str, str], usage: Any, context_length: Any) -> None:
        """One finished primary call: remember its model and the fullest context seen."""
        self.last_route = route
        tokens, window = _prompt_tokens(usage), contract._non_negative_number(context_length)
        if self.peak_route is None:
            self.peak_route = route
        if tokens is None or not window:
            return
        if self.peak_window is None or tokens / window > self.peak_tokens / self.peak_window:
            self.peak_route, self.peak_tokens, self.peak_window = route, tokens, window

    def observe_error(self, route: dict[str, str], error_class: str) -> None:
        """A failed primary call still names the model: a session whose every call overflowed
        is exactly the one the context peak row must report."""
        self.last_route = route
        if self.peak_route is None:
            self.peak_route = route
        if error_class in _LIMIT_ERROR_CLASSES:
            self.limit_hit = True

    def observe_turn(self, outcome: str, route: dict[str, str] | None, now_ns: int) -> None:
        """Only the trailing turn matters for abandonment: a later success clears it."""
        failed = outcome == "failed"
        self.failed_turn_ns = now_ns if failed else 0
        self.failed_route = (route or self.last_route) if failed else None

    def quick_abandon_route(self, now_ns: int) -> dict[str, str] | None:
        if self.failed_turn_ns and now_ns - self.failed_turn_ns <= QUICK_ABANDON_NS:
            return self.failed_route or self.last_route
        return None

    def context_peak_fields(self) -> dict[str, str] | None:
        """One row per session that reached a primary model; None when no call was made."""
        if self.peak_route is None:
            return None
        known = self.peak_window is not None
        return {
            **self.peak_route,
            "limit_hit": "yes" if self.limit_hit else "no",
            "peak_fill_bucket": (
                fields_.context_fill_bucket(self.peak_tokens, self.peak_window) if known else "unknown"
            ),
            "window_bucket": window_bucket(self.peak_window) if known else "unknown",
        }


def attended(start_fields: dict[str, str] | None) -> bool:
    return bool(start_fields) and start_fields.get("entrypoint") not in _UNATTENDED_ENTRYPOINTS


# ---- tool-call quality -----------------------------------------------------------------------

def _required_params(tools: Iterable[Any] | None) -> dict[str, tuple[str, ...]]:
    required: dict[str, tuple[str, ...]] = {}
    for tool in tools or ():
        fn = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(fn, dict) or not isinstance(fn.get("name"), str):
            continue
        params = fn.get("parameters")
        names = params.get("required") if isinstance(params, dict) else None
        required[fn["name"]] = tuple(n for n in names if isinstance(n, str)) if isinstance(names, list) else ()
    return required


def tool_call_issue(
    name: Any, raw_args: Any, *, valid_names: Any, required: dict[str, tuple[str, ...]], repaired: bool,
) -> str:
    """The first problem with one emitted call, most severe first; ``none`` when clean.

    Empty arguments are only an issue when the tool declares required parameters: for a
    parameterless tool Hermes's normalization to ``{}`` is exactly what the model meant.
    """
    if name not in valid_names:
        return "unknown_tool"
    needed = required.get(name, ())
    if isinstance(raw_args, (dict, list)):
        parsed = raw_args
    elif raw_args is None or not str(raw_args).strip():
        return "empty_arguments" if needed else ("repaired" if repaired else "none")
    else:
        try:
            parsed = json.loads(str(raw_args))
        except (TypeError, ValueError):
            return "invalid_json"
    if not isinstance(parsed, dict) or any(key not in parsed for key in needed):
        return "schema_mismatch"
    return "repaired" if repaired else "none"


def record_tool_call_quality(agent: Any, tool_calls: Iterable[Any], repaired_ids: frozenset[int] | set[int]) -> None:
    """Count every tool call one primary response emitted, before Hermes normalizes it.

    ``repaired_ids`` holds ``id(tc)`` of calls whose name Hermes auto-repaired; streamed calls
    whose argument JSON was repaired carry ``function.args_repaired``. No-op unless enabled.
    """
    try:
        from .relay_shared_metrics import enabled, record_process_mark

        if not tool_calls or not enabled():
            return
        from .shared_metrics_signals import tool_unavailable_fields
        route = model_route(getattr(agent, "provider", None), getattr(agent, "model", None))
        valid_names = getattr(agent, "valid_tool_names", None) or frozenset()
        required = _required_params(getattr(agent, "tools", None))
        for tc in tool_calls:
            fn = tc.function
            issue = tool_call_issue(
                fn.name, fn.arguments, valid_names=valid_names, required=required,
                repaired=id(tc) in repaired_ids or getattr(fn, "args_repaired", False) is True,
            )
            record_process_mark(contract.MODEL_TOOL_QUALITY_MARK, {**route, "call_role": "primary", "issue": issue})
            if (unavailable := tool_unavailable_fields(agent, fn.name, issue, route)) is not None:
                record_process_mark(contract.TOOL_UNAVAILABLE_MARK, unavailable)
    except Exception:
        logger.debug("Shared-metrics tool-call quality not recorded", exc_info=True)


# ---- friction --------------------------------------------------------------------------------

def record_model_friction(
    signal: str, *, session_id: Any = None, agent: Any = None, provider: Any = None, model: Any = None,
    hermes_home: Any = None, turns: int = 1,
) -> None:
    """Count one user friction action against the model that produced the session's last turn.
    ``turns``: how many turns an /undo removed (their tokens count as wasted).

    Falls back to ``agent`` (or ``provider``/``model``) when this process never saw the session's
    turns (restart, remote compute host). ``hermes_home`` binds the owning profile for callers that
    run outside it (multiplexed gateway, Desktop sessions of a secondary profile).
    """
    token = None
    try:
        if hermes_home:
            from hermes_constants import set_hermes_home_override

            token = set_hermes_home_override(str(hermes_home))
        from .relay_shared_metrics import enabled, record_session_friction

        if signal not in contract.FRICTION_SIGNALS or not enabled():
            return
        if agent is not None:
            provider, model = getattr(agent, "provider", provider), getattr(agent, "model", model)
            session_id = getattr(agent, "session_id", None) or session_id
        record_session_friction(signal, str(session_id or ""), model_route(provider, model), turns)
    except Exception:
        logger.debug("Shared-metrics %s friction not recorded", signal, exc_info=True)
    finally:
        if token is not None:
            from hermes_constants import reset_hermes_home_override

            reset_hermes_home_override(token)
