"""Model-authored, turn-scoped effort control for verified GPT-5.6 Codex.

The model may request a deeper effort through the existing ``todo`` tool. The
runtime never classifies task text or chooses effort on the model's behalf: it
only validates a static operator policy and applies the request to later API
calls in the exact originating turn.

Control state is agent-local and never restored from persisted receipts. Pro
mode is intentionally absent until its exact endpoint contract is canaried.
"""

from __future__ import annotations

import json
import threading
from copy import deepcopy
from typing import Any, Mapping


_EFFORT_ORDER = ("low", "medium", "high", "xhigh", "max")
_EFFORT_INDEX = {value: index for index, value in enumerate(_EFFORT_ORDER)}
_DEFAULT_BASELINE = "high"
_DEFAULT_CAP = "xhigh"
_MAX_CHANGES_PER_TURN = 4
_CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"


def _normalize_effort(value: Any, *, default: str = "") -> str:
    effort = str(value or "").strip().lower()
    if effort == "minimal":
        effort = "low"
    return effort if effort in _EFFORT_INDEX else default


def _is_verified_gpt56_codex(agent: Any) -> bool:
    """Mechanical capability gate; never examines user or task semantics."""
    if str(getattr(agent, "provider", "") or "").strip().lower() != "openai-codex":
        return False
    if str(getattr(agent, "api_mode", "") or "").strip().lower() != "codex_responses":
        return False

    model = str(getattr(agent, "model", "") or "").strip().lower()
    if model != "gpt-5.6-sol":
        return False
    base_url = str(getattr(agent, "base_url", "") or "").strip().rstrip("/")
    return base_url == _CODEX_BASE_URL


def _policy_from_config(agent_config: Mapping[str, Any] | None) -> dict[str, Any]:
    section: Mapping[str, Any] = {}
    if isinstance(agent_config, Mapping):
        candidate = agent_config.get("adaptive_reasoning", {})
        if isinstance(candidate, Mapping):
            section = candidate
        elif isinstance(candidate, bool):
            # Preserve the natural explicit opt-out shorthand.
            section = {"enabled": candidate}
        else:
            # A present but malformed authority section must not inherit the
            # shipped default-on policy.
            return {"enabled": False, "max_effort": _DEFAULT_CAP}

    # Missing means the shipped default (on); any malformed/non-boolean value
    # fails closed instead of accidentally enabling live authority.
    enabled = section.get("enabled", True) is True
    configured_cap = section.get("max_effort", _DEFAULT_CAP)
    max_effort = _normalize_effort(configured_cap)
    if not max_effort:
        enabled = False
        max_effort = _DEFAULT_CAP
    return {
        "enabled": bool(enabled),
        "max_effort": max_effort,
    }


def _lock_for(agent: Any) -> threading.Lock:
    lock = getattr(agent, "_turn_reasoning_lock", None)
    if lock is None:
        lock = threading.Lock()
        agent._turn_reasoning_lock = lock
    return lock


def configure_adaptive_reasoning(
    agent: Any, agent_config: Mapping[str, Any] | None
) -> None:
    """Initialize policy plus empty per-turn state on ``agent``."""
    lock = _lock_for(agent)
    with lock:
        agent._adaptive_reasoning_policy = _policy_from_config(agent_config)
        agent._turn_reasoning_override = None
        agent._turn_reasoning_changes = 0
        agent._adaptive_reasoning_turn_id = ""


def refresh_adaptive_reasoning_policy(
    agent: Any, agent_config: Mapping[str, Any] | None
) -> None:
    """Refresh live operator policy without rebuilding prompt/tool state.

    Disabling adaptive control or lowering its cap immediately revokes an
    incompatible in-turn override. A fresh gateway turn will then bind a new
    empty turn state through :func:`reset_adaptive_reasoning_turn`.
    """
    policy = _policy_from_config(agent_config)
    lock = _lock_for(agent)
    with lock:
        agent._adaptive_reasoning_policy = policy
        override = getattr(agent, "_turn_reasoning_override", None)
        override_effort = _normalize_effort(
            override.get("effort") if isinstance(override, dict) else None
        )
        cap = policy["max_effort"]
        if not policy["enabled"] or (
            override_effort
            and _EFFORT_INDEX[override_effort] > _EFFORT_INDEX[cap]
        ):
            agent._turn_reasoning_override = None
            agent._turn_reasoning_changes = 0


def reset_adaptive_reasoning_turn(agent: Any, turn_id: str) -> None:
    """Clear prior-turn authority and bind the empty state to ``turn_id``."""
    with _lock_for(agent):
        agent._turn_reasoning_override = None
        agent._turn_reasoning_changes = 0
        agent._adaptive_reasoning_turn_id = str(turn_id or "")


def _base_reasoning_config(agent: Any) -> dict[str, Any] | None:
    configured = getattr(agent, "reasoning_config", None)
    if isinstance(configured, dict):
        if configured.get("enabled") is False:
            return {"enabled": False}
        configured_effort = _normalize_effort(
            configured.get("effort"), default=_DEFAULT_BASELINE
        )
        # The verified GPT-5.6-sol route is quality-first even when an older
        # config still carries ``minimal``, ``low``, or ``medium``.  This is a
        # static capability floor, not a task classifier: no prompt, message,
        # tool argument, or task metadata is inspected.  Explicitly disabling
        # reasoning above remains authoritative, while an operator-selected
        # deeper baseline (xhigh/max) is preserved.
        effort = max(
            (configured_effort, _DEFAULT_BASELINE),
            key=lambda item: _EFFORT_INDEX[item],
        )
        return {"enabled": True, "effort": effort}
    # Only the verified GPT-5.6 Codex route receives the quality-first default.
    if _is_verified_gpt56_codex(agent):
        return {"enabled": True, "effort": _DEFAULT_BASELINE}
    return configured


def effective_reasoning_config(agent: Any) -> dict[str, Any] | None:
    """Return the baseline overlaid by this exact turn's accepted request."""
    # Non-verified providers and transports must retain their byte-for-byte
    # historical reasoning contract. Adaptive normalization is never allowed
    # to spill into Copilot, GitHub, xAI, custom endpoints, or older models.
    if not _is_verified_gpt56_codex(agent):
        return getattr(agent, "reasoning_config", None)

    base = _base_reasoning_config(agent)
    with _lock_for(agent):
        policy = getattr(agent, "_adaptive_reasoning_policy", {}) or {}
        if not policy.get("enabled"):
            return base
        if isinstance(base, dict) and base.get("enabled") is False:
            return base

        turn_id = str(getattr(agent, "_current_turn_id", "") or "")
        bound_turn_id = str(getattr(agent, "_adaptive_reasoning_turn_id", "") or "")
        override = getattr(agent, "_turn_reasoning_override", None)
        if not turn_id or turn_id != bound_turn_id or not isinstance(override, dict):
            return base
        return {"enabled": True, "effort": override["effort"]}


def preserve_verified_reasoning_payload(
    agent: Any,
    authoritative_request: Any,
    candidate_request: Any,
) -> Any:
    """Keep middleware mechanical on the verified adaptive route.

    Request/execution middleware may still rewrite mechanical request fields,
    but it cannot swap the verified model or originate, lower, or raise the
    turn's model-authored effort. Non-verified routes retain the historical
    middleware contract unchanged.
    """
    if not _is_verified_gpt56_codex(agent):
        return candidate_request
    if not isinstance(authoritative_request, Mapping):
        return candidate_request
    if not isinstance(candidate_request, Mapping):
        candidate_request = authoritative_request
    preserved = dict(candidate_request)
    if "model" in authoritative_request:
        preserved["model"] = deepcopy(authoritative_request["model"])
    if "reasoning" in authoritative_request:
        preserved["reasoning"] = deepcopy(authoritative_request["reasoning"])
    else:
        preserved.pop("reasoning", None)
    return preserved


def _receipt(status: str, **values: Any) -> dict[str, Any]:
    receipt = {
        "status": status,
        "scope": "current_turn",
        "expires": "end_of_current_turn",
    }
    receipt.update(values)
    return receipt


def apply_model_reasoning_directive(
    agent: Any,
    directive: Any,
    *,
    originating_turn_id: str,
) -> dict[str, Any]:
    """Atomically validate and apply one originating-turn effort request."""
    if not isinstance(directive, dict):
        return _receipt("rejected", reason="reasoning_must_be_an_object")
    lock = _lock_for(agent)
    with lock:
        if "mode" in directive:
            return _receipt("rejected", reason="reasoning_mode_unverified")
        policy = getattr(agent, "_adaptive_reasoning_policy", {}) or {}
        if not policy.get("enabled"):
            return _receipt("rejected", reason="adaptive_reasoning_disabled")
        if not _is_verified_gpt56_codex(agent):
            return _receipt("rejected", reason="unsupported_model_or_transport")

        current_turn_id = str(getattr(agent, "_current_turn_id", "") or "")
        bound_turn_id = str(getattr(agent, "_adaptive_reasoning_turn_id", "") or "")
        origin = str(originating_turn_id or "")
        if not origin or origin != current_turn_id or origin != bound_turn_id:
            return _receipt("rejected", reason="originating_turn_expired")

        base = _base_reasoning_config(agent)
        if isinstance(base, dict) and base.get("enabled") is False:
            return _receipt("rejected", reason="reasoning_disabled_by_user")
        baseline = _normalize_effort(
            base.get("effort") if isinstance(base, dict) else None,
            default=_DEFAULT_BASELINE,
        )
        effort = _normalize_effort(directive.get("effort"))
        if not effort:
            return _receipt("rejected", reason="unsupported_effort")
        if _EFFORT_INDEX[effort] < _EFFORT_INDEX[baseline]:
            return _receipt(
                "rejected",
                reason="below_user_baseline",
                baseline_effort=baseline,
            )

        cap = _normalize_effort(policy.get("max_effort"), default=_DEFAULT_CAP)
        if _EFFORT_INDEX[effort] > _EFFORT_INDEX[cap]:
            return _receipt("rejected", reason="above_policy_cap", max_effort=cap)

        current = getattr(agent, "_turn_reasoning_override", None)
        current_effort = _normalize_effort(
            current.get("effort") if isinstance(current, dict) else None,
            default=baseline,
        )
        effective_effort = max(
            (current_effort, effort), key=lambda item: _EFFORT_INDEX[item]
        )
        effective = {"effort": effective_effort}
        if current == effective:
            return _receipt(
                "unchanged",
                effective=effective,
                change_count=int(getattr(agent, "_turn_reasoning_changes", 0) or 0),
            )

        changes = int(getattr(agent, "_turn_reasoning_changes", 0) or 0)
        if changes >= _MAX_CHANGES_PER_TURN:
            return _receipt(
                "rejected",
                reason="turn_change_limit_reached",
                max_changes=_MAX_CHANGES_PER_TURN,
            )
        agent._turn_reasoning_override = effective
        agent._turn_reasoning_changes = changes + 1
        return _receipt(
            "applied",
            effective=effective,
            change_count=changes + 1,
        )


def attach_reasoning_receipt(
    agent: Any,
    tool_result: Any,
    directive: Any,
    *,
    originating_turn_id: str,
) -> Any:
    """Attach an application receipt to a successful ``todo`` JSON result."""
    if directive is None or not isinstance(tool_result, str):
        return tool_result
    try:
        payload = json.loads(tool_result)
    except (TypeError, json.JSONDecodeError):
        return tool_result
    if not isinstance(payload, dict) or payload.get("error"):
        return tool_result
    payload["reasoning_control"] = apply_model_reasoning_directive(
        agent,
        directive,
        originating_turn_id=originating_turn_id,
    )
    return json.dumps(payload, ensure_ascii=False)
