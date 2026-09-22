"""Fail-closed per-turn routing through a local semantic classifier.

The classifier may choose only route labels. Hermes owns the allowlisted mapping from a
label to provider, model, credentials, and reasoning configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
import ipaddress
import json
import math
import re
from collections.abc import Callable, Mapping
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


_SECRET_PATTERN = re.compile(r"\b(?:sk|pk|api)[_-][A-Za-z0-9_-]{12,}\b", re.IGNORECASE)
_SENSITIVE_TEXT_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z0-9 ]{1,80}-----.*?-----END [A-Z0-9 ]{1,80}-----", re.IGNORECASE | re.DOTALL),
    re.compile(r"\bBearer\s+[^\s;,]+", re.IGNORECASE),
    re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,})\b", re.IGNORECASE),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b", re.IGNORECASE),
    re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b"),
    re.compile(r"\b(?:password|passwd|secret|token|api[_-]?key|cookie|session(?:id)?|sid)\s*[:=]\s*[^\s;,]+", re.IGNORECASE),
)
_OUTCOME_SIGNALS = (
    ("failure", re.compile(r"\b(?:failed|failure|error|exception|fatal)\b", re.IGNORECASE)),
    ("timeout", re.compile(r"\b(?:timed?\s*out|deadline\s+exceeded)\b", re.IGNORECASE)),
    ("server_error", re.compile(r"\b5\d{2}\b|\b(?:internal|server)\s+error\b", re.IGNORECASE)),
    ("permission", re.compile(r"\b(?:permission|forbidden|unauthorized|access denied)\b", re.IGNORECASE)),
    ("not_found", re.compile(r"\b(?:not found|no such)\b", re.IGNORECASE)),
    ("rate_limited", re.compile(r"\b(?:rate limit|too many requests|quota)\b", re.IGNORECASE)),
)


@dataclass(frozen=True)
class TurnRoute:
    model: str
    runtime: dict[str, Any]
    reasoning_config: dict[str, Any] | None
    decision: str
    reason: str
    request_overrides: dict[str, Any] | None = None


def _router_config(config: Mapping[str, Any] | None) -> Mapping[str, Any]:
    raw = (config or {}).get("model_router")
    return raw if isinstance(raw, Mapping) else {}


def _safe_goal(message: Any, max_chars: int) -> str:
    from agent.monitoring.redaction import redact_for_export
    from agent.redact import REDACTION_UNAVAILABLE, redact_sensitive_text

    try:
        text = redact_sensitive_text(
            str(message or ""), force=True, redact_url_credentials=True,
        )
    except Exception:
        text = REDACTION_UNAVAILABLE
    text = redact_for_export(text) or ""
    text = _SECRET_PATTERN.sub("[REDACTED]", text)
    for pattern in _SENSITIVE_TEXT_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    return text[:max(1, max_chars)]


def _tool_outcome_projection(outcome: Any, max_chars: int) -> dict[str, Any]:
    """Expose bounded difficulty metadata, never the completed tool-result text itself."""
    observed = str(outcome or "")[:max(1, max_chars)]
    signals = [name for name, pattern in _OUTCOME_SIGNALS if pattern.search(observed)]
    return {
        "kind": "tool_outcome_metadata",
        "character_count": len(observed),
        "line_count": observed.count("\n") + 1,
        "signals": signals or ["completed"],
    }


def _controller_auth_token(router: Mapping[str, Any]) -> str:
    """A raw outcome requires both an explicit opt-in and an authenticated controller."""
    token = router.get("controller_auth_token")
    return token.strip() if isinstance(token, str) else ""


def _call_controller(
    controller: Callable[..., Mapping[str, Any]], url: str, timeout_seconds: float,
    state: dict[str, Any], candidates: dict[str, str], *, auth_token: str,
) -> Mapping[str, Any]:
    # Test/embedded controllers retain the established four-argument callable contract. Only
    # the network transport accepts the auth token and writes the Authorization header.
    if controller is post_route_decision:
        return controller(url, timeout_seconds, state, candidates, auth_token=auth_token)
    return controller(url, timeout_seconds, state, candidates)


def _routes(router: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    raw = router.get("routes")
    if not isinstance(raw, Mapping):
        return {}
    return {
        str(name): target for name, target in raw.items()
        if isinstance(name, str) and isinstance(target, Mapping)
        and isinstance(target.get("provider"), str) and target["provider"].strip()
        and isinstance(target.get("model"), str) and target["model"].strip()
        and isinstance(target.get("description"), str) and target["description"].strip()
    }


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        return None


def _require_loopback_http_url(url: str) -> None:
    parsed = urlsplit(url)
    try:
        loopback = bool(parsed.hostname) and ipaddress.ip_address(parsed.hostname).is_loopback
    except ValueError:
        loopback = False
    if parsed.scheme != "http" or not loopback:
        raise ValueError("model_router controller_url must be an IP-literal loopback http URL")


def post_route_decision(
    url: str, timeout_seconds: float, state: dict[str, Any], candidates: dict[str, str], *, auth_token: str = "",
) -> Mapping[str, Any]:
    _require_loopback_http_url(url)
    payload = json.dumps({"state": state, "candidates": candidates}, ensure_ascii=False).encode("utf-8")
    if len(payload) > 20_000:
        raise ValueError("model_router request exceeds 20000 bytes")
    headers = {"Content-Type": "application/json"}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
    request = Request(url, data=payload, headers=headers, method="POST")
    try:
        with build_opener(ProxyHandler({}), _NoRedirect).open(request, timeout=timeout_seconds) as response:  # nosec B310 -- loopback is checked above
            if response.status != 200:
                raise RuntimeError(f"controller HTTP {response.status}")
            raw = response.read(65_537)
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError("controller unavailable") from exc
    if len(raw) > 65_536:
        raise ValueError("controller response exceeds 65536 bytes")
    parsed = json.loads(raw)
    if not isinstance(parsed, Mapping):
        raise ValueError("controller response must be an object")
    return parsed


def _target_route(
    route_name: str,
    routes: Mapping[str, Mapping[str, Any]],
    *,
    reason: str,
    runtime_resolver: Callable[[str, str], Mapping[str, Any]],
    router_config: Mapping[str, Any],
) -> TurnRoute:
    target = routes[route_name]
    provider = str(target["provider"]).strip()
    if provider.lower() in {"auto", "default"}:
        raise ValueError("model_router target must name a concrete provider")
    model = str(target["model"]).strip()
    resolved_runtime = dict(runtime_resolver(provider, model))
    resolved_provider = str(resolved_runtime.get("provider") or "").strip()
    resolved_requested = str(resolved_runtime.get("requested_provider") or "").strip()
    provider_matches = resolved_provider.lower() == provider.lower()
    named_custom_matches = (
        resolved_provider.lower() == "custom"
        and resolved_requested.lower() == provider.lower()
    )
    if not (provider_matches or named_custom_matches) or (
            resolved_requested and resolved_requested.lower() != provider.lower()):
        raise ValueError("model_router resolver did not honor the configured provider")
    if resolved_runtime.get("command") or resolved_runtime.get("args"):
        raise ValueError("model_router does not permit external-command targets")
    if resolved_runtime.get("api_mode") not in {"chat_completions", "responses", "codex_responses"}:
        raise ValueError("model_router target has unsupported API mode")
    runtime = {key: resolved_runtime.get(key) for key in (
        "api_key", "base_url", "provider", "requested_provider", "api_mode", "credential_pool")}
    runtime["requested_provider"] = runtime.get("requested_provider") or provider
    from hermes_constants import parse_reasoning_effort, resolve_reasoning_config
    configured_effort = target.get("reasoning_effort")
    reasoning_config = (
        parse_reasoning_effort(configured_effort)
        if configured_effort not in (None, "")
        else resolve_reasoning_config(dict(router_config), model)
    )
    return TurnRoute(
        model=model,
        runtime=runtime,
        reasoning_config=reasoning_config,
        decision=route_name,
        reason=reason,
        request_overrides=dict(resolved_runtime.get("request_overrides") or {}),
    )


def resolve_turn_route(
    *,
    config: Mapping[str, Any] | None,
    user_message: Any,
    base_model: str,
    base_runtime: Mapping[str, Any],
    runtime_resolver: Callable[[str, str], Mapping[str, Any]],
    controller: Callable[[str, float, dict[str, Any], dict[str, str]], Mapping[str, Any]] = post_route_decision,
) -> TurnRoute:
    """Resolve a configured per-turn route, falling closed to the configured exception route."""
    router = _router_config(config)
    if router.get("enabled") is not True:
        return TurnRoute(base_model, dict(base_runtime), None, "disabled", "disabled")

    routes = _routes(router)
    fallback = str(router.get("fallback_route") or "exception")
    if fallback not in routes:
        raise ValueError("enabled model_router needs a valid fallback_route")
    minimum_confidence = router.get("minimum_confidence")
    if not isinstance(minimum_confidence, (int, float)) or isinstance(minimum_confidence, bool) or not math.isfinite(float(minimum_confidence)) or not 0 <= float(minimum_confidence) <= 1:
        raise ValueError("enabled model_router needs minimum_confidence between zero and one")

    max_chars = int(router.get("message_max_chars") or 4096)
    candidates = {name: str(target["description"]) for name, target in routes.items()}
    try:
        response = _call_controller(
            controller,
            str(router.get("controller_url") or "http://127.0.0.1:18777/v1/route"),
            float(router.get("timeout_seconds") or 5),
            {"goal": _safe_goal(user_message, max_chars)},
            candidates,
            auth_token=_controller_auth_token(router),
        )
        choice = response.get("choice") if isinstance(response, Mapping) else None
        confidence = response.get("confidence") if isinstance(response, Mapping) else None
        if choice not in routes:
            return _target_route(fallback, routes, reason="invalid_choice", runtime_resolver=runtime_resolver, router_config=config or {})
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            return _target_route(fallback, routes, reason="invalid_confidence", runtime_resolver=runtime_resolver, router_config=config or {})
        if confidence < float(minimum_confidence):
            return _target_route(fallback, routes, reason="low_confidence", runtime_resolver=runtime_resolver, router_config=config or {})
        return _target_route(choice, routes, reason="controller", runtime_resolver=runtime_resolver, router_config=config or {})
    except Exception:
        return _target_route(fallback, routes, reason="controller_error", runtime_resolver=runtime_resolver, router_config=config or {})


def resolve_midturn_route(
    *,
    config: Mapping[str, Any] | None,
    user_message: Any,
    tool_outcome: Any,
    base_model: str,
    base_runtime: Mapping[str, Any],
    runtime_resolver: Callable[[str, str], Mapping[str, Any]],
    controller: Callable[[str, float, dict[str, Any], dict[str, str]], Mapping[str, Any]] = post_route_decision,
) -> TurnRoute:
    """Resolve a request-local post-tool route without selecting an implicit fallback.

    Mid-turn callers retain the active route on every uncertain condition.  Unlike the
    turn-start router, this function never turns an unavailable controller into a model
    change because the agent already has a live, validated runtime for the next request.
    """
    router = _router_config(config)
    mid_turn = router.get("mid_turn")
    if router.get("enabled") is not True or not isinstance(mid_turn, Mapping) or mid_turn.get("enabled") is not True:
        return TurnRoute(base_model, dict(base_runtime), None, "disabled", "disabled")

    routes = _routes(router)
    strong_route = str(mid_turn.get("strong_route") or "")
    if strong_route not in routes:
        return TurnRoute(base_model, dict(base_runtime), None, "unchanged", "invalid_strong_route")
    minimum_confidence = router.get("minimum_confidence")
    if not isinstance(minimum_confidence, (int, float)) or isinstance(minimum_confidence, bool) or not math.isfinite(float(minimum_confidence)) or not 0 <= float(minimum_confidence) <= 1:
        return TurnRoute(base_model, dict(base_runtime), None, "unchanged", "invalid_minimum_confidence")
    try:
        goal_max_chars = int(router.get("message_max_chars") or 4096)
        outcome_max_chars = int(mid_turn.get("tool_outcome_max_chars") or 2048)
    except (TypeError, ValueError):
        return TurnRoute(base_model, dict(base_runtime), None, "unchanged", "invalid_max_chars")
    if outcome_max_chars < 1:
        return TurnRoute(base_model, dict(base_runtime), None, "unchanged", "invalid_max_chars")
    if not str(tool_outcome or ""):
        return TurnRoute(base_model, dict(base_runtime), None, "unchanged", "no_tool_outcome")

    authenticated_raw_outcome = (
        mid_turn.get("authenticated_raw_tool_outcome") is True
        and bool(_controller_auth_token(router))
    )
    outcome_state: Any = (
        _safe_goal(tool_outcome, outcome_max_chars)
        if authenticated_raw_outcome
        else _tool_outcome_projection(tool_outcome, outcome_max_chars)
    )

    candidates = {name: str(target["description"]) for name, target in routes.items()}
    try:
        response = _call_controller(
            controller,
            str(router.get("controller_url") or "http://127.0.0.1:18777/v1/route"),
            float(router.get("timeout_seconds") or 5),
            {
                "goal": _safe_goal(user_message, goal_max_chars),
                "completed_tool_outcome": outcome_state,
            },
            candidates,
            auth_token=_controller_auth_token(router),
        )
        choice = response.get("choice") if isinstance(response, Mapping) else None
        confidence = response.get("confidence") if isinstance(response, Mapping) else None
        if choice not in routes:
            return TurnRoute(base_model, dict(base_runtime), None, "unchanged", "invalid_choice")
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            return TurnRoute(base_model, dict(base_runtime), None, "unchanged", "invalid_confidence")
        if confidence < float(minimum_confidence):
            return TurnRoute(base_model, dict(base_runtime), None, "unchanged", "low_confidence")
        if choice in {"planning", "exception"}:
            choice = strong_route
            reason = "strong_route"
        else:
            reason = "controller"
        return _target_route(choice, routes, reason=reason, runtime_resolver=runtime_resolver, router_config=config or {})
    except Exception:
        return TurnRoute(base_model, dict(base_runtime), None, "unchanged", "controller_error")
