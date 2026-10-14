"""Turn-route selection for the Gateway runner.

The route is resolved before agent construction.  Hermes keeps credentials and transport
ownership; middleware only selects a public model/provider identity for this turn.
"""

from __future__ import annotations

import logging
from typing import Optional

from gateway.session import SessionSource

logger = logging.getLogger("gateway.run")


def _project_runtime_agent_kwargs(runtime_kwargs: dict) -> tuple[dict, dict]:
    """Split provider resolution output into public agent runtime and request overrides."""
    runtime = {
        key: runtime_kwargs.get(key) for key in (
            "api_key", "base_url", "provider", "requested_provider", "api_mode", "command", "args",
            "credential_pool", "max_tokens", "capabilities",
        )
    }
    runtime["args"] = list(runtime["args"] or [])
    runtime["capabilities"] = dict(runtime["capabilities"] or {})
    return runtime, dict(runtime_kwargs.get("request_overrides") or {})


class GatewayTurnRoutingMixin:
    """Resolve the effective public route and host-owned runtime for one Gateway turn."""

    def _resolve_turn_agent_config(
        self, user_message: str, model: str, runtime_kwargs: dict,
        *, session_id: Optional[str] = None, session_key: Optional[str] = None,
        source: Optional[SessionSource] = None, conversation_history: Optional[list] = None,
        # Every caller must classify the turn explicitly. Omitting this would silently bypass
        # external-user routing when a new entry point is added.
        internal: bool,
    ) -> dict:
        """Build one turn route; middleware is fail-open and never owns credentials.

        The default keeps internal/background turns out of user-turn middleware. The external
        TurnRunner passes the event's internal identity after loading persisted turn history.
        """
        from agent.fast_mode import STATIC_TIERS
        from gateway.run import _deep_merge_request_overrides
        from hermes_cli.models import resolve_fast_mode_overrides

        runtime, base_request_overrides = _project_runtime_agent_kwargs(runtime_kwargs)
        route = {
            "model": model,
            "runtime": runtime,
            "signature": (
                model, runtime["provider"], runtime["requested_provider"], runtime["base_url"],
                runtime["api_mode"], runtime["command"], tuple(runtime["args"]),
            ),
        }
        if not internal:
            try:
                from hermes_cli.middleware import (
                    apply_turn_route_middleware, normalize_turn_route_selection,
                    public_turn_route, turn_route_credentials_usable,
                )

                result = apply_turn_route_middleware(
                    public_turn_route(route["model"], runtime),
                    user_message=user_message,
                    session_id=session_id,
                    session_key=session_key,
                    source=source.platform.value if source and source.platform else "gateway",
                    is_user_turn=True,
                    is_first_turn=not bool(conversation_history),
                    internal=False,
                    tool_continuation=False,
                )
                # An unchanged route is still a decision; observers need its reason.
                if not result.changed:
                    route["middleware_trace"] = list(result.trace or [])
                elif (selection := normalize_turn_route_selection(route, result.payload)):
                    selected_model, selected_provider = selection
                    current_requested = runtime.get("requested_provider") or runtime.get("provider")
                    if selected_provider != current_requested or selected_model != model:
                        if selected_provider == current_requested:
                            # Resolve target-model settings while retaining this session's endpoint and auth.
                            from hermes_cli.runtime_provider import resolve_runtime_provider

                            target = resolve_runtime_provider(
                                requested=selected_provider, target_model=selected_model,
                                explicit_api_key=runtime.get("api_key"), explicit_base_url=runtime.get("base_url"),
                            )
                            if not turn_route_credentials_usable(target):
                                raise RuntimeError("Turn-route selected OpenRouter without credentials")
                            target_runtime, base_request_overrides = _project_runtime_agent_kwargs(target)
                            target_runtime.update({
                                "api_key": runtime.get("api_key"), "base_url": runtime.get("base_url"),
                                "provider": runtime.get("provider"), "requested_provider": selected_provider,
                                "credential_pool": runtime.get("credential_pool"),
                            })
                            runtime = target_runtime
                        else:
                            if selected_provider == "custom" and runtime.get("base_url"):
                                from hermes_cli.runtime_provider import resolve_runtime_provider

                                target = resolve_runtime_provider(
                                    requested=selected_provider, target_model=selected_model,
                                    explicit_api_key=runtime.get("api_key"), explicit_base_url=runtime.get("base_url"),
                                )
                                if not turn_route_credentials_usable(target):
                                    raise RuntimeError("Turn-route selected OpenRouter without credentials")
                                runtime, base_request_overrides = _project_runtime_agent_kwargs(target)
                            else:
                                from gateway.run import _resolve_runtime_agent_kwargs_for_provider

                                runtime, base_request_overrides = _project_runtime_agent_kwargs(
                                    _resolve_runtime_agent_kwargs_for_provider(
                                        selected_provider, target_model=selected_model
                                    )
                                )
                            if not turn_route_credentials_usable(runtime):
                                raise RuntimeError("Turn-route selected OpenRouter without credentials")
                    runtime["requested_provider"] = selected_provider
                    route["model"] = selected_model
                    route["runtime"] = runtime
                    route["middleware_trace"] = result.trace
                if result.changed and "middleware_trace" not in route:
                    # A rejected route falls back like a failed middleware; say so instead of silently.
                    logger.warning("Turn-route middleware returned an unusable route; using the configured route")
            except Exception as exc:
                logger.warning("Turn-route middleware failed open: %s", exc, exc_info=True)
        tier = getattr(self, "_service_tier", None)
        if tier not in STATIC_TIERS:
            route["request_overrides"] = base_request_overrides
            return route
        try:
            overrides = resolve_fast_mode_overrides(
                route["model"], provider=runtime["provider"], base_url=runtime["base_url"], tier=tier,
            )
        except Exception as exc:
            logger.debug("Fast-mode override resolution failed; using base route: %s", exc, exc_info=True)
            overrides = None
        route["request_overrides"] = _deep_merge_request_overrides(base_request_overrides, overrides or {})
        return route
