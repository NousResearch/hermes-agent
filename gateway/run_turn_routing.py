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
        # Legacy/background callers are internal unless the external TurnRunner opts in explicitly.
        internal: bool = True,
    ) -> dict:
        """Build one turn route; middleware is fail-open and never owns credentials.

        The default keeps internal/background turns out of user-turn middleware. The external
        TurnRunner passes ``internal=False`` after loading the persisted turn history.
        """
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
                from hermes_cli.middleware import apply_turn_route_middleware, public_turn_route

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
                if result.changed and isinstance(result.payload, dict):
                    selected_model = result.payload.get("model")
                    public_runtime = result.payload.get("runtime")
                    public_runtime = public_runtime if isinstance(public_runtime, dict) else {}
                    current_requested = runtime.get("requested_provider") or runtime.get("provider")
                    current_canonical = runtime.get("provider")
                    top_requested = result.payload.get("requested_provider")
                    nested_requested = public_runtime.get("requested_provider")
                    requested = (
                        nested_requested
                        if nested_requested and nested_requested != current_requested
                        else (top_requested or nested_requested)
                    )
                    canonical = result.payload.get("provider") or public_runtime.get("provider")
                    selected_provider = (
                        requested
                        if requested and requested != current_requested
                        else (
                            canonical
                            if canonical and canonical != current_canonical
                            else (requested or canonical or current_requested)
                        )
                    )
                    if (
                        isinstance(selected_model, str)
                        and selected_model.strip()
                        and isinstance(selected_provider, str)
                        and selected_provider.strip()
                    ):
                        selected_model = selected_model.strip()
                        selected_provider = selected_provider.strip()
                        if selected_provider != current_requested:
                            from gateway.run import _resolve_runtime_agent_kwargs_for_provider

                            runtime, base_request_overrides = _project_runtime_agent_kwargs(
                                _resolve_runtime_agent_kwargs_for_provider(selected_provider)
                            )
                        runtime["requested_provider"] = selected_provider
                        route["model"] = selected_model
                        route["runtime"] = runtime
                        route["middleware_trace"] = result.trace
            except Exception as exc:
                logger.warning("Turn-route middleware failed open: %s", exc)
        if getattr(self, "_service_tier", None) != "priority":
            route["request_overrides"] = base_request_overrides
            return route
        try:
            overrides = resolve_fast_mode_overrides(
                route["model"], provider=runtime["provider"], base_url=runtime["base_url"],
            )
        except Exception:
            overrides = None
        route["request_overrides"] = _deep_merge_request_overrides(base_request_overrides, overrides or {})
        return route
