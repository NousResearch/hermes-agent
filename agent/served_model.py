"""Early model-response observers and served-model capture for routing proxies (#54864).

A LiteLLM-style proxy answers with the configured alias in the body's ``model`` field and puts
the deployment it really routed to in a response header (``x-litellm-model-id``, else the
upstream ``x-litellm-model-api-base``). The OpenAI SDK's parsed objects drop headers, so the
capture rides an ``httpx`` response hook on the client Hermes builds for the agent; it stores
the header onto ``agent.last_served_model`` (``None`` when the response carried none, so a
value never outlives the request that produced it). Consumers: ``agent/turn_finalizer.py``
(result ``served_model`` / ``requested_model``) and the opt-in gateway footer field
``served_model`` (``gateway/runtime_footer.py``). Fail-open everywhere.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

SERVED_MODEL_HEADERS: tuple[str, ...] = ("x-litellm-model-id", "x-litellm-model-api-base")
_HOOK_MARK = "_hermes_served_model_hook"


def _response_metadata(response: httpx.Response) -> dict[str, Any]:
    headers = response.headers

    def get_header(name: str) -> str | None:
        if not isinstance(name, str) or not 1 <= len(name) <= 128 or not name.isascii():
            return None
        values = headers.get_list(name)
        return values[0] if len(values) == 1 and len(values[0]) <= 4096 else None

    return {"status_code": response.status_code, "get_header": get_header}


def _notify_response(response: httpx.Response) -> None:
    from hermes_cli import plugins

    try:
        plugins.discover_plugins()
        if plugins.has_hook("llm_http_response"):
            plugins.invoke_hook("llm_http_response", **_response_metadata(response))
    except Exception as exc:  # health: allow BLE001 -- optional observers must not alter SDK response handling
        logger.warning("Model response observer skipped (%s)", type(exc).__name__)


async def _anotify_response(response: httpx.Response) -> None:
    from hermes_cli import plugins

    try:
        plugins.discover_plugins()
        if plugins.has_hook("llm_http_response"):
            await plugins.ainvoke_hook("llm_http_response", **_response_metadata(response))
    except Exception as exc:  # health: allow BLE001 -- optional observers must not alter SDK response handling
        logger.warning("Model response observer skipped (%s)", type(exc).__name__)


def install_llm_response_hook(client: Any) -> Any:
    """Observe headers before SDK parsing, without taking ownership of its HTTP client."""
    http_client = getattr(client, "_client", None)
    if isinstance(http_client, (httpx.Client, httpx.AsyncClient)):
        hook = _anotify_response if isinstance(http_client, httpx.AsyncClient) else _notify_response
        hooks = http_client.event_hooks
        if hook not in hooks.get("response", ()):
            http_client.event_hooks = {**hooks, "response": [hook, *hooks.get("response", ())]}
    return client


def served_model_from_headers(headers: Any) -> Optional[str]:
    """First non-empty served-model header, or ``None``."""
    if headers is None or not hasattr(headers, "get"):
        return None
    for name in SERVED_MODEL_HEADERS:
        value = headers.get(name)
        if value not in (None, ""):
            return str(value).strip()
    return None


def install_served_model_capture(agent: Any, client: Any) -> None:
    """Register the response hook on *client*'s ``httpx`` transport (idempotent per client)."""
    http_client = getattr(client, "_client", None)
    hooks = getattr(http_client, "event_hooks", None)
    if not isinstance(hooks, dict):
        return
    if any(getattr(h, _HOOK_MARK, False) for h in hooks.get("response", ())):
        return

    def _on_response(response: Any) -> None:
        try:
            if 200 <= int(getattr(response, "status_code", 0)) < 300:
                agent.last_served_model = served_model_from_headers(getattr(response, "headers", None))
        except Exception:
            logger.debug("served-model header capture skipped", exc_info=True)

    setattr(_on_response, _HOOK_MARK, True)
    try:
        # httpx copies on assignment; rebuild the mapping instead of mutating the live list.
        http_client.event_hooks = {**hooks, "response": [*hooks.get("response", ()), _on_response]}
    except Exception:
        logger.debug("served-model hook install skipped", exc_info=True)


def result_model_fields(agent: Any) -> dict[str, Optional[str]]:
    """``requested_model`` / ``served_model`` for the turn result: the proxy header when the
    last response carried one, else Hermes' own fallback route (primary → active model)."""
    served = getattr(agent, "last_served_model", None)
    requested = agent.model
    if not served and getattr(agent, "_fallback_activated", False):
        primary = str((getattr(agent, "_primary_runtime", None) or {}).get("model") or "").strip()
        if primary and primary != agent.model:
            requested, served = primary, agent.model
    return {"requested_model": requested, "served_model": served or None}
