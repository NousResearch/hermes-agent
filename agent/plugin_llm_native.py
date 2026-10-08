"""Trust checks and accounting for plugin native Responses requests."""

from __future__ import annotations

import logging
import uuid
from types import SimpleNamespace
from typing import Any

logger = logging.getLogger(__name__)


def complete_native_for_plugin(
    facade: Any, *, native_request: dict[str, Any], route_context: dict[str, Any],
    expected_session_id: str, timeout: float, purpose: str | None, task: str | None,
) -> Any:
    from agent.auxiliary_native import complete_native_request, native_attempt_scope, verify_route_context
    from agent.plugin_llm import PluginLlmNativeResult, _check_overrides, _check_task, _extract_usage

    policy = facade._policy_loader(facade._plugin_id)
    effective_task = _check_task(policy, plugin_id=facade._plugin_id, requested_task=task)
    route = verify_route_context(route_context, expected_session_id)
    _check_overrides(
        policy, requested_provider=route["provider"], requested_model=route["model"],
        requested_agent_id=None, requested_profile=None,
    )
    metadata = {"api_request_id": f"aux-{uuid.uuid4().hex}", "retry_count": 0}
    with native_attempt_scope(aux_task=effective_task or "plugin_native", metadata=metadata, provider=route["provider"]):
        response = complete_native_request(native_request, route, timeout, task=effective_task or "plugin_native")
    usage = _extract_usage(SimpleNamespace(usage=response.get("usage")))
    audit = dict(plugin_id=facade._plugin_id, purpose=purpose or "", task=effective_task or "",
                 provider=route["provider"], model=route["model"], api_request_id=metadata["api_request_id"])
    logger.info("plugin_llm.complete_native plugin=%s provider=%s model=%s task=%s tokens=%d",
                facade._plugin_id, route["provider"], route["model"], effective_task or "", usage.total_tokens)
    return PluginLlmNativeResult(native_response=response, usage=usage, audit=audit)
