"""Azure schema preflight at physical send boundaries, after Relay overrides."""
from __future__ import annotations

import copy
import logging
from typing import Any

from agent.sdk_transform_bypass import bypass_chat_sdk_request_transform
from tools.azure_schema import is_azure_schema_target, prepare_azure_request


def prepare_chat_request(kwargs: dict, client: Any, provider: str = "") -> dict:
    """Keep the SDK bypass, but first validate the final Azure tool collection."""
    provider = provider or getattr(client, "_hermes_aux_effective_provider", "")
    prepared = prepare_azure_request(kwargs, provider=provider, base_url=getattr(client, "base_url", ""))
    return bypass_chat_sdk_request_transform(prepared, client)


def prepare_auxiliary_responses_tools(tools: Any, provider: str, host: str) -> Any:
    """Preserve Azure constraints; retain the existing non-Azure Responses workaround."""
    if not tools:
        return tools
    try:
        from tools.schema_sanitizer import strip_pattern_and_format, strip_slash_enum
        tools = copy.deepcopy(list(tools))
        if not is_azure_schema_target(provider, host):
            tools, _ = strip_pattern_and_format(tools)
            tools, _ = strip_slash_enum(tools)
    except Exception as exc:
        logging.getLogger("agent.auxiliary_client").warning(
            "Auxiliary client: failed to sanitize tool schemas for Codex/xAI Responses path: %s", exc, exc_info=True,
        )
    return tools
