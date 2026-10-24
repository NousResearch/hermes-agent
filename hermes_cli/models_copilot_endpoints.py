"""Copilot per-model API-mode requirements (split out of ``hermes_cli.models``)."""

from __future__ import annotations

import re
from typing import Any, Optional


def copilot_catalog_endpoint_mode(
    model_id: str, catalog: Optional[list[dict[str, Any]]]) -> Optional[str]:
    """Hard per-model API requirement from the live catalog's ``supported_endpoints``.

    Copilot rejects mismatches with HTTP 400 ``model "X" is not accessible via the
    /chat/completions endpoint``. Returns ``"codex_responses"`` for a Responses-only model,
    ``"chat_completions"`` for a model with no Responses endpoint, or ``None`` when the catalog
    is silent or the model advertises both (the id heuristic then decides).
    """
    for item in catalog or []:
        if not isinstance(item, dict) or item.get("id") != model_id:
            continue
        endpoints = item.get("supported_endpoints")
        if not isinstance(endpoints, list) or not endpoints:
            return None
        names = [str(e).strip() for e in endpoints]
        has_chat = any(e.endswith("/chat/completions") for e in names)
        has_responses = any(e.split(":", 1)[-1].endswith("/responses") for e in names)
        if has_responses and not has_chat:
            return "codex_responses"
        if has_chat and not has_responses:
            return "chat_completions"
        return None
    return None


def should_use_copilot_responses_api(
    model_id: str, catalog: Optional[list[dict[str, Any]]] = None) -> bool:
    """opencode's ``shouldUseCopilotResponsesApi``: GPT-5+ uses the Responses API except
    ``gpt-5-mini``; other models use Chat Completions unless the Copilot catalog says they are
    Responses-only (e.g. ``grok-4.x``). No network: uses ``catalog`` or the in-process cache."""
    if catalog is None:
        from hermes_cli import models  # lazy: models re-exports this module
        catalog = models._github_model_catalog_cache
    mode = copilot_catalog_endpoint_mode(model_id, catalog)
    if mode:
        return mode == "codex_responses"
    match = re.match(r"^gpt-(\d+)", model_id)
    return bool(match) and int(match.group(1)) >= 5 and not model_id.startswith("gpt-5-mini")
