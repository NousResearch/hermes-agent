"""Copilot wire selection: advertised HTTP endpoints, then offline family defaults."""

from __future__ import annotations

import re
from typing import Any, Optional

_HTTP_API_MODES = {
    "/chat/completions": "chat_completions",
    "/responses": "codex_responses",
}


def _default_api_mode(model_id: str) -> str:
    if model_id.startswith("grok-"):
        return "codex_responses"
    match = re.match(r"^gpt-(\d+)", model_id)
    responses = bool(match) and int(match.group(1)) >= 5 and not model_id.startswith("gpt-5-mini")
    return "codex_responses" if responses else "chat_completions"


def copilot_model_api_mode(
    model_id: Optional[str], *, catalog: Optional[list[dict[str, Any]]] = None,
    api_key: Optional[str] = None, preferred_api_mode: Optional[str] = None,
) -> str:
    """Choose a supported HTTP wire without requiring a model-name update for new models."""
    from hermes_cli import models

    if catalog is None and api_key:
        catalog = models.fetch_github_model_catalog(api_key=api_key)
    normalized = models.normalize_copilot_model_id(model_id, catalog=catalog or [])
    item = next((item for item in (catalog or []) if item.get("id") == normalized), {})
    endpoints = item.get("supported_endpoints")
    supported = {mode for endpoint, mode in _HTTP_API_MODES.items()
                 if isinstance(endpoints, list) and endpoint in endpoints}
    if len(supported) == 1:
        return next(iter(supported))
    # When both wires are advertised, Responses loses GPT-5-mini reasoning (#46527).
    if normalized.startswith("gpt-5-mini"):
        return "chat_completions"
    if preferred_api_mode and preferred_api_mode in supported:
        return preferred_api_mode
    # Preserve a saved Chat preference without metadata, except for known Responses-only Grok.
    if not supported and preferred_api_mode == "chat_completions" and not normalized.startswith("grok-"):
        return "chat_completions"
    # Messages-only Claude catalog rows still use Copilot's OpenAI-compatible auth path.
    return _default_api_mode(normalized)
