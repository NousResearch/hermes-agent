"""GitHub Copilot catalog filtering for ``hermes_cli.models``.

Split out of ``hermes_cli.models`` to offset that facade's growth (the code-health ratchet
caps files already over 2,000 lines at their current size). Owns the Copilot-side answer to
"which catalog rows are chat-capable text models": the endpoint set Hermes can serve a
Copilot model on, and the per-row predicate the catalog helpers filter through.
"""

from typing import Any

_COPILOT_CHAT_ENDPOINTS = {"/chat/completions", "/responses", "/v1/messages"}


def _copilot_catalog_item_is_text_model(
    item: dict[str, Any], *, ignore_picker_flag: bool = False) -> bool:
    if not str(item.get("id") or "").strip():
        return False
    if not ignore_picker_flag and item.get("model_picker_enabled") is False:
        return False
    capabilities = item.get("capabilities")
    if isinstance(capabilities, dict):
        model_type = str(capabilities.get("type") or "").strip().lower()
        if model_type and model_type != "chat":
            return False
    supported_endpoints = item.get("supported_endpoints")
    if isinstance(supported_endpoints, list):
        endpoints = {e for endpoint in supported_endpoints if (e := str(endpoint).strip())}
        if endpoints and not endpoints & _COPILOT_CHAT_ENDPOINTS:
            return False
    return True
