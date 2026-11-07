"""Reconcile the context window Hermes BELIEVES a local Ollama server has with the one it SERVES.

Startup detection reads ``/api/show`` (Modelfile ``num_ctx``, else the GGUF training max) and the
``custom`` profile sends that value as ``options.num_ctx`` on every request. Ollama's OpenAI-compatible
``/v1`` route maps only the OpenAI parameter allowlist and drops ``num_ctx`` (ollama/ollama#16825 open
since Jun 2026), so the model loads at the server default (4,096 under 24 GB of VRAM, or
``OLLAMA_CONTEXT_LENGTH``). Ollama then truncates the prompt server-side and answers with
``finish_reason=stop`` and a plausible ``prompt_tokens``, so the agent runs with a chopped system prompt
and conversation and nothing on any surface says so (#43900, #132607).

``/api/ps`` reports ``context_length`` for every loaded model, which is the window actually allocated.
After the session's first response the model is loaded, so one GET settles it: warn the user once with
the real numbers and the server-side remedy, and clamp the compressor to the served window so
compaction fires before the server truncates (the same one-directional clamp as
``agent_init._clamp_compressor_to_ollama_num_ctx``).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

_SERVED_WINDOW_MISMATCH = (
    "⚠️ Ollama is serving {model} with a {served:,}-token context window, but Hermes detected "
    "{believed:,} tokens from /api/show and planned around that. Ollama's OpenAI-compatible /v1 route "
    "ignores the per-request num_ctx Hermes sends, so prompts longer than {served:,} tokens are "
    "silently cut by the server. Fix it server-side: OLLAMA_CONTEXT_LENGTH={believed} for `ollama serve`, "
    "or `PARAMETER num_ctx {believed}` in a Modelfile. Compaction now targets the {served:,}-token window."
)


def query_ollama_served_context(model: str, base_url: str, api_key: Any = "") -> Optional[int]:
    """``context_length`` of ``model`` from Ollama ``/api/ps``; None when not loaded, not Ollama, or unreachable."""
    import httpx

    from agent import model_metadata_http
    from agent.model_metadata import _auth_headers, _server_root, _strip_provider_prefix

    bare = _strip_provider_prefix(model)
    server_url = _server_root(base_url)
    try:
        with httpx.Client(timeout=3.0, headers=_auth_headers(api_key), verify=model_metadata_http.resolve_verify(server_url)) as client:
            resp = client.get(f"{server_url}/api/ps")
            if resp.status_code != 200:
                return None
            models = resp.json().get("models") or []
    except Exception as exc:
        logger.debug("Ollama /api/ps probe failed: %s", exc)
        return None
    # Ollama canonicalises an untagged name to ``:latest``; the row carries both ``name`` and ``model``.
    accepted = {bare, f"{bare}:latest"}
    for row in models:
        if not isinstance(row, dict):
            continue
        if accepted & {row.get("name"), row.get("model")}:
            ctx = row.get("context_length")
            return ctx if isinstance(ctx, int) and ctx > 0 else None
    return None


def reconcile_served_context_after_response(agent: Any, api_call_count: int) -> None:
    """Once per (model, base_url) after the first response: warn + clamp when Ollama serves less than detected."""
    believed = getattr(agent, "_ollama_num_ctx", None)
    base_url = getattr(agent, "base_url", None) or ""
    if api_call_count != 1 or not believed or not base_url:
        return
    key = (agent.model, base_url)
    if getattr(agent, "_ollama_served_window_checked", None) == key:
        return
    agent._ollama_served_window_checked = key
    from agent.model_metadata import is_local_endpoint

    if not is_local_endpoint(base_url):
        return
    served = query_ollama_served_context(agent.model, base_url, getattr(agent, "api_key", ""))
    if served is None or served >= believed:
        return
    logger.warning(
        "Ollama serves %s at num_ctx=%d but Hermes detected %d; /v1 dropped the requested num_ctx",
        agent.model, served, believed,
    )
    agent._emit_warning(_SERVED_WINDOW_MISMATCH.format(model=agent.model, served=served, believed=believed))
    compressor = getattr(agent, "context_compressor", None)
    window = int(getattr(compressor, "context_length", 0) or 0)
    if compressor is not None and window and served < window:
        compressor.update_model(
            model=agent.model, context_length=served, base_url=base_url,
            api_key=getattr(agent, "api_key", ""), provider=getattr(agent, "provider", ""),
            api_mode=getattr(agent, "api_mode", ""),
        )
