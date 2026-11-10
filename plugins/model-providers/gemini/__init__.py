"""Google Gemini (AI Studio) provider profile.

Reports api_mode="chat_completions" but runs on GeminiNativeClient; this
profile carries auth/endpoint metadata and the thinking_config translation hook.
"""

import json
import logging
import urllib.parse
import urllib.request
from typing import Any

from providers import register_provider
from providers.base import ProviderProfile

logger = logging.getLogger(__name__)

# AI Studio's catalog marks text models with ``generateContent`` but also gives it to models that
# only make media or serve a dedicated API. These id segments name those families, so a new
# version of one (``gemini-4-flash-tts``) stays out of the agent picker without a list edit. Only
# ``gemini-`` ids are admitted: Gemma is free-tier-only on this API and stays out of the picker by
# design (website/docs/guides/google-gemini.md), reachable by setting the id directly.
_NON_AGENT_ID_SEGMENTS = frozenset({
    "audio", "banana", "computer", "customtools", "embedding", "image", "live", "omni",
    "research", "robotics", "transcribe", "tts",
})
_MAX_CATALOG_PAGES = 10


def _bare_model_id(model_id: Any) -> Any:
    return model_id[len("models/"):] if isinstance(model_id, str) and model_id.startswith("models/") else model_id


def _is_agent_chat_model(entry: Any) -> bool:
    if not isinstance(entry, dict) or "generateContent" not in (entry.get("supportedGenerationMethods") or ()):
        return False
    model_id = str(_bare_model_id(entry.get("name") or "")).lower()
    return model_id.startswith("gemini-") and not _NON_AGENT_ID_SEGMENTS.intersection(model_id.split("-"))


def _list_native_chat_models(base_url: str, api_key: str, timeout: float) -> list[str] | None:
    """Every ``models.list`` page, filtered to agent chat models. Any failed page returns None so
    the picker falls back to the curated list instead of showing a truncated catalog."""
    from hermes_cli.urllib_security import open_credentialed_url

    ids: list[str] = []
    page_token = ""
    for _ in range(_MAX_CATALOG_PAGES):
        query = {"pageSize": "1000", **({"pageToken": page_token} if page_token else {})}
        req = urllib.request.Request(
            f"{base_url}/models?{urllib.parse.urlencode(query)}",
            headers={"x-goog-api-key": api_key, "Accept": "application/json"},
        )
        try:
            with open_credentialed_url(req, timeout=timeout) as resp:
                page = json.loads(resp.read().decode())
        except (OSError, ValueError) as exc:  # URLError/HTTPError/timeout, bad JSON or encoding
            logger.debug("gemini models.list failed: %s", type(exc).__name__)
            return None
        if not isinstance(page, dict):
            return None
        ids.extend(_bare_model_id(m["name"]) for m in page.get("models") or () if _is_agent_chat_model(m))
        page_token = str(page.get("nextPageToken") or "")
        if not page_token:
            return ids or None
    return None


class GeminiProfile(ProviderProfile):
    """Gemini — translate reasoning_config to thinking_config in extra_body."""

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Live chat-model catalog for the picker.

        Native AI Studio hosts are listed through ``models.list`` (header auth, paginated), the only
        surface that says which models answer ``generateContent``; ``/openai/models`` lists TTS,
        embedding, music and video models with nothing to tell them apart. Other base URLs (proxies)
        keep the OpenAI-compatible listing. Ids lose Google's ``models/`` prefix either way, and the
        result flows through the shared curated merge in ``hermes_cli.models``.
        """
        from agent.gemini_native_adapter import is_native_gemini_base_url, is_vertex_express_base_url

        effective_base = (base_url or self.base_url or "").rstrip("/")
        if is_native_gemini_base_url(effective_base) and not is_vertex_express_base_url(effective_base):
            return _list_native_chat_models(effective_base, api_key, timeout) if api_key else None
        models = super().fetch_models(api_key=api_key, base_url=effective_base or None, timeout=timeout)
        return [_bare_model_id(m) for m in models] if models else models

    def build_extra_body(self, *, session_id: str | None = None, **context: Any) -> dict[str, Any]:
        """Native: ``thinking_config``; OpenAI-compat /openai subpath:
        ``extra_body.google.thinking_config`` (snake_case)."""
        from agent.transports.chat_completions import (
            _build_gemini_thinking_config,
            _is_gemini_openai_compat_base_url,
            _snake_case_gemini_thinking_config,
        )

        raw = _build_gemini_thinking_config(context.get("model") or "", context.get("reasoning_config"))
        if not raw:
            return {}
        base_url = context.get("base_url") or self.base_url
        if self.name == "gemini" and _is_gemini_openai_compat_base_url(base_url):
            thinking_config = _snake_case_gemini_thinking_config(raw)
            return {"extra_body": {"google": {"thinking_config": thinking_config}}} if thinking_config else {}
        from agent.gemini_long_decode import long_decode_max_output_tokens
        from agent.gemini_native_adapter import is_native_gemini_base_url

        # Long decode exists only on native generateContent (not /openai, not proxies).
        budget = long_decode_max_output_tokens(context.get("model") or "", context.get("reasoning_config"))
        return {"thinking_config": raw, **({"max_output_tokens": budget} if budget and is_native_gemini_base_url(base_url) else {})}


gemini = GeminiProfile(
    name="gemini", aliases=("google", "google-gemini", "google-ai-studio"), api_mode="chat_completions",
    env_vars=("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta", auth_type="api_key",
    default_aux_model="gemini-3.6-flash",
)

register_provider(gemini)
