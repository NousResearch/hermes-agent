"""Provider profile base class.

A ProviderProfile declares everything about an inference provider in one place:
auth, endpoints, client quirks, request-time quirks. The transport reads this
instead of receiving 20+ boolean flags.

Provider profiles are DECLARATIVE — they describe the provider's behavior.
They do NOT own client construction, credential rotation, or streaming.
Those stay on AIAgent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Sentinel for "omit temperature entirely" (Kimi: server manages it)
OMIT_TEMPERATURE = object()

# Upper bound on a provider model-catalog response body (issue #54735).
# Model lists are normally tens-to-hundreds of KB; 16 MiB is generous
# headroom while still failing closed against a malicious / compromised /
# misconfigured endpoint that streams an arbitrarily large body during model
# discovery (model picker, provider probing, live-list refresh). Oversized
# responses are rejected and callers fall back to the static model list.
_MAX_MODELS_RESPONSE_BYTES = 16 * 1024 * 1024


def _read_json_capped(resp, max_bytes: int | None = None):
    """Read and JSON-parse an HTTP response body with a hard size cap.

    Two-layer guard so a bad endpoint cannot exhaust memory:
      1. Reject up front if the declared ``Content-Length`` exceeds the cap.
      2. Read at most ``max_bytes + 1`` bytes and reject if the body actually
         ran past the cap (covers a missing or lying Content-Length on a
         streamed response).

    Raises ``ValueError`` on oversize so the caller's existing
    ``except Exception: return None`` path falls back to the static list.

    ``max_bytes`` defaults to the module-level ``_MAX_MODELS_RESPONSE_BYTES``
    (read at call time, so the cap can be lowered in tests).
    """
    import json

    if max_bytes is None:
        max_bytes = _MAX_MODELS_RESPONSE_BYTES

    declared = resp.headers.get("Content-Length") if hasattr(resp, "headers") else None
    if declared is not None:
        try:
            if int(declared) > max_bytes:
                raise ValueError(
                    f"model catalog response too large: Content-Length={declared} "
                    f"> {max_bytes}"
                )
        except (TypeError, ValueError) as exc:
            # A non-integer Content-Length is itself suspect — only re-raise
            # our own oversize signal; ignore an unparseable header and fall
            # through to the bounded read below.
            if "too large" in str(exc):
                raise

    raw = resp.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise ValueError(
            f"model catalog response too large: body exceeded {max_bytes} bytes"
        )
    return json.loads(raw.decode())


def _profile_user_agent() -> str:
    """Return a ``hermes-cli/<version>`` UA string, with a stable fallback.

    Used by ``ProviderProfile.fetch_models`` so the catalog probe is not
    served the default ``Python-urllib/<ver>`` UA — some providers
    (OpenCode Zen, etc.) sit behind a WAF that returns 403 for that.
    """
    try:
        from hermes_cli import __version__ as _ver  # lazy: avoid layer cycle at import time
        return f"hermes-cli/{_ver}"
    except Exception:
        return "hermes-cli"


@dataclass
class ProviderProfile:
    """Base provider profile — subclass or instantiate with overrides."""

    # ── Identity ─────────────────────────────────────────────
    name: str
    api_mode: str = "chat_completions"
    aliases: tuple = ()

    # ── Human-readable metadata ───────────────────────────────
    display_name: str = ""       # e.g. "GMI Cloud" — shown in picker/labels
    description: str = ""        # e.g. "GMI Cloud (multi-model direct API)" — picker subtitle
    signup_url: str = ""         # e.g. "https://www.gmicloud.ai/" — shown during setup

    # ── Auth & endpoints ─────────────────────────────────────
    env_vars: tuple = ()
    base_url: str = ""
    models_url: str = ""  # explicit models endpoint; falls back to {base_url}/models
    auth_type: str = "api_key"   # api_key|oauth_device_code|oauth_external|copilot|aws_sdk
    supports_health_check: bool = True  # False → doctor skips /models probe for this provider

    # ── Vision support ────────────────────────────────────────
    # True when the provider's API accepts image content inside
    # tool-result messages natively.  Set on providers that expose
    # multimodal models via tool results (Anthropic Messages API,
    # OpenAI Chat Completions, Gemini, MiniMax, etc.).
    # Falls back to model-catalog lookup when False and the provider
    # has no registered profile.
    supports_vision: bool = False

    # True when the provider's API accepts list-type tool message
    # content (multipart with image_url parts).  Defaults to True for
    # backward compatibility.  Set to False for providers that accept
    # multimodal user messages but reject list-type tool content
    # (e.g. Xiaomi MiMo, which returns 400 "text is not set").
    supports_vision_tool_messages: bool = True

    # ── Model catalog ─────────────────────────────────────────
    # fallback_models: curated list shown in /model picker when live fetch fails.
    # Only agentic models that support tool calling should appear here.
    fallback_models: tuple = ()

    # hostname: base hostname for URL→provider reverse-mapping in model_metadata.py
    # e.g. "api.gmi-serving.com". Derived from base_url when empty.
    hostname: str = ""

    # ── Client-level quirks (set once at client construction) ─
    default_headers: dict[str, str] = field(default_factory=dict)

    # ── Request-level quirks ─────────────────────────────────
    # Temperature: None = use caller's default, OMIT_TEMPERATURE = don't send
    fixed_temperature: Any = None
    default_max_tokens: int | None = None
    default_aux_model: str = (
        ""  # cheap model for auxiliary tasks (compression, vision, etc.)
    )
    # empty = use main model

    # ── Hooks (override in subclass for complex providers) ───

    def get_hostname(self) -> str:
        """Return the provider's base hostname for URL-based detection.

        Uses self.hostname if set explicitly, otherwise derives it from base_url.
        e.g. 'https://api.gmi-serving.com/v1' → 'api.gmi-serving.com'
        """
        if self.hostname:
            return self.hostname
        if self.base_url:
            from urllib.parse import urlparse
            return urlparse(self.base_url).hostname or ""
        return ""

    def prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Provider-specific message preprocessing.

        Called AFTER codex field sanitization, BEFORE developer role swap.
        Default: pass-through.
        """
        return messages

    def build_extra_body(
        self, *, session_id: str | None = None, **context: Any
    ) -> dict[str, Any]:
        """Provider-specific extra_body fields.

        Merged into the API kwargs extra_body. Default: empty dict.
        """
        return {}

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        **context: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Provider-specific kwargs split between extra_body and top-level api_kwargs.

        Returns (extra_body_additions, top_level_kwargs).
        The transport merges extra_body_additions into extra_body, and
        top_level_kwargs directly into api_kwargs.

        This split exists because some providers put reasoning config in
        extra_body (OpenRouter: extra_body.reasoning) while others put it
        as top-level api_kwargs (Kimi: api_kwargs.reasoning_effort).

        Default: ({}, {}).
        """
        return {}, {}

    def get_max_tokens(self, model: str | None) -> int | None:
        """Return the default max_tokens cap for *model*.

        Overrideable hook for providers that need per-model output caps —
        e.g. a relay that fronts several upstream backends, each with a
        different completion-token limit. The transport calls this when
        the user hasn't set an explicit max_tokens.

        Default: return self.default_max_tokens (the static profile field),
        ignoring the model name. Override in a subclass to vary the cap
        per-model.
        """
        return self.default_max_tokens

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Fetch the live model list from the provider's models endpoint.

        Returns a list of model ID strings, or None if the fetch failed or
        the provider does not support live model listing.

        Resolution order for the endpoint URL:
          1. self.models_url  (explicit override — use when the models
             endpoint differs from the inference base URL, e.g. OpenRouter
             exposes a public catalog at /api/v1/models while inference is
             at /api/v1)
          2. base_url (caller override — user-configured model.base_url)
          3. self.base_url + "/models"  (standard OpenAI-compat fallback)

        The default implementation sends Bearer auth when api_key is given
        and forwards self.default_headers. Override to customise auth, path,
        response shape, or to return None for providers with no REST catalog.

        Callers must always fall back to the static _PROVIDER_MODELS list
        when this returns None.
        """
        effective_base = base_url or self.base_url
        url = (self.models_url or "").strip()
        if not url:
            if not effective_base:
                return None
            url = effective_base.rstrip("/") + "/models"

        import urllib.request

        req = urllib.request.Request(url)
        if api_key:
            req.add_header("Authorization", f"Bearer {api_key}")
        req.add_header("Accept", "application/json")
        # Some providers (e.g. OpenCode Zen) sit behind a WAF that blocks
        # the default ``Python-urllib/<ver>`` User-Agent.  Set a generic
        # hermes-cli UA so the catalog endpoint is reachable.
        req.add_header("User-Agent", _profile_user_agent())
        for k, v in self.default_headers.items():
            req.add_header(k, v)

        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = _read_json_capped(resp)
            items = data if isinstance(data, list) else data.get("data", [])
            return [m["id"] for m in items if isinstance(m, dict) and "id" in m]
        except Exception as exc:
            logger.debug("fetch_models(%s): %s", self.name, exc)
            return None
