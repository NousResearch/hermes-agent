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

# Valid values for ProviderProfile.system_prompt_mode.
VALID_SYSTEM_PROMPT_MODES = ("system", "developer", "user")


def _extract_text_content(content: Any) -> str:
    """Flatten a message ``content`` value (string or multimodal part list) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return ""


def _short_system_identity(text: str, max_len: int = 200) -> str:
    """Return a short system identity marker (first non-empty line of the prompt).

    Some OpenAI-compatible relays (notably Gemini-backed ones, see #76783)
    reject long ``systemInstruction`` content with HTTP 429 RESOURCE_EXHAUSTED
    even when identical content as a user message succeeds.  When the runtime
    prompt is moved to the first user message we still keep a brief real
    system message so the model keeps its identity framing.
    """
    for line in text.splitlines():
        line = line.strip()
        if line:
            if len(line) <= max_len:
                return line
            cut = line[:max_len]
            idx = cut.rfind(".")
            if idx > 40:
                return cut[: idx + 1]
            return cut
    return text[:max_len]


def apply_system_prompt_mode(
    messages: list[dict[str, Any]], mode: str | None
) -> list[dict[str, Any]]:
    """Apply a system-prompt compatibility mode to a message list.

    Returns a new list; input messages are never mutated.  ``mode``:

    * ``"system"`` (or None) — pass-through, full backward compatibility.
    * ``"developer"`` — swap the first system message role to ``developer``
      (equivalent to the model-name-based swap used for GPT-5/Codex).
    * ``"user"`` — keep a short system identity marker and prepend the full
      system prompt to the first user message.  Workaround for
      OpenAI-compatible relays backed by Gemini that cannot reliably accept
      Hermes's full runtime prompt as ``system`` content but accept the
      identical content as a ``user`` message (#76783).  Handles both string
      user content and multimodal part lists.

    No-op (returns ``messages`` unchanged) when there is no leading system
    message, the system content is empty, or there is no user message to
    absorb the prompt into.
    """
    if mode not in VALID_SYSTEM_PROMPT_MODES or mode == "system":
        return messages
    if not messages:
        return messages
    first = messages[0]
    if not isinstance(first, dict) or first.get("role") != "system":
        return messages

    out = list(messages)

    if mode == "developer":
        out[0] = {**first, "role": "developer"}
        return out

    # mode == "user"
    sys_text = _extract_text_content(first.get("content", ""))
    if not sys_text.strip():
        return messages

    out[0] = {**first, "content": _short_system_identity(sys_text)}
    wrapper = (
        "[Hermes runtime instructions]\n"
        f"{sys_text}\n"
        "[End runtime instructions]"
    )
    for i, msg in enumerate(out[1:], start=1):
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        user_content = msg.get("content", "")
        if isinstance(user_content, str):
            new_content = wrapper if not user_content else f"{wrapper}\n\n{user_content}"
        elif isinstance(user_content, list):
            new_content = [{"type": "text", "text": wrapper}, *list(user_content)]
        else:
            continue
        out[i] = {**msg, "content": new_content}
        return out

    return messages


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

    # True only when this provider's Chat Completions endpoint explicitly
    # documents ``prompt_cache_key`` as an accepted request body field.  This
    # is deliberately opt-in: many OpenAI-compatible endpoints reject unknown
    # top-level fields rather than ignoring them.
    supports_prompt_cache_key: bool = False

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

    # How the system prompt is represented on the wire:
    #   "system"    (default) — first message role stays "system"
    #   "developer" — first system message role becomes "developer"
    #   "user"      — keep a short system identity marker, prepend the full
    #                 system prompt to the first user message.  Opt-in
    #                 workaround for OpenAI-compatible relays backed by Gemini
    #                 that reject long systemInstruction content with HTTP 429
    #                 RESOURCE_EXHAUSTED while identical user content succeeds
    #                 (#76783).
    system_prompt_mode: str = "system"

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
        Default: pass-through, unless ``system_prompt_mode`` is set to a
        non-default value, in which case the compatibility transformation
        (developer role swap, or moving the system prompt into the first
        user message for Gemini-backed relays, #76783) is applied here.
        """
        if self.system_prompt_mode and self.system_prompt_mode != "system":
            return apply_system_prompt_mode(messages, self.system_prompt_mode)
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

    def default_vision_model(self) -> str | None:
        """Return a default vision model id for this provider, or None.

        Overrideable hook for providers that discover their vision default at
        runtime (e.g. from a live catalog) rather than pinning one in code.
        Keeps provider-specific vision discovery inside the provider's plugin
        instead of a name-check branch in shared vision resolution.

        Default: None (no provider-specific vision model — the caller falls
        back to the user's chat model or the aggregator chain).
        """
        return None

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

        import json
        import urllib.request

        from hermes_cli.urllib_security import open_credentialed_url

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
            with open_credentialed_url(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
            items = data if isinstance(data, list) else data.get("data", [])
            return [m["id"] for m in items if isinstance(m, dict) and "id" in m]
        except Exception as exc:
            logger.debug("fetch_models(%s): %s", self.name, exc)
            return None
