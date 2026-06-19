"""OpenAI image generation backend — ChatGPT/Codex OAuth variant.

Identical model catalog and tier semantics to the ``openai`` image-gen plugin
(``gpt-image-2`` at low/medium/high quality), but routes the request through
the Codex Responses API ``image_generation`` tool instead of the
``images.generate`` REST endpoint. This lets users who are already
authenticated with Codex/ChatGPT generate images without configuring a
separate ``OPENAI_API_KEY``.

Selection precedence for the tier (first hit wins):

1. ``OPENAI_IMAGE_MODEL`` env var (escape hatch for scripts / tests)
2. ``image_gen.openai-codex.model`` in ``config.yaml``
3. ``image_gen.model`` in ``config.yaml`` (when it's one of our tier IDs)
4. :data:`DEFAULT_MODEL` — ``gpt-image-2-medium``

Output is saved as PNG under ``$HERMES_HOME/cache/images/``.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    resolve_aspect_ratio,
    save_b64_image,
    success_response,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model catalog — mirrors the ``openai`` plugin so the picker UX is identical.
# ---------------------------------------------------------------------------

API_MODEL = "gpt-image-2"

_MODELS: Dict[str, Dict[str, Any]] = {
    "gpt-image-2-low": {
        "display": "GPT Image 2 (Low)",
        "speed": "~15s",
        "strengths": "Fast iteration, lowest cost",
        "quality": "low",
    },
    "gpt-image-2-medium": {
        "display": "GPT Image 2 (Medium)",
        "speed": "~40s",
        "strengths": "Balanced — default",
        "quality": "medium",
    },
    "gpt-image-2-high": {
        "display": "GPT Image 2 (High)",
        "speed": "~2min",
        "strengths": "Highest fidelity, strongest prompt adherence",
        "quality": "high",
    },
}

DEFAULT_MODEL = "gpt-image-2-medium"

_SIZES = {
    "landscape": "1536x1024",
    "square": "1024x1024",
    "portrait": "1024x1536",
}

# Codex Responses surface used for the request. The chat model itself is only
# the host that calls the ``image_generation`` tool; the actual image work is
# done by ``API_MODEL``.
_CODEX_CHAT_MODEL = "gpt-5.5"
_CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"
_CODEX_INSTRUCTIONS = (
    "You are an assistant that must fulfill image generation requests by "
    "using the image_generation tool when provided."
)
_CODEX_IMAGE_TOOL = "image_generation"
_CODEX_MODELS_CLIENT_VERSION = "1.0.0"
_CODEX_IMAGE_CAPABILITY_CACHE_TTL = 3600  # 1 hour
_codex_image_capability_cache: Optional[Tuple[float, bool, str]] = None


# ---------------------------------------------------------------------------
# Config + auth helpers
# ---------------------------------------------------------------------------


def _load_image_gen_config() -> Dict[str, Any]:
    """Read ``image_gen`` from config.yaml (returns {} on any failure)."""
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        section = cfg.get("image_gen") if isinstance(cfg, dict) else None
        return section if isinstance(section, dict) else {}
    except Exception as exc:
        logger.debug("Could not load image_gen config: %s", exc)
        return {}


def _resolve_model() -> Tuple[str, Dict[str, Any]]:
    """Decide which tier to use and return ``(model_id, meta)``."""
    import os

    env_override = os.environ.get("OPENAI_IMAGE_MODEL")
    if env_override and env_override in _MODELS:
        return env_override, _MODELS[env_override]

    cfg = _load_image_gen_config()
    sub = cfg.get("openai-codex") if isinstance(cfg.get("openai-codex"), dict) else {}
    candidate: Optional[str] = None
    if isinstance(sub, dict):
        value = sub.get("model")
        if isinstance(value, str) and value in _MODELS:
            candidate = value
    if candidate is None:
        top = cfg.get("model")
        if isinstance(top, str) and top in _MODELS:
            candidate = top

    if candidate is not None:
        return candidate, _MODELS[candidate]

    return DEFAULT_MODEL, _MODELS[DEFAULT_MODEL]


def _read_codex_access_token() -> Optional[str]:
    """Return a usable Codex OAuth token, or None.

    Delegates to the canonical reader in ``agent.auxiliary_client`` so token
    expiry, credential pool selection, and JWT decoding stay in one place.
    """
    try:
        from agent.auxiliary_client import _read_codex_access_token as _reader

        token = _reader()
        if isinstance(token, str) and token.strip():
            return token.strip()
        return None
    except Exception as exc:
        logger.debug("Could not resolve Codex access token: %s", exc)
        return None


def _codex_image_generation_unavailable_message() -> str:
    """Explain the current Codex OAuth image-generation limitation."""
    return (
        "ChatGPT/Codex OAuth currently does not expose the Responses "
        "`image_generation` tool on the Codex backend. Use the API-key-backed "
        "`openai` image provider instead."
    )


def _read_cached_codex_supported_tools() -> Optional[List[str]]:
    """Read experimental_supported_tools for the current Codex chat model."""
    from pathlib import Path

    codex_home = Path(os.getenv("CODEX_HOME", "") or (Path.home() / ".codex")).expanduser()
    cache_path = codex_home / "models_cache.json"
    if not cache_path.exists():
        return None
    try:
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.debug("Could not read Codex models cache: %s", exc)
        return None

    entries = raw.get("models", []) if isinstance(raw, dict) else []
    for item in entries:
        if not isinstance(item, dict):
            continue
        if item.get("slug") != _CODEX_CHAT_MODEL:
            continue
        tools = item.get("experimental_supported_tools")
        if not isinstance(tools, list):
            return None
        return [tool.strip() for tool in tools if isinstance(tool, str) and tool.strip()]
    return None


def _probe_codex_supported_tools(token: str) -> Optional[List[str]]:
    """Return the live Codex model's advertised experimental tools."""
    import httpx
    from agent.auxiliary_client import _codex_cloudflare_headers

    headers = _codex_cloudflare_headers(token)
    headers["Authorization"] = f"Bearer {token}"

    with httpx.Client(timeout=10.0, headers=headers) as http:
        response = http.get(
            f"{_CODEX_BASE_URL}/models?client_version={_CODEX_MODELS_CLIENT_VERSION}"
        )
        response.raise_for_status()

    data = response.json()
    entries = data.get("models", []) if isinstance(data, dict) else []
    for item in entries:
        if not isinstance(item, dict):
            continue
        if item.get("slug") != _CODEX_CHAT_MODEL:
            continue
        tools = item.get("experimental_supported_tools")
        if not isinstance(tools, list):
            return None
        return [tool.strip() for tool in tools if isinstance(tool, str) and tool.strip()]
    return None


def _codex_image_generation_support(
    token: Optional[str] = None,
    *,
    allow_live: bool = False,
) -> Tuple[Optional[bool], str]:
    """Return whether the current Codex OAuth surface supports image generation.

    ``False`` means we have direct evidence that the current backend does not
    expose the ``image_generation`` tool. ``None`` means the capability could
    not be confirmed cheaply and callers may need to rely on the request path
    for a definitive answer.
    """
    global _codex_image_capability_cache

    cached = _codex_image_capability_cache
    now = time.time()
    if cached and now - cached[0] < _CODEX_IMAGE_CAPABILITY_CACHE_TTL:
        return cached[1], cached[2]

    tools = _read_cached_codex_supported_tools()
    if tools is not None:
        supported = _CODEX_IMAGE_TOOL in tools
        message = "" if supported else _codex_image_generation_unavailable_message()
        _codex_image_capability_cache = (now, supported, message)
        return supported, message

    if allow_live and token:
        try:
            tools = _probe_codex_supported_tools(token)
        except Exception as exc:
            logger.debug("Could not probe Codex image-generation capability: %s", exc)
        else:
            if tools is not None:
                supported = _CODEX_IMAGE_TOOL in tools
                message = "" if supported else _codex_image_generation_unavailable_message()
                _codex_image_capability_cache = (now, supported, message)
                return supported, message

    if cached:
        return cached[1], cached[2]
    return None, _codex_image_generation_unavailable_message()


def _build_responses_payload(*, prompt: str, size: str, quality: str) -> Dict[str, Any]:
    """Build the Codex Responses request body for an image_generation call."""
    return {
        "model": _CODEX_CHAT_MODEL,
        "store": False,
        "instructions": _CODEX_INSTRUCTIONS,
        "input": [{
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": prompt}],
        }],
        "tools": [{
            "type": "image_generation",
            "model": API_MODEL,
            "size": size,
            "quality": quality,
            "output_format": "png",
            "background": "opaque",
            "partial_images": 1,
        }],
        "tool_choice": {
            "type": "allowed_tools",
            "mode": "required",
            "tools": [{"type": _CODEX_IMAGE_TOOL}],
        },
        "stream": True,
    }


def _extract_image_b64(value: Any) -> Optional[str]:
    """Return the newest image b64 embedded in a Responses event payload."""
    found: Optional[str] = None
    if isinstance(value, dict):
        if value.get("type") == "image_generation_call":
            result = value.get("result")
            if isinstance(result, str) and result:
                found = result
        partial = value.get("partial_image_b64")
        if isinstance(partial, str) and partial:
            found = partial
        for child in value.values():
            nested = _extract_image_b64(child)
            if nested:
                found = nested
    elif isinstance(value, list):
        for child in value:
            nested = _extract_image_b64(child)
            if nested:
                found = nested
    return found


def _iter_sse_json(response: Any):
    """Yield JSON payloads from an SSE response without OpenAI SDK parsing.

    The ChatGPT/Codex backend can emit image-generation events newer than the
    pinned Python SDK understands. Parsing raw SSE keeps this provider tolerant
    of those event-shape changes.
    """
    event_name: Optional[str] = None
    data_lines: List[str] = []

    def flush():
        nonlocal event_name, data_lines
        if not data_lines:
            event_name = None
            return None
        raw = "\n".join(data_lines).strip()
        event = event_name
        event_name = None
        data_lines = []
        if not raw or raw == "[DONE]":
            return None
        payload = json.loads(raw)
        if isinstance(payload, dict) and event and "type" not in payload:
            payload["type"] = event
        return payload

    for line in response.iter_lines():
        if isinstance(line, bytes):
            line = line.decode("utf-8", errors="replace")
        line = str(line)
        if line == "":
            payload = flush()
            if payload is not None:
                yield payload
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event_name = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:"):].lstrip())

    payload = flush()
    if payload is not None:
        yield payload


def _collect_image_b64(token: str, *, prompt: str, size: str, quality: str) -> Optional[str]:
    """Stream a Codex Responses image_generation call and return the b64 image."""
    import httpx
    from agent.auxiliary_client import _codex_cloudflare_headers

    headers = _codex_cloudflare_headers(token)
    headers.update({
        "Accept": "text/event-stream",
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    })
    payload = _build_responses_payload(prompt=prompt, size=size, quality=quality)
    timeout = httpx.Timeout(300.0, connect=30.0, read=300.0, write=30.0, pool=30.0)

    image_b64: Optional[str] = None
    with httpx.Client(timeout=timeout, headers=headers) as http:
        with http.stream("POST", f"{_CODEX_BASE_URL}/responses", json=payload) as response:
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                exc.response.read()
                body = exc.response.text[:500]
                raise RuntimeError(
                    f"Codex Responses API returned HTTP {exc.response.status_code}: {body}"
                ) from exc
            for event in _iter_sse_json(response):
                found = _extract_image_b64(event)
                if found:
                    image_b64 = found

    return image_b64


def _looks_like_unsupported_image_generation_error(exc: Exception) -> bool:
    """Detect the Codex backend's unsupported-image-generation failure."""
    text = str(exc).lower()
    return (
        _CODEX_IMAGE_TOOL in text
        and "tool choice" in text
        and "not found" in text
    )


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class OpenAICodexImageGenProvider(ImageGenProvider):
    """gpt-image-2 routed through ChatGPT/Codex OAuth instead of an API key."""

    @property
    def name(self) -> str:
        return "openai-codex"

    @property
    def display_name(self) -> str:
        return "OpenAI (Codex auth)"

    def is_available(self) -> bool:
        if not _read_codex_access_token():
            return False
        try:
            import httpx  # noqa: F401
        except ImportError:
            return False
        supported, _message = _codex_image_generation_support()
        return supported is True

    def list_models(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": model_id,
                "display": meta["display"],
                "speed": meta["speed"],
                "strengths": meta["strengths"],
                "price": "varies",
            }
            for model_id, meta in _MODELS.items()
        ]

    def default_model(self) -> Optional[str]:
        return DEFAULT_MODEL

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "OpenAI (Codex auth)",
            "badge": "free",
            "tag": "gpt-image-2 via ChatGPT/Codex OAuth — no API key required",
            "env_vars": [],
            "post_setup_hint": (
                "Sign in with `hermes auth codex` (or `hermes setup` → Codex) "
                "if you haven't already. If this backend stays unavailable, "
                "your current Codex surface likely does not expose "
                "`image_generation` yet."
            ),
        }

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        global _codex_image_capability_cache

        prompt = (prompt or "").strip()
        aspect = resolve_aspect_ratio(aspect_ratio)

        if not prompt:
            return error_response(
                error="Prompt is required and must be a non-empty string",
                error_type="invalid_argument",
                provider="openai-codex",
                aspect_ratio=aspect,
            )

        if not _read_codex_access_token():
            return error_response(
                error=(
                    "No Codex/ChatGPT OAuth credentials available. Run "
                    "`hermes auth codex` (or `hermes setup` → Codex) to sign in."
                ),
                error_type="auth_required",
                provider="openai-codex",
                aspect_ratio=aspect,
            )

        try:
            import httpx  # noqa: F401
        except ImportError:
            return error_response(
                error="httpx Python package not installed (pip install httpx)",
                error_type="missing_dependency",
                provider="openai-codex",
                aspect_ratio=aspect,
            )

        tier_id, meta = _resolve_model()
        size = _SIZES.get(aspect, _SIZES["square"])

        token = _read_codex_access_token()
        if not token:
            return error_response(
                error=(
                    "No Codex/ChatGPT OAuth credentials available. Run "
                    "`hermes auth codex` (or `hermes setup` → Codex) to sign in."
                ),
                error_type="auth_required",
                provider="openai-codex",
                model=tier_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        supported, capability_message = _codex_image_generation_support(
            token,
            allow_live=True,
        )
        if supported is False:
            return error_response(
                error=capability_message,
                error_type="capability_unsupported",
                provider="openai-codex",
                model=tier_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        try:
            b64 = _collect_image_b64(
                token,
                prompt=prompt,
                size=size,
                quality=meta["quality"],
            )
        except Exception as exc:
            logger.debug("Codex image generation failed", exc_info=True)
            if _looks_like_unsupported_image_generation_error(exc):
                _codex_image_capability_cache = (
                    time.time(),
                    False,
                    _codex_image_generation_unavailable_message(),
                )
                return error_response(
                    error=_codex_image_generation_unavailable_message(),
                    error_type="capability_unsupported",
                    provider="openai-codex",
                    model=tier_id,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )
            return error_response(
                error=f"OpenAI image generation via Codex auth failed: {exc}",
                error_type="api_error",
                provider="openai-codex",
                model=tier_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        if not b64:
            return error_response(
                error="Codex response contained no image_generation_call result",
                error_type="empty_response",
                provider="openai-codex",
                model=tier_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        try:
            saved_path = save_b64_image(b64, prefix=f"openai_codex_{tier_id}")
        except Exception as exc:
            return error_response(
                error=f"Could not save image to cache: {exc}",
                error_type="io_error",
                provider="openai-codex",
                model=tier_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        return success_response(
            image=str(saved_path),
            model=tier_id,
            prompt=prompt,
            aspect_ratio=aspect,
            provider="openai-codex",
            extra={"size": size, "quality": meta["quality"]},
        )


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


def register(ctx) -> None:
    """Plugin entry point — register the Codex-backed image-gen provider."""
    ctx.register_image_gen_provider(OpenAICodexImageGenProvider())
