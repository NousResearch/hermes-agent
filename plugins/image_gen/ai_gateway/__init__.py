"""AI Model Gateway image generation backend.

Routes Hermes ``image_generate`` calls through the central llm-gateway so image
usage and cost are metered alongside chat. The gateway picks the upstream
provider (free Pollinations -> OpenRouter image-token) and normalizes the
response to ``b64_json``.

Identity is read from the per-session contextvars (``HERMES_SESSION_*``) so a
Telegram/CLI image call is attributed to the same user/session as their chat
calls.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    resolve_aspect_ratio,
    save_b64_image,
    success_response,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "auto"
_SIZES = {
    "landscape": "1024x576",
    "square": "1024x1024",
    "portrait": "576x1024",
}


def _session_env(name: str, default: str = "") -> str:
    try:
        from gateway.session_context import get_session_env

        value = get_session_env(name, default)
        return str(value or default).strip()
    except Exception:
        return str(os.environ.get(name) or default).strip()


def _gateway_base() -> str:
    base = (
        os.environ.get("AI_GATEWAY_URL")
        or os.environ.get("LLM_GATEWAY_URL")
        or "http://llm-gateway:8321"
    ).strip().rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    return base


class AIGatewayImageGenProvider(ImageGenProvider):
    @property
    def name(self) -> str:
        return "ai_gateway"

    @property
    def display_name(self) -> str:
        return "AI Model Gateway"

    def is_available(self) -> bool:
        try:
            import requests  # noqa: F401
        except ImportError:
            return False
        return True

    def list_models(self) -> List[Dict[str, Any]]:
        return [{
            "id": DEFAULT_MODEL,
            "display": "Gateway auto-route",
            "speed": "~5-150s (provider dependent)",
            "strengths": "Central routing, fallbacks and usage metering",
            "price": "varies by upstream provider",
        }]

    def default_model(self) -> Optional[str]:
        return DEFAULT_MODEL

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "AI Model Gateway",
            "badge": "metered",
            "tag": "Routes through llm-gateway; usage is recorded centrally",
            "env_vars": [],
        }

    def generate(self, prompt: str, aspect_ratio: str = DEFAULT_ASPECT_RATIO,
                 **kwargs: Any) -> Dict[str, Any]:
        prompt = (prompt or "").strip()
        aspect = resolve_aspect_ratio(aspect_ratio)
        if not prompt:
            return error_response(
                error="Prompt is required and must be a non-empty string",
                error_type="invalid_argument",
                provider="ai_gateway",
                aspect_ratio=aspect,
            )

        try:
            import requests
        except ImportError as exc:
            return error_response(
                error=f"requests is required: {exc}",
                error_type="missing_dependency",
                provider="ai_gateway",
                prompt=prompt,
                aspect_ratio=aspect,
            )

        base = _gateway_base()
        headers = {
            "Content-Type": "application/json",
            "X-Hermes-Client": "hermes-agent",
            "X-Hermes-Platform": _session_env("HERMES_SESSION_PLATFORM", "image_gen"),
            "X-Hermes-Job-Type": "image",
        }
        session_id = _session_env("HERMES_SESSION_ID")
        user_id = _session_env("HERMES_SESSION_USER_ID")
        user_name = _session_env("HERMES_SESSION_USER_NAME")
        if session_id:
            headers["X-Hermes-Session-Id"] = session_id
        if user_id:
            headers["X-Hermes-User-Id"] = user_id
        if user_name:
            headers["X-Hermes-User-Label"] = user_name
        api_key = (
            os.environ.get("LLM_GATEWAY_API_KEY")
            or os.environ.get("AI_GATEWAY_API_KEY")
            or ""
        ).strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        model = str(kwargs.get("model") or DEFAULT_MODEL).strip() or DEFAULT_MODEL
        body: Dict[str, Any] = {
            "prompt": prompt,
            "n": 1,
            "size": _SIZES.get(aspect, _SIZES["landscape"]),
        }
        if model.lower() not in ("auto", "default"):
            body["model"] = model

        try:
            resp = requests.post(
                f"{base}/v1/images/generations",
                headers=headers,
                json=body,
                timeout=240,
            )
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
            data = resp.json() or {}
            b64 = ((data.get("data") or [{}])[0] or {}).get("b64_json")
            if not b64:
                raise RuntimeError("gateway returned no b64_json")
            saved = save_b64_image(b64, prefix="ai_gateway")
        except Exception as exc:  # noqa: BLE001
            logger.debug("AI gateway image generation failed", exc_info=True)
            return error_response(
                error=f"AI gateway image generation failed: {exc}",
                error_type="api_error",
                provider="ai_gateway",
                model=model,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        return success_response(
            image=str(saved),
            model=model,
            prompt=prompt,
            aspect_ratio=aspect,
            provider="ai_gateway",
            extra={"gateway": base, "size": body["size"]},
        )


def register(ctx) -> None:
    ctx.register_image_gen_provider(AIGatewayImageGenProvider())
