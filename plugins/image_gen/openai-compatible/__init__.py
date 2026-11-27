"""OpenAI-compatible image generation provider plugin.

This backend targets services that implement ``POST /v1/images/generations``
with OpenAI-like request and response shapes. It accepts normal JSON responses
and Server-Sent Events responses, and saves either ``b64_json`` or ``url`` image
outputs into the Hermes image cache.
"""

from __future__ import annotations

import json
import logging
import math
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlsplit

import requests

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    resolve_aspect_ratio,
    save_b64_image,
    save_url_image,
    success_response,
)
from agent.secret_scope import get_secret

logger = logging.getLogger(__name__)

PROVIDER_NAME = "openai-compatible"
CONFIG_SECTION = "openai_compatible"
ENV_PREFIX = "OPENAI_COMPATIBLE_IMAGE"
DEFAULT_MODEL = "gpt-image-1"

_SIZES = {
    "landscape": "1536x1024",
    "square": "1024x1024",
    "portrait": "1024x1536",
}


def _load_config() -> Dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        cfg = load_config() or {}
        image_gen = cfg.get("image_gen") if isinstance(cfg, dict) else None
        if isinstance(image_gen, dict):
            section = image_gen.get(CONFIG_SECTION)
            if isinstance(section, dict):
                return section
    except Exception as exc:  # pragma: no cover - defensive config fallback
        logger.debug("Could not load %s image_gen config: %s", PROVIDER_NAME, exc)
    return {}


def _cfg_value(name: str, default: Optional[str] = None) -> Optional[str]:
    value = _load_config().get(name)
    if isinstance(value, str) and value.strip():
        return value.strip()
    # Retain the original env settings as profile-scoped compatibility fallbacks.
    env_value = get_secret(f"{ENV_PREFIX}_{name.upper()}")
    if env_value is not None and env_value.strip():
        return env_value.strip()
    return default


def _top_level_image_model() -> Optional[str]:
    try:
        from hermes_cli.config import load_config

        cfg = load_config() or {}
        image_gen = cfg.get("image_gen") if isinstance(cfg, dict) else None
        value = image_gen.get("model") if isinstance(image_gen, dict) else None
        if isinstance(value, str) and value.strip():
            return value.strip()
    except Exception as exc:  # pragma: no cover - defensive config fallback
        logger.debug("Could not load top-level image_gen.model: %s", exc)
    return None


def _resolve_model(override: Optional[str] = None) -> str:
    configured = _cfg_value("model")
    if configured:
        return configured
    if isinstance(override, str) and override.strip():
        return override.strip()
    return _top_level_image_model() or DEFAULT_MODEL


def _endpoint(base_url: str) -> str:
    value = base_url.strip().rstrip("/")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("base_url must be an absolute HTTP(S) API root")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("base_url must not contain credentials, a query, or a fragment")
    if parsed.path.endswith("/images/generations"):
        return value
    if not parsed.path:
        value += "/v1"
    return value + "/images/generations"


def _has_image_data(payload: Dict[str, Any]) -> bool:
    items = payload.get("data")
    return (
        isinstance(items, list)
        and bool(items)
        and isinstance(items[0], dict)
        and (
            bool(items[0].get("b64_json"))
            or bool(items[0].get("url"))
        )
    )


def _as_done_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    if _has_image_data(payload):
        return payload
    if isinstance(payload.get("b64_json"), str) and payload["b64_json"].strip():
        return {"data": [{"b64_json": payload["b64_json"]}]}
    if isinstance(payload.get("url"), str) and payload["url"].strip():
        return {"data": [{"url": payload["url"]}]}
    return payload


def _sse_events(lines: Iterable[str]) -> Iterable[tuple[str, str]]:
    event = ""
    data: List[str] = []
    for raw_line in lines:
        line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
        if not line:
            if data:
                yield event, "\n".join(data)
            event, data = "", []
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]
        if field == "event":
            event = value
        elif field == "data":
            data.append(value)
    # Some image gateways close immediately after their final data line.
    if data:
        yield event, "\n".join(data)


def _parse_sse_lines(lines: Iterable[str]) -> Dict[str, Any]:
    preferred_payload: Optional[Dict[str, Any]] = None
    image_payload: Optional[Dict[str, Any]] = None
    for event, data in _sse_events(lines):
        if data.strip() == "[DONE]":
            break
        payload = json.loads(data)
        if event == "error":
            message = payload
            if isinstance(payload, dict):
                message = payload.get("message") or payload.get("error") or payload
            raise ValueError(f"Image generation provider error: {message}")
        if not isinstance(payload, dict):
            continue
        if payload.get("error"):
            raise ValueError(f"Image generation provider error: {payload['error']}")
        # A preview never becomes a final result merely because the stream ends.
        if event == "partial_image":
            continue
        normalized = _as_done_payload(payload)
        if not _has_image_data(normalized):
            continue
        if event == "done":
            preferred_payload = normalized
        else:
            image_payload = normalized
    result = preferred_payload or image_payload
    if result is None:
        raise ValueError("No final image found in image generation SSE response")
    return result


def _response_json(response: requests.Response) -> Dict[str, Any]:
    content_type = response.headers.get("content-type", "").lower()
    if "text/event-stream" in content_type:
        response.encoding = "utf-8"
        return _parse_sse_lines(response.iter_lines(decode_unicode=True))

    try:
        payload = response.json()
    except ValueError:
        return _parse_sse_lines(response.text.splitlines())
    if not isinstance(payload, dict):
        raise ValueError("Image generation response was not a JSON object")
    return payload


class OpenAICompatibleImageGenProvider(ImageGenProvider):
    @property
    def name(self) -> str:
        return PROVIDER_NAME

    @property
    def display_name(self) -> str:
        return "OpenAI-compatible Images"

    def is_available(self) -> bool:
        try:
            _endpoint(_cfg_value("base_url") or "")
        except (ValueError, RuntimeError):
            return False
        return not bool(_load_config().get("api_key"))

    def list_models(self) -> List[Dict[str, Any]]:
        model = _resolve_model()
        return [
            {
                "id": model,
                "display": model,
                "description": "Configured OpenAI-compatible image generation model",
                "strengths": "OpenAI-compatible /v1/images/generations endpoint",
            }
        ]

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": self.display_name,
            "tag": "OpenAI-compatible /v1/images/generations endpoint — JSON & SSE",
            "env_vars": [
                {
                    "key": f"{ENV_PREFIX}_API_KEY",
                    "prompt": "API key (optional)",
                    "url": "",
                },
            ],
        }

    def capabilities(self) -> Dict[str, Any]:
        return {"modalities": ["text"], "max_reference_images": 0}

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        *,
        image_url: Optional[str] = None,
        reference_image_urls: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if image_url or reference_image_urls:
            return error_response(
                error="OpenAI-compatible Images supports text-to-image only",
                error_type="modality_unsupported", provider=self.name,
                prompt=prompt, aspect_ratio=aspect_ratio,
            )
        if not isinstance(prompt, str) or not prompt.strip():
            return error_response(error="Prompt must be a non-empty string", error_type="invalid_argument", provider=self.name)
        config = _load_config()
        if config.get("api_key"):
            return error_response(
                error=f"Move image_gen.openai_compatible.api_key to the {ENV_PREFIX}_API_KEY secret",
                error_type="invalid_config", provider=self.name,
            )
        base_url = (_cfg_value("base_url") or "").rstrip("/")
        model = _resolve_model(kwargs.get("model"))
        if not base_url:
            return error_response(
                error="OpenAI-compatible image base_url is not configured",
                error_type="missing_config",
                provider=self.name,
                model=model,
                prompt=prompt,
                aspect_ratio=aspect_ratio,
            )

        try:
            endpoint = _endpoint(base_url)
            timeout = float(config.get("timeout", 120.0))
            if not math.isfinite(timeout) or timeout <= 0:
                raise ValueError("timeout must be a finite positive number")
        except (TypeError, ValueError) as exc:
            return error_response(error=str(exc), error_type="invalid_config", provider=self.name)

        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "n": 1,
            "size": kwargs.get("size") or _SIZES.get(resolve_aspect_ratio(aspect_ratio), _SIZES["square"]),
        }
        for key in ("quality", "background", "image_detail", "output_format", "response_format", "seed"):
            value = kwargs.get(key)
            if value is not None:
                payload[key] = value

        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream, application/json",
        }
        api_key = (get_secret(f"{ENV_PREFIX}_API_KEY") or "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        try:
            with requests.post(
                endpoint,
                json=payload,
                headers=headers,
                timeout=timeout,
                stream=True,
                allow_redirects=False,
            ) as response:
                if 300 <= response.status_code < 400:
                    raise ValueError("Image endpoint redirected; configure the final API root explicitly")
                response.raise_for_status()
                data = _response_json(response)
        except Exception as exc:
            return error_response(
                error=str(exc),
                error_type="request_failed",
                provider=self.name,
                model=model,
                prompt=prompt,
                aspect_ratio=aspect_ratio,
            )

        items = data.get("data")
        if not isinstance(items, list) or not items or not isinstance(items[0], dict):
            return error_response(
                error="Image generation response did not contain data[0]",
                error_type="invalid_response",
                provider=self.name,
                model=model,
                prompt=prompt,
                aspect_ratio=aspect_ratio,
            )

        item = items[0]
        try:
            if isinstance(item.get("b64_json"), str) and item["b64_json"].strip():
                image = save_b64_image(item["b64_json"], prefix=PROVIDER_NAME, extension=payload.get("output_format") or "png")
            elif isinstance(item.get("url"), str) and item["url"].strip():
                image = save_url_image(item["url"], prefix=PROVIDER_NAME)
            else:
                raise ValueError("No b64_json or url in first data item")
        except Exception as exc:
            return error_response(
                error=str(exc),
                error_type="save_failed",
                provider=self.name,
                model=model,
                prompt=prompt,
                aspect_ratio=aspect_ratio,
            )

        return success_response(
            image=str(image),
            model=model,
            prompt=prompt,
            aspect_ratio=aspect_ratio,
            provider=self.name,
            extra={"base_url": base_url},
        )


def register(ctx: Any) -> None:
    ctx.register_image_gen_provider(OpenAICompatibleImageGenProvider())
