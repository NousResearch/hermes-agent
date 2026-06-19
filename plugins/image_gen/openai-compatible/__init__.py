"""OpenAI-compatible image generation provider plugin.

This backend targets services that implement ``POST /v1/images/generations``
with OpenAI-like request and response shapes. It accepts normal JSON responses
and Server-Sent Events responses, and saves either ``b64_json`` or ``url`` image
outputs into the Hermes image cache.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Iterable, List, Optional

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

logger = logging.getLogger(__name__)

PROVIDER_NAME = "openai-compatible"
CONFIG_SECTION = "openai_compatible"
ENV_PREFIX = "OPENAI_COMPATIBLE_IMAGE"
DEFAULT_MODEL = "gpt-image-1"


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
    env_name = f"{ENV_PREFIX}_{name.upper()}"
    env_value = os.getenv(env_name)
    if env_value is not None and env_value.strip():
        return env_value.strip()

    value = _load_config().get(name)
    if isinstance(value, str) and value.strip():
        return value.strip()
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
    if isinstance(override, str) and override.strip():
        return override.strip()
    return _cfg_value("model") or _top_level_image_model() or DEFAULT_MODEL


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


def _parse_sse_lines(lines: Iterable[str]) -> Dict[str, Any]:
    current_event: Optional[str] = None
    preferred_payload: Optional[Dict[str, Any]] = None
    image_payload: Optional[Dict[str, Any]] = None
    partial_payload: Optional[Dict[str, Any]] = None
    last_payload: Optional[Dict[str, Any]] = None

    for raw_line in lines:
        if isinstance(raw_line, bytes):
            line = raw_line.decode("utf-8", errors="replace")
        else:
            line = str(raw_line)
        line = line.strip()
        if not line or line.startswith(":"):
            continue
        if line.startswith("event:"):
            current_event = line[6:].strip()
            continue
        if not line.startswith("data:"):
            continue

        data = line[5:].strip()
        if data == "[DONE]":
            continue
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            logger.debug("Skipping non-JSON SSE image payload: %s", data[:200])
            continue
        if not isinstance(payload, dict):
            continue

        if current_event == "error":
            message = payload.get("message") or payload.get("error") or payload
            raise ValueError(f"Image generation provider error: {message}")

        last_payload = payload
        normalized = _as_done_payload(payload)
        if current_event == "done":
            preferred_payload = normalized
        elif _has_image_data(normalized):
            image_payload = normalized
        elif current_event == "partial_image":
            partial_payload = normalized

    result = preferred_payload or image_payload or partial_payload or last_payload
    if result is None:
        raise ValueError("No JSON payload found in image generation SSE response")
    return result


def _response_json(response: requests.Response) -> Dict[str, Any]:
    content_type = response.headers.get("content-type", "").lower()
    if "text/event-stream" in content_type:
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
        return bool(_cfg_value("base_url"))

    def list_models(self) -> List[Dict[str, Any]]:
        model = _resolve_model()
        return [
            {
                "id": model,
                "name": model,
                "description": "Configured OpenAI-compatible image generation model",
                "strengths": "OpenAI-compatible /v1/images/generations endpoint",
            }
        ]

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "env": [f"{ENV_PREFIX}_BASE_URL"],
            "optional_env": [f"{ENV_PREFIX}_API_KEY", f"{ENV_PREFIX}_MODEL"],
            "tag": "OpenAI-compatible /v1/images/generations endpoint",
        }

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        **kwargs: Any,
    ) -> Dict[str, Any]:
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

        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "n": 1,
            "size": kwargs.get("size") or resolve_aspect_ratio(aspect_ratio),
        }
        for key in ("quality", "background", "image_detail", "output_format", "response_format", "seed"):
            value = kwargs.get(key)
            if value is not None:
                payload[key] = value

        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream, application/json",
        }
        api_key = _cfg_value("api_key")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        try:
            response = requests.post(
                f"{base_url}/images/generations",
                json=payload,
                headers=headers,
                timeout=float(kwargs.get("timeout") or 120.0),
            )
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
