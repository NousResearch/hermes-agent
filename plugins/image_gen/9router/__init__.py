"""9router image generation backend.

Routes Hermes `image_generate` through a local/OpenAI-compatible 9router
Responses API endpoint and the `image_generation` tool. Supports optional
reference images by sending them as `input_image` content blocks.
"""

from __future__ import annotations

import base64
import json
import logging
import mimetypes
import os
import urllib.error
import urllib.request
from pathlib import Path
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

API_IMAGE_MODEL = "gpt-image-2"
DEFAULT_MODEL = "gpt-image-2-high"
DEFAULT_CHAT_MODEL = "cx/gpt-5.5"
DEFAULT_BASE_URL = "http://127.0.0.1:20128/v1"

_MODELS: Dict[str, Dict[str, Any]] = {
    "gpt-image-2-low": {
        "display": "GPT Image 2 via 9router (Low)",
        "speed": "~15s",
        "strengths": "Fast iteration, lowest cost",
        "quality": "low",
    },
    "gpt-image-2-medium": {
        "display": "GPT Image 2 via 9router (Medium)",
        "speed": "~40s",
        "strengths": "Balanced",
        "quality": "medium",
    },
    "gpt-image-2-high": {
        "display": "GPT Image 2 via 9router (High)",
        "speed": "~2min",
        "strengths": "Highest fidelity, strongest prompt adherence",
        "quality": "high",
    },
}

_SIZES = {
    "landscape": "1536x1024",
    "square": "1024x1024",
    "portrait": "1024x1536",
}


def _load_config() -> Dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        return cfg if isinstance(cfg, dict) else {}
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not load config for 9router image gen: %s", exc)
        return {}


def _image_config() -> Dict[str, Any]:
    cfg = _load_config()
    section = cfg.get("image_gen")
    return section if isinstance(section, dict) else {}


def _provider_config() -> Dict[str, Any]:
    cfg = _load_config()
    providers = cfg.get("providers")
    if isinstance(providers, dict):
        provider = providers.get("9router")
        if isinstance(provider, dict):
            return provider
    return {}


def _resolve_model() -> Tuple[str, Dict[str, Any]]:
    cfg = _image_config()
    sub = cfg.get("9router") if isinstance(cfg.get("9router"), dict) else {}
    for candidate in (
        os.environ.get("ROUTER_IMAGE_MODEL"),
        sub.get("model") if isinstance(sub, dict) else None,
        cfg.get("model"),
    ):
        if isinstance(candidate, str) and candidate in _MODELS:
            return candidate, _MODELS[candidate]
    return DEFAULT_MODEL, _MODELS[DEFAULT_MODEL]


def _resolve_base_url() -> str:
    cfg = _image_config()
    sub = cfg.get("9router") if isinstance(cfg.get("9router"), dict) else {}
    candidates = [
        os.environ.get("ROUTER_BASE_URL"),
        sub.get("base_url") if isinstance(sub, dict) else None,
        _provider_config().get("base_url"),
        (_load_config().get("model") or {}).get("base_url") if isinstance(_load_config().get("model"), dict) else None,
        DEFAULT_BASE_URL,
    ]
    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value.strip().rstrip("/")
    return DEFAULT_BASE_URL


def _resolve_chat_model() -> str:
    cfg = _image_config()
    sub = cfg.get("9router") if isinstance(cfg.get("9router"), dict) else {}
    for value in (
        os.environ.get("ROUTER_RESPONSES_MODEL"),
        sub.get("responses_model") if isinstance(sub, dict) else None,
        _provider_config().get("responses_model"),
        DEFAULT_CHAT_MODEL,
    ):
        if isinstance(value, str) and value.strip():
            return value.strip()
    return DEFAULT_CHAT_MODEL


def _literal_env(value: str) -> Optional[str]:
    value = value.strip()
    if value.startswith("${") and value.endswith("}"):
        return os.environ.get(value[2:-1])
    return value or None


def _env_from_ref(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    name = value.strip()
    return os.environ.get(name) if name else None


def _read_key_file(path: Path) -> Optional[str]:
    try:
        if path.exists():
            value = path.read_text(encoding="utf-8").strip()
            return value or None
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not read 9router key file %s: %s", path, exc)
    return None


def _resolve_api_key() -> Optional[str]:
    cfg = _image_config()
    sub = cfg.get("9router") if isinstance(cfg.get("9router"), dict) else {}
    provider = _provider_config()
    candidates = [
        os.environ.get("ROUTER_API_KEY"),
        os.environ.get("NINE_ROUTER_API_KEY"),
        os.environ.get("NINEROUTER_API_KEY"),
        sub.get("api_key") if isinstance(sub, dict) else None,
        _env_from_ref(provider.get("key_env")),
        _env_from_ref(provider.get("api_key_env")),
        provider.get("api_key"),
        (_load_config().get("model") or {}).get("api_key") if isinstance(_load_config().get("model"), dict) else None,
    ]
    for value in candidates:
        if isinstance(value, str):
            resolved = _literal_env(value)
            if resolved:
                return resolved

    home = Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes"))).expanduser()
    return _read_key_file(home / ".9router-api-key") or _read_key_file(Path("/opt/9router/.client-api-key"))


def _image_to_data_url(ref: str) -> str:
    if ref.startswith("data:image/") or ref.startswith("http://") or ref.startswith("https://"):
        return ref
    path = Path(ref).expanduser()
    raw = path.read_bytes()
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def _normalize_reference_images(value: Any) -> List[str]:
    if not value:
        return []
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple)):
        values = [v for v in value if isinstance(v, str) and v.strip()]
    else:
        values = []
    return [v.strip() for v in values if v.strip()]


def _extract_image_b64(payload: Dict[str, Any]) -> Optional[str]:
    for item in payload.get("output") or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "image_generation_call" and isinstance(item.get("result"), str):
            return item["result"]
        for content in item.get("content") or []:
            if isinstance(content, dict):
                for key in ("result", "image", "image_b64", "b64_json"):
                    value = content.get(key)
                    if isinstance(value, str) and value:
                        return value
    return None


class NineRouterImageGenProvider(ImageGenProvider):
    @property
    def name(self) -> str:
        return "9router"

    @property
    def display_name(self) -> str:
        return "9router"

    def is_available(self) -> bool:
        return bool(_resolve_api_key())

    def list_models(self) -> List[Dict[str, Any]]:
        return [
            {"id": model_id, "display": meta["display"], "speed": meta["speed"], "strengths": meta["strengths"], "price": "routed"}
            for model_id, meta in _MODELS.items()
        ]

    def default_model(self) -> Optional[str]:
        return DEFAULT_MODEL

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "9router",
            "badge": "local",
            "tag": "GPT Image 2 via local/OpenAI-compatible 9router Responses API",
            "env_vars": [],
        }

    def generate(self, prompt: str, aspect_ratio: str = DEFAULT_ASPECT_RATIO, **kwargs: Any) -> Dict[str, Any]:
        prompt = (prompt or "").strip()
        aspect = resolve_aspect_ratio(aspect_ratio)
        tier_id, meta = _resolve_model()
        size = _SIZES.get(aspect, _SIZES["square"])

        if not prompt:
            return error_response(error="Prompt is required", error_type="invalid_argument", provider=self.name, model=tier_id, aspect_ratio=aspect)

        api_key = _resolve_api_key()
        if not api_key:
            return error_response(error="No 9router API key available", error_type="auth_required", provider=self.name, model=tier_id, prompt=prompt, aspect_ratio=aspect)

        content: List[Dict[str, str]] = [{"type": "input_text", "text": prompt}]
        try:
            for ref in _normalize_reference_images(kwargs.get("reference_images")):
                content.append({"type": "input_image", "image_url": _image_to_data_url(ref)})
        except Exception as exc:  # noqa: BLE001
            return error_response(error=f"Could not load reference image: {exc}", error_type="invalid_reference_image", provider=self.name, model=tier_id, prompt=prompt, aspect_ratio=aspect)

        body = {
            "model": _resolve_chat_model(),
            "input": [{"role": "user", "content": content}],
            "tools": [{
                "type": "image_generation",
                "model": API_IMAGE_MODEL,
                "size": size,
                "quality": meta["quality"],
                "output_format": "png",
                "background": "opaque",
            }],
            "tool_choice": {"type": "allowed_tools", "mode": "required", "tools": [{"type": "image_generation"}]},
        }
        req = urllib.request.Request(
            _resolve_base_url() + "/responses",
            data=json.dumps(body).encode("utf-8"),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=360) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            return error_response(error=f"9router HTTP {exc.code}: {detail}", error_type="api_error", provider=self.name, model=tier_id, prompt=prompt, aspect_ratio=aspect)
        except Exception as exc:  # noqa: BLE001
            return error_response(error=f"9router image generation failed: {exc}", error_type="api_error", provider=self.name, model=tier_id, prompt=prompt, aspect_ratio=aspect)

        b64 = _extract_image_b64(payload)
        if not b64:
            return error_response(error="9router response contained no image_generation_call result", error_type="empty_response", provider=self.name, model=tier_id, prompt=prompt, aspect_ratio=aspect)

        try:
            saved_path = save_b64_image(b64, prefix=f"9router_{tier_id}")
        except Exception as exc:  # noqa: BLE001
            return error_response(error=f"Could not save image to cache: {exc}", error_type="io_error", provider=self.name, model=tier_id, prompt=prompt, aspect_ratio=aspect)

        return success_response(
            image=str(saved_path),
            model=tier_id,
            prompt=prompt,
            aspect_ratio=aspect,
            provider=self.name,
            extra={"size": size, "quality": meta["quality"], "reference_images": len(content) - 1},
        )


def register(ctx) -> None:
    ctx.register_image_gen_provider(NineRouterImageGenProvider())
