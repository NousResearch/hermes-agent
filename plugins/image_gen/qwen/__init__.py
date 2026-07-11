"""
Wanxiang (万相) V2 -- Alibaba DashScope text-to-image backend.

Supports wan2.6-t2i (sync), wan2.5/wan2.2 (async) via the DashScope API.
Models: wan2.6-t2i, wan2.5-t2i-preview, wan2.2-t2i-flash, wan2.2-t2i-plus

API reference:
  https://help.aliyun.com/zh/model-studio/text-to-image-v2-api-reference

Authentication:
  DASHSCOPE_API_KEY environment variable, or
  config.yaml: image_gen.providers.qwen.api_key
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    resolve_aspect_ratio,
)

logger = logging.getLogger(__name__)

# Default public endpoint.
_DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/api/v1"

MODELS: Dict[str, Dict[str, Any]] = {
    "wan2.6-t2i": {
        "display": "Wan 2.6",
        "speed": "~10s",
        "strengths": "Latest model, HTTP sync, free size [1280*1280~1440*1440] px",
        "price": "paid",
        "sync": True,
    },
    "wan2.5-t2i-preview": {
        "display": "Wan 2.5 Preview",
        "speed": "~30s async",
        "strengths": "Free size, ratio [1:4~4:1]",
        "price": "paid",
        "sync": False,
    },
    "wan2.2-t2i-flash": {
        "display": "Wan 2.2 Flash",
        "speed": "~15s async",
        "strengths": "50% faster than 2.1",
        "price": "paid",
        "sync": False,
    },
    "wan2.2-t2i-plus": {
        "display": "Wan 2.2 Plus",
        "speed": "~30s async",
        "strengths": "Higher stability and success rate",
        "price": "paid",
        "sync": False,
    },
}
DEFAULT_MODEL = "wan2.6-t2i"

_FREE_SIZES: Dict[str, str] = {
    "landscape": "1696*960",
    "square": "1280*1280",
    "portrait": "960*1696",
}
_FIXED_SIZES: Dict[str, str] = {
    "landscape": "1408*800",
    "square": "1024*1024",
    "portrait": "800*1408",
}

_MAX_PROMPT_CHARS = 2100
_MAX_NEGATIVE_PROMPT_CHARS = 500
_ASYNC_POLL_INTERVAL = 2
_ASYNC_POLL_MAX_ATTEMPTS = 60
_CIRCUIT_BREAKER_THRESHOLD = 5
_CIRCUIT_BREAKER_COOLDOWN = 120


def _resolve_api_key() -> Optional[str]:
    key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if key:
        return key
    try:
        from hermes_cli.config import cfg_get
        key = (cfg_get("image_gen.providers.qwen.api_key") or "").strip()
        return key or None
    except Exception:
        return None


def _resolve_base_url() -> str:
    return os.environ.get("QWEN_IMAGE_BASE_URL", _DEFAULT_BASE_URL).rstrip("/")


def _resolve_model(model_id: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
    if model_id and model_id in MODELS:
        return model_id, MODELS[model_id]
    try:
        from hermes_cli.config import cfg_get
        configured = cfg_get("image_gen.model")
        if configured and configured in MODELS:
            return configured, MODELS[configured]
    except Exception:
        pass
    return DEFAULT_MODEL, MODELS[DEFAULT_MODEL]


def _resolve_size(aspect_ratio: str, sync: bool) -> str:
    sizes = _FREE_SIZES if sync else _FIXED_SIZES
    return sizes.get(aspect_ratio, "1280*1280" if sync else "1024*1024")


def _download_to_cache(image_url: str, cache_dir: Path) -> Optional[Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time() * 1000)
    url_hash = hashlib.md5(image_url.encode()).hexdigest()[:8]
    output_path = cache_dir / f"wanx_{timestamp}_{url_hash}.png"
    try:
        urllib.request.urlretrieve(image_url, str(output_path))
        if output_path.stat().st_size > 0:
            return output_path
    except Exception:
        logger.debug("Failed to download image from %s", image_url)
    return None


class QwenImageProvider(ImageGenProvider):
    """Wanxiang V2 text-to-image via Alibaba DashScope."""

    def __init__(self) -> None:
        super().__init__()
        self._consecutive_failures: int = 0
        self._breaker_open_until: float = 0.0

    @property
    def name(self) -> str:
        return "qwen"

    @property
    def display_name(self) -> str:
        return "Wanxiang V2 (万相)"

    def is_available(self) -> bool:
        return bool(_resolve_api_key())

    def list_models(self) -> List[Dict[str, Any]]:
        return [
            {"id": mid, "display": meta["display"],
             "speed": meta.get("speed", ""),
             "strengths": meta.get("strengths", ""),
             "price": meta.get("price", "")}
            for mid, meta in MODELS.items()
        ]

    def default_model(self) -> Optional[str]:
        return DEFAULT_MODEL

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Wanxiang V2 (万相)",
            "badge": "paid",
            "tag": "Alibaba Bailian - wan2.6-t2i, wan2.5, wan2.2",
            "env_vars": [{"key": "DASHSCOPE_API_KEY",
                          "prompt": "Alibaba Bailian API Key",
                          "url": "https://help.aliyun.com/zh/model-studio/get-api-key"}],
        }

    def capabilities(self) -> Dict[str, Any]:
        return {"modalities": ["text"], "max_reference_images": 0}

    def generate(self, prompt, aspect_ratio=DEFAULT_ASPECT_RATIO, *,
                 image_url=None, reference_image_urls=None, **kwargs):
        api_key = _resolve_api_key()
        if not api_key:
            return self._error("DASHSCOPE_API_KEY not set", "auth")
        if self._breaker_tripped():
            return self._error("Temporarily unavailable after repeated failures",
                               "circuit_breaker")

        model_id, meta = _resolve_model()
        aspect = resolve_aspect_ratio(aspect_ratio)
        size = _resolve_size(aspect, meta.get("sync", True))
        n = self._clamp_n(kwargs.get("n", 1))

        safe_prompt = prompt[:_MAX_PROMPT_CHARS]
        body = {"model": model_id,
                "input": {"messages": [{"role": "user",
                          "content": [{"text": safe_prompt}]}]},
                "parameters": {"size": size, "watermark": False,
                               "prompt_extend": True, "n": n}}
        if kwargs.get("seed") is not None:
            body["parameters"]["seed"] = int(kwargs["seed"])
        if kwargs.get("negative_prompt"):
            body["parameters"]["negative_prompt"] = str(
                kwargs["negative_prompt"])[:_MAX_NEGATIVE_PROMPT_CHARS]

        base_url = _resolve_base_url()
        try:
            if meta.get("sync"):
                response = self._sync_generate(api_key, base_url, body)
            else:
                response = self._async_generate(api_key, base_url, body)
            self._record_success()
        except Exception as exc:
            self._record_failure()
            return self._error(f"Generation failed: {exc}", type(exc).__name__)

        if not isinstance(response, dict):
            return self._error("Unexpected API response", "provider_contract")

        # API-level error: pass the error code through as error_type
        if response.get("code"):
            return self._error(
                f"{response.get('code')}: {response.get('message', '')}",
                response.get("code", "api_error"),
            )

        image_url = self._extract_image_url(response)
        if not image_url:
            return self._error("No image in response", "empty_response")

        from hermes_constants import get_hermes_home
        cache_dir = get_hermes_home() / "image_cache"
        local = _download_to_cache(image_url, cache_dir)
        result = str(local) if local else image_url

        return {"success": True, "image": result, "model": model_id,
                "prompt": prompt, "aspect_ratio": aspect,
                "modality": "text", "provider": self.name}

    def _sync_generate(self, api_key, base_url, body):
        url = f"{base_url}/services/aigc/multimodal-generation/generation"
        return self._post(api_key, url, body, 120)

    def _async_generate(self, api_key, base_url, body):
        submit_url = f"{base_url}/services/aigc/text2image/image-synthesis"
        h = {"X-DashScope-Async": "enable"}
        resp = self._post(api_key, submit_url, body, 30, h)
        if resp.get("code"):
            return resp
        tid = (resp.get("output") or {}).get("task_id") or resp.get("task_id")
        if not tid:
            return {"code": "parse_error", "message": "No task_id"}

        poll_url = f"{base_url}/tasks/{tid}"
        delay = _ASYNC_POLL_INTERVAL
        for attempt in range(1, _ASYNC_POLL_MAX_ATTEMPTS + 1):
            time.sleep(delay)
            delay = min(delay * 1.2, 10.0)
            try:
                s = self._get(api_key, poll_url, 10)
            except Exception:
                continue
            st = (s.get("output") or {}).get("task_status") or s.get("task_status", "")
            if st == "SUCCEEDED":
                return s
            if st in ("FAILED", "ERROR", "CANCELED"):
                return {"code": "api_error",
                        "message": f"Task {st}: {(s.get('output') or {}).get('message', '')}"}
        return {"code": "timeout", "message": "Async task timed out"}

    @staticmethod
    def _post(api_key, url, body, timeout, extra_headers=None):
        h = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        if extra_headers:
            h.update(extra_headers)
        req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                      headers=h, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            try:
                return json.loads(e.read().decode())
            except Exception:
                return {"code": str(e.code), "message": str(e)}

    @staticmethod
    def _get(api_key, url, timeout):
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())

    @staticmethod
    def _extract_image_url(data):
        output = data.get("output") or {}
        choices = output.get("choices") or []
        if choices:
            parts = choices[0].get("message", {}).get("content") or []
            for p in parts:
                if isinstance(p, dict) and p.get("image"):
                    return str(p["image"])
        results = output.get("results") or []
        if results and isinstance(results[0], dict) and results[0].get("url"):
            return str(results[0]["url"])
        return None

    def _breaker_tripped(self):
        if self._consecutive_failures < _CIRCUIT_BREAKER_THRESHOLD:
            return False
        if time.monotonic() >= self._breaker_open_until:
            self._consecutive_failures = 0
            return False
        return True

    def _record_success(self):
        self._consecutive_failures = 0

    def _record_failure(self):
        self._consecutive_failures += 1
        if self._consecutive_failures >= _CIRCUIT_BREAKER_THRESHOLD:
            self._breaker_open_until = time.monotonic() + _CIRCUIT_BREAKER_COOLDOWN

    @staticmethod
    def _error(msg, typ="unknown"):
        return {"success": False, "image": None, "error": msg,
                "error_type": typ, "provider": "qwen"}

    @staticmethod
    def _clamp_n(n):
        try:
            v = int(n)
        except (TypeError, ValueError):
            return 1
        return max(1, min(v, 4))


def register(ctx):
    ctx.register_image_gen_provider(QwenImageProvider())
