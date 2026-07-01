"""KarinAI trusted image-gateway backend for ``image_generate``.

Managed KarinAI agent containers should not hold raw OpenAI/FAL/Codex image
provider credentials. This provider bridges the upstream ``image_generate`` tool
to the backend-owned image gateway using the scoped ``KARINAI_RUNTIME_TOKEN``
that runtime-manager injects into the container.
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
import os
import re
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    normalize_reference_images,
    resolve_aspect_ratio,
    success_response,
)

logger = logging.getLogger(__name__)

PROVIDER_NAME = "karinai-image-gateway"
DEFAULT_MODEL = "karinai/image-gateway"
_MAX_DATA_URL_FALLBACK_B64_CHARS = 2_000_000
_ALLOWED_DATA_URL_MIME_TYPES = {"image/png", "image/jpeg", "image/webp", "image/gif"}
_ASPECT_TO_GATEWAY = {
    "landscape": "16:9",
    "square": "1:1",
    "portrait": "9:16",
}
_GATEWAY_ASPECTS = {"1:1", "16:9", "9:16", "4:3", "3:4", "21:9"}
_BEARER_RE = re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]+")
_AUTHORIZATION_ASSIGNMENT_RE = re.compile(
    r"(?i)(authorization)([\"']?\s*[:=]\s*[\"']?)(bearer\s+)?([^\"'\s,}]+)"
)
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)(api[_-]?key|token)([\"']?\s*[:=]\s*[\"']?)([^\"'\s,}]+)"
)


def _clean(value: object) -> str:
    return str(value or "").strip()


def _gateway_url() -> str:
    return _clean(os.environ.get("KARINAI_IMAGE_GATEWAY_URL"))


def _runtime_token() -> str:
    return _clean(os.environ.get("KARINAI_RUNTIME_TOKEN"))


def _redact_text(value: object) -> str:
    text = _clean(value)
    if not text:
        return ""
    text = _BEARER_RE.sub("Bearer [REDACTED]", text)
    token = _runtime_token()
    if token:
        text = text.replace(token, "[REDACTED]")

    def _redact_authorization(match: re.Match[str]) -> str:
        scheme = "Bearer " if (match.group(3) or "").strip().lower() == "bearer" else ""
        return f"{match.group(1)}{match.group(2)}{scheme}[REDACTED]"

    text = _AUTHORIZATION_ASSIGNMENT_RE.sub(_redact_authorization, text)
    text = _SECRET_ASSIGNMENT_RE.sub(r"\1\2[REDACTED]", text)
    return text


def _configured_model() -> str:
    return _clean(os.environ.get("KARINAI_IMAGE_GATEWAY_MODEL"))


def _configured_provider_hint() -> str:
    # This is the backend-side provider selected by runtime-manager/image-gateway
    # (for example "openai" or "fake"), not the agent plugin provider name.
    return _clean(os.environ.get("KARINAI_IMAGE_GATEWAY_PROVIDER"))


def _client_timeout_seconds() -> float:
    raw = _clean(os.environ.get("KARINAI_IMAGE_GATEWAY_TIMEOUT_SECONDS"))
    if not raw:
        return 130.0
    try:
        value = float(raw)
    except ValueError:
        return 130.0
    return value if value > 0 else 130.0


def _generation_endpoint(base_url: str) -> str:
    text = _clean(base_url).rstrip("/")
    if not text:
        return ""
    if text.endswith("/internal/images/generations"):
        return text
    if text.endswith("/v1/images/generations"):
        text = text[: -len("/v1/images/generations")].rstrip("/")
    elif text.endswith("/v1"):
        text = text[: -len("/v1")].rstrip("/")
    elif text.endswith("/internal"):
        text = text[: -len("/internal")].rstrip("/")
    return f"{text}/internal/images/generations"


def _gateway_aspect_ratio(value: str) -> str:
    text = _clean(value)
    if text in _GATEWAY_ASPECTS:
        return text
    return _ASPECT_TO_GATEWAY.get(resolve_aspect_ratio(text), "16:9")


def _safe_image_count(value: object) -> int:
    try:
        count = int(str(value or "1"))
    except (TypeError, ValueError):
        return 1
    return 1 if count < 1 else min(count, 1)


def _safe_extension(value: object) -> str:
    ext = _clean(value).lower().lstrip(".") or "png"
    if ext == "jpeg":
        return "jpg"
    if ext in {"png", "jpg", "webp", "gif"}:
        return ext
    return "png"


def _load_json_error(raw: bytes) -> str:
    try:
        data = json.loads(raw.decode("utf-8") or "{}")
    except Exception:
        return ""
    if isinstance(data, dict):
        err = data.get("error")
        if isinstance(err, dict):
            message = _clean(err.get("message"))
            code = _clean(err.get("code"))
            if message and code:
                return _redact_text(f"{message} ({code})")
            return _redact_text(message or code)
        if isinstance(err, str):
            return _redact_text(err)
    return ""


def _post_json(endpoint: str, token: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=raw,
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=_client_timeout_seconds()) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        detail = _load_json_error(exc.read(8192))
        message = f"image gateway returned HTTP {exc.code}"
        if detail:
            message = f"{message}: {detail}"
        raise RuntimeError(message) from exc
    except urllib.error.URLError as exc:
        reason = _redact_text(getattr(exc, "reason", exc)) or "network error"
        raise RuntimeError(f"image gateway is unreachable: {reason}") from exc

    try:
        decoded = json.loads(body.decode("utf-8") or "{}")
    except Exception as exc:
        raise RuntimeError("image gateway returned invalid JSON") from exc
    if not isinstance(decoded, dict):
        raise RuntimeError("image gateway returned a non-object JSON response")
    return decoded


def _compact_asset(asset: Dict[str, Any]) -> Dict[str, Any]:
    allowed = {
        "id",
        "mime_type",
        "size",
        "checksum",
        "width",
        "height",
        "format",
        "signed_url",
    }
    return {key: value for key, value in asset.items() if key in allowed and value is not None}


def _data_url_from_asset(asset: Dict[str, Any]) -> tuple[str, str]:
    raw_b64 = asset.get("b64_json")
    if not isinstance(raw_b64, str) or not raw_b64.strip():
        return "", ""
    b64_json = raw_b64.strip()
    if len(b64_json) > _MAX_DATA_URL_FALLBACK_B64_CHARS:
        return "", "image gateway b64_json asset exceeded the retrievable data URL limit"
    try:
        base64.b64decode(b64_json, validate=True)
    except (binascii.Error, ValueError):
        return "", "image gateway returned invalid b64 image data"
    mime_type = _clean(asset.get("mime_type")) or "image/png"
    if mime_type not in _ALLOWED_DATA_URL_MIME_TYPES:
        return "", "image gateway returned an unsupported image MIME type"
    return f"data:{mime_type};base64,{b64_json}", ""


class KarinAIImageGatewayProvider(ImageGenProvider):
    """Bridge image generation to the trusted KarinAI image gateway."""

    @property
    def name(self) -> str:
        return PROVIDER_NAME

    @property
    def display_name(self) -> str:
        return "KarinAI image gateway"

    def is_available(self) -> bool:
        return bool(_gateway_url() and _runtime_token())

    def list_models(self) -> List[Dict[str, Any]]:
        model = _configured_model() or DEFAULT_MODEL
        return [
            {
                "id": model,
                "display": model,
                "speed": "gateway managed",
                "strengths": "Backend-owned provider credentials, policy, storage, and artifact routing",
                "price": "gateway managed",
            }
        ]

    def default_model(self) -> Optional[str]:
        return _configured_model() or DEFAULT_MODEL

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "KarinAI image gateway",
            "badge": "managed",
            "tag": "Managed-runtime backend that calls the trusted KarinAI image gateway with the scoped runtime token.",
            "env_vars": [],
        }

    def capabilities(self) -> Dict[str, Any]:
        # The first backend image-gateway milestone accepts text prompts only.
        # Reference/image editing should be exposed here only after the gateway
        # supports uploaded reference IDs or another safe product-owned source.
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
        aspect = resolve_aspect_ratio(aspect_ratio)
        model = _clean(kwargs.get("model")) or _configured_model()
        response_model = model or DEFAULT_MODEL
        endpoint = _generation_endpoint(_gateway_url())
        token = _runtime_token()

        refs = normalize_reference_images(reference_image_urls)
        if (isinstance(image_url, str) and image_url.strip()) or refs:
            return error_response(
                error=(
                    "KarinAI image gateway generation currently supports text-to-image only. "
                    "Reference-image/edit support must be added to the backend gateway before "
                    "image_url or reference_image_urls can be used."
                ),
                error_type="modality_unsupported",
                provider=self.name,
                model=response_model,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        if not endpoint or not token:
            return error_response(
                error="KARINAI_IMAGE_GATEWAY_URL and KARINAI_RUNTIME_TOKEN are required for managed image generation",
                error_type="missing_gateway_config",
                provider=self.name,
                model=response_model,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        provider_hint = _configured_provider_hint()
        payload: Dict[str, Any] = {
            "prompt": prompt,
            "aspect_ratio": _gateway_aspect_ratio(aspect_ratio),
            "n": _safe_image_count(kwargs.get("num_images") or kwargs.get("n")),
            "output_format": _safe_extension(kwargs.get("output_format")),
            "include_b64_json": True,
            "wait_for_completion": True,
            "user_id": _clean(os.environ.get("KARINAI_USER_ID")) or None,
            "workspace_id": _clean(os.environ.get("KARINAI_WORKSPACE_ID")) or None,
            "metadata": {
                "source": "karinai-agent-image-generate",
                "agent_provider": self.name,
            },
        }
        if model:
            payload["model"] = model
        if provider_hint:
            payload["provider"] = provider_hint
        try:
            from gateway.session_context import get_session_env

            product_run_id = _clean(get_session_env("HERMES_PRODUCT_RUN_ID", ""))
        except Exception:
            product_run_id = _clean(os.environ.get("HERMES_PRODUCT_RUN_ID"))
        if product_run_id:
            payload["run_id"] = product_run_id

        try:
            result = _post_json(endpoint, token, payload)
        except Exception as exc:  # noqa: BLE001 - provider API returns errors as data
            safe_error = _redact_text(exc)
            logger.warning("KarinAI image gateway request failed: %s", safe_error)
            return error_response(
                error=safe_error,
                error_type="image_gateway_error",
                provider=self.name,
                model=response_model,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        if result.get("status") != "completed":
            return error_response(
                error=f"image gateway returned status {result.get('status') or 'unknown'}",
                error_type="image_gateway_status",
                provider=self.name,
                model=str(result.get("model") or response_model),
                prompt=prompt,
                aspect_ratio=aspect,
            )

        assets = result.get("assets")
        if not isinstance(assets, list) or not assets or not isinstance(assets[0], dict):
            return error_response(
                error="image gateway completed without returning an image asset",
                error_type="image_gateway_contract",
                provider=self.name,
                model=str(result.get("model") or response_model),
                prompt=prompt,
                aspect_ratio=aspect,
            )

        first_asset = assets[0]
        image = _clean(first_asset.get("signed_url"))
        fallback_error = ""
        if not image:
            # Managed product clients need a retrievable reference in the public
            # response. Until the image gateway returns product-signed URLs for
            # every stored asset, allow a bounded, validated data URL fallback for
            # small gateway-returned b64 assets (used by staging fake-provider e2e).
            image, fallback_error = _data_url_from_asset(first_asset)
        if not image:
            return error_response(
                error=fallback_error or "image gateway asset had no signed_url or b64_json image data",
                error_type="image_gateway_contract",
                provider=self.name,
                model=str(result.get("model") or response_model),
                prompt=prompt,
                aspect_ratio=aspect,
            )

        compact_assets = [_compact_asset(asset) for asset in assets if isinstance(asset, dict)]
        extra = {
            "generation_id": result.get("id"),
            "asset_id": first_asset.get("id"),
            "gateway_provider": result.get("provider"),
            "gateway_model": result.get("model"),
            "assets": compact_assets,
        }
        return success_response(
            image=image,
            model=str(result.get("model") or response_model),
            prompt=prompt,
            aspect_ratio=aspect,
            provider=self.name,
            modality="text",
            extra={key: value for key, value in extra.items() if value is not None},
        )
