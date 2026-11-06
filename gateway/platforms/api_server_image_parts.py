"""Image content parts of API-server requests, validated into the canonical vision shape.

Split out of ``api_server.py`` (the facade keeps the multimodal walk; this holds the one
part-level validator) so the facade stops growing.
"""

from __future__ import annotations

from typing import Any


def _normalize_image_part(part: dict[str, Any]) -> dict[str, Any]:
    """Validate one image part (Responses top-level ``image_url`` string or Chat Completions
    ``{"url", "detail"}`` dict) into the canonical vision shape; raises ValueError."""
    detail = part.get("detail")
    image_ref = part.get("image_url")
    if isinstance(image_ref, dict):
        url_value = image_ref.get("url")
        detail = image_ref.get("detail", detail)
    else:
        url_value = image_ref
    if not isinstance(url_value, str) or not url_value.strip():
        raise ValueError("invalid_image_url:Image parts must include a non-empty image URL.")
    url_value = url_value.strip()
    lowered = url_value.lower()
    if lowered.startswith("data:"):
        if not lowered.startswith("data:image/") or "," not in url_value:
            raise ValueError(
                "unsupported_content_type:Only image data URLs are supported. "
                "Non-image data payloads are not supported.")
    elif not (lowered.startswith("http://") or lowered.startswith("https://")):
        raise ValueError(
            "invalid_image_url:Image inputs must use http(s) URLs or data:image/... URLs.")
    image_part: dict[str, Any] = {"type": "image_url", "image_url": {"url": url_value}}
    if detail is not None:
        if not isinstance(detail, str) or not detail.strip():
            raise ValueError("invalid_content_part:Image detail must be a non-empty string when provided.")
        image_part["image_url"]["detail"] = detail.strip()
    return image_part
