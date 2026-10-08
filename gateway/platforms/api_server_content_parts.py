"""Pure content-part normalization helpers for the API server."""

from typing import Any, Dict

MAX_NORMALIZED_TEXT_LENGTH = 65_536  # 64 KB cap for normalized content parts
MAX_CONTENT_LIST_SIZE = 1_000  # Max items when content is an array


def _cap_text(text: str) -> str:
    return text[:MAX_NORMALIZED_TEXT_LENGTH] if len(text) > MAX_NORMALIZED_TEXT_LENGTH else text


def _cap_list(items: list) -> list:
    return items[:MAX_CONTENT_LIST_SIZE] if len(items) > MAX_CONTENT_LIST_SIZE else items


# Chat Completions / Responses part-type spellings; emitted shape is always the canonical
# ``{"type": "text", ...}`` / ``{"type": "image_url", ...}`` the agent pipeline understands.
_TEXT_PART_TYPES = frozenset({"text", "input_text", "output_text"})
_IMAGE_PART_TYPES = frozenset({"image_url", "input_image"})
_FILE_PART_TYPES = frozenset({"file", "input_file"})


def _normalize_image_part(part: Dict[str, Any]) -> Dict[str, Any]:
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
    image_part: Dict[str, Any] = {"type": "image_url", "image_url": {"url": url_value}}
    if detail is not None:
        if not isinstance(detail, str) or not detail.strip():
            raise ValueError("invalid_content_part:Image detail must be a non-empty string when provided.")
        image_part["image_url"]["detail"] = detail.strip()
    return image_part


def _content_has_visible_payload(content: Any) -> bool:
    """True when content has any text or image attachment.  Used to reject empty turns."""
    if isinstance(content, str):
        return bool(content.strip())
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict):
                ptype = str(part.get("type") or "").strip().lower()
                if ptype in _IMAGE_PART_TYPES or (
                        ptype in _TEXT_PART_TYPES and str(part.get("text") or "").strip()):
                    return True
    return False
