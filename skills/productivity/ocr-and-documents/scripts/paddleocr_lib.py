"""
PaddleOCR Library

API wrapper for PaddleOCR document parsing and text recognition capabilities.
"""

import base64
import logging
import math
import os
from pathlib import Path
from typing import Any, Optional
from urllib.parse import unquote, urlparse

import httpx

logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

DEFAULT_TIMEOUT = 600  # seconds (10 minutes) for layout, 120s for ocr
API_GUIDE_URL = "https://paddleocr.com"
FILE_TYPE_PDF = 0
FILE_TYPE_IMAGE = 1
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp")


# =============================================================================
# Environment
# =============================================================================


def _get_env(key: str) -> str:
    """Get environment variable, defaulting to empty string with whitespace stripped."""
    return os.getenv(key, "").strip()


def _http_timeout_from_env(env_key: str, default_seconds: float) -> float:
    """
    Read HTTP client timeout in seconds from the environment.
    """
    raw = os.getenv(env_key)
    if raw is None:
        return float(default_seconds)
    stripped = raw.strip()
    if not stripped:
        return float(default_seconds)
    try:
        timeout = float(stripped)
    except (ValueError, TypeError):
        logger.warning(
            "Invalid %s value %r; using default %ss",
            env_key,
            raw,
            default_seconds,
        )
        return float(default_seconds)
    if not math.isfinite(timeout) or timeout <= 0:
        logger.warning(
            "%s must be a finite number > 0 (got %r); using default %ss",
            env_key,
            raw,
            default_seconds,
        )
        return float(default_seconds)
    return timeout


def _resolve_api_url(api_url: str, env_var: str) -> str:
    """Require https; allow host-only values by prepending https://."""
    if api_url.startswith("http://"):
        raise ValueError(f"{env_var} must use https://; http:// is not allowed.")
    if not api_url.startswith("https://"):
        return f"https://{api_url}"
    return api_url


def get_config(is_layout: bool = True) -> tuple[str, str, float]:
    """
    Get API URL, token, and timeout from environment.

    Args:
        is_layout: True for layout parsing, False for OCR

    Returns:
        tuple of (api_url, token, timeout)
    """
    url_env = "PADDLEOCR_DOC_PARSING_API_URL" if is_layout else "PADDLEOCR_OCR_API_URL"
    timeout_env = "PADDLEOCR_DOC_PARSING_TIMEOUT" if is_layout else "PADDLEOCR_OCR_TIMEOUT"
    default_timeout = 600.0 if is_layout else 120.0
    endpoint_suffix = "/layout-parsing" if is_layout else "/ocr"

    api_url = _get_env(url_env)
    token = _get_env("PADDLEOCR_ACCESS_TOKEN")

    if not api_url:
        raise ValueError(f"{url_env} not configured. Get your API at: {API_GUIDE_URL}")
    if not token:
        raise ValueError(f"PADDLEOCR_ACCESS_TOKEN not configured. Get your API at: {API_GUIDE_URL}")

    api_url = _resolve_api_url(api_url, url_env)
    api_path = urlparse(api_url).path.rstrip("/")
    if not api_path.endswith(endpoint_suffix):
        raise ValueError(
            f"{url_env} must be a full endpoint ending with {endpoint_suffix}. "
            f"Example: https://your-service.paddleocr.com{endpoint_suffix}"
        )

    timeout = _http_timeout_from_env(timeout_env, default_timeout)

    return api_url, token, timeout


# =============================================================================
# File Utilities
# =============================================================================


def _detect_file_type(path_or_url: str) -> int:
    """Detect file type: 0=PDF, 1=Image."""
    path = path_or_url.lower()
    if path.startswith(("http://", "https://")):
        path = unquote(urlparse(path).path)

    if path.endswith(".pdf"):
        return FILE_TYPE_PDF
    elif path.endswith(IMAGE_EXTENSIONS):
        return FILE_TYPE_IMAGE
    else:
        raise ValueError(f"Unsupported file format: {path_or_url}")


def _load_file_as_base64(file_path: str) -> str:
    """Load local file and encode as base64."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if path.stat().st_size == 0:
        raise ValueError(f"File is empty (0 bytes): {file_path}")

    return base64.b64encode(path.read_bytes()).decode("utf-8")


# =============================================================================
# API Request
# =============================================================================


def _make_api_request(
    api_url: str, token: str, timeout: float, params: dict[str, Any]
) -> dict[str, Any]:
    """
    Make PaddleOCR API request.
    """
    headers = {
        "Authorization": f"token {token}",
        "Content-Type": "application/json",
        "Client-Platform": "official-skill",
    }

    try:
        with httpx.Client(timeout=timeout) as client:
            try:
                resp = client.post(api_url, json=params, headers=headers)
            except TypeError as e:
                raise RuntimeError(
                    "Request parameters cannot be JSON-encoded; use only JSON-serializable "
                    f"option values ({e})"
                ) from e
    except httpx.TimeoutException:
        raise RuntimeError(f"API request timed out after {timeout}s")
    except httpx.RequestError as e:
        raise RuntimeError(f"API request failed: {e}")

    if resp.status_code != 200:
        error_detail = ""
        try:
            error_body = resp.json()
            if isinstance(error_body, dict):
                error_detail = str(error_body.get("errorMsg", "")).strip()
        except Exception:
            pass

        if not error_detail:
            error_detail = (resp.text[:200] or "No response body").strip()

        if resp.status_code == 403:
            raise RuntimeError(f"Authentication failed (403): {error_detail}")
        elif resp.status_code == 429:
            raise RuntimeError(f"API rate limit exceeded (429): {error_detail}")
        elif resp.status_code >= 500:
            raise RuntimeError(
                f"API service error ({resp.status_code}): {error_detail}"
            )
        else:
            raise RuntimeError(f"API error ({resp.status_code}): {error_detail}")

    try:
        result = resp.json()
    except Exception:
        raise RuntimeError(f"Invalid JSON response: {resp.text[:200]}")

    if not isinstance(result, dict):
        raise RuntimeError(
            f"Unexpected JSON shape (expected object): {resp.text[:200]}"
        )

    if result.get("errorCode", 0) != 0:
        msg = result.get("errorMsg", "Unknown error")
        raise RuntimeError(f"API error: {msg}")

    return result


# =============================================================================
# Main API
# =============================================================================


def _process_document(
    is_layout: bool,
    file_path: Optional[str] = None,
    file_url: Optional[str] = None,
    file_type: Optional[int] = None,
    **options: Any,
) -> dict[str, Any]:
    """
    Internal function to process document with PaddleOCR.
    """
    if file_path is not None and not isinstance(file_path, str):
        return _error("INPUT_ERROR", "file_path must be a string or None")
    if file_url is not None and not isinstance(file_url, str):
        return _error("INPUT_ERROR", "file_url must be a string or None")

    fp = file_path.strip() if file_path else ""
    fu = file_url.strip() if file_url else ""
    if fp and fu:
        return _error(
            "INPUT_ERROR",
            "Provide only one of file_path or file_url, not both",
        )
    if not fp and not fu:
        return _error("INPUT_ERROR", "file_path or file_url required")
    if file_type is not None and file_type not in (FILE_TYPE_PDF, FILE_TYPE_IMAGE):
        return _error("INPUT_ERROR", "file_type must be 0 (PDF) or 1 (Image)")

    try:
        api_url, token, timeout = get_config(is_layout=is_layout)
    except ValueError as e:
        return _error("CONFIG_ERROR", str(e))

    # Build request params
    try:
        resolved_file_type: Optional[int] = None
        if fu:
            params = {"file": fu}
            if file_type is not None:
                resolved_file_type = file_type
            else:
                try:
                    resolved_file_type = _detect_file_type(fu)
                except ValueError:
                    resolved_file_type = None
        else:
            resolved_file_type = (
                file_type if file_type is not None else _detect_file_type(fp)
            )
            params = {"file": _load_file_as_base64(fp)}

        params["visualize"] = False  # reduce response payload
        params.update(options)
        if resolved_file_type is not None:
            params["fileType"] = resolved_file_type
        else:
            params.pop("fileType", None)

    except (ValueError, OSError, MemoryError) as e:
        return _error("INPUT_ERROR", str(e))

    try:
        result = _make_api_request(api_url, token, timeout, params)
    except RuntimeError as e:
        return _error("API_ERROR", str(e))

    try:
        text = _extract_text_layout(result) if is_layout else _extract_text_ocr(result)
    except ValueError as e:
        return _error("API_ERROR", str(e))

    return {
        "ok": True,
        "text": text,
        "result": result,
        "error": None,
    }


def parse_document(
    file_path: Optional[str] = None,
    file_url: Optional[str] = None,
    file_type: Optional[int] = None,
    **options: Any,
) -> dict[str, Any]:
    """Parse document with PaddleOCR."""
    return _process_document(True, file_path, file_url, file_type, **options)


def ocr(
    file_path: Optional[str] = None,
    file_url: Optional[str] = None,
    file_type: Optional[int] = None,
    **options: Any,
) -> dict[str, Any]:
    """Perform OCR on image or PDF with PaddleOCR."""
    return _process_document(False, file_path, file_url, file_type, **options)


def _extract_text_layout(result: dict[str, Any]) -> str:
    """Extract text from document parsing result."""
    if not isinstance(result, dict):
        raise ValueError("Invalid API response: top-level response must be an object")

    raw_result = result.get("result")
    if not isinstance(raw_result, dict):
        raise ValueError("Invalid API response: missing 'result' object")

    pages = raw_result.get("layoutParsingResults")
    if not isinstance(pages, list):
        raise ValueError(
            "Invalid API response: result.layoutParsingResults must be an array"
        )

    texts = []
    for i, page in enumerate(pages):
        if not isinstance(page, dict):
            raise ValueError(
                f"Invalid API response: result.layoutParsingResults[{i}] must be an object"
            )

        markdown = page.get("markdown")
        if not isinstance(markdown, dict):
            raise ValueError(
                f"Invalid API response: result.layoutParsingResults[{i}].markdown must be an object"
            )

        text = markdown.get("text")
        if not isinstance(text, str):
            raise ValueError(
                f"Invalid API response: result.layoutParsingResults[{i}].markdown.text must be a string"
            )
        texts.append(text)

    return "\n\n".join(texts)


def _extract_text_ocr(result: dict[str, Any]) -> str:
    """Extract text from OCR result."""
    if not isinstance(result, dict):
        raise ValueError("Invalid API response: top-level response must be an object")

    raw_result = result.get("result")
    if not isinstance(raw_result, dict):
        raise ValueError("Invalid API response: missing 'result' object")

    pages = raw_result.get("ocrResults")
    if not isinstance(pages, list):
        raise ValueError("Invalid API response: result.ocrResults must be an array")

    all_text = []
    for i, item in enumerate(pages):
        if not isinstance(item, dict):
            raise ValueError(
                f"Invalid API response: result.ocrResults[{i}] must be an object"
            )

        pruned = item.get("prunedResult")
        if not isinstance(pruned, dict):
            raise ValueError(
                f"Invalid API response: result.ocrResults[{i}].prunedResult must be an object"
            )

        texts = pruned.get("rec_texts", [])
        if not isinstance(texts, list):
            raise ValueError(
                f"Invalid API response: result.ocrResults[{i}].prunedResult.rec_texts must be an array"
            )

        line_parts: list[str] = []
        for j, t in enumerate(texts):
            if not isinstance(t, str):
                raise ValueError(
                    f"Invalid API response: result.ocrResults[{i}].prunedResult.rec_texts[{j}] must be a string"
                )
            line_parts.append(t)
        if line_parts:
            all_text.append("\n".join(line_parts))

    return "\n\n".join(all_text)


def _error(code: str, message: str) -> dict[str, Any]:
    """Create error response."""
    return {
        "ok": False,
        "text": "",
        "result": None,
        "error": {"code": code, "message": message},
    }
