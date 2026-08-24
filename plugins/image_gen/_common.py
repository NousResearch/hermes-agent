"""Shared helpers for the bundled ``image_gen`` provider plugins. Providers are loaded by path
(``hermes_plugins.image_gen__<name>``) and resolve this via the repo root on ``sys.path``;
not a plugin itself (the scanner only looks at directories)."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from agent.image_gen_provider import (
    ImageGenProvider, error_response, normalize_reference_images, save_b64_image, save_url_image)

logger = logging.getLogger(__name__)

# OpenAI-style ``size`` per semantic aspect, shared by every OpenAI-compatible backend.
OPENAI_SIZES: Dict[str, str] = {"landscape": "1536x1024", "square": "1024x1024", "portrait": "1024x1536"}

# gpt-image-2 quality tiers as virtual model ids (same API model, different ``quality`` knob).
GPT_IMAGE_2_API_MODEL = "gpt-image-2"
GPT_IMAGE_2_DEFAULT = "gpt-image-2-medium"
GPT_IMAGE_2_TIERS: Dict[str, Dict[str, Any]] = {
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

PROMPT_REQUIRED = "Prompt is required and must be a non-empty string"
OPENAI_MISSING = "openai Python package not installed (pip install openai)"

ErrorFn = Callable[..., Dict[str, Any]]


def size_for(aspect: str) -> str:
    """OpenAI ``size`` string for a semantic aspect (square when unknown)."""
    return OPENAI_SIZES.get(aspect, OPENAI_SIZES["square"])


def load_image_gen_config(sub: Optional[str] = None) -> Dict[str, Any]:
    """Read ``image_gen`` (or ``image_gen.<sub>``) from config.yaml; ``{}`` on any failure."""
    label = "image_gen" if sub is None else f"image_gen.{sub}"
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        section = cfg.get("image_gen") if isinstance(cfg, dict) else None
        if sub is not None:
            section = section.get(sub) if isinstance(section, dict) else None
        return section if isinstance(section, dict) else {}
    except Exception as exc:  # noqa: BLE001 - config is best-effort
        logger.debug("Could not load %s config: %s", label, exc)
        return {}


def resolve_static_model(
    models: Dict[str, Dict[str, Any]], default: str, *, env_var: str, config_key: str,
    explicit: Optional[str] = None, include_top_level: bool = True,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """``(model_id, meta)`` from a fixed catalog; first *known* id wins (unknown ids fall through):
    explicit → ``env_var`` → ``image_gen.<config_key>.model`` → ``image_gen.model`` → ``default``."""
    if isinstance(explicit, str) and explicit.strip() in models:
        return explicit.strip(), models[explicit.strip()]
    env_override = os.environ.get(env_var)
    if env_override and env_override in models:
        return env_override, models[env_override]
    cfg = load_image_gen_config() if config is None else config
    scoped = cfg.get(config_key)
    candidates = [scoped.get("model") if isinstance(scoped, dict) else None]
    if include_top_level:
        candidates.append(cfg.get("model"))
    for candidate in candidates:
        if isinstance(candidate, str) and candidate in models:
            return candidate, models[candidate]
    return default, models[default]


def collect_source_images(
    image_url: Optional[str], reference_image_urls: Optional[List[str]], limit: Optional[int] = None
) -> List[str]:
    """Primary ``image_url`` first, then normalized references, clamped to ``limit``."""
    sources: List[str] = []
    if isinstance(image_url, str) and image_url.strip():
        sources.append(image_url.strip())
    sources.extend(normalize_reference_images(reference_image_urls) or [])
    return sources[:limit] if limit is not None else sources


def catalog_rows(
    models: Dict[str, Dict[str, Any]],
    fields: Iterable[str] = ("display", "speed", "strengths", "price"), *,
    price: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Picker rows: ``id`` + ``fields`` (missing ``display`` → id, else ``""``); ``price`` overrides."""
    rows = []
    for model_id, meta in models.items():
        row: Dict[str, Any] = {"id": model_id}
        for field in fields:
            row[field] = meta.get(field, model_id if field == "display" else "")
        if price is not None:
            row["price"] = price
        rows.append(row)
    return rows


def api_key_setup_schema(
    name: str, badge: str, tag: str, *, key: str, prompt: str, url: str
) -> Dict[str, Any]:
    """``get_setup_schema()`` dict for a provider authenticated by one env var."""
    return {
        "name": name, "badge": badge, "tag": tag, "env_vars": [{"key": key, "prompt": prompt, "url": url}],
    }


class StaticImageGenProvider(ImageGenProvider):
    """Identity + picker surface from class attributes: ``provider_id``/``label``; fixed catalog via
    ``models`` (+ ``default_model_id``, ``price``, ``catalog_fields``); single-env-var auth via
    ``setup`` (kwargs for :func:`api_key_setup_schema`). Dynamic providers override methods."""

    provider_id: str
    label: str
    models: Dict[str, Dict[str, Any]] = {}
    default_model_id: Optional[str] = None
    price: Optional[str] = None
    catalog_fields: Tuple[str, ...] = ("display", "speed", "strengths", "price")
    setup: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        return self.provider_id

    @property
    def display_name(self) -> str:
        return self.label

    def list_models(self) -> List[Dict[str, Any]]:
        return catalog_rows(self.models, self.catalog_fields, price=self.price)

    def default_model(self) -> Optional[str]:
        return self.default_model_id

    def get_setup_schema(self) -> Dict[str, Any]:
        return api_key_setup_schema(**self.setup)


def error_factory(provider: str, aspect: str, *, model: str = "", prompt: str = "") -> ErrorFn:
    """Return ``fail(error, error_type, **override)`` pre-bound to this call's context."""

    def fail(error: str, error_type: str, **override: Any) -> Dict[str, Any]:
        kwargs = dict(provider=provider, model=model, prompt=prompt, aspect_ratio=aspect)
        kwargs.update(override)
        return error_response(error=error, error_type=error_type, **kwargs)

    return fail


def prompt_required_error(provider: str, aspect: str) -> Dict[str, Any]:
    return error_factory(provider, aspect)(PROMPT_REQUIRED, "invalid_argument")


def openai_importable() -> bool:
    return import_openai("", "")[0] is not None


def import_openai(provider: str, aspect: str) -> Tuple[Any, Optional[Dict[str, Any]]]:
    """Return ``(openai_module, None)`` or ``(None, missing_dependency error)``."""
    try:
        import openai
    except ImportError:
        return None, error_factory(provider, aspect)(OPENAI_MISSING, "missing_dependency")
    return openai, None


def materialize_image(
    b64: Optional[str], url: Optional[str], *, prefix: str, label: str, provider: str, model: str,
    prompt: str, aspect: str, log: logging.Logger = logger,
    on_url_fail: Optional[Callable[[Exception], None]] = None,
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """``(image_ref, None)`` or ``(None, error)`` for a ``(b64_json, url)`` pair. Base64 is always
    cached (write failure → ``io_error``); a URL is cached best-effort, falling back to the bare URL."""
    fail = error_factory(provider, aspect, model=model, prompt=prompt)
    if b64:
        try:
            return str(save_b64_image(b64, prefix=prefix)), None
        except Exception as exc:  # noqa: BLE001
            return None, fail(f"Could not save image to cache: {exc}", "io_error")
    if url:
        return cache_url_best_effort(url, prefix=prefix, label=label, log=log, on_fail=on_url_fail), None
    return None, fail(f"{label} response contained neither b64_json nor URL", "empty_response")


def cache_url_best_effort(
    url: str, *, prefix: str, label: str, log: logging.Logger = logger,
    on_fail: Optional[Callable[[Exception], None]] = None,
) -> str:
    """Cache ``url`` locally; on failure warn (or call ``on_fail``) and return the bare URL."""
    try:
        return str(save_url_image(url, prefix=prefix))
    except Exception as exc:  # noqa: BLE001
        if on_fail is not None:
            on_fail(exc)
        else:
            log.warning("%s image URL %s could not be cached (%s); falling back to bare URL.", label, url, exc)
        return url


def requests_error_message(response: Any, exc: Exception) -> str:
    """``error.message`` from an HTTP error body, else the first 300 chars of it."""
    try:
        return response.json().get("error", {}).get("message", response.text[:300])
    except Exception:  # noqa: BLE001 - non-JSON / non-dict body
        return response.text[:300] if response is not None else str(exc)


@dataclass
class HttpFailure:
    """One failed ``post_json`` attempt as ``(error, error_type)``. ``kind`` ∈ http / timeout /
    connection / request / invalid_json; ``message`` = HTTP error text or decode error;
    ``status`` / ``response`` are set for ``http`` only."""

    kind: str
    error: str
    error_type: str
    status: int = 0
    message: str = ""
    response: Any = None


def post_json(
    url: str, *, headers: Dict[str, str], payload: Dict[str, Any], timeout: Any, label: str,
    error_message: Callable[[Any, Exception], str] = requests_error_message,
    catch_request_exception: bool = False,
) -> Tuple[Optional[Any], Optional[HttpFailure]]:
    """POST ``payload`` → ``(json_body, None)`` or ``(None, failure)``. ``timeout`` goes to ``requests``
    verbatim (message reports the read component); ``error_message(response, exc)`` extracts the
    backend-specific HTTP error text."""
    import requests

    read_timeout = timeout[1] if isinstance(timeout, tuple) else timeout
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=timeout)
        response.raise_for_status()
    except requests.HTTPError as exc:
        resp = exc.response
        status = resp.status_code if resp is not None else 0
        message = error_message(resp, exc)
        return None, HttpFailure(
            "http", f"{label} image generation failed ({status}): {message}", "api_error",
            status=status, message=message, response=resp)
    except requests.Timeout:
        return None, HttpFailure(
            "timeout", f"{label} image generation timed out ({int(read_timeout)}s)", "timeout")
    except requests.ConnectionError as exc:
        return None, HttpFailure("connection", f"{label} connection error: {exc}", "connection_error")
    except requests.RequestException as exc:
        if not catch_request_exception:
            raise
        return None, HttpFailure("request", f"{label} request failed: {exc}", "api_error")
    try:
        return response.json(), None
    except Exception as exc:  # noqa: BLE001
        return None, HttpFailure(
            "invalid_json", f"{label} returned invalid JSON: {exc}", "invalid_response", message=str(exc),
        )
