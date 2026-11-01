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
OPENAI_SIZES: dict[str, str] = {"landscape": "1536x1024", "square": "1024x1024", "portrait": "1024x1536"}

# gpt-image-2 quality tiers as virtual model ids (same API model, different ``quality`` knob).
GPT_IMAGE_2_API_MODEL = "gpt-image-2"
GPT_IMAGE_2_DEFAULT = "gpt-image-2-medium"
GPT_IMAGE_2_TIERS: dict[str, dict[str, Any]] = {
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

ErrorFn = Callable[..., dict[str, Any]]


def size_for(aspect: str) -> str:
    """OpenAI ``size`` string for a semantic aspect (square when unknown)."""
    return OPENAI_SIZES.get(aspect, OPENAI_SIZES["square"])


def load_image_gen_config(sub: Optional[str] = None) -> dict[str, Any]:
    """Read ``image_gen`` (or ``image_gen.<sub>``) from config.yaml; ``{}`` on any failure."""
    label = "image_gen" if sub is None else f"image_gen.{sub}"
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        section = cfg.get("image_gen") if isinstance(cfg, dict) else None
        if sub is not None:
            section = section.get(sub) if isinstance(section, dict) else None
        return section if isinstance(section, dict) else {}
    except Exception as exc:
        logger.debug("Could not load %s config: %s", label, exc)
        return {}


def resolve_static_model(
    models: dict[str, dict[str, Any]], default: str, *, env_var: str, config_key: str,
    explicit: Optional[str] = None, include_top_level: bool = True,
    config: Optional[dict[str, Any]] = None, passthrough: bool = False,
) -> tuple[str, dict[str, Any]]:
    """``(model_id, meta)`` from a fixed catalog; first *known* id wins (unknown ids fall through):
    explicit → ``env_var`` → ``image_gen.<config_key>.model`` → ``image_gen.model`` → ``default``.

    ``passthrough``: an unknown id from ``env_var`` or the provider-scoped ``image_gen.<config_key>.model``
    is sent verbatim as the API model with no ``quality`` (OpenAI-compatible gateways serve their own
    image model names and reject unknown enum values, #97928). The shared top-level ``image_gen.model``
    never passes through — it may hold another provider's id.

    The provider-scoped and env selections are explicit intent for THIS provider, so they are decided
    before the shared top-level knob is consulted. Testing them against the catalog first instead let a
    stale top-level catalog id shadow a scoped custom id: a gateway user with
    ``image_gen.openai.model: ag/gemini-3.1-flash-image`` plus a leftover
    ``image_gen.model: gpt-image-2-medium`` got the latter sent to the gateway, which rejected the
    unknown model as a missing-credential error."""
    if isinstance(explicit, str) and explicit.strip() in models:
        return explicit.strip(), models[explicit.strip()]
    env_override = os.environ.get(env_var)
    if env_override and env_override in models:
        return env_override, models[env_override]
    cfg = load_image_gen_config() if config is None else config
    scoped = cfg.get(config_key)
    scoped_model = scoped.get("model") if isinstance(scoped, dict) else None
    for candidate in (env_override, scoped_model):
        if not isinstance(candidate, str) or not candidate.strip():
            continue
        candidate = candidate.strip()
        if candidate in models:
            return candidate, models[candidate]
        if passthrough:
            return candidate, {"display": candidate, "api_model": candidate, "quality": None}
    top_level = cfg.get("model") if include_top_level else None
    if isinstance(top_level, str) and top_level in models:
        return top_level, models[top_level]
    return default, models[default]


def collect_source_images(
    image_url: Optional[str], reference_image_urls: Optional[list[str]], limit: Optional[int] = None
) -> list[str]:
    """Primary ``image_url`` first, then normalized references, clamped to ``limit``."""
    sources: list[str] = []
    if isinstance(image_url, str) and image_url.strip():
        sources.append(image_url.strip())
    sources.extend(normalize_reference_images(reference_image_urls) or [])
    return sources[:limit] if limit is not None else sources


def catalog_rows(
    models: dict[str, dict[str, Any]],
    fields: Iterable[str] = ("display", "speed", "strengths", "price"), *,
    price: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Picker rows: ``id`` + ``fields`` (missing ``display`` → id, else ``""``); ``price`` overrides."""
    rows = []
    for model_id, meta in models.items():
        row: dict[str, Any] = {"id": model_id}
        for field in fields:
            row[field] = meta.get(field, model_id if field == "display" else "")
        if price is not None:
            row["price"] = price
        rows.append(row)
    return rows


def api_key_setup_schema(
    name: str, badge: str, tag: str, *, key: str, prompt: str, url: str
) -> dict[str, Any]:
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
    models: dict[str, dict[str, Any]] = {}
    default_model_id: Optional[str] = None
    price: Optional[str] = None
    catalog_fields: tuple[str, ...] = ("display", "speed", "strengths", "price")
    setup: dict[str, Any] = {}

    @property
    def name(self) -> str:
        return self.provider_id

    @property
    def display_name(self) -> str:
        return self.label

    def list_models(self) -> list[dict[str, Any]]:
        return catalog_rows(self.models, self.catalog_fields, price=self.price)

    def default_model(self) -> Optional[str]:
        return self.default_model_id

    def get_setup_schema(self) -> dict[str, Any]:
        return api_key_setup_schema(**self.setup)


def error_factory(provider: str, aspect: str, *, model: str = "", prompt: str = "") -> ErrorFn:
    """Return ``fail(error, error_type, **override)`` pre-bound to this call's context."""

    def fail(error: str, error_type: str, **override: Any) -> dict[str, Any]:
        kwargs = dict(provider=provider, model=model, prompt=prompt, aspect_ratio=aspect)
        kwargs.update(override)
        return error_response(error=error, error_type=error_type, **kwargs)

    return fail


def prompt_required_error(provider: str, aspect: str) -> dict[str, Any]:
    return error_factory(provider, aspect)(PROMPT_REQUIRED, "invalid_argument")


def openai_importable() -> bool:
    return import_openai("", "")[0] is not None


def import_openai(provider: str, aspect: str) -> tuple[Any, Optional[dict[str, Any]]]:
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
) -> tuple[Optional[str], Optional[dict[str, Any]]]:
    """``(image_ref, None)`` or ``(None, error)`` for a ``(b64_json, url)`` pair. Base64 is always
    cached (write failure → ``io_error``); a URL is cached best-effort, falling back to the bare URL."""
    fail = error_factory(provider, aspect, model=model, prompt=prompt)
    if b64:
        try:
            return str(save_b64_image(b64, prefix=prefix)), None
        except Exception as exc:
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
    except Exception as exc:
        if on_fail is not None:
            on_fail(exc)
        else:
            log.warning("%s image URL %s could not be cached (%s); falling back to bare URL.", label, url, exc)
        return url


def requests_error_message(response: Any, exc: Exception) -> str:
    """``error.message`` from an HTTP error body, else the first 300 chars of it."""
    try:
        return response.json().get("error", {}).get("message", response.text[:300])
    except Exception:
        return response.text[:300] if response is not None else str(exc)


@dataclass
class HttpFailure:
    """One failed ``post_json`` attempt as ``(error, error_type)``. ``kind`` ∈ http / timeout /
    unreachable (DNS failure or refused connection: nothing was sent) / connection (dropped
    after connecting) / request / invalid_json; ``message`` = HTTP error text or decode error;
    ``status`` / ``response`` are set for ``http`` only."""

    kind: str
    error: str
    error_type: str
    status: int = 0
    message: str = ""
    response: Any = None


def record_token_usage(usage: Any, *, model: str, provider: str, base_url: Optional[str] = None) -> None:
    """Record a token-billed image call against the ambient session as task ``image_generation``.

    Token-metered image models (OpenRouter chat-image and Image API models, OpenAI ``gpt-image``)
    bill exactly like a chat completion, so they get a ``session_model_usage`` row through the
    same chokepoint as auxiliary calls. ``usage`` is the response's usage block (dict or SDK
    object). Per-image backends (FAL, xAI, Krea, ...) return no token usage and never call this;
    a body without tokens is a no-op inside ``record_aux_usage``, as is running outside a turn.
    """
    if not usage:
        return
    from types import SimpleNamespace

    from agent.aux_accounting import record_aux_usage

    record_aux_usage(
        SimpleNamespace(model=model, usage=usage), "image_generation", provider=provider, base_url=base_url)


def _never_connected(exc: Exception) -> bool:
    """True when the last connection attempt never connected, so no server accepted the request body
    (a redirect hop may have answered earlier, but a redirect accepts nothing)."""
    from urllib3.exceptions import MaxRetryError, NameResolutionError, NewConnectionError

    cause = exc.args[0] if exc.args else None
    reason = cause.reason if isinstance(cause, MaxRetryError) else cause
    return isinstance(reason, (NewConnectionError, NameResolutionError))


def post_json(
    url: str, *, headers: dict[str, str], payload: dict[str, Any], timeout: Any, label: str,
    error_message: Callable[[Any, Exception], str] = requests_error_message,
    catch_request_exception: bool = False,
) -> tuple[Optional[Any], Optional[HttpFailure]]:
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
        kind = "unreachable" if _never_connected(exc) else "connection"
        return None, HttpFailure(kind, f"{label} connection error: {exc}", "connection_error")
    except requests.RequestException as exc:
        if not catch_request_exception:
            raise
        return None, HttpFailure("request", f"{label} request failed: {exc}", "api_error")
    try:
        return response.json(), None
    except Exception as exc:
        return None, HttpFailure(
            "invalid_json", f"{label} returned invalid JSON: {exc}", "invalid_response", message=str(exc),
        )
