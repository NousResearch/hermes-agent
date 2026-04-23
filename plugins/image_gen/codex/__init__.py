"""Codex / ChatGPT-subscription image generation backend.

Uses Hermes's existing ``openai-codex`` OAuth session (stored in
``~/.hermes/auth.json``) to call the ChatGPT Codex Responses backend with the
built-in ``image_generation`` tool. No ``OPENAI_API_KEY`` is required.

Selection precedence (first hit wins):

1. ``CODEX_IMAGE_MODEL`` env var
2. ``image_gen.codex.model`` in ``config.yaml``
3. ``image_gen.model`` in ``config.yaml`` (when it matches our catalog)
4. :data:`DEFAULT_MODEL`
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from agent.auxiliary_client import _codex_cloudflare_headers
from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    resolve_aspect_ratio,
    save_b64_image,
    success_response,
)
from hermes_cli.auth import resolve_codex_runtime_credentials

logger = logging.getLogger(__name__)


_MODELS: Dict[str, Dict[str, Any]] = {
    "gpt-5.4-low": {
        "display": "GPT-5.4 via ChatGPT Subscription (Low)",
        "speed": "~15s",
        "strengths": "Fast iteration, lowest cost",
        "tool_model": "gpt-image-2",
        "quality": "low",
        "api_model": "gpt-5.4",
    },
    "gpt-5.4-medium": {
        "display": "GPT-5.4 via ChatGPT Subscription (Medium)",
        "speed": "~40s",
        "strengths": "Balanced — default",
        "tool_model": "gpt-image-2",
        "quality": "medium",
        "api_model": "gpt-5.4",
    },
    "gpt-5.4-high": {
        "display": "GPT-5.4 via ChatGPT Subscription (High)",
        "speed": "~2min",
        "strengths": "Highest fidelity",
        "tool_model": "gpt-image-2",
        "quality": "high",
        "api_model": "gpt-5.4",
    },
}

DEFAULT_MODEL = "gpt-5.4-medium"


def _load_codex_config() -> Dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        section = cfg.get("image_gen") if isinstance(cfg, dict) else None
        return section if isinstance(section, dict) else {}
    except Exception as exc:
        logger.debug("Could not load image_gen config: %s", exc)
        return {}



def _resolve_model() -> Tuple[str, Dict[str, Any]]:
    env_override = os.environ.get("CODEX_IMAGE_MODEL")
    if env_override and env_override in _MODELS:
        return env_override, _MODELS[env_override]

    cfg = _load_codex_config()
    codex_cfg = cfg.get("codex") if isinstance(cfg.get("codex"), dict) else {}
    candidate: Optional[str] = None
    if isinstance(codex_cfg, dict):
        value = codex_cfg.get("model")
        if isinstance(value, str) and value in _MODELS:
            candidate = value
    if candidate is None:
        top = cfg.get("model")
        if isinstance(top, str) and top in _MODELS:
            candidate = top

    if candidate is not None:
        return candidate, _MODELS[candidate]
    return DEFAULT_MODEL, _MODELS[DEFAULT_MODEL]



def _extract_output(response: Any) -> List[Any]:
    if isinstance(response, list):
        terminal_response = None
        output_items: List[Any] = []
        for event in response:
            etype = _item_get(event, "type")
            if etype == "response.output_item.done":
                item = _item_get(event, "item")
                if item is not None:
                    output_items.append(item)
            if etype in {"response.completed", "response.failed", "response.incomplete"}:
                terminal_response = _item_get(event, "response")
                break
        if terminal_response is not None:
            terminal_output = _extract_output(terminal_response)
            if terminal_output:
                return terminal_output
        return output_items
    if hasattr(response, "__iter__") and not isinstance(response, (dict, str, bytes)):
        terminal_response = None
        output_items: List[Any] = []
        try:
            for event in response:
                etype = _item_get(event, "type")
                if etype == "response.output_item.done":
                    item = _item_get(event, "item")
                    if item is not None:
                        output_items.append(item)
                if etype in {"response.completed", "response.failed", "response.incomplete"}:
                    terminal_response = _item_get(event, "response")
                    break
        finally:
            close_fn = getattr(response, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    pass
        if terminal_response is not None:
            terminal_output = _extract_output(terminal_response)
            if terminal_output:
                return terminal_output
        return output_items
    if isinstance(response, dict):
        output = response.get("output")
    else:
        output = getattr(response, "output", None)
    return output if isinstance(output, list) else []



def _item_get(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)



def _find_image_generation_item(output: List[Any]) -> Any:
    for item in output:
        if _item_get(item, "type") == "image_generation_call":
            return item
    return None


class CodexImageGenProvider(ImageGenProvider):
    @property
    def name(self) -> str:
        return "codex"

    @property
    def display_name(self) -> str:
        return "Codex"

    def is_available(self) -> bool:
        try:
            creds = resolve_codex_runtime_credentials(refresh_if_expiring=False)
            api_key = str(creds.get("api_key", "") or "").strip()
            if not api_key:
                return False
            import openai  # noqa: F401
        except Exception:
            return False
        return True

    def list_models(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": model_id,
                "display": meta["display"],
                "speed": meta["speed"],
                "strengths": meta["strengths"],
                "price": "included with ChatGPT/Codex auth",
            }
            for model_id, meta in _MODELS.items()
        ]

    def default_model(self) -> Optional[str]:
        return DEFAULT_MODEL

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Codex",
            "badge": "chatgpt",
            "tag": "No OpenAI API key required. Uses your Hermes OpenAI Codex / ChatGPT login if available. Please login OpenAI Codex via OAuth using hermes model first before using this.",
            "env_vars": [],
        }

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        prompt = (prompt or "").strip()
        aspect = resolve_aspect_ratio(aspect_ratio)

        if not prompt:
            return error_response(
                error="Prompt is required and must be a non-empty string",
                error_type="invalid_argument",
                provider="codex",
                aspect_ratio=aspect,
            )

        try:
            creds = resolve_codex_runtime_credentials(refresh_if_expiring=True)
        except Exception as exc:
            return error_response(
                error=(
                    "Codex authentication not available. Run `hermes auth` and log in "
                    f"to OpenAI Codex / ChatGPT first. ({exc})"
                ),
                error_type="auth_required",
                provider="codex",
                prompt=prompt,
                aspect_ratio=aspect,
            )

        api_key = str(creds.get("api_key", "") or "").strip()
        base_url = str(creds.get("base_url", "") or "").strip().rstrip("/")
        if not api_key or not base_url:
            return error_response(
                error="Codex credentials are missing api_key or base_url.",
                error_type="auth_required",
                provider="codex",
                prompt=prompt,
                aspect_ratio=aspect,
            )

        try:
            import openai
        except ImportError:
            return error_response(
                error="openai Python package not installed (pip install openai)",
                error_type="missing_dependency",
                provider="codex",
                prompt=prompt,
                aspect_ratio=aspect,
            )

        model_id, meta = _resolve_model()
        payload: Dict[str, Any] = {
            "model": meta["api_model"],
            "instructions": "You are a helpful assistant.",
            "input": [{"role": "user", "content": prompt}],
            "tools": [{
                "type": "image_generation",
                "output_format": "png",
                "quality": meta["quality"],
            }],
            "store": False,
            "stream": True,
        }

        try:
            client = openai.OpenAI(
                api_key=api_key,
                base_url=base_url,
                default_headers=_codex_cloudflare_headers(api_key),
            )
            response = client.responses.create(**payload)
        except Exception as exc:
            logger.debug("Codex image generation failed", exc_info=True)
            return error_response(
                error=f"Codex image generation failed: {exc}",
                error_type="api_error",
                provider="codex",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        item = _find_image_generation_item(_extract_output(response))
        if item is None:
            return error_response(
                error="Codex returned no image generation output",
                error_type="empty_response",
                provider="codex",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        result_b64 = _item_get(item, "result")
        revised_prompt = _item_get(item, "revised_prompt")
        if not isinstance(result_b64, str) or not result_b64.strip():
            return error_response(
                error="Codex image generation output was missing base64 result data",
                error_type="empty_response",
                provider="codex",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        try:
            saved_path = save_b64_image(result_b64, prefix=f"codex_{model_id.replace('.', '_')}")
        except Exception as exc:
            return error_response(
                error=f"Could not save image to cache: {exc}",
                error_type="io_error",
                provider="codex",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        extra: Dict[str, Any] = {"tool_model": meta["tool_model"]}
        if revised_prompt:
            extra["revised_prompt"] = revised_prompt

        return success_response(
            image=str(saved_path),
            model=model_id,
            prompt=prompt,
            aspect_ratio=aspect,
            provider="codex",
            extra=extra,
        )



def register(ctx) -> None:
    ctx.register_image_gen_provider(CodexImageGenProvider())
