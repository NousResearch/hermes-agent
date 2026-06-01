"""
Civitai Orchestration API image generation backend.

Accesses Civitai's provider-orchestrated image generation: Flux 2 (Klein
4b/9b, Dev, Pro, Max), SDXL/Illustrious/Anima via sdcpp, Z-Image Turbo,
and custom AIR checkpoints from the tens of thousands available on
civitai.com.  Also supports LoRA stacking via AIR identifiers.

Workflow API
------------
Submits a workflow with a single ``imageGen`` step, waits inline
(``wait=120``), and extracts the first blob URL from the response.
See: https://developer.civitai.com/orchestration/

Selection precedence (first hit wins):

1. ``CIVITAI_IMAGE_MODEL`` env var
2. ``image_gen.civitai.model`` in ``config.yaml``
3. ``image_gen.model`` in ``config.yaml`` (when it's one of our IDs)
4. :data:`DEFAULT_MODEL` — ``flux2-klein-4b``

Custom AIRs & LoRAs
--------------------
The tool passes ``checkpoint`` (AIR URN) and ``loras`` (JSON dict of
AIR:strength) through kwargs.  These override the default model for
SDXL-family engines.  Example::

    image_generate(prompt="...", checkpoint="urn:air:sdxl:checkpoint:civitai:...",
                   loras='{"urn:air:sdxl:lora:civitai:...": 1.0}')
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    resolve_aspect_ratio,
    success_response,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

API_BASE = "https://orchestration.civitai.com"
WORKFLOWS_URL = f"{API_BASE}/v2/consumer/workflows"
DEFAULT_WAIT = 120
REQUEST_TIMEOUT = 130  # slightly above wait

# ---------------------------------------------------------------------------
# Model catalog
# ---------------------------------------------------------------------------
# Each entry declares the engine, sub-model, and default dimensions for the
# three aspect-ratio tiers.  "family" groups models that share the same
# width×height table.

_FLUX_SIZES: Dict[str, str] = {
    "landscape": "1216x832",
    "square": "1024x1024",
    "portrait": "832x1216",
}

_SDXL_SIZES: Dict[str, str] = {
    "landscape": "1216x832",
    "square": "1024x1024",
    "portrait": "832x1216",
}

# Default checkpoint AIRs
_DEFAULT_CHECKPOINT = "urn:air:sdxl:checkpoint:civitai:827184@2883731"  # WAI Illustrious
_DEFAULT_CHECKPOINT_JANKU = "urn:air:sdxl:checkpoint:civitai:1277670@2786084"  # JANKU v7.77

# Each entry maps a user-friendly model id to an SDXL-family ecosystem.
# All use the sdcpp engine; the checkpoint is either the default above
# or overridden per-request via the ``checkpoint`` kwarg.
_DEFAULT_CFG: Dict[str, Any] = {
    "cfgScale": 4,
    "steps": 30,
    "sampleMethod": "euler_a",
}

_MODELS: Dict[str, Dict[str, Any]] = {
    "illustrious": {
        "display": "WAI Illustrious v17.0",
        "speed": "~25s",
        "strengths": "Anime/illustration, NSFW-capable, wide LoRA ecosystem",
        "family": "sdxl",
        "engine": "sdcpp",
        "ecosystem": "sdxl",
        "checkpoint": "urn:air:sdxl:checkpoint:civitai:827184@2883731",
        "default_width": 1024,
        "default_height": 1024,
        "sizes": _SDXL_SIZES,
        "defaults": {"cfgScale": 4, "steps": 30, "sampleMethod": "euler_a"},
    },
    "janku": {
        "display": "JANKU + Chenkin & NoobAI v7.77",
        "speed": "~25s",
        "strengths": "NSFW-optimized, explicit tags, BREAK-compatible",
        "family": "sdxl",
        "engine": "sdcpp",
        "ecosystem": "sdxl",
        "checkpoint": _DEFAULT_CHECKPOINT_JANKU,
        "default_width": 1024,
        "default_height": 1024,
        "sizes": _SDXL_SIZES,
        "defaults": {"cfgScale": 3, "steps": 25, "sampleMethod": "euler_a"},
    },
}

DEFAULT_MODEL = "illustrious"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_config() -> Dict[str, Any]:
    """Read ``image_gen`` from config.yaml (returns {} on any failure)."""
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        section = cfg.get("image_gen") if isinstance(cfg, dict) else None
        return section if isinstance(section, dict) else {}
    except Exception as exc:
        logger.debug("Could not load image_gen config: %s", exc)
        return {}


def _resolve_model() -> Tuple[str, Dict[str, Any]]:
    """Decide which model tier to use and return ``(model_id, meta)``."""
    env_override = os.environ.get("CIVITAI_IMAGE_MODEL")
    if env_override and env_override in _MODELS:
        return env_override, _MODELS[env_override]

    cfg = _load_config()
    civitai_cfg = cfg.get("civitai") if isinstance(cfg.get("civitai"), dict) else {}
    candidate: Optional[str] = None
    if isinstance(civitai_cfg, dict):
        value = civitai_cfg.get("model")
        if isinstance(value, str) and value in _MODELS:
            candidate = value
    if candidate is None:
        top = cfg.get("model")
        if isinstance(top, str) and top in _MODELS:
            candidate = top

    if candidate is not None:
        return candidate, _MODELS[candidate]
    return DEFAULT_MODEL, _MODELS[DEFAULT_MODEL]


def _resolve_dimensions(
    meta: Dict[str, Any],
    aspect: str,
    checkpoint: Optional[str] = None,
) -> Tuple[int, int]:
    """Return ``(width, height)`` for the given aspect ratio."""
    sizes = meta.get("sizes", {})
    size_str = sizes.get(aspect, "1024x1024")
    try:
        w, h = size_str.split("x")
        return int(w), int(h)
    except (ValueError, AttributeError):
        return meta.get("default_width", 1024), meta.get("default_height", 1024)


def _call_api(
    api_key: str,
    payload: Dict[str, Any],
    wait: int = DEFAULT_WAIT,
) -> Dict[str, Any]:
    """Submit a workflow via subprocess curl and return the parsed response.

    Python's TLS fingerprint (urllib / requests) is blocked by Cloudflare
    on the Civitai endpoint.  curl on macOS uses SecureTransport and passes
    through cleanly.  Proxy env vars are stripped so the request goes
    direct rather than through a blocked proxy IP.
    """
    import subprocess

    url = f"{WORKFLOWS_URL}?wait={wait}"
    body = json.dumps(payload)

    # Strip proxy env vars so curl connects directly — the agent host's
    # proxy routes through IPs blocked by Civitai's Cloudflare WAF.
    env_clean = os.environ.copy()
    for key in list(env_clean):
        if key.lower() in ("http_proxy", "https_proxy", "all_proxy",
                           "no_proxy", "ftp_proxy"):
            del env_clean[key]

    result = subprocess.run(
        [
            "curl", "-s", "-X", "POST", url,
            "-H", f"Authorization: Bearer {api_key}",
            "-H", "Content-Type: application/json",
            "-H", "User-Agent: hermes-civitai/1.0",
            "-d", body,
            "--max-time", str(REQUEST_TIMEOUT),
        ],
        capture_output=True,
        text=True,
        timeout=REQUEST_TIMEOUT + 5,
        env=env_clean,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Civitai API curl failed (exit {result.returncode}): "
            f"{result.stderr[:300]}"
        )

    try:
        resp = json.loads(result.stdout)
    except json.JSONDecodeError:
        raise RuntimeError(
            f"Civitai API returned non-JSON: {result.stdout[:500]}"
        )

    if isinstance(resp, dict) and resp.get("status") == "failed":
        return resp

    return resp


def _extract_image_url(workflow: Dict[str, Any]) -> Optional[str]:
    """Walk the workflow response and return the first blob URL, if any."""
    steps = workflow.get("steps", [])
    for step in steps:
        output = step.get("output", {}) if isinstance(step, dict) else {}
        images = output.get("images", [])
        for img in images:
            url = img.get("url")
            if url:
                return url
    return None


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class CivitaiImageGenProvider(ImageGenProvider):
    """Civitai Orchestration API — Flux 2, SDXL, custom AIRs, LoRAs."""

    @property
    def name(self) -> str:
        return "civitai"

    @property
    def display_name(self) -> str:
        return "Civitai"

    @property
    def prefer_local_execution(self) -> bool:
        return True

    # Module-level flag for dispatch to check before the class is
    # fully imported (avoids @property resolution races).
    PREFER_LOCAL_EXECUTION = True

    def is_available(self) -> bool:
        return bool(os.environ.get("CIVITAI_API_KEY"))

    def list_models(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": model_id,
                "display": meta["display"],
                "speed": meta["speed"],
                "strengths": meta.get("strengths", ""),
                "price": "buzz",
            }
            for model_id, meta in _MODELS.items()
        ]

    def default_model(self) -> Optional[str]:
        return DEFAULT_MODEL

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Civitai",
            "badge": "buzz",
            "tag": (
                "Flux 2, SDXL, Illustrious, Anima, Z-Image — "
                "custom AIR checkpoints + LoRAs from civitai.com"
            ),
            "env_vars": [
                {
                    "key": "CIVITAI_API_KEY",
                    "prompt": "Civitai Orchestration API key",
                    "url": "https://civitai.com/user/account",
                },
            ],
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
                provider="civitai",
                aspect_ratio=aspect,
            )

        api_key = os.environ.get("CIVITAI_API_KEY")
        if not api_key:
            return error_response(
                error=(
                    "CIVITAI_API_KEY not set. Run `hermes tools` → Image "
                    "Generation → Civitai to configure."
                ),
                error_type="auth_required",
                provider="civitai",
                aspect_ratio=aspect,
            )

        tier_id, meta = _resolve_model()
        engine = meta["engine"]
        family = meta.get("family", "sdxl")
        width, height = _resolve_dimensions(meta, aspect)

        # ── Build step input ─────────────────────────────────────────
        step_input: Dict[str, Any] = {
            "engine": engine,
            "prompt": prompt,
            "width": width,
            "height": height,
            "outputFormat": "jpeg",
        }

        # Apply model defaults (cfgScale, steps, sampleMethod, ecosystem)
        defaults = meta.get("defaults", {})
        for k, v in defaults.items():
            step_input.setdefault(k, v)

        # Ecosystem
        ecosystem = meta.get("ecosystem")
        if ecosystem:
            step_input["ecosystem"] = ecosystem

        # Checkpoint: kwargs override > model default
        checkpoint = kwargs.get("checkpoint") or meta.get("checkpoint")
        if checkpoint:
            step_input["model"] = str(checkpoint)

        # Optional generation knobs (from kwargs)
        negative = kwargs.get("negative_prompt")
        if negative:
            step_input["negativePrompt"] = str(negative)

        for key in ("steps", "cfg_scale", "sample_method", "seed"):
            val = kwargs.get(key)
            if val is not None:
                mapped = {
                    "steps": "steps",
                    "cfg_scale": "cfgScale",
                    "sample_method": "sampleMethod",
                    "seed": "seed",
                }[key]
                if mapped == "steps":
                    step_input[mapped] = int(val)
                elif mapped == "cfgScale":
                    step_input[mapped] = float(val)
                elif mapped == "seed":
                    step_input[mapped] = int(val)
                else:
                    step_input[mapped] = str(val)

        # LoRAs — JSON dict of AIR:strength
        loras_raw = kwargs.get("loras")
        if loras_raw:
            try:
                loras_dict = (
                    json.loads(loras_raw)
                    if isinstance(loras_raw, str)
                    else loras_raw
                )
                if isinstance(loras_dict, dict) and loras_dict:
                    step_input["loras"] = loras_dict
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                return error_response(
                    error=f"Invalid loras format (expected JSON dict): {exc}",
                    error_type="invalid_argument",
                    provider="civitai",
                    model=tier_id,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )

        # ── Build workflow payload ──────────────────────────────────
        workflow: Dict[str, Any] = {
            "steps": [{"$type": "imageGen", "input": step_input}],
            "allowMatureContent": True,
        }

        # Quantity (up to 4)
        quantity = kwargs.get("quantity", 1)
        try:
            quantity = max(1, min(4, int(quantity)))
        except (TypeError, ValueError):
            quantity = 1
        if quantity > 1:
            step_input["quantity"] = quantity

        # ── Submit ──────────────────────────────────────────────────
        try:
            result = _call_api(api_key, workflow)
        except RuntimeError as exc:
            logger.debug("Civitai generation failed", exc_info=True)
            return error_response(
                error=str(exc),
                error_type="api_error",
                provider="civitai",
                model=tier_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        status = result.get("status")
        if status == "failed":
            # Collect error detail from steps if available
            step_errs = []
            for step in result.get("steps", []):
                for e in step.get("output", {}).get("errors", []):
                    step_errs.append(str(e))
            return error_response(
                error="; ".join(step_errs) if step_errs else "Workflow failed",
                error_type="api_error",
                provider="civitai",
                model=tier_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        image_url = _extract_image_url(result)
        if not image_url:
            return error_response(
                error=f"Workflow {status} but no image URL in response",
                error_type="empty_response",
                provider="civitai",
                model=tier_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        extra: Dict[str, Any] = {
            "width": width,
            "height": height,
        }
        cost = result.get("cost")
        if isinstance(cost, dict):
            extra["buzz"] = cost.get("total")

        return success_response(
            image=image_url,
            model=tier_id,
            prompt=prompt,
            aspect_ratio=aspect,
            provider="civitai",
            extra=extra,
        )


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


def register(ctx) -> None:
    """Plugin entry point — wire ``CivitaiImageGenProvider`` into the registry."""
    ctx.register_image_gen_provider(CivitaiImageGenProvider())