"""DeepInfra video generation backend.

DeepInfra serves video over the OpenAI-compatible ``/v1/openai/videos`` endpoint (``create`` → poll →
``download_content``), so all SDK plumbing lives in :class:`agent.video_gen_provider.OpenAICompatibleVideoGenProvider`.
This plugin only declares identity, credentials, and live model discovery — no hardcoded model ids, so retired
models drop out without a patch. Mirrors ``plugins/image_gen/deepinfra``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from agent.video_gen_provider import OpenAICompatibleVideoGenProvider

logger = logging.getLogger(__name__)


class DeepInfraVideoGenProvider(OpenAICompatibleVideoGenProvider):
    """Text-to-video and image-to-video via DeepInfra's OpenAI-compatible API."""

    name = "deepinfra"
    display_name = "DeepInfra"
    _env_key = "DEEPINFRA_API_KEY"
    _default_base_url = "https://api.deepinfra.com/v1/openai"

    def list_models(self) -> List[Dict[str, Any]]:
        """``video-gen``-tagged models from the live catalog; empty when unreachable (nothing beats a retired model)."""
        try:
            from hermes_cli.models import _fetch_deepinfra_models_by_tag
        except Exception as exc:  # noqa: BLE001 — never break the picker
            logger.debug("Cannot import _fetch_deepinfra_models_by_tag: %s", exc)
            return []
        return [{"id": item["id"], "display": item["id"].split("/")[-1],
                 "strengths": ((item.get("metadata", {}) or {}).get("description") or "")[:80]}
                for item in (_fetch_deepinfra_models_by_tag("video-gen") or []) if item.get("id")]

    def capabilities(self) -> Dict[str, Any]:
        return {"modalities": ["text", "image"], "aspect_ratios": ["16:9", "9:16", "1:1"], "resolutions": ["480p", "720p", "1080p"],
                "max_duration": 10, "min_duration": 1, "supports_audio": False, "supports_negative_prompt": True,
                "supports_seed": True, "supports_upscale": False, "max_reference_images": 0}

    def get_setup_schema(self) -> Dict[str, Any]:
        return {"name": "DeepInfra", "badge": "paid",
                "tag": "Wan, p-video, … — live catalog from api.deepinfra.com; text-to-video & image-to-video",
                "env_vars": [{"key": "DEEPINFRA_API_KEY", "prompt": "DeepInfra API key", "url": "https://deepinfra.com/dash/api_keys"}]}


def register(ctx) -> None:
    """Plugin entry point — wire ``DeepInfraVideoGenProvider`` into the registry."""
    ctx.register_video_gen_provider(DeepInfraVideoGenProvider())
