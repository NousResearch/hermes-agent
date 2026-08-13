"""DeepInfra video generation backend.

DeepInfra serves video over the OpenAI-compatible ``/v1/openai/videos``
endpoint (async job: ``create`` → poll → ``download_content``), so all the
SDK plumbing lives in
:class:`agent.video_gen_provider.OpenAICompatibleVideoGenProvider`. This
plugin only declares DeepInfra's identity, credentials, and live model
discovery — no hardcoded model ids, so retired models drop out of hermes the
next time the catalog is fetched without a patch.

Mirrors ``plugins/image_gen/deepinfra`` (which does the same for
``/v1/openai/images/generations``).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from agent.video_gen_provider import OpenAICompatibleVideoGenProvider

logger = logging.getLogger(__name__)


class DeepInfraVideoGenProvider(OpenAICompatibleVideoGenProvider):
    """Text-to-video and image-to-video via DeepInfra's OpenAI-compatible API."""

    name = "deepinfra"
    _env_key = "DEEPINFRA_API_KEY"
    _default_base_url = "https://api.deepinfra.com/v1/openai"

    @property
    def display_name(self) -> str:
        return "DeepInfra"

    def list_models(self) -> List[Dict[str, Any]]:
        """Return ``video-gen``-tagged DeepInfra models from the live catalog.

        Empty list when the catalog is unreachable — the picker then shows no
        options rather than routing to a possibly-retired model.
        """
        try:
            from hermes_cli.models import _fetch_deepinfra_models_by_tag
        except Exception as exc:  # noqa: BLE001 — never break the picker
            logger.debug("Cannot import _fetch_deepinfra_models_by_tag: %s", exc)
            return []
        items = _fetch_deepinfra_models_by_tag("video-gen") or []
        out: List[Dict[str, Any]] = []
        for item in items:
            mid = item.get("id")
            if not mid:
                continue
            meta = item.get("metadata", {}) if isinstance(item, dict) else {}
            out.append({
                "id": mid,
                "display": mid.split("/")[-1],
                "strengths": (meta.get("description") or "")[:80],
            })
        return out

    def capabilities(self) -> Dict[str, Any]:
        return {
            "modalities": ["text", "image"],
            "aspect_ratios": ["16:9", "9:16", "1:1"],
            "resolutions": ["480p", "720p", "1080p"],
            "max_duration": 10,
            "min_duration": 1,
            "supports_audio": False,
            "supports_negative_prompt": True,
            "max_reference_images": 0,
        }

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "DeepInfra",
            "badge": "paid",
            "tag": "Wan, p-video, … — live catalog from api.deepinfra.com; text-to-video & image-to-video",
            "env_vars": [
                {
                    "key": "DEEPINFRA_API_KEY",
                    "prompt": "DeepInfra API key",
                    "url": "https://deepinfra.com/dash/api_keys",
                },
            ],
        }


def register(ctx) -> None:
    """Plugin entry point — wire ``DeepInfraVideoGenProvider`` into the registry."""
    ctx.register_video_gen_provider(DeepInfraVideoGenProvider())
