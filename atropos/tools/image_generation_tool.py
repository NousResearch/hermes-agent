"""
Image generation tool (external).

This is intentionally minimal for the Phase 4.6 "external tool demo":
- executed via ToolServer (no secrets in sandboxes)
- by default returns a tiny inline PNG data URL (no network required)
- can optionally proxy to an OpenAI-compatible images endpoint (e.g. a local service)
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from typing import Literal, Optional

import httpx

from .base import Tool, ToolResult, ToolSchema


def _tiny_png_data_url() -> str:
    # 1x1 transparent PNG
    png_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMB/6X0kQAAAABJRU5ErkJggg=="
    )
    return f"data:image/png;base64,{base64.b64encode(png_bytes).decode('ascii')}"


@dataclass
class ImageGenerateConfig:
    backend: Literal["inline", "openai"] = "inline"
    base_url: Optional[str] = None
    model: Optional[str] = None
    timeout_s: float = 120.0

    @classmethod
    def from_env(cls) -> "ImageGenerateConfig":
        backend = (os.getenv("IMAGE_GENERATE_BACKEND") or "inline").strip().lower()
        if backend not in {"inline", "openai"}:
            backend = "inline"

        base_url = os.getenv("IMAGE_GENERATE_BASE_URL") or os.getenv("OLLAMA_BASE_URL")
        model = os.getenv("IMAGE_GENERATE_MODEL") or os.getenv("OLLAMA_IMAGE_MODEL")

        timeout_s = float(os.getenv("IMAGE_GENERATE_TIMEOUT_S", "120.0"))
        return cls(
            backend=backend,  # type: ignore[arg-type]
            base_url=base_url,
            model=model,
            timeout_s=timeout_s,
        )


class ImageGenerateTool(Tool):
    def __init__(self, config: Optional[ImageGenerateConfig] = None) -> None:
        self._config = config or ImageGenerateConfig.from_env()

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="image_generate",
            description=(
                "Generate an image from a text prompt. Returns a JSON object containing an image URL "
                "(often a data URL) plus metadata."
            ),
            parameters={
                "prompt": {"type": "string", "description": "The image prompt. Be detailed."},
                "aspect_ratio": {
                    "type": "string",
                    "enum": ["landscape", "square", "portrait"],
                    "description": "Desired aspect ratio.",
                    "default": "landscape",
                },
            },
            required=["prompt"],
            external=True,
        )

    def is_available(self) -> tuple[bool, str | None]:
        # Availability of this external tool is primarily governed by whether a ToolServer
        # is configured/enabled. The tool itself will return a clear error if an upstream
        # image endpoint is required but not configured.
        return True, None

    async def execute(self, prompt: str, aspect_ratio: str = "landscape") -> ToolResult:  # noqa: ARG002
        cfg = self._config

        if cfg.backend == "inline":
            payload = {
                "url": _tiny_png_data_url(),
                "backend": "inline",
                "width": 1,
                "height": 1,
            }
            return ToolResult(success=True, output=json.dumps(payload))

        # OpenAI-compatible images generation endpoint.
        # Expected: POST {base_url}/v1/images/generations with OpenAI-style body.
        base_url = cfg.base_url or ""
        base_url = base_url.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"
        url = f"{base_url}/images/generations"

        body = {
            "prompt": prompt,
            "n": 1,
            "response_format": "url",
        }
        if cfg.model:
            body["model"] = cfg.model

        try:
            async with httpx.AsyncClient(timeout=cfg.timeout_s) as client:
                resp = await client.post(url, json=body)
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:
            return ToolResult(success=False, error=f"image_generate upstream call failed: {e}")

        try:
            image_url = data["data"][0]["url"]
        except Exception:
            return ToolResult(success=False, error=f"Unexpected image response format: {data!r}")

        payload = {"url": image_url, "backend": "openai"}
        return ToolResult(success=True, output=json.dumps(payload))
