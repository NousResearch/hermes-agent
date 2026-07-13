"""Google Vertex AI image generation backend.

Generates high-quality images via Google Vertex AI's gemini-3.1-flash-image or
gemini-3-pro-image models. Authentication is managed dynamically via your
active gcloud CLI credentials.
"""

from __future__ import annotations

import base64
import logging
import os
import subprocess
from typing import Any, Dict, List, Optional

import requests

from agent.image_gen_provider import (
    ImageGenProvider,
    error_response,
    save_b64_image,
    success_response,
)

logger = logging.getLogger(__name__)


class VertexImageGenProvider(ImageGenProvider):
    @property
    def name(self) -> str:
        """Stable short identifier used in config."""
        return "vertex"

    @property
    def display_name(self) -> str:
        """Human-readable label shown in hermes tools."""
        return "Google Vertex AI"

    def is_available(self) -> bool:
        """Check if gcloud command is available on the system."""
        try:
            subprocess.run(
                ["gcloud", "--version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
        except Exception:
            return False

    def list_models(self) -> List[Dict[str, Any]]:
        """List models supported by this backend."""
        return [
            {
                "id": "gemini-3.1-flash-image",
                "display": "Gemini 3.1 Flash Image",
                "speed": "~10s",
                "strengths": "Fast high-quality image generation",
            },
            {
                "id": "gemini-3-pro-image",
                "display": "Gemini 3 Pro Image",
                "speed": "~15s",
                "strengths": "Highest fidelity image generation",
            },
        ]

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = "landscape",
        image_url: Optional[str] = None,
        reference_image_urls: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate an image using Vertex AI's Gemini Image models."""
        try:
            # Mint OAuth token via active gcloud session
            token = (
                subprocess.check_output(["gcloud", "auth", "print-access-token"])
                .decode()
                .strip()
            )
        except Exception as exc:
            return error_response(
                error=f"Failed to retrieve Google Cloud OAuth token via gcloud CLI: {exc}",
                error_type="gcloud_token_error",
            )

        # We know hi@gptindia.pro has access to the winter-environs project
        project_id = "winter-environs-427409-r8"
        location = "global"

        # Resolve model from config
        from hermes_cli.config import load_config

        cfg = load_config()
        img_cfg = cfg.get("image_gen", {}) if isinstance(cfg, dict) else {}
        model_id = img_cfg.get("model") or "gemini-3.1-flash-image"
        if model_id not in [
            "gemini-3.1-flash-image",
            "gemini-3-pro-image",
            "gemini-3.1-flash-lite-image",
            "gemini-3.1-flash-image-preview",
        ]:
            model_id = "gemini-3.1-flash-image"

        url = f"https://aiplatform.googleapis.com/v1beta1/projects/{project_id}/locations/{location}/publishers/google/models/{model_id}:generateContent"

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "X-Goog-User-Project": project_id,
        }

        # Adapt prompt based on requested aspect ratio since generateContent
        # API doesn't accept a native ratio param (that belongs to predict payload)
        full_prompt = prompt
        if aspect_ratio == "landscape":
            full_prompt += " (Landscape aspect ratio, 16:9)"
        elif aspect_ratio == "portrait":
            full_prompt += " (Portrait aspect ratio, 9:16)"
        elif aspect_ratio == "square":
            full_prompt += " (Square aspect ratio, 1:1)"

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": full_prompt}],
                }
            ]
        }

        try:
            response = requests.post(url, headers=headers, json=payload)
            if response.status_code != 200:
                return error_response(
                    error=f"Vertex AI API call failed (HTTP {response.status_code}): {response.text}",
                    error_type="vertex_api_error",
                )

            res_json = response.json()
            candidates = res_json.get("candidates", [])
            if not candidates:
                return error_response(
                    error="No image candidates returned by Vertex AI.",
                    error_type="no_candidates",
                )

            parts = candidates[0].get("content", {}).get("parts", [])
            b64_data = None
            for p in parts:
                if "inlineData" in p:
                    b64_data = p["inlineData"].get("data")
                    break

            if not b64_data:
                return error_response(
                    error="No inline image bytes found in Vertex AI response.",
                    error_type="no_image_data",
                )

            # Save generated image to cache directory ($HERMES_HOME/cache/images/)
            img_path = save_b64_image(b64_data, prefix="vertex")

            return success_response(
                image=str(img_path),
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect_ratio,
                modality="text",
                provider="vertex",
            )
        except Exception as e:
            return error_response(
                error=f"Exception during Vertex generation: {e}",
                error_type="vertex_exception",
            )


def register(ctx) -> None:
    """Plugin entry point — wire ``VertexImageGenProvider`` into the registry."""
    ctx.register_image_gen_provider(VertexImageGenProvider())
