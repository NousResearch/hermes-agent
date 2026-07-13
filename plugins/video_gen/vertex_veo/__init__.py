"""Google Vertex AI Veo video generation backend.

Generates high-quality cinematic videos via Google's Veo 2.0 model on Vertex AI.
Authentication is managed dynamically via your active gcloud CLI credentials.
"""

from __future__ import annotations

import base64
import logging
import mimetypes
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from agent.video_gen_provider import (
    VideoGenProvider,
    error_response,
    save_b64_video,
    success_response,
)

logger = logging.getLogger(__name__)


class VertexVeoVideoGenProvider(VideoGenProvider):
    @property
    def name(self) -> str:
        """Stable short identifier used in config."""
        return "vertex_veo"

    @property
    def display_name(self) -> str:
        """Human-readable label shown in hermes tools."""
        return "Google Vertex AI Veo"

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
                "id": "veo-2.0-generate-001",
                "display": "Google Veo 2.0 (Vertex AI)",
                "speed": "~120-180s",
                "strengths": "High-fidelity cinematic generations with start/end keyframe interpolation support.",
                "modalities": ["text", "image"],
            }
        ]

    def default_model(self) -> Optional[str]:
        return "veo-2.0-generate-001"

    def capabilities(self) -> Dict[str, Any]:
        return {
            "modalities": ["text", "image"],
            "aspect_ratios": ["16:9", "9:16"],
            "resolutions": ["720p", "1080p"],
            "min_duration": 4,
            "max_duration": 8,
            "supports_audio": False,
            "supports_negative_prompt": False,
            "max_reference_images": 1,
        }

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": self.display_name,
            "badge": "gcloud",
            "tag": "Requires active gcloud CLI authentication",
            "env_vars": [],
        }

    def generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        image_url: Optional[str] = None,
        reference_image_urls: Optional[List[str]] = None,
        duration: Optional[int] = None,
        aspect_ratio: str = "16:9",
        resolution: str = "720p",
        negative_prompt: Optional[str] = None,
        audio: Optional[bool] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        # Mint OAuth token via active gcloud session
        try:
            token = (
                subprocess.check_output(["gcloud", "auth", "print-access-token"])
                .decode()
                .strip()
            )
        except Exception as exc:
            return error_response(
                error=f"Failed to retrieve Google Cloud OAuth token via gcloud CLI: {exc}",
                error_type="gcloud_token_error",
                provider=self.name,
                model=model or "veo-2.0-generate-001",
                prompt=prompt,
                aspect_ratio=aspect_ratio,
            )

        # Resolve project and location
        project_id = os.getenv("VERTEX_PROJECT_ID") or "winter-environs-427409-r8"
        location = os.getenv("VERTEX_LOCATION") or "us-central1"
        model_id = model or "veo-2.0-generate-001"

        if model_id not in ["veo-2.0-generate-001"]:
            model_id = "veo-2.0-generate-001"

        # Build endpoints
        url = f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/publishers/google/models/{model_id}:predictLongRunning"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "X-Goog-User-Project": project_id,
        }

        # Sub-helper to load image (either URL or local file path) into Base64 inlineData
        def _load_image_payload(ref: str) -> Dict[str, Any]:
            ref = ref.strip()
            if ref.startswith(("http://", "https://")):
                resp = requests.get(ref, timeout=30)
                resp.raise_for_status()
                content = resp.content
                mime_type = resp.headers.get("Content-Type") or mimetypes.guess_type(ref)[0] or "image/png"
            else:
                path = Path(ref).expanduser()
                if not path.is_file():
                    raise FileNotFoundError(f"Image path not found: {ref}")
                content = path.read_bytes()
                mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
            
            b64_data = base64.b64encode(content).decode("utf-8")
            return {
                "bytesBase64Encoded": b64_data,
                "mimeType": mime_type
            }

        # Build Instances payload
        instances_payload: Dict[str, Any] = {
            "prompt": prompt,
        }

        # 1. Image-to-Video Start Image mapping
        if image_url:
            try:
                instances_payload["image"] = _load_image_payload(image_url)
            except Exception as exc:
                return error_response(
                    error=f"Failed to load start image: {exc}",
                    error_type="image_load_error",
                    provider=self.name,
                    model=model_id,
                    prompt=prompt,
                    aspect_ratio=aspect_ratio,
                )

        # 2. Keyframe Interpolation End Image mapping (first element of reference_image_urls maps to lastFrame)
        if reference_image_urls:
            try:
                instances_payload["lastFrame"] = _load_image_payload(reference_image_urls[0])
            except Exception as exc:
                return error_response(
                    error=f"Failed to load end frame (lastFrame): {exc}",
                    error_type="image_load_error",
                    provider=self.name,
                    model=model_id,
                    prompt=prompt,
                    aspect_ratio=aspect_ratio,
                )

        # Build Parameters payload
        parameters_payload: Dict[str, Any] = {
            "numberOfVideos": 1,
        }
        if aspect_ratio:
            parameters_payload["aspectRatio"] = aspect_ratio
        if resolution:
            parameters_payload["resolution"] = resolution
        if duration:
            parameters_payload["durationSeconds"] = int(duration)
        else:
            # Default to 8 seconds for high-quality standard runs
            parameters_payload["durationSeconds"] = 8

        payload = {
            "instances": [instances_payload],
            "parameters": parameters_payload,
        }

        # Trigger Long Running Operation
        try:
            logger.info("Submitting Vertex Veo request: %s", url)
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            if response.status_code != 200:
                return error_response(
                    error=f"Vertex AI Veo API call failed (HTTP {response.status_code}): {response.text}",
                    error_type="vertex_api_error",
                    provider=self.name,
                    model=model_id,
                    prompt=prompt,
                    aspect_ratio=aspect_ratio,
                )
            
            res_json = response.json()
            operation_name = res_json.get("name")
            if not operation_name:
                return error_response(
                    error=f"No operation name returned in Vertex AI response: {res_json}",
                    error_type="no_operation_name",
                    provider=self.name,
                    model=model_id,
                    prompt=prompt,
                    aspect_ratio=aspect_ratio,
                )
        except Exception as e:
            return error_response(
                error=f"Exception during Vertex Veo request: {e}",
                error_type="vertex_exception",
                provider=self.name,
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect_ratio,
            )

        # Poll operation using fetchPredictOperation
        poll_url = f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/publishers/google/models/{model_id}:fetchPredictOperation"
        poll_payload = {"operationName": operation_name}
        
        max_attempts = 60  # ~10 minutes (10s intervals)
        attempt = 0
        video_b64 = None
        
        logger.info("Polling operation: %s", operation_name)
        while attempt < max_attempts:
            time.sleep(10)
            attempt += 1
            try:
                poll_resp = requests.post(poll_url, headers=headers, json=poll_payload, timeout=30)
                if poll_resp.status_code != 200:
                    logger.warning("Vertex Veo polling failed (HTTP %s): %s", poll_resp.status_code, poll_resp.text)
                    continue
                
                poll_json = poll_resp.json()
                if poll_json.get("done"):
                    # Check for execution error
                    if "error" in poll_json:
                        err_msg = poll_json["error"].get("message", "Unknown error")
                        return error_response(
                            error=f"Veo generation failed: {err_msg}",
                            error_type="veo_generation_error",
                            provider=self.name,
                            model=model_id,
                            prompt=prompt,
                            aspect_ratio=aspect_ratio,
                        )
                    
                    # Extract the completed video
                    response_obj = poll_json.get("response", {})
                    videos = response_obj.get("videos", [])
                    if not videos:
                        return error_response(
                            error="No videos returned in completed operation response.",
                            error_type="no_video_returned",
                            provider=self.name,
                            model=model_id,
                            prompt=prompt,
                            aspect_ratio=aspect_ratio,
                        )
                    
                    video_b64 = videos[0].get("bytesBase64Encoded")
                    break
            except Exception as e:
                logger.warning("Exception during Vertex Veo polling: %s", e)
                continue
        
        if not video_b64:
            return error_response(
                error="Video generation timed out after 10 minutes.",
                error_type="generation_timeout",
                provider=self.name,
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect_ratio,
            )

        # Save and return final cached file
        try:
            video_path = save_b64_video(video_b64, prefix="vertex_veo")
            return success_response(
                video=str(video_path),
                model=model_id,
                prompt=prompt,
                modality="image" if image_url else "text",
                aspect_ratio=aspect_ratio,
                duration=duration or 8,
                provider=self.name,
            )
        except Exception as e:
            return error_response(
                error=f"Failed to save generated video file: {e}",
                error_type="save_file_error",
                provider=self.name,
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect_ratio,
            )


def register(ctx) -> None:
    """Plugin entry point — wire ``VertexVeoVideoGenProvider`` into the registry."""
    ctx.register_video_gen_provider(VertexVeoVideoGenProvider())
