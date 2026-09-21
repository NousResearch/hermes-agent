"""
AI video generation with provider fallback — Content Engine v2.2.

Unpluggable provider backends modelled after image_generator.py.
Primary: FAL.ai (text-to-video, image-to-video, video-extend).
Future: xAI Grok-Imagine (when API key available).

Operations:
  generate  — text-to-video
  image2vid — image-to-video (uses static card or FAL image as frame 1)
  extend    — extend existing video

Usage:
    from ai_video_generator import generate_video
    path = generate_video(
        brand="plenishd",
        operation="generate",
        prompt="A stressed parent staring at an empty fridge, 4K cinematic",
        duration=8,
        aspect_ratio="9:16",
    )

Cost tracking: every call logs estimated spend to stdout.
"""

import os
import time
import uuid
from pathlib import Path
from typing import Optional, Literal

import requests

# ── Provider registry ───────────────────────────────────────────────

VIDEO_PROVIDERS = [
    {
        "name": "fal_video",
        "base_url": "https://fal.ai/api",
        "cost_per_second": 0.064,  # WAN 2.2 720p — adjust per model
        "api_key_env": "FAL_KEY",
        "priority": 1,
    },
    {
        "name": "xai_video",
        "base_url": "https://api.x.ai",
        "cost_per_second": 0.0,  # TBD — xAI pricing not public yet
        "api_key_env": "XAI_API_KEY",
        "priority": 2,
    },
]

# Model catalog per provider
FAL_VIDEO_MODELS = {
    # General text-to-video
    "wan_22_480p": "fal-ai/wan/v2.2/480p",
    "wan_22_720p": "fal-ai/wan/v2.2/720p",
    "wan_27": "fal-ai/wan/v2.7",
    "seedance_fast": "fal-ai/seedance/v2.0/fast",
    # From Hermes PR #25126
    "veo3.1": "fal-ai/veo3.1",
    "veo3.1_image2vid": "fal-ai/veo3.1/image-to-video",
    "kling_o3": "fal-ai/kling-video/o3/standard/image-to-video",
    "pixverse_v6": "fal-ai/pixverse/v6/image-to-video",
}

XAI_VIDEO_MODELS = {
    "grok_imagine": "grok-imagine-video",
}

# Default model per operation
DEFAULT_MODELS = {
    "generate": "seedance_fast",   # cheapest for text-to-video
    "image2vid": "kling_o3",       # best quality for image-to-video
    "extend": "wan_22_720p",       # reliable for extending
}

# Cost per model (USD per second, approximate)
MODEL_COST = {
    "seedance_fast": 0.022,
    "wan_22_480p": 0.032,
    "wan_22_720p": 0.064,
    "wan_27": 0.080,
    "veo3.1": 0.050,
    "veo3.1_image2vid": 0.050,
    "kling_o3": 0.060,
    "pixverse_v6": 0.040,
}


def _fal_headers(api_key: str) -> dict:
    return {"Authorization": f"Key {api_key}", "Content-Type": "application/json"}


def _poll_fal_result(model_id: str, request_id: str, api_key: str, timeout: int = 300) -> Optional[dict]:
    """Poll FAL.ai queue for async video result."""
    poll_url = f"https://queue.fal.run/{model_id}/requests/{request_id}"
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(poll_url, headers=_fal_headers(api_key), timeout=15)
            if r.status_code == 200:
                data = r.json()
                status = data.get("status", "")
                if status == "COMPLETED":
                    return data
                elif status in ("FAILED", "CANCELLED"):
                    print(f"FAL video queue failed: {status}")
                    return None
            time.sleep(5)
        except Exception as e:
            print(f"FAL video poll error: {e}")
            return None
    print("FAL video: polling timeout")
    return None


def _fal_generate(
    api_key: str,
    model: str,
    operation: Literal["generate", "image2vid", "extend"],
    prompt: str,
    image_path: Optional[str] = None,
    video_path: Optional[str] = None,
    duration: int = 5,
    aspect_ratio: str = "16:9",
    output_dir: str = "/home/kensei/repos/KenseiAgent/content_engine/output",
    filename: Optional[str] = None,
) -> Optional[str]:
    """Generate video via FAL.ai. Returns saved MP4 path or None."""

    model_id = FAL_VIDEO_MODELS.get(model, FAL_VIDEO_MODELS["seedance_fast"])

    # Build payload based on operation
    payload: dict = {"prompt": prompt}

    if operation == "generate":
        payload.update({
            "duration": duration,
            "aspect_ratio": aspect_ratio,
        })
    elif operation == "image2vid":
        if image_path and Path(image_path).exists():
            # FAL image-to-video expects image_url or base64
            # For now, upload and get URL, or use existing public URL
            # Simpler: read file and upload via FAL file upload if needed
            # For MVP, we'll pass the prompt + image_url if we can serve it
            # Actually, FAL supports passing image as base64 data URI
            import base64
            img_bytes = Path(image_path).read_bytes()
            b64 = base64.b64encode(img_bytes).decode()
            mime = "image/png" if image_path.endswith(".png") else "image/jpeg"
            payload["image_url"] = f"data:{mime};base64,{b64}"
        payload.update({
            "duration": duration,
            "aspect_ratio": aspect_ratio,
        })
    elif operation == "extend":
        if video_path and Path(video_path).exists():
            import base64
            vid_bytes = Path(video_path).read_bytes()
            b64 = base64.b64encode(vid_bytes).decode()
            payload["video_url"] = f"data:video/mp4;base64,{b64}"
        payload.update({
            "duration": duration,
        })

    # FAL video models are async-only via queue
    queue_url = f"https://queue.fal.run/{model_id}"

    try:
        qr = requests.post(queue_url, json=payload, headers=_fal_headers(api_key), timeout=30)
        if qr.status_code != 200:
            print(f"FAL video queue submit failed: {qr.status_code} {qr.text[:200]}")
            return None

        qdata = qr.json()
        request_id = qdata.get("request_id")
        if not request_id:
            print("FAL video: no request_id in queue response")
            return None

        est_cost = duration * MODEL_COST.get(model, 0.05)
        print(f"FAL video queued: model={model} request_id={request_id} est_cost=${est_cost:.3f}")

        result = _poll_fal_result(model_id, request_id, api_key, timeout=300)
        if not result:
            return None

        # Extract video URL
        video_url = None
        for key in ["video", "output"]:
            if key in result:
                val = result[key]
                if isinstance(val, dict):
                    video_url = val.get("url")
                elif isinstance(val, str):
                    video_url = val
                if video_url:
                    break

        if not video_url:
            print(f"FAL video: no URL. Result keys: {list(result.keys())[:10]}")
            return None

        # Download
        vid_r = requests.get(video_url, timeout=120)
        if vid_r.status_code != 200:
            print(f"FAL video download failed: {vid_r.status_code}")
            return None

        out_path = Path(output_dir) / "ai_videos"
        out_path.mkdir(parents=True, exist_ok=True)
        fname = filename or f"{model}_{str(uuid.uuid4())[:8]}.mp4"
        fpath = out_path / fname
        fpath.write_bytes(vid_r.content)
        actual_cost = duration * MODEL_COST.get(model, 0.05)
        print(f"FAL video saved: {fpath} ({len(vid_r.content)} bytes, ~${actual_cost:.3f})")
        return str(fpath)

    except Exception as e:
        print(f"FAL video generation error: {e}")
        return None


def _xai_generate(
    api_key: str,
    model: str,
    operation: Literal["generate", "image2vid", "extend"],
    prompt: str,
    image_path: Optional[str] = None,
    video_path: Optional[str] = None,
    duration: int = 5,
    aspect_ratio: str = "16:9",
    output_dir: str = "/home/kensei/repos/KenseiAgent/content_engine/output",
    filename: Optional[str] = None,
) -> Optional[str]:
    """xAI Grok-Imagine video generation. Stub — implement when API available."""
    print("xAI video generation not yet implemented — no API endpoint documented.")
    print("Set XAI_API_KEY and this provider will activate automatically.")
    return None


# ── Public API ──────────────────────────────────────────────────────

def generate_video(
    brand: str,
    operation: Literal["generate", "image2vid", "extend"] = "generate",
    prompt: str = "",
    image_path: Optional[str] = None,
    video_path: Optional[str] = None,
    model: Optional[str] = None,
    duration: int = 5,
    aspect_ratio: str = "9:16",
    output_dir: str = "/home/kensei/repos/KenseiAgent/content_engine/output",
    filename: Optional[str] = None,
) -> Optional[str]:
    """
    Generate AI video with automatic provider fallback.

    Args:
        brand: Brand key (for output path routing)
        operation: generate | image2vid | extend
        prompt: Text prompt describing the scene
        image_path: For image2vid — path to source image
        video_path: For extend — path to source video
        model: Override model choice (e.g. "kling_o3", "seedance_fast")
        duration: Seconds (default 5, provider may clamp)
        aspect_ratio: "9:16", "16:9", "1:1", etc.
        output_dir: Base output directory
        filename: Override output filename

    Returns:
        Absolute path to saved MP4, or None on failure.
    """
    # Pick model
    chosen_model = model or DEFAULT_MODELS.get(operation, "seedance_fast")

    # Brand-specific subdir
    brand_dir = Path(output_dir) / "ai_videos" / brand
    brand_dir.mkdir(parents=True, exist_ok=True)

    # Try providers in priority order
    for provider in sorted(VIDEO_PROVIDERS, key=lambda p: p["priority"]):
        api_key_env = provider["api_key_env"]
        api_key = os.getenv(api_key_env, "")
        if not api_key:
            continue

        provider_name = provider["name"]
        print(f"Trying provider: {provider_name} (model={chosen_model})")

        if provider_name == "fal_video":
            result = _fal_generate(
                api_key=api_key,
                model=chosen_model,
                operation=operation,
                prompt=prompt,
                image_path=image_path,
                video_path=video_path,
                duration=duration,
                aspect_ratio=aspect_ratio,
                output_dir=str(brand_dir),
                filename=filename,
            )
            if result:
                return result
        elif provider_name == "xai_video":
            result = _xai_generate(
                api_key=api_key,
                model=chosen_model,
                operation=operation,
                prompt=prompt,
                image_path=image_path,
                video_path=video_path,
                duration=duration,
                aspect_ratio=aspect_ratio,
                output_dir=str(brand_dir),
                filename=filename,
            )
            if result:
                return result

    print("All AI video providers failed. No video generated.")
    return None


def list_available_models() -> dict:
    """Return models grouped by provider, filtering by available API keys."""
    available = {}
    for provider in VIDEO_PROVIDERS:
        key = os.getenv(provider["api_key_env"], "")
        if key:
            if provider["name"] == "fal_video":
                available["fal"] = {
                    "models": list(FAL_VIDEO_MODELS.keys()),
                    "default_model": DEFAULT_MODELS["generate"],
                }
            elif provider["name"] == "xai_video":
                available["xai"] = {
                    "models": list(XAI_VIDEO_MODELS.keys()),
                    "default_model": "grok_imagine",
                }
    return available


def estimate_cost(model: str, duration: int) -> float:
    """Estimate USD cost for a video generation."""
    return duration * MODEL_COST.get(model, 0.05)


# ── Convenience wrappers for content engine ─────────────────────────

def generate_for_draft(
    draft: dict,
    operation: Literal["generate", "image2vid", "extend"] = "generate",
    model: Optional[str] = None,
    duration: int = 5,
) -> Optional[str]:
    """
    Generate AI video for an approved draft dict.
    Uses draft["visual_description"] as prompt.
    """
    brand = draft.get("brand", "matchdaymaestro")
    visual_desc = draft.get("visual_description", "")
    prompt = visual_desc or draft.get("body_text", "")[:200]

    # Determine aspect ratio from platform
    platform = draft.get("platform", "twitter")
    aspect_map = {
        "twitter": "1:1",
        "instagram": "4:5",
        "tiktok": "9:16",
        "linkedin": "1.91:1",
    }
    aspect = aspect_map.get(platform, "9:16")

    return generate_video(
        brand=brand,
        operation=operation,
        prompt=prompt,
        model=model,
        duration=duration,
        aspect_ratio=aspect,
        filename=f"{draft['id']}.mp4",
    )


if __name__ == "__main__":
    # Quick self-test
    print("Available models:", list_available_models())
    print("Cost estimate (seedance_fast, 5s):", estimate_cost("seedance_fast", 5))
