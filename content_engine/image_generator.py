"""
Multi-provider image generation with automatic fallback.
Tries providers in priority order until one succeeds.
"""
import _no_proxy  # noqa: F401  (strips dead Privoxy from env on import)

import os
import time
import uuid
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import requests

from config import IMAGE_PROVIDERS


def _pollinations_generate(prompt: str, width: int = 1024, height: int = 1024, seed: int = None) -> bytes:
    """Pollinations.ai — free, no API key, poor quality. Last resort fallback."""
    seed = seed or int(time.time())
    # Pollinations takes the prompt in the URL path. Art-directed prompts can be
    # thousands of chars (full style guides with markdown tables), which blows
    # past the server's URL limit and 404s — defeating the free fallback. Collapse
    # whitespace and cap to a concise lead so the fallback actually fires.
    clean = " ".join(str(prompt).split())[:480]
    encoded = quote(clean)
    url = f"https://image.pollinations.ai/prompt/{encoded}?width={width}&height={height}&nologo=true&seed={seed}"
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    return resp.content


def _fal_generate(prompt: str, api_key: str, width: int = 1024, height: int = 1024) -> bytes:
    """Fal.ai FLUX.2 Schnell — $0.003/image, best quality per penny."""
    url = "https://fal.ai/api/flux/schnell"
    headers = {
        "Authorization": f"Key {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "prompt": prompt,
        "image_size": f"{width}x{height}",
        "num_inference_steps": 4,
        "guidance_scale": 3.5,
        "enable_safety_checker": True,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    # Fal returns image URL in response
    image_url = data.get("images", [{}])[0].get("url") or data.get("image_url")
    if not image_url:
        raise RuntimeError(f"No image URL in Fal response: {data}")
    img_resp = requests.get(image_url, timeout=60)
    img_resp.raise_for_status()
    return img_resp.content


def _replicate_generate(prompt: str, api_key: str, width: int = 1024, height: int = 1024) -> bytes:
    """Replicate FLUX.1 Schnell — $0.003/image."""
    # Create prediction
    url = "https://api.replicate.com/v1/models/black-forest-labs/flux-schnell/predictions"
    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "application/json",
        "Prefer": "wait",
    }
    payload = {
        "input": {
            "prompt": prompt,
            "aspect_ratio": f"{width}:{height}",
            "output_format": "png",
            "output_quality": 80,
        }
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    # Replicate returns output URL directly with Prefer: wait
    output = data.get("output")
    if isinstance(output, list) and output:
        image_url = output[0]
    elif isinstance(output, str):
        image_url = output
    else:
        raise RuntimeError(f"No output in Replicate response: {data}")
    img_resp = requests.get(image_url, timeout=60)
    img_resp.raise_for_status()
    return img_resp.content


def _together_generate(prompt: str, api_key: str, width: int = 1024, height: int = 1024) -> bytes:
    """Together.ai FLUX.1 Schnell — free tier available, then paid."""
    url = "https://api.together.xyz/v1/images/generations"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "black-forest-labs/FLUX.1-schnell",
        "prompt": prompt,
        "width": width,
        "height": height,
        "steps": 4,
        "n": 1,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    image_url = data.get("data", [{}])[0].get("url")
    if not image_url:
        raise RuntimeError(f"No image URL in Together response: {data}")
    img_resp = requests.get(image_url, timeout=60)
    img_resp.raise_for_status()
    return img_resp.content


PROVIDER_HANDLERS = {
    "fal_flux_schnell": _fal_generate,
    "replicate_flux_schnell": _replicate_generate,
    "together_flux": _together_generate,
    "pollinations": _pollinations_generate,
}


def generate_image(
    prompt: str,
    output_path: Optional[str] = None,
    width: int = 1024,
    height: int = 1024,
    seed: int = None,
) -> str:
    """
    Generate an image with automatic provider fallback.
    Returns the path to the saved image file.
    Raises RuntimeError if all providers fail.
    """
    # Sort providers by priority
    sorted_providers = sorted(IMAGE_PROVIDERS, key=lambda p: p["priority"])
    
    last_error = None
    image_bytes = None
    used_provider = None
    
    for provider in sorted_providers:
        handler = PROVIDER_HANDLERS.get(provider["name"])
        if not handler:
            continue
            
        # Check API key availability
        api_key = None
        if provider["api_key_env"]:
            api_key = os.getenv(provider["api_key_env"])
            if not api_key:
                continue  # Skip, no key available
        
        try:
            if provider["name"] == "pollinations":
                image_bytes = handler(prompt, width, height, seed)
            else:
                image_bytes = handler(prompt, api_key, width, height)
            used_provider = provider["name"]
            break
        except Exception as e:
            last_error = e
            continue
    
    if image_bytes is None:
        raise RuntimeError(
            f"All image providers failed. Last error: {last_error}"
        )
    
    # Save to disk
    if output_path is None:
        output_dir = Path.home() / "apps" / "content-engine" / "assets" / "generated"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{uuid.uuid4().hex}.png"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    Path(output_path).write_bytes(image_bytes)
    
    return str(output_path), used_provider


def estimate_monthly_cost(images_per_month: int, provider: str = "fal_flux_schnell") -> float:
    """Estimate USD cost for N images with a given provider."""
    for p in IMAGE_PROVIDERS:
        if p["name"] == provider:
            return images_per_month * p["cost_per_image"]
    return 0.0
