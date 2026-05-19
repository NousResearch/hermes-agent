"""
image_gen.py - Evelyn Image Generation (FAL.ai / FLUX)
========================================================
Commands:
  /generate <prompt>

Natural language triggers (detected in bot.py):
  "buat gambar...", "generate nft...", "buat pfp..."

Features:
- FAL.ai FLUX model support
- Async image generation
- Returns image URL for Telegram send_photo
- Evelyn personality in responses

Env:
  FAL_KEY=
  IMAGE_MODEL=fal-ai/flux/dev

SAFETY: Never generates harmful/NSFW content.
"""

import os
import httpx

FAL_KEY = os.getenv("FAL_KEY", "")
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "fal-ai/flux/dev")
FAL_BASE_URL = "https://fal.run"


async def generate_image(prompt: str) -> dict:
    """
    Generate image using FAL.ai FLUX.

    Args:
        prompt: Image generation prompt

    Returns:
        dict with 'url' (image URL) or 'error'
    """
    if not FAL_KEY:
        return {"error": "FAL_KEY belum di-set sayang. Tambahin di .env ya."}

    headers = {
        "Authorization": f"Key {FAL_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "prompt": prompt,
        "image_size": "square_hd",
        "num_images": 1,
        "enable_safety_checker": True,
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{FAL_BASE_URL}/{IMAGE_MODEL}",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        # FAL returns images array
        images = data.get("images", [])
        if images:
            return {"url": images[0].get("url", ""), "prompt": prompt}
        else:
            return {"error": "No image generated."}

    except httpx.HTTPStatusError as e:
        try:
            err = e.response.json().get("detail", str(e.response.status_code))
        except Exception:
            err = str(e.response.status_code)
        return {"error": f"FAL API error: {err}"}

    except httpx.TimeoutException:
        return {"error": "Timeout generating image. Coba lagi ya sayang 🥺"}

    except Exception as e:
        return {"error": f"Error: {str(e)[:100]}"}


def is_image_request(text: str) -> bool:
    """Detect if user message is an image generation request."""
    triggers = [
        "buat gambar", "generate gambar", "bikin gambar",
        "generate nft", "buat nft", "bikin nft",
        "buat pfp", "generate pfp", "bikin pfp",
        "buat image", "generate image", "bikin image",
        "gambarkan", "visualisasi",
    ]
    text_lower = text.lower()
    return any(trigger in text_lower for trigger in triggers)


def extract_image_prompt(text: str) -> str:
    """Extract the actual prompt from user message."""
    # Remove common prefixes
    prefixes = [
        "buat gambar", "generate gambar", "bikin gambar",
        "generate nft", "buat nft", "bikin nft",
        "buat pfp", "generate pfp", "bikin pfp",
        "buat image", "generate image", "bikin image",
        "gambarkan", "visualisasi",
    ]
    text_lower = text.lower()
    for prefix in prefixes:
        if text_lower.startswith(prefix):
            return text[len(prefix):].strip()
    return text
