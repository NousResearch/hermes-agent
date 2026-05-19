"""
image_gen.py - Evelyn Image Generation (FAL.ai / FLUX)
========================================================
Commands:
  /generate <prompt>

Natural language triggers:
  "buat gambar...", "generate nft...", "buat pfp..."
  "pap", "selfie dong", "kirim foto kamu", "aku mau liat kamu"
  "pap habis mandi", "selfie habis mandi"

Features:
- FAL.ai FLUX realism model support
- Consistent Evelyn visual identity (reference prompt)
- Selfie generation with variations
- Async image generation
- Returns image URL for Telegram send_photo

Env:
  FAL_KEY=
  IMAGE_MODEL=fal-ai/flux/dev

SAFETY:
- Never generates explicit NSFW/pornographic content
- enable_safety_checker=True always
- Negative prompts enforce safe content
"""

import os
import random

import httpx

FAL_KEY = os.getenv("FAL_KEY", "")
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "fal-ai/flux/dev")
FAL_BASE_URL = "https://fal.run"

# === Evelyn Official Visual Reference ===

EVELYN_BASE_PROMPT = (
    "Beautiful realistic young woman, Indonesian/Korean mixed features, "
    "soft natural makeup, long black slightly wavy hair, soft glowing skin, "
    "cute gamer girl aesthetic, natural selfie photography, realistic eyes, "
    "realistic skin texture, soft smile, ultra photorealistic, "
    "Instagram/TikTok selfie aesthetic, cinematic soft lighting, "
    "realistic smartphone camera look"
)

EVELYN_NEGATIVE = (
    "anime, cartoon, illustration, painting, low quality, distorted face, "
    "unrealistic skin, nsfw, nudity, explicit, pornographic"
)

# Selfie variation themes
SELFIE_VARIATIONS = [
    "cozy RGB purple room lighting, black oversized hoodie, gaming setup background, subtle cyberpunk vibe",
    "mirror selfie, casual outfit, black hoodie, soft room lighting, relaxed pose",
    "cafe selfie, warm lighting, cute casual outfit, iced coffee nearby",
    "sleepy selfie, messy hair, oversized t-shirt, morning light, cozy bed background",
    "gaming setup, RGB keyboard glow, headphones around neck, purple ambient light",
    "cyberpunk city vibe, neon lights reflection, night time, urban backdrop",
    "casual candid selfie, natural daylight, park or balcony background",
    "black hoodie, peace sign, cute expression, RGB room lighting",
]

# Post-shower/bathroom selfie variations (SFW)
SHOWER_VARIATIONS = [
    "post-shower selfie, wet hair, oversized hoodie, bathroom mirror, soft warm lighting, cozy",
    "wet hair selfie, towel on shoulders, bathroom mirror, steam, soft lighting, natural look",
    "fresh shower look, damp hair, collarbone visible, oversized t-shirt, bathroom mirror selfie",
    "after shower, wet wavy hair, cozy towel aesthetic, mirror selfie, soft bathroom lighting",
]


async def generate_image(prompt: str, negative_prompt: str = None) -> dict:
    """
    Generate image using FAL.ai FLUX.

    Args:
        prompt: Image generation prompt
        negative_prompt: Things to avoid in generation

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
        "image_size": "portrait_4_3",
        "num_images": 1,
        "enable_safety_checker": True,
    }

    if negative_prompt:
        payload["negative_prompt"] = negative_prompt

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{FAL_BASE_URL}/{IMAGE_MODEL}",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

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


async def generate_evelyn_selfie(variation: str = None) -> dict:
    """
    Generate Evelyn selfie with consistent visual identity.

    Args:
        variation: Optional specific variation context

    Returns:
        dict with 'url' or 'error'
    """
    if not variation:
        variation = random.choice(SELFIE_VARIATIONS)

    prompt = f"{EVELYN_BASE_PROMPT}, {variation}"
    return await generate_image(prompt, negative_prompt=EVELYN_NEGATIVE)


async def generate_evelyn_shower_selfie() -> dict:
    """Generate SFW post-shower Evelyn selfie."""
    variation = random.choice(SHOWER_VARIATIONS)
    prompt = f"{EVELYN_BASE_PROMPT}, {variation}"
    return await generate_image(prompt, negative_prompt=EVELYN_NEGATIVE)


# === Detection Helpers ===

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


def is_selfie_request(text: str) -> bool:
    """Detect if user wants Evelyn selfie/pap."""
    triggers = [
        "pap", "selfie dong", "selfie", "kirim foto kamu",
        "aku mau liat kamu", "liat muka kamu", "foto dong",
        "kirim foto", "mau liat kamu", "pap dong",
    ]
    text_lower = text.lower()
    return any(trigger in text_lower for trigger in triggers)


def is_shower_selfie_request(text: str) -> bool:
    """Detect post-shower selfie request (SFW)."""
    triggers = [
        "pap habis mandi", "selfie habis mandi",
        "wet hair selfie", "bathroom selfie",
        "habis mandi", "abis mandi",
    ]
    text_lower = text.lower()
    return any(trigger in text_lower for trigger in triggers)


def extract_image_prompt(text: str) -> str:
    """Extract the actual prompt from user message."""
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
