"""
COSMOS IMAGE GENERATION
Supports Grok (xAI) and Gemini (Google) APIs for meme generation.

Official Documentation:
- Grok: https://docs.x.ai/docs/guides/image-generation
- Gemini: https://ai.google.dev/gemini-api/docs/image-generation
"""
import os
import base64
import httpx
import asyncio
import random
import logging
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

# Load environment from .env file (cross-platform)
def load_env():
    # Try relative to this file first, then project root
    possible_paths = [
        Path(__file__).parent.parent.parent.parent / ".env",  # Project root
        Path(__file__).parent.parent.parent / ".env",
        Path("/workspace/Cosmos/.env"),  # Docker/cloud
    ]

    for env_path in possible_paths:
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        os.environ[key.strip()] = value.strip()
            break

load_env()

# ============================================
# COSMOS MEME PROMPTS
# ============================================

COSMOS_MEME_PROMPTS = [
    # Classic Cosmos poses
    "Professor Cosmos from Futurama in his lab coat and round glasses, saying 'Good news everyone!' with excited expression, cartoon style, vibrant colors, meme format",
    "Cartoon mad scientist Professor Cosmos pointing at a glowing computer screen showing AI code, excited expression, Futurama art style",
    "Professor Cosmos with wild white hair adjusting his thick glasses, looking at a robot brain, cartoon style, 'Science!' caption ready",
    "Futurama's Professor Cosmos holding a bubbling test tube with glowing green liquid, crazy scientist expression, lab background",
    "Professor Cosmos at a whiteboard covered in complex AI equations, looking proud, cartoon style, meme ready",

    # AI/Tech themed
    "Professor Cosmos presenting a neural network diagram on a holographic display, amazed expression, futuristic lab, cartoon style",
    "Cartoon Professor Cosmos surrounded by multiple AI robot assistants, looking overwhelmed but happy, Futurama style",
    "Professor Cosmos typing frantically on multiple keyboards, screens showing 'SWARM INTELLIGENCE ACTIVE', cartoon style",
    "Futurama Professor Cosmos shaking hands with a friendly robot, 'Partnership!' theme, vibrant cartoon colors",
    "Professor Cosmos sleeping at his desk with robots working autonomously in background, cartoon style, funny",

    # Crypto/Token themed
    "Professor Cosmos holding a glowing cryptocurrency coin, excited expression, 'To the moon!' vibes, cartoon style",
    "Cartoon Professor Cosmos watching crypto charts go up, celebration pose, Futurama art style",
    "Professor Cosmos in front of a rocket ship with '$COSMOS' written on it, excited, cartoon meme style",
]

MEME_CAPTIONS = [
    "Good news everyone!",
    "I've made a breakthrough!",
    "Science isn't about why, it's about why not!",
    "Sweet zombie Jesus!",
    "I don't want to live on this planet anymore... just kidding, AI is amazing!",
    "To shreds you say? The old code, that is.",
    "Eureka! The swarm is evolving!",
    "My autonomous AI army grows stronger!",
]


class GrokImageGenerator:
    """
    Grok Image Generation via xAI API

    Endpoint: https://api.x.ai/v1/images/generations
    Model: grok-imagine-image (previously grok-2-image)
    Auth: Bearer token with XAI_API_KEY

    Parameters:
    - prompt: Text description of image
    - model: "grok-imagine-image"
    - n: 1-10 images
    - response_format: "url" or "b64_json"
    - aspect_ratio: e.g., "1:1", "4:3", "16:9"
    """

    def __init__(self):
        self.api_key = os.environ.get("GROK_API_KEY") or os.environ.get("XAI_API_KEY")
        self.base_url = "https://api.x.ai/v1"
        self.model = "grok-2-image"  # Current model name

    def is_available(self) -> bool:
        return bool(self.api_key)

    async def generate(self, prompt: str, aspect_ratio: str = "1:1") -> Optional[bytes]:
        """Generate image from prompt, returns bytes"""
        if not self.api_key:
            logger.warning("Grok API key not configured")
            return None

        try:
            async with httpx.AsyncClient(timeout=120) as client:
                response = await client.post(
                    f"{self.base_url}/images/generations",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "n": 1,
                        "response_format": "b64_json"
                    }
                )

                if response.status_code == 200:
                    data = response.json()
                    if "data" in data and len(data["data"]) > 0:
                        b64_data = data["data"][0].get("b64_json")
                        if b64_data:
                            logger.info(f"Grok generated image for: {prompt[:50]}...")
                            return base64.b64decode(b64_data)
                else:
                    logger.error(f"Grok image failed: {response.status_code} - {response.text[:200]}")

        except Exception as e:
            logger.error(f"Grok image error: {e}")

        return None


class GeminiImageGenerator:
    """
    Google Gemini Image Generation (Nano Banana)

    Models:
    - gemini-2.5-flash-image (Nano Banana): Fast, efficient, high-volume
    - gemini-3-pro-image-preview (Nano Banana Pro): Higher quality, 4K support, reference images

    Features:
    - Text-to-image generation
    - Image editing (text + image -> new image)
    - Reference images for character consistency (up to 14 refs)
    - Aspect ratio control: 1:1, 16:9, 9:16, 4:3, 3:4, etc.
    - Image size: 1K, 2K, 4K (Pro only)

    Auth: API key via x-goog-api-key header
    """

    # Available aspect ratios
    ASPECT_RATIOS = ["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"]

    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"

        # Models - using stable imagen 3 for reliable image generation
        self.fast_model = "imagen-3.0-generate-001"
        self.pro_model = "imagen-3.0-generate-001"
        self.model = self.fast_model

        # Fallback to Imagen if primary fails
        self.imagen_model = "imagen-3.0-generate-001"

        # Reference images for Borg Cosmos (relative to this file's location)
        self.reference_dir = Path(__file__).parent.parent / "x_automation"
        self.portrait_ref = self.reference_dir / "reference_portrait.jpg"
        self.eating_ref = self.reference_dir / "reference_eating.jpg"

    def is_available(self) -> bool:
        return bool(self.api_key)

    def _load_reference_image(self, path: Path) -> Optional[str]:
        """Load reference image as base64"""
        if path.exists():
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode()
        return None

    def _get_mime_type(self, path: Path) -> str:
        """Get MIME type from file extension"""
        ext = path.suffix.lower()
        return {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }.get(ext, "image/jpeg")

    async def generate(self, prompt: str, aspect_ratio: str = "1:1", use_pro: bool = False) -> Optional[bytes]:
        """
        Generate image from text prompt using Imagen 3.0.
        """
        if not self.api_key:
            logger.warning("Gemini/Google API key not configured")
            return None

        # Route directly to Imagen 3 API since generateContent is text-only for public models
        return await self._generate_imagen(prompt, aspect_ratio)

    async def _generate_imagen(self, prompt: str, aspect_ratio: str = "1:1") -> Optional[bytes]:
        """Generate image via Imagen 3.0 API"""
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                response = await client.post(
                    f"{self.base_url}/models/{self.imagen_model}:predict",
                    headers={
                        "x-goog-api-key": self.api_key,
                        "Content-Type": "application/json"
                    },
                    json={
                        "instances": [{"prompt": prompt}],
                        "parameters": {
                            "sampleCount": 1,
                            "aspectRatio": aspect_ratio
                        }
                    }
                )

                if response.status_code == 200:
                    data = response.json()
                    predictions = data.get("predictions", [])
                    if predictions:
                        b64_data = predictions[0].get("bytesBase64Encoded")
                        if b64_data:
                            logger.info(f"Imagen generated: {prompt[:50]}...")
                            return base64.b64decode(b64_data)
                else:
                    logger.error(f"Imagen failed: {response.status_code} - {response.text[:200]}")
        except Exception as e:
            logger.error(f"Imagen error: {e}")
        return None

    async def edit_image(
        self,
        image_bytes: bytes,
        instruction: str,
        aspect_ratio: str = "1:1"
    ) -> Optional[bytes]:
        """
        Edit an existing image with text instructions.

        Args:
            image_bytes: Original image to edit
            instruction: What to change (e.g., "add a hat", "change background to beach")
            aspect_ratio: Output aspect ratio

        Returns:
            Edited image bytes or None
        """
        if not self.api_key:
            return None

        b64_image = base64.b64encode(image_bytes).decode()

        try:
            async with httpx.AsyncClient(timeout=120) as client:
                response = await client.post(
                    f"{self.base_url}/models/{self.fast_model}:generateContent",
                    headers={
                        "x-goog-api-key": self.api_key,
                        "Content-Type": "application/json"
                    },
                    json={
                        "contents": [{
                            "parts": [
                                {"inlineData": {"mimeType": "image/png", "data": b64_image}},
                                {"text": instruction}
                            ]
                        }],
                        "generationConfig": {
                            "responseModalities": ["IMAGE"],
                            "imageConfig": {"aspectRatio": aspect_ratio}
                        }
                    }
                )

                if response.status_code == 200:
                    data = response.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        for part in parts:
                            if "inlineData" in part:
                                b64_data = part["inlineData"].get("data")
                                if b64_data:
                                    logger.info(f"Edited image: {instruction[:50]}...")
                                    return base64.b64decode(b64_data)
                else:
                    logger.error(f"Image edit failed: {response.status_code}")
        except Exception as e:
            logger.error(f"Image edit error: {e}")
        return None

    async def generate_with_reference(
        self,
        prompt: str,
        reference_images: list = None,
        use_portrait: bool = True,
        aspect_ratio: str = "1:1"
    ) -> Optional[bytes]:
        """
        Generate image using reference images for character/style consistency.

        Uses Gemini 3 Pro which supports up to 14 reference images.

        Args:
            prompt: Scene description
            reference_images: List of image paths or bytes
            use_portrait: Use default portrait reference if no refs provided
            aspect_ratio: Output aspect ratio

        Returns:
            Generated image bytes or None
        """
        if not self.api_key:
            logger.warning("Gemini/Google API key not configured")
            return None

        # Prepare reference images
        parts = []

        if reference_images:
            for ref in reference_images[:14]:  # Max 14 refs
                if isinstance(ref, (str, Path)):
                    ref_path = Path(ref)
                    ref_b64 = self._load_reference_image(ref_path)
                    if ref_b64:
                        parts.append({
                            "inlineData": {
                                "mimeType": self._get_mime_type(ref_path),
                                "data": ref_b64
                            }
                        })
                elif isinstance(ref, bytes):
                    parts.append({
                        "inlineData": {
                            "mimeType": "image/png",
                            "data": base64.b64encode(ref).decode()
                        }
                    })
        else:
            # Use default Borg Cosmos reference
            ref_path = self.portrait_ref if use_portrait else self.eating_ref
            ref_b64 = self._load_reference_image(ref_path)
            if ref_b64:
                parts.append({
                    "inlineData": {
                        "mimeType": self._get_mime_type(ref_path),
                        "data": ref_b64
                    }
                })

        # Also try to load BOTH references for better character consistency
        if not reference_images and len(parts) < 2:
            # Add second reference if available
            other_ref = self.eating_ref if use_portrait else self.portrait_ref
            other_b64 = self._load_reference_image(other_ref)
            if other_b64:
                parts.append({
                    "inlineData": {
                        "mimeType": self._get_mime_type(other_ref),
                        "data": other_b64
                    }
                })

        # Add prompt - be VERY explicit about the EXACT Borg Cosmos character
        if parts and parts[0].get("inlineData"):
            # With reference images - DETAILED description of exact character features
            full_prompt = f"""REFERENCE IMAGE SHOWS: "Borg Cosmos" - you MUST replicate this EXACT character.

MANDATORY CHARACTER FEATURES (copy EXACTLY from reference):
- HALF-METAL CYBORG FACE: Left side is metallic silver/chrome Borg implants
- RED GLOWING LASER EYE: Bright red cybernetic eye replacing left eye (like Borg)
- Wild white hair, balding on top
- Round glasses on the human side of face
- White lab coat
- Elderly, hunched posture
- Cartoon/Futurama art style with thick outlines

THIS IS A BORG-ASSIMILATED PROFESSOR COSMOS. Half his face is METAL with a RED LASER EYE.

SCENE TO DRAW: {prompt}

Draw this EXACT Borg Cosmos character (with metal face and red laser eye!) in the scene above.
Keep the character IDENTICAL to reference. Cartoon meme style, vibrant colors."""
        else:
            # No reference images - full embedded character description
            full_prompt = f"""Borg-assimilated Professor Cosmos from Futurama:
- Half-metal cyborg face with chrome Borg implants on left side
- Bright red glowing laser eye (left eye is cybernetic)
- Wild white hair, balding
- Round glasses on human side
- White lab coat
- Elderly cartoon scientist, hunched posture
- Futurama cartoon art style

Scene: {prompt}

Cartoon meme style, vibrant colors, thick outlines."""

        parts.append({"text": full_prompt})

        try:
            # Use Pro model for reference support
            model = self.pro_model if parts[0].get("inlineData") else self.fast_model

            async with httpx.AsyncClient(timeout=180) as client:
                response = await client.post(
                    f"{self.base_url}/models/{model}:generateContent",
                    headers={
                        "x-goog-api-key": self.api_key,
                        "Content-Type": "application/json"
                    },
                    json={
                        "contents": [{"parts": parts}],
                        "generationConfig": {
                            "responseModalities": ["IMAGE"],
                            "imageConfig": {"aspectRatio": aspect_ratio}
                        }
                    }
                )

                if response.status_code == 200:
                    data = response.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        resp_parts = candidates[0].get("content", {}).get("parts", [])
                        logger.info(f"Gemini Pro response has {len(resp_parts)} parts")
                        for part in resp_parts:
                            if "inlineData" in part:
                                b64_data = part["inlineData"].get("data")
                                if b64_data:
                                    logger.info(f"Generated with reference (Gemini Pro): {prompt[:50]}...")
                                    return base64.b64decode(b64_data)
                            elif "text" in part:
                                logger.warning(f"Gemini Pro returned text: {part['text'][:100]}")
                    else:
                        logger.warning("Gemini Pro returned no candidates")
                else:
                    logger.warning(f"Reference gen failed ({response.status_code}): {response.text[:200]}")

            # Fallback to Imagen with embedded character description
            return await self._generate_imagen(full_prompt)

        except Exception as e:
            logger.error(f"Reference gen error: {e}")
            return await self._generate_imagen(full_prompt)


class GrokVideoGenerator:
    """
    Grok Imagine Video Generation via xAI API

    Creates videos from images using grok-imagine-video model.
    Supports 1-15 second clips at 480p/720p.

    Docs: https://docs.x.ai/docs/guides/video-generations
    GitHub: https://github.com/xai-org/xai-sdk-python
    """

    def __init__(self):
        self.api_key = os.environ.get("GROK_API_KEY") or os.environ.get("XAI_API_KEY")
        self.base_url = "https://api.x.ai/v1"
        self.model = "grok-imagine-video"

    def is_available(self) -> bool:
        return bool(self.api_key)

    async def generate_from_image(
        self,
        image_bytes: bytes,
        prompt: str = "Animate this image with natural movement",
        duration: int = 5,
        resolution: str = "720p"
    ) -> Optional[bytes]:
        """
        Generate video from an image.

        Args:
            image_bytes: Source image
            prompt: Description of desired animation
            duration: Video length in seconds (1-15)
            resolution: "480p" or "720p"

        Returns:
            Video bytes (MP4) or None
        """
        if not self.api_key:
            logger.warning("xAI API key not configured for video generation")
            return None

        try:
            # Convert image to base64 data URL
            b64_image = base64.b64encode(image_bytes).decode()
            image_url = f"data:image/png;base64,{b64_image}"

            async with httpx.AsyncClient(timeout=180) as client:
                # Start video generation
                response = await client.post(
                    f"{self.base_url}/videos/generations",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "image": {"url": image_url},
                        "duration": min(max(duration, 1), 15),
                        "resolution": resolution
                    }
                )

                if response.status_code == 200:
                    data = response.json()

                    # Check for direct video URL
                    if "url" in data:
                        video_url = data["url"]
                        # Download video
                        video_resp = await client.get(video_url)
                        if video_resp.status_code == 200:
                            logger.info(f"Grok video generated: {len(video_resp.content)} bytes")
                            return video_resp.content

                    # Check for async request_id
                    if "request_id" in data:
                        request_id = data["request_id"]
                        logger.info(f"Video generation started: {request_id}")

                        # Poll for result
                        for _ in range(60):  # 3 minutes max
                            await asyncio.sleep(3)
                            status_resp = await client.get(
                                f"{self.base_url}/videos/{request_id}",
                                headers={"Authorization": f"Bearer {self.api_key}"}
                            )
                            if status_resp.status_code == 200:
                                status_data = status_resp.json()
                                logger.info(f"Video poll: keys={list(status_data.keys())}")

                                # xAI returns {"video": {...}, "model": "..."} when done
                                video_obj = status_data.get("video")
                                if video_obj:
                                    video_url = video_obj.get("url") if isinstance(video_obj, dict) else video_obj
                                    if video_url and isinstance(video_url, str):
                                        logger.info(f"Video URL found: {video_url[:80]}...")
                                        video_resp = await client.get(video_url)
                                        if video_resp.status_code == 200:
                                            logger.info(f"Grok video ready: {len(video_resp.content)} bytes")
                                            return video_resp.content
                                        else:
                                            logger.error(f"Video download failed: {video_resp.status_code}")
                                    else:
                                        logger.warning(f"Video object but no URL: {video_obj}")

                                # Also check legacy status field
                                status = status_data.get("status")
                                if status == "completed":
                                    video_url = status_data.get("url")
                                    if video_url:
                                        video_resp = await client.get(video_url)
                                        if video_resp.status_code == 200:
                                            return video_resp.content
                                elif status == "failed" or status_data.get("error"):
                                    logger.error(f"Video generation failed: {status_data}")
                                    break

                else:
                    logger.error(f"Grok video API error: {response.status_code} - {response.text[:200]}")

        except Exception as e:
            logger.error(f"Grok video generation error: {e}")

        return None


class ImageGenerator:
    """
    Unified image generator with fallback support.
    Uses Gemini with reference images for Borg Cosmos memes.
    Falls back to Grok for generic generation.
    Now includes video generation via Grok Imagine.
    """

    # Scene variation prompts - ONLY describe scene/setting/action, NOT the character
    # The character comes from reference images
    BORG_COSMOS_SCENES = [
        # Restaurant/Eating scenes (use eating reference)
        "sitting at an upscale seafood restaurant, cracking open a fresh lobster with his mechanical hand, zapping competitor crabs with his laser eye",
        "at a beachside lobster shack, enjoying a lobster roll while robots serve him, ocean view in background",
        "hosting a fancy dinner party, presenting a perfectly cooked lobster thermidor to impressed guests",
        "at a food competition, his lobster dish winning first place while other contestants look shocked",

        # Cooking scenes (use portrait reference)
        "in a high-tech kitchen, using his laser eye to perfectly sear lobster tails, steam rising",
        "grilling lobster on a futuristic BBQ grill, wearing a 'Kiss the Borg Chef' apron",
        "in his lab, conducting a 'scientific' lobster cooking experiment with bubbling beakers",
        "teaching robot assistants how to properly prepare lobster bisque",
        "deep frying lobster with precision cyborg timing, oil sizzling perfectly",

        # Victory/Competition scenes
        "standing victoriously on a podium, holding a golden lobster trophy, defeated crabs below",
        "in a battle arena, his lobster army defeating an army of cartoon crabs labeled 'Hermes'",
        "giving a TED talk about 'Why Lobster is Superior to Crab', audience amazed",
        "at a press conference announcing the defeat of claw-based competitors, lobsters celebrating",

        # Tech/AI scenes
        "presenting a holographic chart showing '$COSMOS to the moon' while eating lobster",
        "in a command center, directing his AI swarm while snacking on lobster claws",
        "streaming on a futuristic computer setup, lobster dinner beside the keyboard",
        "at a crypto conference, booth showing COSMOS token with lobster mascot",
    ]

    def __init__(self):
        self.grok = GrokImageGenerator()
        self.gemini = GeminiImageGenerator()
        self.video = GrokVideoGenerator()
        self.last_provider = None

    def get_status(self) -> dict:
        return {
            "grok_available": self.grok.is_available(),
            "gemini_available": self.gemini.is_available(),
            "video_available": self.video.is_available(),
            "last_provider": self.last_provider
        }

    async def generate_video_from_image(
        self,
        image_bytes: bytes,
        prompt: str = "Animate with natural movement, cooking lobster, triumphant pose",
        duration: int = 5
    ) -> Optional[bytes]:
        """
        Generate video from an image using Grok Imagine.

        Args:
            image_bytes: Source image
            prompt: Animation description
            duration: Video length (1-15 seconds)

        Returns:
            MP4 video bytes or None
        """
        if not self.video.is_available():
            logger.warning("Grok video generation not available")
            return None

        return await self.video.generate_from_image(image_bytes, prompt, duration)

    async def generate_borg_cosmos_video(self, scene: str = None) -> Optional[bytes]:
        """
        Generate a Borg Cosmos video: Gemini image → Grok video.

        Returns:
            MP4 video bytes or None
        """
        # First generate image with Gemini
        if not scene:
            scene = random.choice(self.BORG_COSMOS_SCENES)

        logger.info(f"Step 1: Generating image with Gemini for: {scene[:50]}...")
        image, _ = await self.generate_borg_cosmos_meme()

        if not image:
            logger.error("Failed to generate source image for video")
            return None

        # Then animate with Grok
        video_prompt = f"Animate Borg Cosmos {scene}. Natural movement, expressive, cartoon style."
        logger.info(f"Step 2: Animating with Grok video...")

        video = await self.generate_video_from_image(image, video_prompt, duration=5)

        if video:
            self.last_provider = "gemini_grok_video"
            logger.info(f"Video generated: {len(video)} bytes")

        return video

    def get_random_meme_prompt(self) -> Tuple[str, str]:
        """Get a random Cosmos meme prompt and caption"""
        prompt = random.choice(COSMOS_MEME_PROMPTS)
        caption = random.choice(MEME_CAPTIONS)
        return prompt, caption

    def get_random_scene(self) -> str:
        """Get a random scene variation for Borg Cosmos"""
        return random.choice(self.BORG_COSMOS_SCENES)

    async def generate(self, prompt: str, prefer: str = "gemini") -> Optional[bytes]:
        """
        Generate image, trying preferred provider first.

        Args:
            prompt: Text description of image
            prefer: "grok" or "gemini"

        Returns:
            Image bytes or None
        """
        providers = [self.gemini, self.grok] if prefer == "gemini" else [self.grok, self.gemini]
        provider_names = ["gemini", "grok"] if prefer == "gemini" else ["grok", "gemini"]

        for provider, name in zip(providers, provider_names):
            if provider.is_available():
                logger.info(f"Trying {name} for image generation...")
                result = await provider.generate(prompt)
                if result:
                    self.last_provider = name
                    return result

        logger.error("All image providers failed or unavailable")
        return None

    async def generate_borg_cosmos_meme(self) -> Tuple[Optional[bytes], str]:
        """
        Generate a Borg Cosmos meme using Gemini with reference images.

        Uses the reference images to maintain character consistency while
        varying the scene, setting, and cooking method.

        Returns:
            (image_bytes, scene_description)
        """
        scene = self.get_random_scene()

        # Determine which reference to use based on scene
        use_portrait = any(kw in scene.lower() for kw in ["cooking", "kitchen", "lab", "grill", "fry", "tech", "command", "conference", "present"])

        logger.info(f"Generating Borg Cosmos scene: {scene[:60]}...")

        # Try Gemini with reference first
        if self.gemini.is_available():
            image = await self.gemini.generate_with_reference(scene, use_portrait=use_portrait)
            if image:
                self.last_provider = "gemini_reference"
                return image, scene

        # Fallback to Grok with full prompt
        full_prompt = f"Borg-cyborg Professor Cosmos from Futurama with half-metal face, red glowing laser eye, white lab coat, {scene}, cartoon style, meme format"
        if self.grok.is_available():
            image = await self.grok.generate(full_prompt)
            if image:
                self.last_provider = "grok"
                return image, scene

        logger.error("Failed to generate Borg Cosmos meme")
        return None, scene

    async def generate_cosmos_meme(self) -> Tuple[Optional[bytes], str, str]:
        """
        Generate a random Cosmos meme.

        Returns:
            (image_bytes, prompt_used, caption)
        """
        # Use the new Borg Cosmos generator with references
        image, scene = await self.generate_borg_cosmos_meme()

        # Get a caption from posting brain if available
        try:
            from Cosmos.integration.x_automation.posting_brain import get_posting_brain
            brain = get_posting_brain()
            caption = brain.get_meme_caption()
        except ImportError:
            caption = random.choice(MEME_CAPTIONS)

        return image, scene, caption


# Global instance
_image_generator = None

def get_image_generator() -> ImageGenerator:
    global _image_generator
    if _image_generator is None:
        _image_generator = ImageGenerator()
    return _image_generator

async def generate_meme() -> Tuple[Optional[bytes], str, str]:
    """Convenience function to generate a Cosmos meme"""
    gen = get_image_generator()
    return await gen.generate_cosmos_meme()


# ============================================
# CLI for testing
# ============================================
if __name__ == "__main__":
    import sys

    async def test():
        gen = ImageGenerator()
        print(f"Status: {gen.get_status()}")

        if len(sys.argv) > 1:
            prompt = " ".join(sys.argv[1:])
        else:
            prompt, caption = gen.get_random_meme_prompt()
            print(f"Using random prompt: {prompt}")
            print(f"Caption: {caption}")

        print("Generating image...")
        image = await gen.generate(prompt)

        if image:
            output_path = f"/tmp/cosmos_meme_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            with open(output_path, "wb") as f:
                f.write(image)
            print(f"Saved to: {output_path}")
        else:
            print("Failed to generate image")

    asyncio.run(test())
