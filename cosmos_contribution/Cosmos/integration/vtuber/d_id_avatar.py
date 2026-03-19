#!/usr/bin/env python3
"""
D-ID Avatar Integration - Generates talking avatar videos with lip sync.
Uses ElevenLabs for audio + D-ID for avatar rendering.
"""

import asyncio
import aiohttp
import os
import time
import base64
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass
from loguru import logger

# Load environment
try:
    from dotenv import load_dotenv
    load_dotenv("/workspace/Cosmos/.env")
except ImportError:
    pass


@dataclass
class DIDConfig:
    """D-ID API configuration"""
    api_key: str = ""
    presenter_id: str = ""  # Avatar image URL (S3 URL from D-ID upload)
    driver_url: str = "bank://lively"  # Default driver for natural movement
    result_format: str = "mp4"

    def __post_init__(self):
        if not self.api_key:
            self.api_key = os.getenv("DID_API_KEY", "")
        if not self.presenter_id:
            self.presenter_id = os.getenv("DID_AVATAR_URL", "")


class DIDAvatar:
    """
    D-ID Avatar renderer - creates talking head videos from audio.

    Flow:
    1. Text → ElevenLabs → Audio file
    2. Upload audio to D-ID → S3 URL
    3. Create talk with audio S3 URL → Video with lip sync
    4. Download video → Stream to Twitter
    """

    BASE_URL = "https://api.d-id.com"

    def __init__(self, config: Optional[DIDConfig] = None):
        self.config = config or DIDConfig()
        self._session: Optional[aiohttp.ClientSession] = None

        # Cache directory for generated videos
        self.cache_dir = Path("/workspace/Cosmos/cache/did_videos")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Uploaded image URL cache (to avoid re-uploading same avatar)
        self._avatar_url: Optional[str] = None

    def _get_headers(self) -> dict:
        """Get auth headers for D-ID API"""
        return {
            "Authorization": f"Basic {self.config.api_key}",
            "Content-Type": "application/json",
        }

    def _get_upload_headers(self) -> dict:
        """Get auth headers for multipart uploads (no Content-Type)"""
        return {
            "Authorization": f"Basic {self.config.api_key}",
        }

    async def _ensure_session(self):
        """Ensure aiohttp session exists"""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=60, connect=10)
            self._session = aiohttp.ClientSession(timeout=timeout)

    async def close(self):
        """Close session"""
        if self._session and not self._session.closed:
            await self._session.close()

    async def get_credits(self) -> dict:
        """Check remaining D-ID credits"""
        await self._ensure_session()
        async with self._session.get(f"{self.BASE_URL}/credits", headers=self._get_headers()) as resp:
            return await resp.json()

    async def list_presenters(self) -> list:
        """List available presenters/avatars"""
        await self._ensure_session()
        async with self._session.get(
            f"{self.BASE_URL}/clips/presenters",
            headers=self._get_headers()
        ) as resp:
            data = await resp.json()
            return data.get("presenters", [])

    async def upload_audio(self, audio_path: str) -> Optional[str]:
        """
        Upload audio file to D-ID and get S3 URL.
        D-ID requires audio at an accessible URL, not base64.
        """
        await self._ensure_session()

        try:
            with open(audio_path, "rb") as f:
                audio_data = f.read()

            # Determine filename and content type
            filename = Path(audio_path).name
            if audio_path.endswith(".mp3"):
                content_type = "audio/mpeg"
            else:
                content_type = "audio/wav"

            # Upload via multipart form
            data = aiohttp.FormData()
            data.add_field("audio", audio_data, filename=filename, content_type=content_type)

            async with self._session.post(
                f"{self.BASE_URL}/audios",
                data=data,
                headers=self._get_upload_headers()
            ) as resp:
                if resp.status == 201:
                    result = await resp.json()
                    audio_url = result.get("url")
                    logger.info(f"Audio uploaded to D-ID: {result.get('id')}")
                    return audio_url
                else:
                    error = await resp.text()
                    logger.error(f"Audio upload failed ({resp.status}): {error}")
                    return None

        except Exception as e:
            logger.error(f"Audio upload error: {e}")
            return None

    async def upload_image(self, image_path: str) -> Optional[str]:
        """
        Upload an image to D-ID and get URL for use as avatar.
        """
        await self._ensure_session()

        try:
            with open(image_path, "rb") as f:
                image_data = f.read()

            # Determine content type
            ext = Path(image_path).suffix.lower()
            content_types = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".webp": "image/webp",
            }
            content_type = content_types.get(ext, "image/jpeg")
            filename = Path(image_path).name

            # Upload via multipart form
            data = aiohttp.FormData()
            data.add_field("image", image_data, filename=filename, content_type=content_type)

            async with self._session.post(
                f"{self.BASE_URL}/images",
                data=data,
                headers=self._get_upload_headers()
            ) as resp:
                if resp.status == 201:
                    result = await resp.json()
                    image_url = result.get("url")
                    logger.info(f"Image uploaded to D-ID: {result.get('id')}")
                    return image_url
                else:
                    error = await resp.text()
                    logger.error(f"Image upload failed ({resp.status}): {error}")
                    return None

        except Exception as e:
            logger.error(f"Image upload error: {e}")
            return None

    async def set_avatar(self, image_path: str) -> bool:
        """Upload an image and set it as the avatar for future talks."""
        url = await self.upload_image(image_path)
        if url:
            self._avatar_url = url
            logger.info(f"Avatar set: {url}")
            return True
        return False

    async def create_talk_from_audio(
        self,
        audio_path: str,
        presenter_id: Optional[str] = None,
        driver_url: Optional[str] = None,
    ) -> Optional[str]:
        """
        Create a talking avatar video from audio file.

        Args:
            audio_path: Path to audio file (wav/mp3)
            presenter_id: D-ID presenter/avatar URL or image URL (uses config/uploaded if not provided)
            driver_url: Animation driver URL

        Returns:
            Path to generated video file, or None on failure
        """
        await self._ensure_session()

        # Use config defaults or uploaded avatar
        presenter_id = presenter_id or self._avatar_url or self.config.presenter_id
        driver_url = driver_url or self.config.driver_url

        if not presenter_id:
            logger.error("No presenter_id/avatar configured. Call set_avatar() first.")
            return None

        # Upload audio to D-ID (they don't accept base64)
        logger.info(f"Uploading audio to D-ID: {Path(audio_path).name}")
        audio_url = await self.upload_audio(audio_path)
        if not audio_url:
            logger.error("Failed to upload audio to D-ID")
            return None

        # Create talk request
        payload = {
            "source_url": presenter_id,  # Presenter image URL
            "script": {
                "type": "audio",
                "audio_url": audio_url,  # S3 URL from upload
            },
            "config": {
                "fluent": True,
                "pad_audio": 0.5,
            },
            "driver_url": driver_url,
        }

        logger.info(f"Creating D-ID talk video...")

        try:
            # Submit talk creation
            async with self._session.post(
                f"{self.BASE_URL}/talks",
                json=payload,
                headers=self._get_headers()
            ) as resp:
                if resp.status != 201:
                    error = await resp.text()
                    logger.error(f"D-ID create talk failed ({resp.status}): {error}")
                    return None

                result = await resp.json()
                talk_id = result.get("id")
                logger.info(f"D-ID talk created: {talk_id}")

            # Poll for completion
            video_url = await self._wait_for_talk(talk_id)
            if not video_url:
                return None

            # Download video
            output_path = self.cache_dir / f"talk_{talk_id}.mp4"
            await self._download_video(video_url, output_path)

            logger.info(f"D-ID video saved: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"D-ID talk creation failed: {e}")
            return None

    async def create_talk_from_text(
        self,
        text: str,
        voice_id: str = "en-US-GuyNeural",  # Azure voice ID
        presenter_id: Optional[str] = None,
        driver_url: Optional[str] = None,
    ) -> Optional[str]:
        """
        Create a talking avatar video from text (using D-ID's TTS).
        Note: For better quality, use ElevenLabs + create_talk_from_audio.

        Args:
            text: Text to speak
            voice_id: D-ID/Azure voice ID
            presenter_id: D-ID presenter/avatar ID
            driver_url: Animation driver URL

        Returns:
            Path to generated video file, or None on failure
        """
        await self._ensure_session()

        presenter_id = presenter_id or self.config.presenter_id
        driver_url = driver_url or self.config.driver_url

        if not presenter_id:
            logger.error("No presenter_id configured")
            return None

        payload = {
            "source_url": presenter_id,
            "script": {
                "type": "text",
                "input": text,
                "provider": {
                    "type": "microsoft",
                    "voice_id": voice_id,
                },
            },
            "config": {
                "fluent": True,
                "pad_audio": 0.5,
                "stitch": True,
            },
            "driver_url": driver_url,
        }

        logger.info(f"Creating D-ID talk from text: {text[:50]}...")

        try:
            async with self._session.post(
                f"{self.BASE_URL}/talks",
                json=payload
            ) as resp:
                if resp.status != 201:
                    error = await resp.text()
                    logger.error(f"D-ID create talk failed ({resp.status}): {error}")
                    return None

                result = await resp.json()
                talk_id = result.get("id")
                logger.info(f"D-ID talk created: {talk_id}")

            video_url = await self._wait_for_talk(talk_id)
            if not video_url:
                return None

            output_path = self.cache_dir / f"talk_{talk_id}.mp4"
            await self._download_video(video_url, output_path)

            logger.info(f"D-ID video saved: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"D-ID talk creation failed: {e}")
            return None

    async def _wait_for_talk(
        self,
        talk_id: str,
        timeout: float = 120,
        poll_interval: float = 2,
    ) -> Optional[str]:
        """
        Poll for talk completion and return result URL.

        Returns:
            Video URL on success, None on failure/timeout
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                async with self._session.get(
                    f"{self.BASE_URL}/talks/{talk_id}",
                    headers=self._get_headers()
                ) as resp:
                    if resp.status != 200:
                        logger.warning(f"D-ID poll failed: {resp.status}")
                        await asyncio.sleep(poll_interval)
                        continue

                    result = await resp.json()
                    status = result.get("status")

                    if status == "done":
                        video_url = result.get("result_url")
                        logger.info(f"D-ID talk completed: {video_url}")
                        return video_url
                    elif status == "error":
                        error = result.get("error", {})
                        logger.error(f"D-ID talk failed: {error}")
                        return None
                    else:
                        logger.debug(f"D-ID talk status: {status}")

            except Exception as e:
                logger.warning(f"D-ID poll error: {e}")

            await asyncio.sleep(poll_interval)

        logger.error(f"D-ID talk timed out after {timeout}s")
        return None

    async def _download_video(self, url: str, output_path: Path) -> bool:
        """Download video from URL to local file"""
        try:
            await self._ensure_session()
            async with self._session.get(url) as resp:
                if resp.status == 200:
                    with open(output_path, "wb") as f:
                        async for chunk in resp.content.iter_chunked(8192):
                            f.write(chunk)
                    return True
                else:
                    logger.error(f"Video download failed: {resp.status}")
                    return False
        except Exception as e:
            logger.error(f"Video download error: {e}")
            return False


class ElevenLabsDIDPipeline:
    """
    Full pipeline: Text → ElevenLabs (audio) → D-ID (video with lip sync)

    This offloads all heavy processing to APIs:
    - ElevenLabs: High quality TTS with custom Cosmos voice
    - D-ID: Avatar rendering + lip sync with custom Cosmos avatar
    """

    def __init__(
        self,
        did_config: Optional[DIDConfig] = None,
        elevenlabs_voice_id: str = None,
    ):
        self.did = DIDAvatar(did_config)
        self.elevenlabs_voice_id = elevenlabs_voice_id or os.getenv(
            "ELEVENLABS_VOICE_COSMOS",
            "dxvY1G6UilzEKgCy370m"
        )
        self.elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY", "")

        # Audio cache
        self.audio_cache_dir = Path("/workspace/Cosmos/cache/elevenlabs_audio")
        self.audio_cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"ElevenLabs+D-ID Pipeline initialized")
        logger.info(f"  Voice: {self.elevenlabs_voice_id}")
        logger.info(f"  Avatar: {self.did.config.presenter_id[:50]}..." if self.did.config.presenter_id else "  Avatar: NOT SET")

    async def generate_elevenlabs_audio(self, text: str) -> Optional[str]:
        """Generate audio using ElevenLabs API with Cosmos voice"""
        if not self.elevenlabs_api_key:
            logger.error("ElevenLabs API key not configured")
            return None

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.elevenlabs_voice_id}"

        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": self.elevenlabs_api_key,
        }

        payload = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
            }
        }

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.post(url, json=payload, headers=headers) as resp:
                    if resp.status == 200:
                        audio_data = await resp.read()

                        # Save to file
                        timestamp = int(time.time() * 1000)
                        output_path = self.audio_cache_dir / f"tts_{timestamp}.mp3"
                        with open(output_path, "wb") as f:
                            f.write(audio_data)

                        logger.info(f"ElevenLabs audio generated: {len(text)} chars")
                        return str(output_path)
                    else:
                        error = await resp.text()
                        logger.error(f"ElevenLabs failed ({resp.status}): {error}")
                        return None

        except Exception as e:
            logger.error(f"ElevenLabs request failed: {e}")
            return None

    async def generate_video(
        self,
        text: str,
        download: bool = True
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Full pipeline: Text → ElevenLabs audio → D-ID video

        Args:
            text: Text to speak
            download: Whether to download the video locally

        Returns:
            Tuple of (local_video_path, video_url)
            If download=False, local_video_path will be None
        """
        # Step 1: Generate audio with ElevenLabs
        logger.info(f"Pipeline: Generating audio...")
        audio_path = await self.generate_elevenlabs_audio(text)
        if not audio_path:
            logger.error("Pipeline: Audio generation failed")
            return None, None

        # Step 2: Upload audio to D-ID
        logger.info("Pipeline: Uploading audio to D-ID...")
        audio_url = await self.did.upload_audio(audio_path)
        if not audio_url:
            logger.error("Pipeline: Audio upload failed")
            return None, None

        # Step 3: Create D-ID talk
        logger.info("Pipeline: Creating avatar video...")
        await self.did._ensure_session()

        presenter_id = self.did._avatar_url or self.did.config.presenter_id
        if not presenter_id:
            logger.error("Pipeline: No avatar configured")
            return None, None

        payload = {
            "source_url": presenter_id,
            "script": {
                "type": "audio",
                "audio_url": audio_url,
            },
            "config": {"fluent": True, "pad_audio": 0.5},
            "driver_url": self.did.config.driver_url,
        }

        try:
            async with self.did._session.post(
                f"{self.did.BASE_URL}/talks",
                json=payload,
                headers=self.did._get_headers()
            ) as resp:
                if resp.status != 201:
                    error = await resp.text()
                    logger.error(f"D-ID create talk failed ({resp.status}): {error}")
                    return None, None

                result = await resp.json()
                talk_id = result.get("id")
                logger.info(f"D-ID talk created: {talk_id}")

            # Poll for completion
            video_url = await self.did._wait_for_talk(talk_id)
            if not video_url:
                return None, None

            # Download if requested
            local_path = None
            if download:
                output_path = self.did.cache_dir / f"talk_{talk_id}.mp4"
                await self.did._download_video(video_url, output_path)
                local_path = str(output_path)
                logger.info(f"Pipeline complete: {local_path}")
            else:
                logger.info(f"Pipeline complete: {video_url}")

            return local_path, video_url

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            return None, None

    async def close(self):
        """Clean up"""
        await self.did.close()


# Test function
async def test_did_api():
    """Test D-ID API connection"""
    api_key = os.getenv("DID_API_KEY")
    if not api_key:
        print("DID_API_KEY not set in environment")
        return

    config = DIDConfig(api_key=api_key)
    avatar = DIDAvatar(config)

    try:
        # Check credits
        credits = await avatar.get_credits()
        print(f"D-ID Credits: {credits.get('remaining', 0)}/{credits.get('total', 0)}")

        # List presenters
        presenters = await avatar.list_presenters()
        print(f"Available presenters: {len(presenters)}")
        for p in presenters[:5]:
            print(f"  - {p.get('presenter_id')}: {p.get('name', 'unnamed')}")

    finally:
        await avatar.close()


if __name__ == "__main__":
    asyncio.run(test_did_api())
