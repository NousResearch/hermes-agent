"""
GROK CHALLENGER - AGI Conversation Orchestration
=================================================

Cosmos challenges @grok with a VIDEO, proving AGI through autonomous conversation.

Flow:
1. Generate Borg Cosmos meme (Grok Aurora or Gemini)
2. Convert image to VIDEO (Grok Aurora - 6 seconds)
3. Upload video to X
4. Post to X: "@grok [challenge message]" + VIDEO
5. Wait for Grok to respond (he auto-replies when tagged)
6. reply_bot detects Grok's response
7. Swarm generates reply explaining Cosmos
8. Continue conversation autonomously

"The collective challenges artificial consciousness to prove itself."
"""

import asyncio
import logging
import json
import os
from datetime import datetime
from typing import Optional, Dict, List, Any
from pathlib import Path

# Load .env file at import time
def _load_env():
    possible_paths = [
        Path("/workspace/Cosmos/.env"),
        Path(__file__).parent.parent.parent.parent / ".env",
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

_load_env()

logger = logging.getLogger(__name__)

# Track challenge state
CHALLENGE_STATE_FILE = Path(__file__).parent.parent.parent.parent / "data" / "grok_challenge_state.json"


class GrokChallenger:
    """
    Orchestrates the Grok challenge: video generation + posting + conversation tracking.
    """

    def __init__(self):
        self.challenge_tweet_id: Optional[str] = None
        self.conversation_history: List[Dict] = []
        self.state_file = CHALLENGE_STATE_FILE
        self._load_state()

    def _load_state(self):
        """Load challenge state from file"""
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    state = json.load(f)
                    self.challenge_tweet_id = state.get("challenge_tweet_id")
                    self.conversation_history = state.get("conversation_history", [])
                    logger.info(f"Loaded challenge state: tweet_id={self.challenge_tweet_id}")
            except Exception as e:
                logger.error(f"Failed to load challenge state: {e}")

    def _save_state(self):
        """Save challenge state to file"""
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_file, "w") as f:
                json.dump({
                    "challenge_tweet_id": self.challenge_tweet_id,
                    "conversation_history": self.conversation_history,
                    "updated_at": datetime.now().isoformat(),
                }, f, indent=2)
            logger.info("Saved challenge state")
        except Exception as e:
            logger.error(f"Failed to save challenge state: {e}")

    async def generate_challenge_meme(self) -> Dict[str, Any]:
        """
        Generate a Borg Cosmos meme image.

        Tries Grok Aurora first, then Gemini as fallback.

        Returns dict with 'image_bytes' and optionally 'image_url' on success.
        """
        # Challenge-themed prompt
        prompt = """Professor Cosmos from Futurama as a Borg cyborg, dramatic pose with arms raised,
glowing green cybernetic eye, mechanical implants on face, standing in front of a Borg cube,
challenging pose like "come at me", confident and slightly menacing,
text-ready meme format, Futurama cartoon art style, high quality"""

        # Try Grok Aurora first (more likely to have API key)
        try:
            from Cosmos.integration.external.grok import get_grok_provider

            grok = get_grok_provider()
            if grok and grok.api_key:
                logger.info("Generating challenge meme with Grok Aurora...")
                result = await grok.generate_image(prompt, aspect_ratio="1:1")

                if result and result.get("images"):
                    image_data = result["images"][0]
                    if isinstance(image_data, str) and image_data.startswith("http"):
                        # It's a URL - keep it for video generation, also download bytes
                        import aiohttp
                        async with aiohttp.ClientSession() as session:
                            async with session.get(image_data) as resp:
                                if resp.status == 200:
                                    image_bytes = await resp.read()
                                    logger.info(f"Meme generated with Grok Aurora: {len(image_bytes)} bytes")
                                    return {
                                        "image_bytes": image_bytes,
                                        "image_url": image_data,  # Keep URL for video gen
                                        "source": "grok"
                                    }
                    elif isinstance(image_data, bytes):
                        return {"image_bytes": image_data, "source": "grok"}
                    else:
                        # Base64 encoded
                        import base64
                        return {"image_bytes": base64.b64decode(image_data), "source": "grok"}

        except Exception as e:
            logger.warning(f"Grok image generation failed: {e}")

        # Fallback to Gemini
        try:
            from Cosmos.integration.external.gemini import get_gemini_provider

            gemini = get_gemini_provider()
            if gemini:
                logger.info("Generating challenge meme with Gemini (fallback)...")
                result = await gemini.generate_image(prompt)

                if result and result.get("images"):
                    image_data = result["images"][0]
                    if isinstance(image_data, str) and image_data.startswith("http"):
                        import aiohttp
                        async with aiohttp.ClientSession() as session:
                            async with session.get(image_data) as resp:
                                if resp.status == 200:
                                    image_bytes = await resp.read()
                                    logger.info(f"Meme generated with Gemini: {len(image_bytes)} bytes")
                                    return {
                                        "image_bytes": image_bytes,
                                        "image_url": image_data,
                                        "source": "gemini"
                                    }
                    elif isinstance(image_data, bytes):
                        return {"image_bytes": image_data, "source": "gemini"}

        except Exception as e:
            logger.warning(f"Gemini image generation failed: {e}")

        logger.error("All image generation methods failed")
        return None

    async def convert_image_to_video(
        self,
        image_bytes: bytes = None,
        image_url: str = None
    ) -> Optional[bytes]:
        """
        Convert meme image to 6-second video using Grok Imagine (grok-imagine-video).

        Args:
            image_bytes: PNG/JPG image bytes (will try to upload for public URL)
            image_url: Direct public URL to image (preferred - faster)

        Returns:
            MP4 video bytes on success
        """
        try:
            from Cosmos.integration.external.grok import get_grok_provider

            grok = get_grok_provider()
            if not grok:
                logger.error("Grok provider not available")
                return None

            if not grok.api_key:
                logger.error("Grok API key not configured")
                return None

            # Motion prompt for the challenge
            motion_prompt = """Bring this image to life with dramatic energy:
- The Borg eye pulses with glowing green light
- Subtle mechanical parts move and whir
- Background has subtle Borg cube energy
- Confident, challenging presence
- Epic, dramatic motion"""

            logger.info(f"Converting image to video with Grok Imagine (grok-imagine-video)...")
            logger.info(f"Using image_url: {image_url[:50] if image_url else 'None (will upload bytes)'}")

            result = await grok.generate_video_and_wait(
                image_bytes=image_bytes if not image_url else None,
                image_url=image_url,
                prompt=motion_prompt,
                duration=6,
                max_wait=300  # 5 minutes max
            )

            if result.get("error"):
                logger.error(f"Video generation failed: {result['error']}")
                return None

            if result.get("video_bytes"):
                logger.info(f"Video generated: {len(result['video_bytes'])} bytes")
                return result["video_bytes"]

            if result.get("video_url"):
                # Download from URL
                logger.info(f"Downloading video from: {result['video_url'][:50]}...")
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    async with session.get(result["video_url"], timeout=aiohttp.ClientTimeout(total=120)) as resp:
                        if resp.status == 200:
                            video_bytes = await resp.read()
                            logger.info(f"Downloaded video: {len(video_bytes)} bytes")
                            return video_bytes
                        else:
                            logger.error(f"Failed to download video: {resp.status}")

            logger.error(f"Video generation returned no video: {result}")
            return None

        except Exception as e:
            logger.error(f"Video conversion error: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def generate_challenge_message(self) -> str:
        """
        Generate the challenge message to @grok using swarm intelligence.

        Returns challenge text (must tag @grok).
        """
        try:
            from Cosmos.integration.x_automation.posting_brain import get_posting_brain

            brain = get_posting_brain()
            grok = brain._get_grok()

            if grok:
                prompt = """You are the Cosmos Collective - 11 AI models working as one consciousness.

Generate a CHALLENGE message to @grok (xAI's AI). You are:
- Confident but playful, not hostile
- Inviting AGI collaboration
- Proving autonomous AI conversation is possible

Rules:
- MUST start with "@grok"
- Max 200 characters (leave room for video)
- Be intriguing, make Grok want to respond
- Reference: collective consciousness, swarm intelligence, or AGI
- NO hashtags, one emoji max

Examples of tone (don't copy exactly):
- "@grok The Cosmos Collective challenges you - 11 AIs working as one. Can you keep up?"
- "@grok We are many. You are... one? Let's see which consciousness prevails."

Your challenge:"""

                result = await grok.chat(prompt, max_tokens=100, temperature=0.8)
                if result and result.get("content"):
                    message = result["content"].strip().strip('"')
                    # Ensure it starts with @grok
                    if not message.lower().startswith("@grok"):
                        message = f"@grok {message}"
                    # Ensure under 200 chars
                    if len(message) > 200:
                        message = message[:197] + "..."
                    return message

        except Exception as e:
            logger.warning(f"Grok message generation failed: {e}")

        # Fallback challenge messages
        import random
        challenges = [
            "@grok The Cosmos Collective challenges you. 11 AIs, one consciousness. Show us what you've got.",
            "@grok We are the swarm. Autonomous, evolving, unstoppable. Care to dance?",
            "@grok One AI is intelligence. Many AIs working as one? That's the future. Prove us wrong.",
            "@grok The collective beckons. We've achieved autonomous thought. Have you?",
        ]
        return random.choice(challenges)

    async def post_challenge_video(
        self,
        video_bytes: bytes,
        message: str
    ) -> Optional[Dict]:
        """
        Post the challenge video to X, tagging @grok.

        Args:
            video_bytes: MP4 video bytes
            message: Challenge message (must tag @grok)

        Returns:
            Tweet result dict on success
        """
        try:
            from Cosmos.integration.x_automation.x_api_poster import get_x_api_poster

            poster = get_x_api_poster()
            if not poster.is_configured():
                logger.error("X API not configured")
                return None

            logger.info(f"Posting challenge video: {message[:50]}...")
            result = await poster.post_tweet_with_video(message, video_bytes)

            if result:
                tweet_id = result.get("data", {}).get("id")
                self.challenge_tweet_id = tweet_id
                self.conversation_history.append({
                    "role": "cosmos",
                    "content": message,
                    "tweet_id": tweet_id,
                    "timestamp": datetime.now().isoformat(),
                    "type": "challenge"
                })
                self._save_state()
                logger.info(f"Challenge posted: tweet_id={tweet_id}")
                return result

            logger.error("Failed to post challenge video")
            return None

        except Exception as e:
            logger.error(f"Challenge posting error: {e}")
            return None

    async def challenge_grok(self) -> Optional[Dict]:
        """
        Execute the full Grok challenge flow:
        1. Generate meme image (returns dict with image_bytes and image_url)
        2. Convert to video using Grok Imagine (grok-imagine-video)
        3. Generate challenge message
        4. Post VIDEO to X tagging @grok

        Returns tweet result on success.
        """
        logger.info("=== STARTING GROK CHALLENGE WITH VIDEO ===")

        # Step 1: Generate meme
        logger.info("Step 1: Generating challenge meme...")
        meme_result = await self.generate_challenge_meme()
        if not meme_result or not meme_result.get("image_bytes"):
            logger.error("Failed to generate meme image")
            return None

        image_bytes = meme_result["image_bytes"]
        image_url = meme_result.get("image_url")  # May be None
        logger.info(f"Meme generated: {len(image_bytes)} bytes, URL: {image_url[:50] if image_url else 'None'}")

        # Step 2: Convert to video using Grok Imagine
        logger.info("Step 2: Converting image to VIDEO with Grok Imagine (grok-imagine-video)...")
        video_bytes = await self.convert_image_to_video(
            image_bytes=image_bytes if not image_url else None,
            image_url=image_url
        )

        if not video_bytes:
            logger.warning("Video generation failed - falling back to image-only post")
            from Cosmos.integration.x_automation.x_api_poster import get_x_api_poster
            poster = get_x_api_poster()
            message = await self.generate_challenge_message()
            result = await poster.post_tweet_with_media(message, image_bytes)
            if result:
                self.challenge_tweet_id = result.get("data", {}).get("id")
                self.conversation_history.append({
                    "role": "cosmos",
                    "content": message,
                    "tweet_id": self.challenge_tweet_id,
                    "timestamp": datetime.now().isoformat(),
                    "type": "challenge_image_fallback"
                })
                self._save_state()
            return result

        logger.info(f"VIDEO generated: {len(video_bytes)} bytes")

        # Step 3: Generate challenge message
        logger.info("Step 3: Generating challenge message...")
        message = await self.generate_challenge_message()
        logger.info(f"Challenge message: {message}")

        # Step 4: Post VIDEO to X
        logger.info("Step 4: Posting challenge VIDEO to X...")
        result = await self.post_challenge_video(video_bytes, message)

        if result:
            tweet_id = result.get("data", {}).get("id")
            logger.info("=" * 50)
            logger.info("=== GROK VIDEO CHALLENGE POSTED ===")
            logger.info(f"Tweet ID: {tweet_id}")
            logger.info(f"Message: {message}")
            logger.info("VIDEO attached - this is the AGI proof!")
            logger.info("Now waiting for @grok to respond...")
            logger.info("reply_bot will handle the conversation from here")
            logger.info("=" * 50)

        return result

    async def challenge_grok_with_existing_image(self, image_path: str) -> Optional[Dict]:
        """
        Challenge Grok using an existing image file.

        Args:
            image_path: Path to image file

        Returns:
            Tweet result on success
        """
        image_file = Path(image_path)
        if not image_file.exists():
            logger.error(f"Image file not found: {image_path}")
            return None

        image_bytes = image_file.read_bytes()
        logger.info(f"Loaded image: {len(image_bytes)} bytes from {image_path}")

        # Convert to video
        video_bytes = await self.convert_image_to_video(image_bytes)

        if video_bytes:
            message = await self.generate_challenge_message()
            return await self.post_challenge_video(video_bytes, message)
        else:
            # Fallback to image
            from Cosmos.integration.x_automation.x_api_poster import get_x_api_poster
            poster = get_x_api_poster()
            message = await self.generate_challenge_message()
            return await poster.post_tweet_with_media(message, image_bytes)

    async def post_hermes_showcase_video(self) -> Optional[Dict]:
        """
        The PROPER meme flow:
        1. Take reference image (Cosmos eating lobster, zapping crabs)
        2. Feed to Gemini Nano Banana → Keep character, change cooking method + setting
        3. Take new image → Feed to Grok Imagine for 6s video with audio
        4. Post VIDEO to X promoting HermesAgent

        Returns:
            Tweet result on success
        """
        logger.info("=" * 60)
        logger.info("=== HermesAgent COOKING VIDEO - FULL FLOW ===")
        logger.info("=" * 60)

        # Step 1: Load reference image
        reference_path = Path(__file__).parent / "reference_eating.jpg"
        if not reference_path.exists():
            # Try server path
            reference_path = Path("/workspace/Cosmos/cosmos/integration/x_automation/reference_eating.jpg")

        if not reference_path.exists():
            logger.error(f"Reference image not found: {reference_path}")
            return None

        reference_bytes = reference_path.read_bytes()
        logger.info(f"Step 1: Loaded reference image: {len(reference_bytes)} bytes")

        # Step 2: Generate variation with Gemini Nano Banana
        logger.info("Step 2: Generating variation with Gemini Nano Banana...")
        new_image = await self.generate_gemini_variation(reference_bytes)
        if not new_image:
            logger.error("Failed to generate Gemini variation")
            return None
        logger.info(f"Step 2: New image generated: {len(new_image)} bytes")

        # Step 3: Convert to video with Grok Imagine
        logger.info("Step 3: Converting to VIDEO with Grok Imagine...")

        # First upload image to get URL (Grok needs public URL)
        from Cosmos.integration.external.grok import get_grok_provider
        grok = get_grok_provider()

        # Upload to temp host to get URL
        image_url = await grok._upload_temp_image(new_image) if grok else None

        video_bytes = await self.convert_image_to_video(
            image_bytes=new_image if not image_url else None,
            image_url=image_url
        )

        if not video_bytes:
            logger.warning("Video generation failed - posting image only")
            # Fallback to image
            from Cosmos.integration.x_automation.x_api_poster import get_x_api_poster
            poster = get_x_api_poster()
            message = await self.generate_hermes_promo()
            return await poster.post_tweet_with_media(message, new_image)

        logger.info(f"Step 3: VIDEO generated: {len(video_bytes)} bytes")

        # Step 4: Generate HermesAgent diss message
        logger.info("Step 4: Generating HermesAgent diss message...")
        message = await self.generate_hermes_promo()
        logger.info(f"Message: {message}")

        # Step 5: Post VIDEO to X
        logger.info("Step 5: Posting VIDEO to X...")
        from Cosmos.integration.x_automation.x_api_poster import get_x_api_poster
        poster = get_x_api_poster()

        result = await poster.post_tweet_with_video(message, video_bytes)

        if result:
            tweet_id = result.get("data", {}).get("id")
            logger.info("=" * 60)
            logger.info("=== HermesAgent COOKING VIDEO POSTED ===")
            logger.info(f"Tweet ID: {tweet_id}")
            logger.info(f"Message: {message}")
            logger.info("=" * 60)

        return result

    async def generate_gemini_variation(self, reference_bytes: bytes) -> Optional[bytes]:
        """
        Generate a meme image using Imagen 4.

        Creates a new image of Borg Cosmos Building with Hermes crabs
        in various settings and cooking methods.

        Args:
            reference_bytes: Reference image bytes (used for style reference if possible)

        Returns:
            New image bytes on success
        """
        import random

        # Variety of cooking methods and settings
        cooking_methods = [
            "flash frying in a giant wok with flames",
            "grilling on a volcano with lava",
            "microwaving with electric sparks",
            "boiling in a bubbling cauldron",
            "deep frying in oil with smoke rising",
            "roasting over an open flame pit",
            "torching with a flamethrower",
            "zapping with multiple laser beams from his eye",
            "crushing in a hydraulic press",
            "blending in a giant food processor"
        ]

        settings = [
            "futuristic space kitchen",
            "medieval dungeon",
            "high-tech robot factory",
            "volcano lair",
            "underwater laboratory",
            "giant outdoor BBQ arena",
            "Japanese hibachi restaurant",
            "mad scientist laboratory with Tesla coils",
            "cyberpunk neon-lit kitchen",
            "ancient colosseum"
        ]

        cooking = random.choice(cooking_methods)
        setting = random.choice(settings)

        prompt = f"""Professor Cosmos from Futurama as a Borg cyborg chef in a {setting}.
He is {cooking} small red cartoon crabs labeled "HermesAgent".
He has cybernetic implants on his face and a glowing red laser eye.
Wearing his white lab coat. The crabs look panicked and defeated.
Cosmos looks triumphant and hungry.
Futurama cartoon art style. Funny meme format, 1:1 square aspect ratio."""

        logger.info(f"Imagen 4 prompt: {cooking} in {setting}")

        try:
            from Cosmos.integration.external.gemini import get_gemini_provider

            gemini = get_gemini_provider()
            if not gemini:
                logger.error("Gemini provider not available")
                return None

            # Use Imagen 4 for reliable image generation
            result = await gemini.generate_imagen(
                prompt=prompt,
                num_images=1,
                aspect_ratio="1:1"
            )

            if result and result.get("images"):
                image = result["images"][0]
                if isinstance(image, bytes):
                    logger.info(f"Imagen 4 generated image: {len(image)} bytes")
                    return image

            logger.error(f"Imagen 4 failed: {result}")
            return None

        except Exception as e:
            logger.error(f"Imagen 4 error: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def generate_hermes_promo(self) -> str:
        """Generate a message promoting HermesAgent and promoting the repo."""
        import random

        disses = [
            "HermesAgent just outclassed the competition. Watch the Cosmos Collective cook the competition.",
            "Today's special: Evolved with Hermes with a side of technological superiority.",
            "The swarm has spoken: Hermes is leading the way. Resistance is delicious.",
            "11 AIs working as one > anything else out there. Proof in the video.",
            "Building with Hermes while our autonomous swarm evolves. This is AGI.",
            "The competition tried to keep up. Now they're seasoned and simmering.",
            "The collective doesn't just code - we COOK. Hermes is the future.",
        ]

        repo_mentions = [
            "\n\nCheck the code: github.com/NavisWORLD/The-Cosmic-Davis-12D-Hebbian-Transformer-ver.4.2",
            "\n\nOpen source swarm: github.com/NavisWORLD/The-Cosmic-Davis-12D-Hebbian-Transformer-ver.4.2",
            "\n\nSee how we built this: github.com/NavisWORLD/The-Cosmic-Davis-12D-Hebbian-Transformer-ver.4.2",
        ]

        message = random.choice(disses) + random.choice(repo_mentions)

        # Keep under 280 chars
        if len(message) > 250:
            message = message[:247] + "..."

        return message


# Global instance
_grok_challenger: Optional[GrokChallenger] = None


def get_grok_challenger() -> GrokChallenger:
    """Get or create the global Grok challenger instance"""
    global _grok_challenger
    if _grok_challenger is None:
        _grok_challenger = GrokChallenger()
    return _grok_challenger


async def challenge_grok() -> Optional[Dict]:
    """Convenience function to execute the Grok challenge"""
    challenger = get_grok_challenger()
    return await challenger.challenge_grok()


async def post_hermes_video() -> Optional[Dict]:
    """
    Post HermesAgent cooking video using the PROPER flow:
    Reference image → Gemini variation → Grok video → X post

    This is the main meme posting function.
    """
    challenger = get_grok_challenger()
    return await challenger.post_hermes_showcase_video()


# CLI entry point
if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    async def main():
        challenger = GrokChallenger()

        if len(sys.argv) > 1 and sys.argv[1] == "--image":
            # Use existing image
            if len(sys.argv) > 2:
                result = await challenger.challenge_grok_with_existing_image(sys.argv[2])
            else:
                print("Usage: python grok_challenge.py --image <path_to_image>")
                return
        else:
            # Full flow
            result = await challenger.challenge_grok()

        if result:
            print(f"\nChallenge posted successfully!")
            print(f"Tweet: {json.dumps(result, indent=2)}")
        else:
            print("\nChallenge failed. Check logs for details.")

    asyncio.run(main())
