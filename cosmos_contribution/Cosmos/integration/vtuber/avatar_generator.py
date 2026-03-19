"""
Avatar Generator - Creates VTuber avatar using Gemini Nano Banana PRO
Generates Borg Cosmos avatar images with consistent character appearance
"""

import asyncio
import os
from pathlib import Path
from typing import Optional, Tuple
from loguru import logger

try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# Avatar output directory
AVATAR_DIR = Path(__file__).parent / "avatars"
AVATAR_DIR.mkdir(parents=True, exist_ok=True)


class VTuberAvatarGenerator:
    """
    Generates VTuber avatar images for Cosmos using the image generation system.

    Creates:
    - Base neutral avatar for streaming
    - Expression variants (happy, thinking, excited, etc.)
    - Speaking frames with different mouth positions
    """

    def __init__(self):
        self._image_generator = None
        self._gemini = None

    async def _get_generator(self):
        """Lazy load the image generator"""
        if self._image_generator is None:
            try:
                from Cosmos.integration.image_gen.generator import get_image_generator
                self._image_generator = get_image_generator()
            except ImportError as e:
                logger.error(f"Failed to import image generator: {e}")
        return self._image_generator

    async def generate_base_avatar(self, save_path: Optional[str] = None) -> Optional[bytes]:
        """
        Generate the base VTuber avatar image.

        This creates a neutral pose, front-facing Borg Cosmos suitable
        for VTuber use with consistent character appearance.
        """
        generator = await self._get_generator()
        if not generator:
            logger.error("Image generator not available")
            return None

        # Special prompt for VTuber avatar (front-facing, neutral, suitable for animation)
        avatar_prompt = """
        Professor Cosmos from Futurama as a Borg cyborg, FRONT FACING PORTRAIT:
        - Half-metal chrome Borg implants on LEFT side of face
        - Red glowing laser eye on the cybernetic side
        - Normal eye on the human side
        - White wild Einstein-like hair
        - White lab coat with high collar
        - Neutral expression, mouth closed
        - Dark space/tech background
        - High detail, suitable for animation
        - Centered composition, shoulders visible
        - Cartoon/anime style, clean lines
        - 4K quality, sharp details
        """

        try:
            # Use Gemini with reference for consistency
            if hasattr(generator, 'gemini') and generator.gemini.is_available():
                image_bytes = await generator.gemini.generate_with_reference(
                    avatar_prompt,
                    use_portrait=True
                )
            else:
                # Fallback to regular generation
                image_bytes = await generator.generate(avatar_prompt)

            if image_bytes and save_path:
                with open(save_path, 'wb') as f:
                    f.write(image_bytes)
                logger.info(f"Avatar saved to: {save_path}")

            return image_bytes

        except Exception as e:
            logger.error(f"Avatar generation failed: {e}")
            return None

    async def generate_expression_set(self, output_dir: Optional[str] = None) -> dict:
        """
        Generate a set of expression images for the avatar.

        Returns dict mapping expression names to image bytes.
        """
        if output_dir is None:
            output_dir = str(AVATAR_DIR)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        expressions = {
            "neutral": "neutral expression, mouth closed, calm demeanor",
            "happy": "happy smile, eyes slightly closed, warm expression",
            "excited": "excited wide eyes, big smile, energetic pose",
            "thinking": "thoughtful expression, one eyebrow raised, looking slightly up",
            "surprised": "surprised wide eyes, mouth slightly open, raised eyebrows",
            "speaking_1": "mouth slightly open as if speaking, mid-word",
            "speaking_2": "mouth more open, vowel sound, engaged expression",
            "speaking_3": "mouth wide open, emphatic speaking, animated",
        }

        results = {}
        generator = await self._get_generator()

        if not generator:
            logger.error("Image generator not available")
            return results

        base_prompt = """
        Professor Cosmos from Futurama as a Borg cyborg, FRONT FACING PORTRAIT:
        - Half-metal chrome Borg implants on LEFT side of face
        - Red glowing laser eye on the cybernetic side
        - White wild Einstein-like hair, white lab coat
        - Dark space/tech background
        - Cartoon/anime style, centered, shoulders visible
        - Expression: {expression}
        """

        for name, expression_desc in expressions.items():
            logger.info(f"Generating expression: {name}")

            prompt = base_prompt.format(expression=expression_desc)

            try:
                if hasattr(generator, 'gemini') and generator.gemini.is_available():
                    image_bytes = await generator.gemini.generate_with_reference(
                        prompt,
                        use_portrait=True
                    )
                else:
                    image_bytes = await generator.generate(prompt)

                if image_bytes:
                    # Save to file
                    save_path = output_path / f"cosmos_{name}.png"
                    with open(save_path, 'wb') as f:
                        f.write(image_bytes)

                    results[name] = image_bytes
                    logger.info(f"Generated {name} expression -> {save_path}")

                # Rate limiting - don't spam the API
                await asyncio.sleep(2)

            except Exception as e:
                logger.error(f"Failed to generate {name}: {e}")

        return results

    async def generate_speaking_frames(self, num_frames: int = 5,
                                       output_dir: Optional[str] = None) -> list:
        """
        Generate a sequence of speaking frames for lip sync animation.

        These can be blended/morphed for smooth lip sync.
        """
        if output_dir is None:
            output_dir = str(AVATAR_DIR / "speaking")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Different mouth positions for visemes
        mouth_positions = [
            ("closed", "mouth completely closed, lips together"),
            ("slight", "mouth slightly open, relaxed"),
            ("medium", "mouth medium open, mid-vowel"),
            ("wide", "mouth wide open, 'ah' sound"),
            ("round", "mouth rounded, 'oh' sound"),
        ]

        frames = []
        generator = await self._get_generator()

        if not generator:
            return frames

        base_prompt = """
        Professor Cosmos Borg cyborg PORTRAIT, SPEAKING:
        - Half-metal chrome face with red laser eye
        - White hair, white lab coat
        - Mouth position: {mouth}
        - Engaged, animated speaking pose
        - Cartoon style, front facing, dark background
        """

        for i, (name, mouth_desc) in enumerate(mouth_positions[:num_frames]):
            prompt = base_prompt.format(mouth=mouth_desc)

            try:
                if hasattr(generator, 'gemini') and generator.gemini.is_available():
                    image_bytes = await generator.gemini.generate_with_reference(
                        prompt,
                        use_portrait=True
                    )
                else:
                    image_bytes = await generator.generate(prompt)

                if image_bytes:
                    save_path = output_path / f"speaking_{i:02d}_{name}.png"
                    with open(save_path, 'wb') as f:
                        f.write(image_bytes)

                    frames.append({
                        "name": name,
                        "path": str(save_path),
                        "bytes": image_bytes
                    })

                await asyncio.sleep(2)

            except Exception as e:
                logger.error(f"Failed to generate speaking frame {i}: {e}")

        return frames

    def load_avatar_set(self, avatar_dir: Optional[str] = None) -> dict:
        """
        Load a previously generated avatar set from disk.

        Returns dict mapping expression names to numpy arrays (if cv2 available)
        or file paths.
        """
        if avatar_dir is None:
            avatar_dir = str(AVATAR_DIR)

        avatar_path = Path(avatar_dir)
        results = {}

        if not avatar_path.exists():
            logger.warning(f"Avatar directory not found: {avatar_dir}")
            return results

        for img_file in avatar_path.glob("cosmos_*.png"):
            name = img_file.stem.replace("cosmos_", "")

            if HAS_CV2:
                img = cv2.imread(str(img_file), cv2.IMREAD_UNCHANGED)
                if img is not None:
                    results[name] = img
                    logger.debug(f"Loaded avatar: {name}")
            else:
                results[name] = str(img_file)

        logger.info(f"Loaded {len(results)} avatar images")
        return results


# Convenience functions
async def generate_vtuber_avatar(save_path: Optional[str] = None) -> Optional[bytes]:
    """Generate a single VTuber avatar image"""
    generator = VTuberAvatarGenerator()
    return await generator.generate_base_avatar(save_path)


async def generate_full_avatar_set(output_dir: Optional[str] = None) -> dict:
    """Generate a complete set of avatar expressions"""
    generator = VTuberAvatarGenerator()
    return await generator.generate_expression_set(output_dir)


# CLI
if __name__ == "__main__":
    import sys

    async def main():
        generator = VTuberAvatarGenerator()

        if len(sys.argv) > 1 and sys.argv[1] == "--full":
            print("Generating full avatar set...")
            results = await generator.generate_expression_set()
            print(f"Generated {len(results)} expressions")
        else:
            print("Generating base avatar...")
            output_path = str(AVATAR_DIR / "cosmos_base.png")
            result = await generator.generate_base_avatar(output_path)
            if result:
                print(f"Avatar saved to: {output_path}")
            else:
                print("Failed to generate avatar")

    asyncio.run(main())
