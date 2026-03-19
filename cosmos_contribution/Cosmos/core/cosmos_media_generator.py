"""
COSMOS Media Generator
======================
Generates video and images using Google Gemini APIs,
with prompts enriched by the Cosmos 54D Transformer and CST Physics.

Flow:
    User Prompt → Cosmos Transformer (54D CST enrichment) → Enhanced Prompt → Gemini → Media

Image Generation: Uses Gemini's NATIVE image generation (gemini-2.0-flash-exp)
    - Works on the FREE tier
    - Returns images inline via generate_content
    - No separate Imagen API needed

Video Generation: Uses Veo (requires GCP billing)
    - veo-2.0-generate-001 (standard)
    - veo-3.1-generate-preview (best quality)
    - Falls back to clear error if billing not enabled

Prompt Enrichment: Uses Gemini via the NEW google-genai SDK
    - Falls back to Ollama if Gemini is rate-limited
    - Falls back to manual CST enrichment if both fail
"""

import asyncio
import base64
import os
import time
import uuid
import logging
from typing import Optional
from pathlib import Path

logger = logging.getLogger("COSMOS_MEDIA")

# φ constant
PHI = 1.618033988749895

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GEMINI SDK DETECTION (NEW google-genai ONLY)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    logger.warning("[MEDIA] google-genai SDK not installed. Run: pip install google-genai")


class CosmosMediaGenerator:
    """
    Generates media (video/image) powered by:
    1. Cosmos 54D Transformer — enriches prompts with CST physics
    2. Gemini Native Image Gen — generates images via generate_content (FREE tier)
    3. Google Veo — generates video (requires GCP billing)
    """

    # Available video models (require GCP billing)
    VIDEO_MODELS = {
        "veo-2": "veo-2.0-generate-001",
        "veo-3.1": "veo-3.1-generate-preview",
        "veo-3.1-fast": "veo-3.1-fast-generate-preview",
    }

    IMAGE_MODEL = "gemini-2.0-flash"

    # Text model for prompt enrichment
    TEXT_MODEL = "gemini-2.5-flash"

    def __init__(self, api_key: str = None, output_dir: str = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.client = None
        self.available = False

        # Output directory for generated media
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            # Default: project's static/generated/ directory
            self.output_dir = Path(__file__).parent.parent / "web" / "static" / "generated"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize
        self._initialize()

    def _initialize(self):
        """Initialize the Gemini client using the NEW google-genai SDK."""
        if not self.api_key:
            logger.warning("[MEDIA] No Gemini API key found. Media generation disabled.")
            return

        if GENAI_AVAILABLE:
            try:
                self.client = genai.Client(api_key=self.api_key)
                self.available = True
                logger.info("[MEDIA] Gemini Media Generator initialized (google-genai SDK)")
            except Exception as e:
                logger.error(f"[MEDIA] Failed to initialize google-genai client: {e}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # CST PROMPT ENRICHMENT (The Cosmos Transformer)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def get_cst_context(self) -> dict:
        """
        Read the current CST state from the SynapticField.
        Returns emotional context, dark matter dynamics, and φ-aesthetic parameters.
        """
        cst = {
            "emotional_state": "NEUTRAL",
            "dark_matter_w": 0.1,
            "geometric_phase_rad": 0.0,
            "informational_mass": 5.0,
            "emeth_weights": {"percussion": 0.33, "strings": 0.33, "brass": 0.34},
            "phi_composition": PHI,
            "quantum_entropy": 0.5,
        }

        # Try to read live state from the CosmoSynapse engine
        try:
            from cosmosynapse.engine.synaptic_field import get_field
            field = get_field()
            if field:
                physics = field.user_physics or {}
                cst_physics = physics.get("cst_physics", {})

                cst["emotional_state"] = cst_physics.get("cst_state", "NEUTRAL")
                cst["geometric_phase_rad"] = cst_physics.get("geometric_phase_rad", 0.0)
                cst["informational_mass"] = physics.get("derived_state", {}).get("informational_mass", 5.0)

                # Dark matter
                dm = field.dark_matter_state if hasattr(field, "dark_matter_state") else {}
                cst["dark_matter_w"] = dm.get("w", 0.1)

                # Quantum entropy
                from Cosmos.core.quantum_bridge import get_quantum_bridge
                qb = get_quantum_bridge()
                if qb:
                    cst["quantum_entropy"] = qb.get_entropy(physics)
        except Exception as e:
            logger.debug(f"[MEDIA] CST context read failed (using defaults): {e}")

        return cst

    async def enhance_prompt(self, raw_prompt: str, media_type: str = "video") -> str:
        """
        Use the Cosmos Transformer (via Gemini text) to enrich the user's prompt
        with CST-aware cinematic/visual directions.

        Uses the NEW google-genai SDK. Falls back to Ollama, then manual enrichment.
        """
        cst = self.get_cst_context()

        # Map emotional state to visual style
        emotion_styles = {
            "JOY": "warm golden lighting, vibrant colors, dynamic movement",
            "SADNESS": "blue-tinted atmosphere, slow motion, rain or mist",
            "ANGER": "red and orange tones, sharp cuts, intense close-ups",
            "FEAR": "dark shadows, flickering light, tight framing",
            "SURPRISE": "bright flash, wide angle, rapid zoom",
            "TRUST": "soft natural light, earth tones, steady camera",
            "ANTICIPATION": "building tension, crescendo movement, dawn breaking",
            "NEUTRAL": "balanced composition, natural colors, cinematic wide shots",
            "CALM": "serene atmosphere, gentle movement, pastel tones",
            "CALIBRATING": "digital glitch aesthetic, circuit patterns, neon glow",
        }
        emotion_style = emotion_styles.get(cst["emotional_state"], emotion_styles["NEUTRAL"])

        # Map dark matter chaos to visual turbulence
        chaos_level = min(1.0, abs(cst["dark_matter_w"]) / 10.0)
        if chaos_level > 0.7:
            chaos_visual = "chaotic swirling particles, fractals, intense energy"
        elif chaos_level > 0.3:
            chaos_visual = "dynamic flowing patterns, moderate visual complexity"
        else:
            chaos_visual = "calm and ordered composition, minimal visual noise"

        # Map Emeth weights to creative direction
        emeth = cst["emeth_weights"]
        if emeth.get("brass", 0) > 0.5:
            creative_dir = "highly creative and experimental visuals"
        elif emeth.get("strings", 0) > 0.5:
            creative_dir = "emotionally resonant and empathetic visuals"
        elif emeth.get("percussion", 0) > 0.5:
            creative_dir = "precise and logically structured composition"
        else:
            creative_dir = "balanced artistic vision"

        # φ-scaled composition hint
        phi_hint = f"Use golden ratio (φ={PHI:.3f}) composition for key elements"

        enrichment_prompt = f"""You are COSMOS, the 54D Cosmic Synapse Transformer.
Your task: Enhance this {media_type} generation prompt with cinematic direction.

ORIGINAL PROMPT: "{raw_prompt}"

CURRENT CST STATE:
- Emotional Resonance: {cst['emotional_state']} → {emotion_style}
- Dark Matter Chaos Level: {chaos_level:.2f} → {chaos_visual}
- Geometric Phase: {cst['geometric_phase_rad']:.2f} rad
- Creative Direction (Emeth): {creative_dir}
- Composition: {phi_hint}
- Quantum Entropy: {cst['quantum_entropy']:.4f}

RULES:
1. Keep the user's core idea intact
2. Add specific cinematic/visual directions based on the CST state
3. Include camera movements, lighting, color palette, and mood
4. Keep the enhanced prompt under 200 words
5. Output ONLY the enhanced prompt, nothing else"""

        # Try 1: Use Gemini via NEW google-genai SDK for enrichment
        if self.client:
            try:
                # Dynamically scale the output depth and creativity based on the current Quantum Entropy
                entropy_val = float(cst.get('quantum_entropy', 0.5))
                max_tokens = int(300 * (1.0 + (entropy_val * 4.0))) # Scale 300 up to 1500 tokens for wildly detailed visual prompts
                temperature = 0.5 + (entropy_val * 0.5) # Scale temp from 0.5 to 1.0 based on chaos
                
                logger.info(f"[QUANTUM-MEDIA] Token Limit: {max_tokens} | Creativity Temp: {temperature:.2f} | Entropy: {entropy_val:.4f}")

                response = await asyncio.to_thread(
                    self.client.models.generate_content,
                    model=self.TEXT_MODEL,
                    contents=enrichment_prompt,
                    config=types.GenerateContentConfig(
                        max_output_tokens=max_tokens,
                        temperature=temperature,
                    )
                )
                if response and response.text:
                    enhanced = response.text.strip()
                    if len(enhanced) > 20:
                        logger.info(f"[MEDIA] Cosmos Transformer enhanced prompt: {enhanced[:80]}...")
                        return enhanced
            except Exception as e:
                logger.warning(f"[MEDIA] Gemini prompt enrichment failed: {e}")

        # Try 2: Use Ollama as fallback for prompt enrichment
        try:
            import ollama
            ollama_response = await asyncio.to_thread(
                ollama.chat,
                model="llama3.1:8b",
                messages=[{"role": "user", "content": enrichment_prompt}],
            )
            enhanced = ollama_response.get("message", {}).get("content", "").strip()
            if enhanced and len(enhanced) > 20:
                logger.info(f"[MEDIA] Ollama enriched prompt: {enhanced[:80]}...")
                return enhanced
        except Exception as e:
            logger.debug(f"[MEDIA] Ollama prompt enrichment failed: {e}")

        # Fallback 3: manual enrichment using CST state
        fallback = f"{raw_prompt}. Style: {emotion_style}. {chaos_visual}. {phi_hint}."
        logger.info(f"[MEDIA] Using fallback enrichment: {fallback[:80]}...")
        return fallback

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # IMAGE GENERATION (Gemini Native — FREE TIER)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def generate_image(
        self,
        prompt: str,
        enhance: bool = True,
    ) -> dict:
        """
        Generate an image natively or using Gemini.

        Args:
            prompt: User's text prompt for image generation
            enhance: Whether to enrich prompt with CST physics

        Returns:
            dict with {success, file_path, file_url, enhanced_prompt, model, error}
        """
        # --- NATIVE 12D FALLBACK OR OVERRIDE ---
        if not self.available or not self.client or "native" in prompt.lower():
            logger.info("[MEDIA] Executing NATIVE 12D Inverse Synthesis for Image...")
            try:
                from Cosmos.core.multimodal.visual_engine import generate_image_from_12d
                from PIL import Image
                
                cst = self.get_cst_context()
                
                # Construct 12D state from CST Phase variables
                state_12d = [0.0] * 12
                state_12d[0] = cst.get("informational_mass", 5.0) / 10.0   # D1 Energy
                state_12d[1] = cst.get("informational_mass", 5.0) / 5.0    # D2 Mass
                state_12d[2] = cst.get("geometric_phase_rad", 0.0)       # D3 Phi
                state_12d[3] = cst.get("dark_matter_w", 0.1)             # D4 Chaos
                state_12d[8] = cst.get("quantum_entropy", 0.5)           # D9 Cosmic
                state_12d[10] = cst.get("quantum_entropy", 0.5) * PHI    # D11 Freq
                
                # Math Synthesis
                img_rgb = generate_image_from_12d(state_12d, width=768, height=768)
                
                # Save
                filename = f"cosmos_native_image_{uuid.uuid4().hex[:8]}.png"
                filepath = self.output_dir / filename
                Image.fromarray(img_rgb).save(filepath)
                file_url = f"/static/generated/{filename}"
                
                return {
                    "success": True,
                    "file_path": str(filepath),
                    "file_url": file_url,
                    "enhanced_prompt": "Synthesized natively via Inverse 12D CST Phase Math.",
                    "original_prompt": prompt,
                    "model": "Cosmos-Native-12D",
                }
            except Exception as e:
                logger.error(f"[MEDIA] Native 12D synthesis failed: {e}")
                if not self.available:
                    return {"success": False, "error": "Both native and cloud generators failed."}

        # Step 1: Cosmos Transformer enrichment
        enhanced_prompt = prompt
        if enhance:
            enhanced_prompt = await self.enhance_prompt(prompt, media_type="image")

        # Step 2: Generate image via Gemini native image generation (with retry for 429)
        max_attempts = 2
        for attempt in range(max_attempts):
            try:
                logger.info(f"[MEDIA] Generating image with {self.IMAGE_MODEL} (attempt {attempt + 1})...")

                response = await asyncio.to_thread(
                    self.client.models.generate_content,
                    model=self.IMAGE_MODEL,
                    contents=enhanced_prompt,
                    config=types.GenerateContentConfig(
                        response_modalities=["IMAGE", "TEXT"],
                    )
                )

                # Extract image from response parts
                if response and response.candidates:
                    for part in response.candidates[0].content.parts:
                        if part.inline_data is not None:
                            # Save the image
                            filename = f"cosmos_image_{uuid.uuid4().hex[:8]}.png"
                            filepath = self.output_dir / filename

                            # Decode and save
                            image_data = part.inline_data.data
                            if isinstance(image_data, str):
                                image_data = base64.b64decode(image_data)

                            with open(filepath, "wb") as f:
                                f.write(image_data)

                            file_url = f"/static/generated/{filename}"
                            logger.info(f"[MEDIA] Image saved: {filepath}")

                            return {
                                "success": True,
                                "file_path": str(filepath),
                                "file_url": file_url,
                                "enhanced_prompt": enhanced_prompt,
                                "original_prompt": prompt,
                                "model": self.IMAGE_MODEL,
                            }

                return {"success": False, "error": "No image data in response. The model may have declined the prompt.", "enhanced_prompt": enhanced_prompt}

            except Exception as e:
                error_msg = str(e)
                logger.error(f"[MEDIA] Image generation failed (attempt {attempt + 1}): {error_msg}")

                # Retry on rate limit with delay
                if ("429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg) and attempt < max_attempts - 1:
                    retry_delay = 20
                    logger.info(f"[MEDIA] Rate limited. Retrying in {retry_delay}s...")
                    await asyncio.sleep(retry_delay)
                    continue

                # Final attempt — return helpful error
                if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                    error_msg = "Rate limited by Gemini API — please wait a minute and try again."
                elif "SAFETY" in error_msg.upper():
                    error_msg = "The prompt was declined by safety filters. Try rephrasing."

                return {"success": False, "error": error_msg, "enhanced_prompt": enhanced_prompt}

        return {"success": False, "error": "Image generation failed after retries.", "enhanced_prompt": enhanced_prompt}

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # VIDEO GENERATION (Veo — requires GCP billing)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def generate_video(
        self,
        prompt: str,
        model: str = "veo-2",
        enhance: bool = True,
        timeout: int = 300,
    ) -> dict:
        """
        Generate a video natively or using Gemini Veo.

        Args:
            prompt: User's text prompt for video generation
            model: Which Veo model to use (veo-2, veo-3.1, veo-3.1-fast)
            enhance: Whether to enrich prompt with CST physics
            timeout: Max wait time in seconds

        Returns:
            dict with {success, file_path, file_url, enhanced_prompt, model, error}
        """
        # --- NATIVE 54D FALLBACK OR OVERRIDE ---
        if not self.available or not self.client or "native" in prompt.lower():
            logger.info("[MEDIA] Executing NATIVE 54D Inverse Synthesis for Video...")
            try:
                from Cosmos.core.multimodal.visual_engine import generate_video_from_54d
                import imageio
                import numpy as np
                
                cst = self.get_cst_context()
                state_54d = [0.0] * 54
                state_54d[0] = cst.get("informational_mass", 5.0) / 10.0   # D1
                state_54d[1] = cst.get("informational_mass", 5.0) / 5.0    # D2
                state_54d[2] = cst.get("geometric_phase_rad", 0.0)       # D3
                state_54d[3] = cst.get("dark_matter_w", 0.1)             # D4
                state_54d[8] = cst.get("quantum_entropy", 0.5)           # D9
                state_54d[10] = cst.get("quantum_entropy", 0.5) * PHI    # D11
                
                emeth = cst.get("emeth_weights", {})
                state_54d[36] = emeth.get("percussion", 0.33) * 10.0
                state_54d[37] = emeth.get("strings", 0.33) * 10.0
                state_54d[38] = emeth.get("brass", 0.34) * 10.0
                
                # Math Synthesis (60 frames)
                frames_rgb = generate_video_from_54d(state_54d, frames=45, width=512, height=512)
                
                # Save as MP4
                filename = f"cosmos_native_video_{uuid.uuid4().hex[:8]}.mp4"
                filepath = self.output_dir / filename
                
                # imageio requires ffmpeg or similar
                imageio.mimwrite(str(filepath), frames_rgb, fps=15, codec='libx264')
                
                file_url = f"/static/generated/{filename}"
                return {
                    "success": True,
                    "file_path": str(filepath),
                    "file_url": file_url,
                    "enhanced_prompt": "Synthesized natively via Inverse 54D Lorenz Chaos Math.",
                    "original_prompt": prompt,
                    "model": "Cosmos-Native-54D",
                }
            except Exception as e:
                logger.error(f"[MEDIA] Native 54D video synthesis failed: {e}")
                if not self.available:
                    return {"success": False, "error": "Both native and cloud generators failed."}

        model_name = self.VIDEO_MODELS.get(model, self.VIDEO_MODELS["veo-2"])

        # Step 1: Cosmos Transformer enrichment
        enhanced_prompt = prompt
        if enhance:
            enhanced_prompt = await self.enhance_prompt(prompt, media_type="video")

        # Step 2: Generate video via Veo API
        try:
            logger.info(f"[MEDIA] Generating video with {model_name}...")
            logger.info(f"[MEDIA] Enhanced prompt: {enhanced_prompt[:100]}...")

            # Submit generation job
            operation = await asyncio.to_thread(
                self.client.models.generate_videos,
                model=model_name,
                prompt=enhanced_prompt,
            )

            # Step 3: Poll for completion
            start_time = time.time()
            while not operation.done:
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    return {"success": False, "error": f"Video generation timed out after {timeout}s"}

                logger.info(f"[MEDIA] Video generating... ({elapsed:.0f}s elapsed)")
                await asyncio.sleep(10)
                operation = await asyncio.to_thread(
                    self.client.operations.get, operation
                )

            # Step 4: Download and save
            generated_video = operation.response.generated_videos[0]

            # Download the video file
            await asyncio.to_thread(
                self.client.files.download,
                file=generated_video.video
            )

            # Save with unique filename
            filename = f"cosmos_video_{uuid.uuid4().hex[:8]}.mp4"
            filepath = self.output_dir / filename
            await asyncio.to_thread(
                generated_video.video.save, str(filepath)
            )

            file_url = f"/static/generated/{filename}"
            logger.info(f"[MEDIA] Video saved: {filepath}")

            return {
                "success": True,
                "file_path": str(filepath),
                "file_url": file_url,
                "enhanced_prompt": enhanced_prompt,
                "original_prompt": prompt,
                "model": model_name,
            }

        except Exception as e:
            error_msg = str(e)
            logger.error(f"[MEDIA] Video generation failed: {error_msg}")

            # Provide helpful error messages
            if "FAILED_PRECONDITION" in error_msg or "billing" in error_msg.lower():
                error_msg = ("Video generation (Veo) requires Google Cloud Platform billing. "
                             "Enable billing at https://console.cloud.google.com/billing "
                             "or use Image Generation (free) instead.")
            elif "429" in error_msg:
                error_msg = "Rate limited — please wait and try again."

            return {"success": False, "error": error_msg, "enhanced_prompt": enhanced_prompt}

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # STATUS & INFO
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def get_status(self) -> dict:
        """Get current media generator status."""
        return {
            "available": self.available,
            "api_key_set": bool(self.api_key),
            "genai_sdk": GENAI_AVAILABLE,
            "image_model": self.IMAGE_MODEL,
            "image_note": "FREE tier — uses Gemini native image gen",
            "video_models": list(self.VIDEO_MODELS.keys()),
            "video_note": "Requires GCP billing enabled",
            "output_dir": str(self.output_dir),
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SINGLETON
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_media_generator: Optional[CosmosMediaGenerator] = None


def get_media_generator() -> CosmosMediaGenerator:
    """Get or create the global media generator."""
    global _media_generator
    if _media_generator is None:
        _media_generator = CosmosMediaGenerator()
    return _media_generator
