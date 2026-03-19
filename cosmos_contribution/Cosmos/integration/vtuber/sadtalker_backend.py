"""
SadTalker Backend - Full face animation for VTuber streaming.

Uses SadTalker's 3DMM pipeline: audio → expression coefficients → full face
rendering with head motion, lip sync, eye blinks, and expressions.

Produces D-ID-level animation quality. Pre-renders frames for each utterance,
then streams them during audio playback.

Architecture:
  1. One-time: detect face, extract 3DMM coefficients, load neural networks
  2. Runtime: audio → mel → Audio2Coeff → AnimateFromCoeff → raw frames
  3. Composite animated crop back onto full-frame background
  4. Frame deque bridges async generation with the render loop
"""

import asyncio
import os
import sys
import tempfile
import time
import threading
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple
from pathlib import Path
from loguru import logger

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class SadTalkerConfig:
    """Configuration for SadTalker backend"""
    sadtalker_dir: str = "/workspace/SadTalker"

    # Face image
    face_image_path: str = ""

    # Output dimensions (stream resolution)
    output_width: int = 854
    output_height: int = 480
    fps: int = 25

    # Inference
    size: int = 256          # 256 or 512, lower = faster
    batch_size: int = 8
    device: str = "cuda:0"
    preprocess: str = "crop"  # "crop", "resize", "full"
    still_mode: bool = True   # Reduce head motion
    exp_scale: float = 1.0    # Expression intensity
    use_blink: bool = True
    pose_style: int = 0       # Head motion style (0=from audio)

    # Frame queue
    max_frames: int = 500


class SadTalkerBackend:
    """Full-face animation backend using SadTalker's 3DMM pipeline."""

    def __init__(self, config: SadTalkerConfig):
        self._config = config
        self._initialized = False
        self._frame_queue: deque = deque(maxlen=config.max_frames)
        self._lock = threading.Lock()

        # SadTalker components (loaded in initialize())
        self._preprocess_model = None
        self._audio_to_coeff = None
        self._animate_from_coeff = None
        self._sadtalker_paths = None

        # Pre-computed from source image
        self._source_image = None      # Full-res background
        self._idle_frame = None        # Static frame when not speaking
        self._crop_info = None         # Crop coordinates from face detection
        self._first_coeff_path = None  # 3DMM coefficients of source face
        self._crop_pic_path = None     # Cropped face image path
        self._orig_img_h = 0           # Original image height (for coordinate scaling)
        self._orig_img_w = 0           # Original image width (for coordinate scaling)
        self._rendering = False        # True while rendering frames
        self._pending_frames: List[np.ndarray] = []  # Buffered during pre-render, released with audio

        # Pre-rendered filler clips: list of (frames, audio_path) tuples
        self._filler_clips: List[Tuple[List[np.ndarray], str]] = []
        self._filler_index = 0

        self._device = config.device

    async def initialize(self) -> bool:
        """Load SadTalker models and preprocess the source face."""
        if not HAS_TORCH or not HAS_CV2:
            logger.error("SadTalker requires torch and cv2")
            return False

        try:
            # Apply compatibility patches
            self._apply_patches()

            # Enable TF32 for Ampere GPUs
            if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                logger.info("TF32 enabled (Ampere GPU detected)")

            # Import SadTalker modules
            sadtalker_dir = self._config.sadtalker_dir
            if sadtalker_dir not in sys.path:
                sys.path.insert(0, sadtalker_dir)

            from src.utils.init_path import init_path
            from src.utils.preprocess import CropAndExtract
            from src.test_audio2coeff import Audio2Coeff
            from src.facerender.animate import AnimateFromCoeff

            checkpoint_path = os.path.join(sadtalker_dir, "checkpoints")
            config_path = os.path.join(sadtalker_dir, "src/config")

            self._sadtalker_paths = init_path(
                checkpoint_path, config_path,
                self._config.size, False, self._config.preprocess
            )
            logger.info(f"SadTalker paths initialized (size={self._config.size})")

            # Load models
            logger.info("Loading SadTalker models...")
            self._preprocess_model = CropAndExtract(self._sadtalker_paths, self._device)
            self._audio_to_coeff = Audio2Coeff(self._sadtalker_paths, self._device)
            self._animate_from_coeff = AnimateFromCoeff(self._sadtalker_paths, self._device)
            logger.info("SadTalker models loaded")

            # Preprocess source face
            success = await self._preprocess_face()
            if not success:
                logger.error("Failed to preprocess source face")
                return False

            self._initialized = True
            logger.info(
                f"SadTalker backend ready  |  size={self._config.size}  |  "
                f"device={self._device}  |  output={self._config.output_width}x{self._config.output_height}"
            )
            return True

        except Exception as e:
            logger.error(f"SadTalker initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _apply_patches(self):
        """Apply compatibility patches for numpy/torchvision."""
        import types

        # Patch torchvision.transforms.functional_tensor (removed in newer torchvision)
        try:
            import torchvision.transforms.functional_tensor
        except ImportError:
            import torchvision.transforms.functional as F
            mod = types.ModuleType('torchvision.transforms.functional_tensor')
            mod.rgb_to_grayscale = F.rgb_to_grayscale
            sys.modules['torchvision.transforms.functional_tensor'] = mod

    async def _preprocess_face(self) -> bool:
        """Detect face and extract 3DMM coefficients from source image."""
        face_path = self._config.face_image_path
        if not face_path or not os.path.exists(face_path):
            logger.error(f"Face image not found: {face_path}")
            return False

        try:
            out_w, out_h = self._config.output_width, self._config.output_height

            # Load and store full-res background
            img = cv2.imread(face_path)
            if img is None:
                logger.error(f"Failed to read image: {face_path}")
                return False
            self._orig_img_h, self._orig_img_w = img.shape[:2]
            self._source_image = cv2.resize(img, (out_w, out_h))
            self._idle_frame = self._source_image.copy()
            logger.info(f"Original image: {self._orig_img_w}x{self._orig_img_h} → output: {out_w}x{out_h}")

            # SadTalker needs the image file, not numpy array
            # Copy to temp location (SadTalker moves the file)
            import shutil
            work_dir = tempfile.mkdtemp(prefix="sadtalker_")
            face_copy = os.path.join(work_dir, "face.png")
            shutil.copy2(face_path, face_copy)

            first_frame_dir = os.path.join(work_dir, "first_frame")
            os.makedirs(first_frame_dir, exist_ok=True)

            # Extract 3DMM coefficients and crop face
            self._first_coeff_path, self._crop_pic_path, self._crop_info = \
                self._preprocess_model.generate(
                    face_copy, first_frame_dir,
                    self._config.preprocess, True, self._config.size
                )

            if self._first_coeff_path is None:
                logger.error("3DMM extraction failed - no face detected")
                return False

            logger.info(
                f"Face preprocessed: crop_info={self._crop_info[0] if self._crop_info else 'None'}, "
                f"coeff_path={self._first_coeff_path}"
            )
            return True

        except Exception as e:
            logger.error(f"Face preprocessing failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def process_audio(
        self,
        audio_data: Optional[np.ndarray] = None,
        sample_rate: int = 16000,
        audio_file: Optional[str] = None,
    ):
        """Generate animated face frames from audio.

        Runs the full SadTalker pipeline:
          audio → mel → Audio2Coeff → AnimateFromCoeff → frames

        Frames are appended to self._frame_queue for streaming.
        """
        if not self._initialized:
            logger.error("SadTalker not initialized")
            return

        # Resolve audio file
        wav_path = None
        tmp_created = False

        if audio_file and os.path.exists(audio_file):
            wav_path = audio_file
        elif audio_data is not None:
            wav_path = tempfile.mktemp(suffix=".wav")
            tmp_created = True
            import soundfile as sf
            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=-1)
            sf.write(wav_path, audio_data, sample_rate)

        if not wav_path or not os.path.exists(wav_path):
            logger.error("No valid audio source")
            return

        try:
            await asyncio.get_event_loop().run_in_executor(
                None, self._process_audio_sync, wav_path
            )
        finally:
            if tmp_created and wav_path and os.path.exists(wav_path):
                try:
                    os.unlink(wav_path)
                except OSError:
                    pass

    def _process_audio_sync(self, wav_path: str):
        """Synchronous audio → frames pipeline (runs in thread)."""
        t0 = time.time()
        self._rendering = True

        try:
            from src.generate_batch import get_data
            from src.generate_facerender_batch import get_facerender_data

            # Step 1: Audio → 3DMM motion coefficients
            save_dir = tempfile.mkdtemp(prefix="sadtalker_gen_")

            batch = get_data(
                self._first_coeff_path, wav_path, self._device,
                ref_eyeblink_coeff_path=None,
                still=self._config.still_mode,
            )

            coeff_path = self._audio_to_coeff.generate(
                batch, save_dir, self._config.pose_style, ref_pose_coeff_path=None,
            )

            # Step 2: Prepare face render data
            data = get_facerender_data(
                coeff_path, self._crop_pic_path,
                self._first_coeff_path, wav_path,
                self._config.batch_size,
                input_yaw_list=None, input_pitch_list=None, input_roll_list=None,
                expression_scale=self._config.exp_scale,
                still_mode=self._config.still_mode,
                preprocess=self._config.preprocess,
                size=self._config.size,
            )

            # Step 3: Render animated face frames
            frames = self._render_frames(data)

            # Step 4: Composite onto full background
            self._composite_frames(frames)

            elapsed = time.time() - t0
            audio_len = len(frames) / self._config.fps
            logger.info(
                f"SadTalker: generated {len(frames)} frames from {audio_len:.1f}s audio "
                f"in {elapsed:.1f}s (RTF={elapsed/max(audio_len,0.1):.1f}x)"
            )

            # Cleanup temp dir
            import shutil
            shutil.rmtree(save_dir, ignore_errors=True)

        except Exception as e:
            logger.error(f"SadTalker audio processing failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._rendering = False

    def _render_frames(self, data: dict) -> List[np.ndarray]:
        """Run the face animation neural network and return raw frames.

        Bypasses SadTalker's video-writing code to get numpy arrays directly.
        """
        from src.facerender.modules.make_animation import make_animation
        from skimage import img_as_ubyte

        source_image = data['source_image'].type(torch.FloatTensor).to(self._device)
        source_semantics = data['source_semantics'].type(torch.FloatTensor).to(self._device)
        target_semantics = data['target_semantics_list'].type(torch.FloatTensor).to(self._device)

        yaw_c_seq = data.get('yaw_c_seq')
        pitch_c_seq = data.get('pitch_c_seq')
        roll_c_seq = data.get('roll_c_seq')
        if yaw_c_seq is not None:
            yaw_c_seq = yaw_c_seq.type(torch.FloatTensor).to(self._device)
        if pitch_c_seq is not None:
            pitch_c_seq = pitch_c_seq.type(torch.FloatTensor).to(self._device)
        if roll_c_seq is not None:
            roll_c_seq = roll_c_seq.type(torch.FloatTensor).to(self._device)

        frame_num = data['frame_num']

        afc = self._animate_from_coeff
        with torch.no_grad():
            predictions = make_animation(
                source_image, source_semantics, target_semantics,
                afc.generator, afc.kp_extractor, afc.he_estimator, afc.mapping,
                yaw_c_seq, pitch_c_seq, roll_c_seq, use_exp=True,
            )

        predictions = predictions.reshape((-1,) + predictions.shape[2:])
        predictions = predictions[:frame_num]

        # Convert tensor → numpy uint8 frames
        frames = []
        for idx in range(predictions.shape[0]):
            image = predictions[idx]
            image = np.transpose(image.data.cpu().numpy(), [1, 2, 0]).astype(np.float32)
            frames.append(image)

        result = img_as_ubyte(frames)

        # Resize to match crop aspect ratio if needed
        original_size = self._crop_info[0] if self._crop_info else None
        if original_size:
            size = self._config.size
            result = [
                cv2.resize(r, (size, int(size * original_size[1] / original_size[0])))
                for r in result
            ]

        return result

    def _composite_frames(self, frames: List[np.ndarray]):
        """Composite animated face crop back onto full-frame background.

        Uses crop_info from preprocessing to place the face at the correct
        position in the output frame. Crop coordinates are in the original
        image space and must be scaled to output resolution.
        """
        if not frames:
            return

        out_w, out_h = self._config.output_width, self._config.output_height

        # Scale factors: original image space → output space
        scale_x = out_w / max(self._orig_img_w, 1)
        scale_y = out_h / max(self._orig_img_h, 1)

        if self._crop_info and len(self._crop_info) >= 3:
            # crop_info format varies:
            #   (original_size, lm, [clx, cly, crx, cry])          - 4 values
            #   (original_size, lm, [rsize, clx, cly, crx, cry])   - 5 values
            # All coordinates are in original image pixel space
            _, _, crop_params = self._crop_info
            if crop_params is not None and len(crop_params) >= 5:
                _, clx, cly, crx, cry = crop_params[:5]
            elif crop_params is not None and len(crop_params) >= 4:
                clx, cly, crx, cry = crop_params[:4]
            else:
                clx, cly = 0, 0
                crx, cry = float(self._orig_img_w), float(self._orig_img_h)

            # Scale from original image space to output space
            clx = int(clx * scale_x)
            cly = int(cly * scale_y)
            crx = int(crx * scale_x)
            cry = int(cry * scale_y)
        else:
            # Fallback: center the face
            fh, fw = frames[0].shape[:2]
            clx = max(0, (out_w - fw) // 2)
            cly = max(0, (out_h - fh) // 2)
            crx = min(out_w, clx + fw)
            cry = min(out_h, cly + fh)

        logger.debug(f"Composite region: ({clx},{cly})-({crx},{cry}) on {out_w}x{out_h}")

        for frame in frames:
            # Create output by pasting animated crop onto background
            output = self._source_image.copy()

            # Resize frame to fit the crop region
            crop_w = crx - clx
            crop_h = cry - cly
            if crop_w > 0 and crop_h > 0:
                # Convert RGB to BGR if needed (SadTalker outputs RGB)
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                resized = cv2.resize(frame_bgr, (crop_w, crop_h), interpolation=cv2.INTER_LANCZOS4)

                # Clamp to output bounds
                dst_y1 = max(0, cly)
                dst_y2 = min(out_h, cry)
                dst_x1 = max(0, clx)
                dst_x2 = min(out_w, crx)

                src_y1 = dst_y1 - cly
                src_y2 = src_y1 + (dst_y2 - dst_y1)
                src_x1 = dst_x1 - clx
                src_x2 = src_x1 + (dst_x2 - dst_x1)

                output[dst_y1:dst_y2, dst_x1:dst_x2] = resized[src_y1:src_y2, src_x1:src_x2]

            self._pending_frames.append(output)

    def release_frames(self):
        """Move buffered frames to the playback queue. Call after audio is queued."""
        for frame in self._pending_frames:
            self._frame_queue.append(frame)
        count = len(self._pending_frames)
        self._pending_frames.clear()
        return count

    @property
    def idle_frame(self) -> Optional[np.ndarray]:
        """Return static face frame for idle periods."""
        return self._idle_frame

    def get_next_frame(self) -> Optional[np.ndarray]:
        """Pop the next animated frame from the queue."""
        if self._frame_queue:
            return self._frame_queue.popleft()
        return None

    @property
    def has_frames(self) -> bool:
        return len(self._frame_queue) > 0

    @property
    def is_rendering(self) -> bool:
        """True while the render thread is generating frames."""
        return self._rendering

    @property
    def frame_count(self) -> int:
        return len(self._frame_queue)

    async def prerender_fillers(self, tts_func) -> int:
        """Pre-render short filler animations for use during processing gaps.

        Args:
            tts_func: async function(text) -> audio_file_path

        Returns number of clips pre-rendered.
        """
        fillers = [
            "Hmm, let me think about that for a moment.",
            "Interesting, very interesting indeed.",
            "Now this is a fascinating topic.",
            "Let me consult with the collective on this.",
            "Oh my, where to begin with this one.",
            "The data is quite revealing here.",
        ]

        rendered = 0
        for text in fillers:
            try:
                audio_path = await tts_func(text)
                if not audio_path or not os.path.exists(audio_path):
                    continue

                # Render frames (stores in self._frame_queue)
                old_queue = list(self._frame_queue)
                self._frame_queue.clear()

                await self.process_audio(audio_file=audio_path)

                clip_frames = list(self._frame_queue)
                self._frame_queue.clear()

                # Restore any frames that were in the queue
                for f in old_queue:
                    self._frame_queue.append(f)

                if clip_frames:
                    self._filler_clips.append((clip_frames, audio_path))
                    rendered += 1
                    logger.info(f"Pre-rendered filler {rendered}: '{text[:30]}...' ({len(clip_frames)} frames)")

            except Exception as e:
                logger.warning(f"Failed to pre-render filler '{text[:20]}': {e}")

        logger.info(f"Pre-rendered {rendered} filler clips")
        return rendered

    def queue_filler(self) -> Optional[str]:
        """Queue a pre-rendered filler clip's frames and return its audio path.

        Returns the audio file path to play, or None if no fillers available.
        """
        if not self._filler_clips:
            return None

        frames, audio_path = self._filler_clips[self._filler_index % len(self._filler_clips)]
        self._filler_index += 1

        for frame in frames:
            self._frame_queue.append(frame)

        logger.info(f"Queued filler clip: {len(frames)} frames")
        return audio_path

    async def cleanup(self):
        """Release GPU memory and resources."""
        self._preprocess_model = None
        self._audio_to_coeff = None
        self._animate_from_coeff = None
        self._frame_queue.clear()
        self._filler_clips.clear()

        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._initialized = False
        logger.info("SadTalker backend cleaned up")
