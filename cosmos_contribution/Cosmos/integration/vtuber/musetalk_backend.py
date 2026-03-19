"""
MuseTalk Backend - Real-time neural lip sync for VTuber streaming.

Wraps TMElyralab/MuseTalk for photorealistic audio-driven face animation.
Single-step latent inpainting at 30+ FPS on V100. ~4GB VRAM.

Architecture:
  1. One-time: detect face, crop 256x256, VAE encode, generate blend masks
  2. Runtime: audio -> Whisper features -> UNet (single step) -> VAE decode -> composite
  3. Frame deque bridges async generation with the render loop

Requires MuseTalk cloned at MUSETALK_DIR with weights downloaded.
"""

import asyncio
import math
import os
import sys
import tempfile
import time
import threading
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
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
class MuseTalkConfig:
    """Configuration for MuseTalk backend"""
    # MuseTalk installation
    musetalk_dir: str = "/workspace/MuseTalk"
    model_version: str = "v15"  # "v10" or "v15"

    # Face image
    face_image_path: str = ""

    # Output dimensions (stream resolution)
    output_width: int = 854
    output_height: int = 480
    fps: int = 25

    # Inference
    batch_size: int = 16
    device: str = "cuda:0"
    use_float16: bool = True

    # Blending (v1.5)
    left_cheek_width: int = 90
    right_cheek_width: int = 90
    extra_margin: int = 10
    bbox_shift: int = 0

    # Proxy face mode: use a clean human face for MuseTalk, transfer mouth to cyborg
    proxy_face_path: str = ""  # If set, this face is used for MuseTalk inference

    # Frame buffer
    max_frame_buffer: int = 200


class MuseTalkBackend:
    """
    MuseTalk-powered avatar backend for photorealistic lip sync.

    Usage::

        config = MuseTalkConfig(face_image_path="/path/to/face.png")
        backend = MuseTalkBackend(config)
        await backend.initialize()

        # When speaking (call from background thread or executor):
        await backend.process_audio(audio_numpy_16khz)

        # In render loop (called every frame):
        frame = backend.get_frame()  # pops from queue or returns idle frame
    """

    def __init__(self, config: MuseTalkConfig):
        self._config = config

        # Models (loaded on init)
        self._vae = None
        self._unet = None
        self._pe = None
        self._whisper = None
        self._audio_processor = None
        self._face_parser = None
        self._device = None
        self._weight_dtype = None
        self._timesteps = None

        # MuseTalk module references
        self._load_all_model = None
        self._datagen = None
        self._get_landmark_and_bbox = None
        self._get_image_prepare_material = None
        self._get_image_blending = None

        # Preprocessed avatar data
        self._source_image = None       # Full image at output resolution
        self._input_latents = []        # VAE-encoded face crops
        self._coord_list = []           # Bounding box coords [(x1,y1,x2,y2), ...]
        self._frame_list = []           # Original frames for compositing
        self._mask_list = []            # Blending masks
        self._mask_coords_list = []     # Mask crop coordinates

        # Proxy face mode state
        self._proxy_mode = False
        self._cyborg_crop_256: Optional[np.ndarray] = None  # Original cyborg 256x256
        self._proxy_ref_256: Optional[np.ndarray] = None    # Proxy reference 256x256
        self._mouth_roi = None        # (x, y, w, h) in 256x256 space
        self._mouth_blend_mask = None  # Feathered alpha mask for mouth

        # Frame output
        self._frame_queue: deque = deque(maxlen=config.max_frame_buffer)
        self._idle_frame: Optional[np.ndarray] = None
        self._is_processing = False
        self._initialized = False

        # Stats
        self._total_frames_generated = 0
        self._total_audio_processed = 0.0

    async def initialize(self) -> bool:
        """Load MuseTalk models and preprocess source face image."""
        if not HAS_TORCH:
            logger.error("PyTorch required for MuseTalk backend")
            return False
        if not HAS_CV2:
            logger.error("OpenCV required for MuseTalk backend")
            return False

        try:
            # Add MuseTalk to sys.path
            mdir = self._config.musetalk_dir
            if not os.path.isdir(mdir):
                logger.error(f"MuseTalk directory not found: {mdir}")
                return False
            if mdir not in sys.path:
                sys.path.insert(0, mdir)

            # Import MuseTalk modules
            if not self._import_musetalk():
                return False

            self._device = torch.device(self._config.device)

            # Enable TensorFloat32 for Ampere+ GPUs (A40, A100, etc.)
            if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                logger.info("TF32 enabled (Ampere GPU detected)")

            # Load VAE + UNet + PositionalEncoding
            # Must CWD to MuseTalk dir because load_all_model uses relative paths
            logger.info("Loading MuseTalk models...")
            prev_cwd = os.getcwd()
            os.chdir(mdir)

            try:
                models_dir = os.path.join(mdir, "models")

                if self._config.model_version == "v15":
                    unet_path = os.path.join(models_dir, "musetalkV15", "unet.pth")
                    unet_cfg = os.path.join(models_dir, "musetalkV15", "musetalk.json")
                else:
                    unet_path = os.path.join(models_dir, "musetalk", "pytorch_model.bin")
                    unet_cfg = os.path.join(models_dir, "musetalk", "musetalk.json")

                if not os.path.exists(unet_path):
                    logger.error(f"UNet weights not found: {unet_path}")
                    return False

                self._vae, self._unet, self._pe = self._load_all_model(
                    unet_model_path=unet_path,
                    vae_type="sd-vae-ft-mse",
                    unet_config=unet_cfg,
                    device=self._device,
                )

                # FP16 for speed
                self._timesteps = torch.tensor([0], device=self._device)
                self._pe = self._pe.half().to(self._device)
                self._vae.vae = self._vae.vae.half().to(self._device)
                self._unet.model = self._unet.model.half().to(self._device)
                self._weight_dtype = self._unet.model.dtype

                # Whisper (audio feature extraction)
                whisper_path = os.path.join(models_dir, "whisper")
                from musetalk.utils.audio_processor import AudioProcessor
                self._audio_processor = AudioProcessor(
                    feature_extractor_path=whisper_path
                )

                from transformers import WhisperModel
                self._whisper = WhisperModel.from_pretrained(whisper_path)
                self._whisper = self._whisper.to(
                    device=self._device, dtype=self._weight_dtype
                ).eval()
                self._whisper.requires_grad_(False)

                # Face parser (for blending masks) - also uses relative paths
                from musetalk.utils.face_parsing import FaceParsing
                self._face_parser = FaceParsing(
                    left_cheek_width=self._config.left_cheek_width,
                    right_cheek_width=self._config.right_cheek_width,
                )

                logger.info("MuseTalk models loaded successfully")
            finally:
                os.chdir(prev_cwd)

            # Preprocess source face
            if not self._config.face_image_path:
                logger.error("No face_image_path provided")
                return False

            # Proxy face mode: use clean human face for MuseTalk,
            # transfer mouth to cyborg for display
            if self._config.proxy_face_path and os.path.exists(self._config.proxy_face_path):
                logger.info(f"PROXY MODE: MuseTalk on proxy, mouth transfer to cyborg")
                success = await self._init_proxy_mode(
                    self._config.proxy_face_path,
                    self._config.face_image_path,
                )
            else:
                success = await self._preprocess_face(self._config.face_image_path)

            if not success:
                return False

            self._initialized = True
            mode_str = "proxy" if self._proxy_mode else "direct"
            logger.info(
                f"MuseTalk backend ready  |  "
                f"version={self._config.model_version}  |  "
                f"device={self._device}  |  "
                f"output={self._config.output_width}x{self._config.output_height}  |  "
                f"mode={mode_str}"
            )
            return True

        except Exception as e:
            logger.error(f"MuseTalk initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _import_musetalk(self) -> bool:
        """Import MuseTalk modules. Returns False if missing.

        We skip musetalk.utils.preprocessing (requires mmpose/DWPose)
        and use face_alignment directly for face bbox detection.
        """
        try:
            from musetalk.utils.utils import load_all_model, datagen
            from musetalk.utils.blending import (
                get_image_prepare_material,
                get_image_blending,
            )

            self._load_all_model = load_all_model
            self._datagen = datagen
            self._get_image_prepare_material = get_image_prepare_material
            self._get_image_blending = get_image_blending

            logger.info("MuseTalk modules imported successfully")
            return True

        except ImportError as e:
            logger.error(
                f"MuseTalk not installed at {self._config.musetalk_dir}: {e}\n"
                f"Run: scripts/setup_musetalk.sh to install"
            )
            return False

    async def _preprocess_face(self, image_path: str) -> bool:
        """Detect face, crop, VAE-encode, and generate blending masks.

        Uses face_alignment for face detection (no mmpose/DWPose needed).
        """
        try:
            logger.info(f"Preprocessing face: {image_path}")

            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"Cannot read image: {image_path}")
                return False

            # Resize to output resolution
            output_img = cv2.resize(
                img, (self._config.output_width, self._config.output_height)
            )
            self._source_image = output_img.copy()
            self._idle_frame = output_img.copy()

            # Detect face bbox using face_alignment
            face_box = self._detect_face_bbox(output_img)
            if face_box is None:
                logger.error("Face detection failed - no face found in image")
                return False

            x1, y1, x2, y2 = face_box
            logger.info(f"Face detected: bbox=({x1},{y1},{x2},{y2})")

            self._coord_list = [face_box]
            self._frame_list = [output_img]

            # Crop face and encode with VAE
            crop = output_img[y1:y2, x1:x2]
            if crop.size == 0:
                logger.error(f"Empty face crop: bbox={face_box}")
                return False
            crop_256 = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_LANCZOS4)

            latent = self._vae.get_latents_for_unet(crop_256)
            self._input_latents = [latent]

            # Generate blending mask
            # Try BiSeNet first, fall back to elliptical mask for non-standard faces
            try:
                mask, mask_coord = self._get_image_prepare_material(
                    output_img, face_box, fp=self._face_parser
                )
                # Verify mask covers enough area (BiSeNet may fail on cyborg faces)
                mask_coverage = np.count_nonzero(mask) / mask.size
                if mask_coverage < 0.25:
                    logger.warning(
                        f"BiSeNet mask too sparse ({mask_coverage:.1%}), "
                        "using elliptical mask instead"
                    )
                    mask, mask_coord = self._create_elliptical_mask(face_box, output_img.shape)
                else:
                    logger.info(f"BiSeNet mask coverage: {mask_coverage:.1%}")
            except Exception as e:
                logger.warning(f"BiSeNet mask failed ({e}), using elliptical mask")
                mask, mask_coord = self._create_elliptical_mask(face_box, output_img.shape)

            self._mask_list = [mask]
            self._mask_coords_list = [mask_coord]

            logger.info(
                f"Face preprocessed: bbox=({x1},{y1},{x2},{y2}), "
                f"crop={crop.shape}, latent cached"
            )
            return True

        except Exception as e:
            logger.error(f"Face preprocessing failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def _init_proxy_mode(self, proxy_path: str, cyborg_path: str) -> bool:
        """Set up proxy face mode.

        - Preprocesses PROXY face for MuseTalk inference (bbox, VAE, etc.)
        - Loads CYBORG face for display
        - Gets 68-point landmarks for both to compute mouth transfer
        """
        try:
            import face_alignment as fa_lib

            out_w, out_h = self._config.output_width, self._config.output_height

            # Load and resize both images to output resolution
            cyborg_img = cv2.imread(cyborg_path)
            proxy_img = cv2.imread(proxy_path)
            if cyborg_img is None or proxy_img is None:
                logger.error("Failed to read proxy or cyborg image")
                return False

            cyborg_resized = cv2.resize(cyborg_img, (out_w, out_h))
            proxy_resized = cv2.resize(proxy_img, (out_w, out_h))

            # Store cyborg as the display frame
            self._source_image = cyborg_resized.copy()
            self._idle_frame = cyborg_resized.copy()

            # Get landmarks for proxy face
            fa = fa_lib.FaceAlignment(fa_lib.LandmarksType.TWO_D, device="cuda")

            proxy_lm = fa.get_landmarks(proxy_resized)
            if proxy_lm is None or len(proxy_lm) == 0:
                logger.error("No face landmarks found on proxy image")
                del fa
                return False
            proxy_lm = proxy_lm[0]  # 68 landmarks

            cyborg_lm = fa.get_landmarks(cyborg_resized)
            if cyborg_lm is None or len(cyborg_lm) == 0:
                logger.warning("No landmarks on cyborg, using proxy landmarks")
                cyborg_lm = proxy_lm
            else:
                cyborg_lm = cyborg_lm[0]
            del fa

            # Compute proxy face bbox using landmarks (nose-centered crop)
            nose_y = proxy_lm[30][1]
            chin_y = proxy_lm[8][1]
            brow_y = proxy_lm[19][1]
            face_cx = (proxy_lm[0][0] + proxy_lm[16][0]) / 2
            face_w = proxy_lm[16][0] - proxy_lm[0][0]

            crop_h = (chin_y - brow_y) * 1.3
            crop_top = nose_y - crop_h * 0.4
            crop_w = face_w * 1.3
            x1 = max(0, int(face_cx - crop_w / 2))
            y1 = max(0, int(crop_top))
            x2 = min(out_w, int(face_cx + crop_w / 2))
            y2 = min(out_h, int(crop_top + crop_h))

            proxy_bbox = (x1, y1, x2, y2)
            logger.info(f"Proxy face bbox: {proxy_bbox}")

            # Preprocess proxy face for MuseTalk (VAE encode, masks)
            proxy_crop = proxy_resized[y1:y2, x1:x2]
            proxy_256 = cv2.resize(proxy_crop, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            self._proxy_ref_256 = proxy_256.copy()

            latent = self._vae.get_latents_for_unet(proxy_256)
            self._input_latents = [latent]
            self._coord_list = [proxy_bbox]
            self._frame_list = [proxy_resized]  # proxy for MuseTalk compositing

            # Use elliptical mask for proxy (it's a clean face, but simpler)
            mask, mask_coord = self._create_elliptical_mask(proxy_bbox, proxy_resized.shape)
            self._mask_list = [mask]
            self._mask_coords_list = [mask_coord]

            # --- Mouth transfer setup ---
            # Mouth region in 256x256 proxy crop
            # Proxy mouth landmarks relative to proxy bbox
            proxy_mouth_pts = proxy_lm[48:68]  # 20 mouth landmarks
            mouth_x_min = proxy_mouth_pts[:, 0].min()
            mouth_x_max = proxy_mouth_pts[:, 0].max()
            mouth_y_min = proxy_mouth_pts[:, 1].min()
            mouth_y_max = proxy_mouth_pts[:, 1].max()

            # Convert to 256x256 space
            scale_x = 256.0 / (x2 - x1)
            scale_y = 256.0 / (y2 - y1)
            mx1 = int((mouth_x_min - x1) * scale_x)
            my1 = int((mouth_y_min - y1) * scale_y)
            mx2 = int((mouth_x_max - x1) * scale_x)
            my2 = int((mouth_y_max - y1) * scale_y)

            # Expand mouth ROI with padding
            pad_x = int((mx2 - mx1) * 0.35)
            pad_y = int((my2 - my1) * 0.5)
            mx1 = max(0, mx1 - pad_x)
            my1 = max(0, my1 - pad_y)
            mx2 = min(256, mx2 + pad_x)
            my2 = min(256, my2 + pad_y)
            self._mouth_roi = (mx1, my1, mx2 - mx1, my2 - my1)

            # Compute CYBORG face bbox from cyborg landmarks
            c_nose_y = cyborg_lm[30][1]
            c_chin_y = cyborg_lm[8][1]
            c_brow_y = cyborg_lm[19][1]
            c_face_cx = (cyborg_lm[0][0] + cyborg_lm[16][0]) / 2
            c_face_w = cyborg_lm[16][0] - cyborg_lm[0][0]

            c_crop_h = (c_chin_y - c_brow_y) * 1.3
            c_crop_top = c_nose_y - c_crop_h * 0.4
            c_crop_w = c_face_w * 1.3
            cx1 = max(0, int(c_face_cx - c_crop_w / 2))
            cy1 = max(0, int(c_crop_top))
            cx2 = min(out_w, int(c_face_cx + c_crop_w / 2))
            cy2 = min(out_h, int(c_crop_top + c_crop_h))
            self._cyborg_bbox = (cx1, cy1, cx2, cy2)
            logger.info(f"Cyborg face bbox: {self._cyborg_bbox}")

            # Pre-compute cyborg crop at ITS OWN bbox
            cyborg_crop = cyborg_resized[cy1:cy2, cx1:cx2]
            self._cyborg_crop_256 = cv2.resize(
                cyborg_crop, (256, 256), interpolation=cv2.INTER_LANCZOS4
            )

            # Cyborg mouth ROI in cyborg 256x256 space
            # Use cyborg landmarks for CENTER, but scale to proxy proportions
            # so the animated mouth has enough room to display lip movement
            cyborg_mouth = cyborg_lm[48:68]
            c_scale_x = 256.0 / (cx2 - cx1)
            c_scale_y = 256.0 / (cy2 - cy1)
            cyborg_mouth_cx = ((cyborg_mouth[:, 0].min() + cyborg_mouth[:, 0].max()) / 2 - cx1) * c_scale_x
            cyborg_mouth_cy = ((cyborg_mouth[:, 1].min() + cyborg_mouth[:, 1].max()) / 2 - cy1) * c_scale_y

            # Use proxy mouth ROI dimensions (already padded) for target size
            # Scale by ratio of face sizes to preserve proportionality
            proxy_face_w = x2 - x1
            cyborg_face_w = cx2 - cx1
            face_ratio = cyborg_face_w / max(proxy_face_w, 1)
            target_mw = int((mx2 - mx1) * face_ratio)
            target_mh = int((my2 - my1) * face_ratio)

            cmx1 = max(0, int(cyborg_mouth_cx - target_mw / 2))
            cmy1 = max(0, int(cyborg_mouth_cy - target_mh / 2))
            cmx2 = min(256, cmx1 + target_mw)
            cmy2 = min(256, cmy1 + target_mh)
            self._cyborg_mouth_roi = (cmx1, cmy1, cmx2 - cmx1, cmy2 - cmy1)
            logger.info(
                f"Mouth ROI sizing: proxy={self._mouth_roi}, "
                f"cyborg={self._cyborg_mouth_roi}, face_ratio={face_ratio:.2f}"
            )

            # Create blend mask: large opaque core with thin feathered edges
            # This ensures the animated mouth is clearly visible
            mw, mh = cmx2 - cmx1, cmy2 - cmy1
            mouth_mask = np.zeros((mh, mw), dtype=np.float32)
            # Fill large ellipse at full opacity (85% of area)
            cv2.ellipse(
                mouth_mask,
                (mw // 2, mh // 2),
                (int(mw * 0.48), int(mh * 0.48)),
                0, 0, 360, 1.0, -1,
            )
            # Thin feather edge (10% of smallest dimension)
            ksize = max(3, int(min(mw, mh) * 0.12) | 1)
            mouth_mask = cv2.GaussianBlur(mouth_mask, (ksize, ksize), 0)
            self._mouth_blend_mask = mouth_mask

            self._proxy_mode = True
            logger.info(
                f"Proxy mode ready: proxy_mouth_roi={self._mouth_roi}, "
                f"cyborg_mouth_roi={self._cyborg_mouth_roi}, "
                f"cyborg_bbox={self._cyborg_bbox}"
            )
            return True

        except Exception as e:
            logger.error(f"Proxy mode init failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _transfer_mouth(self, generated_256: np.ndarray) -> np.ndarray:
        """Transfer mouth from proxy to cyborg using Poisson seamless cloning.

        Uses cv2.seamlessClone (MIXED_CLONE) for automatic gradient-domain
        blending that preserves the cyborg's skin texture at boundaries
        while transferring the proxy's lip movement.
        """
        # Extract animated proxy mouth
        pmx, pmy, pmw, pmh = self._mouth_roi
        proxy_mouth = generated_256[pmy:pmy + pmh, pmx:pmx + pmw]

        # Target location on cyborg
        cmx, cmy, cmw, cmh = self._cyborg_mouth_roi
        cyborg = self._cyborg_crop_256.copy()

        # Resize proxy mouth to cyborg mouth size
        proxy_resized = cv2.resize(
            proxy_mouth, (cmw, cmh), interpolation=cv2.INTER_LANCZOS4
        )

        # Create binary elliptical mask for seamlessClone (needs uint8, 255=include)
        clone_mask = np.zeros((cmh, cmw), dtype=np.uint8)
        cv2.ellipse(
            clone_mask,
            (cmw // 2, cmh // 2),
            (int(cmw * 0.45), int(cmh * 0.45)),
            0, 0, 360, 255, -1,
        )

        # Center point for seamless clone (in cyborg crop coords)
        center = (cmx + cmw // 2, cmy + cmh // 2)

        # MIXED_CLONE preserves gradients from BOTH source and target
        # This keeps cyborg skin texture while adding proxy lip movements
        try:
            result = cv2.seamlessClone(
                proxy_resized, cyborg, clone_mask, center, cv2.MIXED_CLONE
            )
            return result
        except cv2.error:
            # Fallback: direct alpha blend if seamlessClone fails
            mask = self._mouth_blend_mask
            if mask.shape[:2] != (cmh, cmw):
                mask = cv2.resize(mask, (cmw, cmh))
            mask_3ch = mask[:, :, np.newaxis]
            cyborg_region = cyborg[cmy:cmy + cmh, cmx:cmx + cmw].astype(np.float32)
            blended = proxy_resized.astype(np.float32) * mask_3ch + cyborg_region * (1.0 - mask_3ch)
            cyborg[cmy:cmy + cmh, cmx:cmx + cmw] = np.clip(blended, 0, 255).astype(np.uint8)
            return cyborg

    @staticmethod
    def _color_transfer(source: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Lab color transfer: match hue/saturation to target, keep luminance from source.

        This preserves the lip shape detail (luminance variation from MuseTalk)
        while matching the skin tone of the cyborg face (hue/saturation).
        """
        src_lab = cv2.cvtColor(source, cv2.COLOR_BGR2Lab).astype(np.float32)
        tgt_lab = cv2.cvtColor(target, cv2.COLOR_BGR2Lab).astype(np.float32)

        # L channel: mostly preserve source luminance (lip shape detail)
        # Shift brightness to match target, but keep source contrast
        s_l_mean, s_l_std = src_lab[:, :, 0].mean(), max(src_lab[:, :, 0].std(), 1e-6)
        t_l_mean, t_l_std = tgt_lab[:, :, 0].mean(), max(tgt_lab[:, :, 0].std(), 1e-6)
        # Normalize, keep 85% source contrast, shift to 60/40 mean
        norm_l = (src_lab[:, :, 0] - s_l_mean) / s_l_std
        mixed_std = s_l_std * 0.85 + t_l_std * 0.15
        mixed_mean = s_l_mean * 0.4 + t_l_mean * 0.6
        src_lab[:, :, 0] = norm_l * mixed_std + mixed_mean

        # a/b channels: fully match target (transfers skin tone/hue)
        for c in [1, 2]:
            s_mean = src_lab[:, :, c].mean()
            s_std = max(src_lab[:, :, c].std(), 1e-6)
            t_mean = tgt_lab[:, :, c].mean()
            t_std = max(tgt_lab[:, :, c].std(), 1e-6)
            src_lab[:, :, c] = (src_lab[:, :, c] - s_mean) * (t_std / s_std) + t_mean

        src_lab = np.clip(src_lab, 0, 255).astype(np.uint8)
        return cv2.cvtColor(src_lab, cv2.COLOR_Lab2BGR)

    def _detect_face_bbox(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect face bounding box using face_alignment.

        Returns (x1, y1, x2, y2) or None if no face found.
        Falls back to manual ROI or center estimate for cyborg faces.
        """
        h, w = image.shape[:2]

        # Try face_alignment
        try:
            from face_detection import FaceAlignment, LandmarksType
            fa = FaceAlignment(LandmarksType._2D, flip_input=False, device="cuda")
            detections = fa.get_detections_for_batch(np.array([image]))
            if detections and detections[0] is not None:
                det = detections[0]
                x1, y1, x2, y2 = int(det[0]), int(det[1]), int(det[2]), int(det[3])
                # MuseTalk crops lower face for lip inpainting
                # Keep more of the face to ensure lips are captured
                face_h = y2 - y1
                y1_adjusted = y1 + int(face_h * 0.15)
                # Clamp
                x1 = max(0, x1)
                y1_adjusted = max(0, y1_adjusted)
                x2 = min(w, x2)
                y2 = min(h, y2 + self._config.extra_margin)
                logger.info(f"face_alignment detection: ({x1},{y1_adjusted},{x2},{y2})")
                return (x1, y1_adjusted, x2, y2)
        except Exception as e:
            logger.warning(f"face_alignment detection failed: {e}")

        # Fallback: center estimate for cyborg face
        # Assumes face occupies roughly the center 60% of the image
        cx, cy = w // 2, int(h * 0.5)
        fw, fh = int(w * 0.5), int(h * 0.5)
        x1 = max(0, cx - fw // 2)
        y1 = max(0, cy - fh // 4)  # upper boundary ~40% down
        x2 = min(w, cx + fw // 2)
        y2 = min(h, cy + fh // 2)
        logger.info(f"Using center-estimate face bbox: ({x1},{y1},{x2},{y2})")
        return (x1, y1, x2, y2)

    def _create_elliptical_mask(
        self,
        face_box: Tuple[int, int, int, int],
        img_shape: Tuple[int, ...],
        expand: float = 1.5,
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """Create a soft elliptical blending mask matching MuseTalk's format.

        MuseTalk's get_image_blending expects:
        - 2D uint8 mask (H, W), values 0-255
        - Sized to the expanded crop_box (1.5x face bbox)
        - crop_box coordinates as (x_s, y_s, x_e, y_e)

        Used when BiSeNet face parser fails (e.g. on cyborg/non-standard faces).
        """
        x, y, x1, y1 = face_box
        h, w = img_shape[:2]

        # Compute expanded crop box (same as MuseTalk's get_crop_box)
        x_c, y_c = (x + x1) // 2, (y + y1) // 2
        bw, bh = x1 - x, y1 - y
        s = int(max(bw, bh) // 2 * expand)
        crop_box = (x_c - s, y_c - s, x_c + s, y_c + s)
        x_s, y_s, x_e, y_e = crop_box
        cw, ch = x_e - x_s, y_e - y_s

        # Create mask sized to the crop_box
        mask = np.zeros((ch, cw), dtype=np.uint8)

        # Face position within the crop
        fx1, fy1 = x - x_s, y - y_s
        fx2, fy2 = x1 - x_s, y1 - y_s
        face_w, face_h = fx2 - fx1, fy2 - fy1

        # Ellipse centered on the mouth area within the face bbox
        cx = (fx1 + fx2) // 2
        cy = fy1 + int(face_h * 0.65)  # 65% down from top of face = mouth
        ax = int(face_w * 0.48)  # horizontal radius - wide enough for cheeks
        ay = int(face_h * 0.40)  # vertical radius - covers mouth + chin

        cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)

        # Feather edges with Gaussian blur
        ksize = int(0.1 * cw // 2 * 2) + 1
        mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)

        coverage = np.count_nonzero(mask) / mask.size
        logger.info(
            f"Elliptical mask: center=({cx},{cy}), radii=({ax},{ay}), "
            f"crop_box={crop_box}, coverage={coverage:.1%}"
        )
        return mask, crop_box

    # ------------------------------------------------------------------
    # Audio processing → frame generation
    # ------------------------------------------------------------------

    async def process_audio(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000,
        audio_file: Optional[str] = None,
    ):
        """Generate lip-synced frames from audio.

        Runs MuseTalk inference and pushes frames to the internal deque.
        Call get_frame() from the render loop to consume them.

        Args:
            audio_data: Audio as numpy array (float32 or int16).
            sample_rate: Source sample rate. Resampled to 16 kHz if different.
            audio_file: Optional path to WAV file. If provided, used directly
                        (avoids writing a temp file).
        """
        if not self._initialized:
            logger.warning("MuseTalk not initialized - skipping audio")
            return

        self._is_processing = True

        try:
            # Run in thread executor to avoid blocking the render loop
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._process_audio_sync,
                audio_data,
                sample_rate,
                audio_file,
            )
        except Exception as e:
            logger.error(f"MuseTalk audio processing failed: {e}")
        finally:
            self._is_processing = False

    def _process_audio_sync(
        self,
        audio_data: Optional[np.ndarray],
        sample_rate: int,
        audio_file: Optional[str],
    ):
        """Synchronous audio processing (runs in thread)."""
        try:
            # Always prefer loading from audio_file (audio_data may be raw bytes)
            if audio_file and os.path.exists(audio_file):
                import soundfile as sf
                audio_data, sample_rate = sf.read(audio_file, dtype="float32")
            elif audio_data is not None and not isinstance(audio_data, np.ndarray):
                # audio_data is raw bytes - try to decode
                import soundfile as sf
                import io
                audio_data, sample_rate = sf.read(
                    io.BytesIO(audio_data), dtype="float32"
                )

            if audio_data is None:
                logger.warning("No audio data or file provided")
                return

            if not isinstance(audio_data, np.ndarray):
                logger.warning(f"audio_data is {type(audio_data)}, expected ndarray")
                return

            # Convert stereo to mono (MuseTalk/Whisper needs mono)
            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=-1)

            # Ensure 16 kHz float32
            if sample_rate != 16000:
                import librosa
                audio_data = librosa.resample(
                    audio_data.astype(np.float32),
                    orig_sr=sample_rate,
                    target_sr=16000,
                )

            if audio_data.dtype != np.float32:
                if audio_data.dtype == np.int16:
                    audio_data = audio_data.astype(np.float32) / 32768.0
                else:
                    audio_data = audio_data.astype(np.float32)

            # Normalize
            peak = np.abs(audio_data).max()
            if peak > 1.0:
                audio_data = audio_data / peak
            elif 0 < peak < 0.01:
                audio_data = audio_data / peak * 0.5

            # Save to temp WAV (AudioProcessor needs a file path)
            wav_path = audio_file
            tmp_created = False
            if not wav_path or not os.path.exists(wav_path):
                import soundfile as sf
                wav_path = os.path.join(
                    tempfile.gettempdir(),
                    f"musetalk_audio_{int(time.time()*1000)}.wav",
                )
                sf.write(wav_path, audio_data, 16000)
                tmp_created = True

            # Extract Whisper features
            whisper_feats, lib_length = self._audio_processor.get_audio_feature(
                wav_path, weight_dtype=self._weight_dtype
            )

            whisper_chunks = self._audio_processor.get_whisper_chunk(
                whisper_feats,
                self._device,
                self._weight_dtype,
                self._whisper,
                lib_length,
                fps=self._config.fps,
                audio_padding_length_left=2,
                audio_padding_length_right=2,
            )

            if whisper_chunks is None or (hasattr(whisper_chunks, '__len__') and len(whisper_chunks) == 0):
                logger.warning("No whisper chunks extracted from audio")
                return

            # Cycle latents to match audio length
            n_chunks = len(whisper_chunks)
            n_lat = len(self._input_latents)
            latent_cycle = (self._input_latents * (n_chunks // n_lat + 1))[:n_chunks]

            # Generate frames batch by batch
            gen = self._datagen(whisper_chunks, latent_cycle, self._config.batch_size)

            frame_idx = 0
            for whisper_batch, latent_batch in gen:
                audio_feat = self._pe(whisper_batch.to(self._device))
                latent_batch = latent_batch.to(
                    device=self._device, dtype=self._weight_dtype
                )

                with torch.no_grad():
                    pred = self._unet.model(
                        latent_batch,
                        self._timesteps,
                        encoder_hidden_states=audio_feat,
                    ).sample

                pred = pred.to(device=self._device, dtype=self._vae.vae.dtype)
                recon = self._vae.decode_latents(pred)

                for res_frame in recon:
                    if frame_idx >= n_chunks:
                        break

                    if self._proxy_mode:
                        # PROXY MODE: transfer mouth from generated proxy
                        # to cyborg face, then composite cyborg
                        cyborg_with_mouth = self._transfer_mouth(
                            res_frame.astype(np.uint8)
                        )
                        # Composite onto cyborg full frame at CYBORG bbox
                        cx1, cy1, cx2, cy2 = self._cyborg_bbox
                        resized = cv2.resize(
                            cyborg_with_mouth, (cx2 - cx1, cy2 - cy1),
                            interpolation=cv2.INTER_LANCZOS4,
                        )
                        final = self._source_image.copy()
                        final[cy1:cy2, cx1:cx2] = resized
                    else:
                        # DIRECT MODE: standard MuseTalk compositing
                        ci = frame_idx % len(self._coord_list)
                        coord = self._coord_list[ci]
                        ori = self._frame_list[ci].copy()
                        mask = self._mask_list[ci]
                        mask_coord = self._mask_coords_list[ci]

                        x1, y1, x2, y2 = coord
                        res_frame = cv2.resize(
                            res_frame.astype(np.uint8), (x2 - x1, y2 - y1)
                        )

                        final = self._get_image_blending(
                            ori, res_frame, coord, mask, mask_coord
                        )

                    self._frame_queue.append(final)
                    frame_idx += 1

            # Clean up temp file
            if tmp_created:
                try:
                    os.unlink(wav_path)
                except OSError:
                    pass

            audio_dur = len(audio_data) / 16000
            self._total_frames_generated += frame_idx
            self._total_audio_processed += audio_dur
            logger.info(
                f"MuseTalk: generated {frame_idx} frames from {audio_dur:.1f}s audio"
            )

        except Exception as e:
            logger.error(f"MuseTalk sync processing error: {e}")
            import traceback
            traceback.print_exc()

    # ------------------------------------------------------------------
    # Frame output (called by render loop)
    # ------------------------------------------------------------------

    def get_frame(self) -> Optional[np.ndarray]:
        """Return next lip-synced frame, or idle frame if queue empty."""
        if self._frame_queue:
            return self._frame_queue.popleft()
        if self._idle_frame is not None:
            return self._idle_frame.copy()
        return None

    @property
    def is_processing(self) -> bool:
        return self._is_processing

    @property
    def has_frames(self) -> bool:
        return len(self._frame_queue) > 0

    @property
    def frames_remaining(self) -> int:
        return len(self._frame_queue)

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "initialized": self._initialized,
            "is_processing": self._is_processing,
            "frames_buffered": len(self._frame_queue),
            "total_frames_generated": self._total_frames_generated,
            "total_audio_seconds": round(self._total_audio_processed, 1),
            "model_version": self._config.model_version,
            "device": str(self._device) if self._device else "none",
        }

    async def cleanup(self):
        """Release all GPU memory and models."""
        for attr in ("_vae", "_unet", "_pe", "_whisper"):
            obj = getattr(self, attr, None)
            if obj is not None:
                del obj
                setattr(self, attr, None)

        self._audio_processor = None
        self._face_parser = None
        self._input_latents = []
        self._frame_queue.clear()

        if HAS_TORCH:
            torch.cuda.empty_cache()

        self._initialized = False
        logger.info("MuseTalk backend cleaned up")
