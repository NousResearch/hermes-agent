"""
Local Avatar Animation - Wav2Lip neural + Face Rig with tracking points
Replaces D-ID cloud dependency with local rendering for the cyborg Cosmos avatar.

Primary: Wav2Lip neural lip sync (GPU)
Fallback: FaceRigAnimator - manual tracking points + Delaunay mesh warping (CPU)
"""

import asyncio
import math
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from loguru import logger

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .avatar_controller import AvatarState


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class LocalAnimationConfig:
    """Configuration for local avatar animation"""
    face_image_path: str = ""
    output_width: int = 1280
    output_height: int = 720
    wav2lip_model_path: Optional[str] = None
    manual_mouth_roi: Optional[Dict[str, float]] = None  # {"x","y","w","h"} normalised
    warp_intensity: float = 1.0
    blend_radius: int = 20
    use_mediapipe: bool = True
    mediapipe_confidence: float = 0.3
    idle_head_sway: float = 0.5
    idle_blink_interval: float = 4.0
    jaw_drop_max: int = 30  # max jaw drop in pixels at output resolution
    # Tracking point overrides (normalised coords, or None to use defaults)
    tracking_points: Optional[List[Tuple[str, float, float, str]]] = None


# ---------------------------------------------------------------------------
# Mouth region
# ---------------------------------------------------------------------------

@dataclass
class MouthRegion:
    """Detected mouth region information"""
    bbox: Tuple[int, int, int, int]  # x, y, w, h in pixels
    landmarks: Optional[np.ndarray] = None  # Nx2 mouth landmark points
    detection_method: str = "unknown"


# ---------------------------------------------------------------------------
# Face region detector  (3-tier)
# ---------------------------------------------------------------------------

class FaceRegionDetector:
    """Finds mouth region with 3-tier fallback:
    1. MediaPipe Face Mesh (468 landmarks)
    2. Manual ROI from config
    3. Centre estimate heuristic
    """

    # MediaPipe outer mouth landmark indices
    _OUTER_MOUTH_IDS = [
        61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
        308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78,
    ]
    # Inner mouth
    _INNER_MOUTH_IDS = [
        78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
        324, 318, 402, 317, 14, 87, 178, 88, 95,
    ]

    def __init__(self, config: LocalAnimationConfig):
        self._config = config
        self._mp_face_mesh = None

    def detect(self, image: np.ndarray) -> MouthRegion:
        """Detect mouth region in image using 3-tier fallback"""
        h, w = image.shape[:2]

        # Tier 1: MediaPipe
        if self._config.use_mediapipe and HAS_MEDIAPIPE:
            region = self._detect_mediapipe(image, w, h)
            if region is not None:
                return region

        # Tier 2: Manual ROI
        if self._config.manual_mouth_roi:
            return self._detect_manual(w, h)

        # Tier 3: Centre estimate
        return self._detect_centre_estimate(w, h)

    def _detect_mediapipe(self, image: np.ndarray, w: int, h: int) -> Optional[MouthRegion]:
        """Try MediaPipe Face Mesh detection"""
        try:
            if self._mp_face_mesh is None:
                self._mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=self._config.mediapipe_confidence,
                )

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.shape[2] >= 3 else image
            results = self._mp_face_mesh.process(rgb)

            if not results.multi_face_landmarks:
                logger.debug("MediaPipe: no face detected (cyborg face may need manual ROI)")
                return None

            face = results.multi_face_landmarks[0]
            all_ids = list(set(self._OUTER_MOUTH_IDS + self._INNER_MOUTH_IDS))
            pts = np.array(
                [[int(face.landmark[i].x * w), int(face.landmark[i].y * h)] for i in all_ids],
                dtype=np.int32,
            )

            x_min, y_min = pts.min(axis=0)
            x_max, y_max = pts.max(axis=0)
            pad = int(max(x_max - x_min, y_max - y_min) * 0.25)
            x_min = max(0, x_min - pad)
            y_min = max(0, y_min - pad)
            x_max = min(w, x_max + pad)
            y_max = min(h, y_max + pad)

            logger.info(f"MediaPipe mouth detected: ({x_min},{y_min}) -> ({x_max},{y_max})")
            return MouthRegion(
                bbox=(x_min, y_min, x_max - x_min, y_max - y_min),
                landmarks=pts,
                detection_method="mediapipe",
            )
        except Exception as e:
            logger.warning(f"MediaPipe detection failed: {e}")
            return None

    def _detect_manual(self, w: int, h: int) -> MouthRegion:
        """Use manual ROI from config (normalised coords)"""
        roi = self._config.manual_mouth_roi
        x = int(roi["x"] * w)
        y = int(roi["y"] * h)
        rw = int(roi["w"] * w)
        rh = int(roi["h"] * h)
        logger.info(f"Using manual mouth ROI: ({x},{y},{rw},{rh})")
        return MouthRegion(bbox=(x, y, rw, rh), detection_method="manual")

    def _detect_centre_estimate(self, w: int, h: int) -> MouthRegion:
        """Heuristic: mouth is ~70% down the face, centre horizontally"""
        cx, cy = w // 2, int(h * 0.7)
        rw, rh = int(w * 0.35), int(h * 0.15)
        x = cx - rw // 2
        y = cy - rh // 2
        logger.info(f"Using centre-estimate mouth ROI: ({x},{y},{rw},{rh})")
        return MouthRegion(bbox=(x, y, rw, rh), detection_method="centre_estimate")

    def cleanup(self):
        if self._mp_face_mesh is not None:
            self._mp_face_mesh.close()
            self._mp_face_mesh = None


# ---------------------------------------------------------------------------
# Wav2Lip renderer (primary, neural)
# ---------------------------------------------------------------------------

class Wav2LipRenderer:
    """Neural lip sync using Wav2Lip model.

    Expects ``wav2lip_gan.pth`` (or compatible checkpoint).
    Returns None on any failure so the caller can fall back to OpenCV.
    """

    def __init__(self, model_path: str):
        self._model = None
        self._device = None
        self._model_path = model_path

    def load(self) -> bool:
        """Load the Wav2Lip model. Returns False if unavailable."""
        if not HAS_TORCH:
            logger.info("PyTorch not available - Wav2Lip disabled")
            return False

        model_file = Path(self._model_path)
        if not model_file.exists():
            logger.info(f"Wav2Lip model not found at {self._model_path}")
            return False

        try:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            checkpoint = torch.load(self._model_path, map_location=self._device, weights_only=False)

            try:
                from wav2lip.models import Wav2Lip as Wav2LipModel
            except ImportError:
                logger.info("wav2lip package not installed - neural renderer disabled")
                return False

            self._model = Wav2LipModel()
            state_dict = checkpoint.get("state_dict", checkpoint)
            cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}
            self._model.load_state_dict(cleaned)
            self._model = self._model.to(self._device).eval()

            logger.info(f"Wav2Lip model loaded on {self._device}")
            return True

        except Exception as e:
            logger.warning(f"Failed to load Wav2Lip model: {e}")
            self._model = None
            return False

    @property
    def available(self) -> bool:
        return self._model is not None

    def render(
        self,
        face_image: np.ndarray,
        audio_chunk: np.ndarray,
        mouth_region: MouthRegion,
    ) -> Optional[np.ndarray]:
        """Render a lip-synced frame."""
        if self._model is None:
            return None

        try:
            x, y, w, h = mouth_region.bbox
            img_h, img_w = face_image.shape[:2]

            face_size = max(w, h) * 2
            cx, cy = x + w // 2, y + h // 2
            half = face_size // 2
            fx1 = max(0, cx - half)
            fy1 = max(0, cy - half)
            fx2 = min(img_w, cx + half)
            fy2 = min(img_h, cy + half)

            face_crop = face_image[fy1:fy2, fx1:fx2]
            face_96 = cv2.resize(face_crop, (96, 96))

            mel = self._audio_to_mel(audio_chunk)
            if mel is None:
                return None

            with torch.no_grad():
                img_tensor = (
                    torch.FloatTensor(face_96.transpose(2, 0, 1)[np.newaxis] / 255.0)
                    .to(self._device)
                )
                mel_tensor = torch.FloatTensor(mel[np.newaxis, np.newaxis]).to(self._device)
                pred = self._model(mel_tensor, img_tensor)

            pred_frame = (pred.squeeze().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            pred_resized = cv2.resize(pred_frame, (fx2 - fx1, fy2 - fy1))

            result = face_image.copy()
            mask = self._create_blend_mask(fx2 - fx1, fy2 - fy1, 20)
            mask_3ch = np.stack([mask] * 3, axis=-1)
            roi = result[fy1:fy2, fx1:fx2].astype(np.float32)
            blended = roi * (1 - mask_3ch) + pred_resized.astype(np.float32) * mask_3ch
            result[fy1:fy2, fx1:fx2] = blended.astype(np.uint8)

            return result

        except Exception as e:
            logger.debug(f"Wav2Lip render failed: {e}")
            return None

    def _audio_to_mel(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """Convert audio chunk to mel spectrogram (80x16)"""
        try:
            target_len = 3200
            if len(audio) < target_len:
                audio = np.pad(audio, (0, target_len - len(audio)))
            else:
                audio = audio[:target_len]

            n_fft = 800
            hop = 200
            n_mels = 80
            n_frames = 16

            specs = []
            for i in range(n_frames):
                start = i * hop
                end = start + n_fft
                if end > len(audio):
                    frame = np.pad(audio[start:], (0, end - len(audio)))
                else:
                    frame = audio[start:end]

                window = np.hanning(len(frame))
                spectrum = np.abs(np.fft.rfft(frame * window))
                bin_size = max(1, len(spectrum) // n_mels)
                mel_frame = np.array([
                    spectrum[j * bin_size:(j + 1) * bin_size].mean()
                    for j in range(n_mels)
                ])
                specs.append(mel_frame)

            mel = np.array(specs).T
            mel = np.log(np.maximum(mel, 1e-5))
            return mel

        except Exception as e:
            logger.debug(f"Mel conversion failed: {e}")
            return None

    @staticmethod
    def _create_blend_mask(w: int, h: int, radius: int) -> np.ndarray:
        """Elliptical feathered blend mask"""
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.ellipse(mask, (w // 2, h // 2), (w // 2 - radius, h // 2 - radius),
                     0, 0, 360, 1.0, -1)
        if radius > 0:
            mask = cv2.GaussianBlur(mask, (radius * 2 + 1, radius * 2 + 1), radius / 2)
        return mask

    def cleanup(self):
        if self._model is not None:
            del self._model
            self._model = None
            if HAS_TORCH and torch.cuda.is_available():
                torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Face Rig Animator - Tracking points + Delaunay mesh warping
# ---------------------------------------------------------------------------

class FaceRigAnimator:
    """Face animation using manually placed tracking points and mesh warping.

    Like D-ID but free: places ~32 tracking points on the face image,
    builds a Delaunay triangle mesh, then warps the mesh per-frame based
    on speaking state, visemes, blinking, and idle motion.

    Key features:
    - Manual tracking points accurate for the cyborg face
    - Boundary anchor points prevent edge artifacts
    - Large displacements (20-30px) for clearly visible animation
    - Procedural speaking animation (self-driven, doesn't need external lip sync)
    - Smooth Delaunay triangle warping for natural deformation
    """

    # Default tracking points for the cyborg Cosmos portrait.
    # Format: (name, x_norm, y_norm, group)
    # Calibrated for the 942x875 cyborg image (human lower face, Borg upper).
    _DEFAULT_LANDMARKS = [
        # ---- Boundary anchors (fixed, never move) ----
        ("bound_tl",  0.00, 0.00, "boundary"),
        ("bound_tc",  0.50, 0.00, "boundary"),
        ("bound_tr",  1.00, 0.00, "boundary"),
        ("bound_ml",  0.00, 0.50, "boundary"),
        ("bound_mr",  1.00, 0.50, "boundary"),
        ("bound_bl",  0.00, 1.00, "boundary"),
        ("bound_bc",  0.50, 1.00, "boundary"),
        ("bound_br",  1.00, 1.00, "boundary"),
        # Edge midpoints for denser boundary
        ("bound_tl2", 0.25, 0.00, "boundary"),
        ("bound_tr2", 0.75, 0.00, "boundary"),
        ("bound_bl2", 0.00, 0.75, "boundary"),
        ("bound_br2", 1.00, 0.75, "boundary"),

        # ---- Jaw contour (moves down when mouth opens) ----
        ("jaw_l",     0.28, 0.82, "jaw"),
        ("jaw_ml",    0.37, 0.88, "jaw"),
        ("chin",      0.50, 0.93, "jaw"),
        ("jaw_mr",    0.63, 0.88, "jaw"),
        ("jaw_r",     0.72, 0.82, "jaw"),

        # ---- Upper lip (moves slightly up when mouth opens) ----
        ("lip_up_l",  0.41, 0.745, "mouth_upper"),
        ("lip_up_c",  0.50, 0.735, "mouth_upper"),
        ("lip_up_r",  0.59, 0.745, "mouth_upper"),

        # ---- Lower lip (moves down with jaw) ----
        ("lip_lo_l",  0.41, 0.775, "mouth_lower"),
        ("lip_lo_c",  0.50, 0.785, "mouth_lower"),
        ("lip_lo_r",  0.59, 0.775, "mouth_lower"),

        # ---- Mouth corners (spread/narrow for visemes) ----
        ("mouth_cl",  0.36, 0.76, "mouth_corner"),
        ("mouth_cr",  0.64, 0.76, "mouth_corner"),

        # ---- Nose (mostly stable) ----
        ("nose_tip",  0.50, 0.64, "nose"),

        # ---- Cheeks (anchor, slight deformation with jaw) ----
        ("cheek_l",   0.26, 0.66, "cheek"),
        ("cheek_r",   0.74, 0.66, "cheek"),

        # ---- Eyes (for blinking) ----
        ("eye_l_top", 0.37, 0.43, "eye_l"),
        ("eye_l_bot", 0.37, 0.48, "eye_l"),
        ("eye_r_top", 0.63, 0.43, "eye_r"),
        ("eye_r_bot", 0.63, 0.48, "eye_r"),

        # ---- Forehead (for brow raise) ----
        ("brow_l",    0.37, 0.36, "brow"),
        ("brow_r",    0.63, 0.36, "brow"),
    ]

    # Viseme -> (mouth_width_scale, mouth_open_scale) relative to max
    _VISEME_SHAPES: Dict[str, Tuple[float, float]] = {
        "sil": (1.0, 0.0),
        "PP":  (0.80, 0.05),
        "FF":  (0.90, 0.10),
        "TH":  (0.90, 0.15),
        "DD":  (0.85, 0.30),
        "kk":  (0.80, 0.25),
        "CH":  (0.70, 0.35),
        "SS":  (0.60, 0.15),
        "nn":  (0.80, 0.20),
        "RR":  (0.75, 0.35),
        "aa":  (1.20, 1.00),
        "E":   (1.30, 0.50),
        "ih":  (1.10, 0.40),
        "oh":  (0.70, 0.80),
        "ou":  (0.60, 0.70),
    }

    def __init__(self, config: LocalAnimationConfig):
        self._config = config
        self._base_image: Optional[np.ndarray] = None
        self._rest_pts: Optional[np.ndarray] = None   # Nx2 rest positions (pixels)
        self._triangles: Optional[List[Tuple[int, int, int]]] = None
        self._landmarks: List[Tuple[str, float, float, str]] = []
        self._group_indices: Dict[str, List[int]] = {}
        self._boundary_indices: set = set()
        self._img_w: int = 0
        self._img_h: int = 0
        self._max_drop: int = 30
        self._initialised = False

        # Procedural speaking state
        self._speaking_start: float = 0.0
        self._was_speaking: bool = False

        # Blinking state
        self._start_time = time.time()
        self._last_blink_time = time.time()
        self._blink_progress = 0.0
        self._is_blinking = False

        # Debug frame counter
        self._frame_count = 0
        self._last_debug_log = time.time()

    def setup(self, image: np.ndarray, mouth_region: MouthRegion) -> bool:
        """Initialise tracking points and Delaunay mesh."""
        if not HAS_CV2:
            return False

        self._base_image = image.copy()
        self._img_h, self._img_w = image.shape[:2]
        self._max_drop = self._config.jaw_drop_max

        # Use custom tracking points if provided, otherwise defaults
        self._landmarks = (
            self._config.tracking_points
            if self._config.tracking_points
            else list(self._DEFAULT_LANDMARKS)
        )

        # Convert normalised coords to pixel positions
        self._rest_pts = np.array(
            [[lm[1] * self._img_w, lm[2] * self._img_h] for lm in self._landmarks],
            dtype=np.float32,
        )

        # Build group index map
        self._group_indices = {}
        self._boundary_indices = set()
        for i, lm in enumerate(self._landmarks):
            group = lm[3]
            if group not in self._group_indices:
                self._group_indices[group] = []
            self._group_indices[group].append(i)
            if group == "boundary":
                self._boundary_indices.add(i)

        # Build Delaunay triangulation
        self._triangles = self._build_delaunay(self._rest_pts)

        n_interior = len(self._landmarks) - len(self._boundary_indices)
        logger.info(
            f"FaceRigAnimator initialised: {len(self._landmarks)} tracking points "
            f"({n_interior} interior, {len(self._boundary_indices)} boundary), "
            f"{len(self._triangles)} triangles, max_drop={self._max_drop}px"
        )
        self._initialised = True
        return True

    @property
    def available(self) -> bool:
        return self._initialised

    def render(self, state: AvatarState) -> Optional[np.ndarray]:
        """Render an animated frame by warping the face mesh."""
        if not self._initialised or self._base_image is None:
            return None

        try:
            t = time.time() - self._start_time

            # Compute target positions for all tracking points
            target_pts = self._compute_targets(state, t)

            # Check if there's any meaningful movement
            max_displacement = np.max(np.abs(target_pts - self._rest_pts))
            if max_displacement < 0.5:
                return self._base_image.copy()

            # Debug: log displacement stats every 3 seconds
            self._frame_count += 1
            now = time.time()
            if now - self._last_debug_log > 3.0:
                self._last_debug_log = now
                logger.info(
                    f"FaceRig frame #{self._frame_count}: max_disp={max_displacement:.1f}px, "
                    f"speaking={state.is_speaking}, mouth_open={state.mouth_open:.2f}, "
                    f"viseme={state.current_viseme}"
                )

            # Warp each Delaunay triangle
            warped = self._warp_mesh(self._base_image, self._rest_pts, target_pts)

            return warped

        except Exception as e:
            logger.debug(f"FaceRig render failed: {e}")
            return self._base_image.copy()

    def _compute_targets(self, state: AvatarState, t: float) -> np.ndarray:
        """Compute target positions for all tracking points based on state."""
        targets = self._rest_pts.copy()

        # --- Mouth opening ---
        mouth_open = state.mouth_open
        viseme = state.current_viseme or "sil"
        w_scale, v_open = self._VISEME_SHAPES.get(viseme, (1.0, 0.0))

        # Blend in viseme opening
        mouth_open = max(mouth_open, v_open * 0.6)

        # Procedural speaking: if is_speaking but mouth_open is low, self-animate
        if state.is_speaking:
            if not self._was_speaking:
                self._speaking_start = time.time()
                self._was_speaking = True
            speak_t = time.time() - self._speaking_start
            # Natural oscillation: layered sinusoids for organic movement
            procedural = (
                0.3 + 0.5 * abs(math.sin(speak_t * 7.0))
                * (0.4 + 0.6 * abs(math.sin(speak_t * 3.3)))
            )
            mouth_open = max(mouth_open, procedural)
        else:
            self._was_speaking = False

        # Scale to pixels
        drop_px = mouth_open * self._max_drop * self._config.warp_intensity

        # Jaw: move down
        for i in self._group_indices.get("jaw", []):
            targets[i, 1] += drop_px * 0.75

        # Lower lip: move down (most movement)
        for i in self._group_indices.get("mouth_lower", []):
            targets[i, 1] += drop_px

        # Upper lip: move up slightly
        for i in self._group_indices.get("mouth_upper", []):
            targets[i, 1] -= drop_px * 0.15

        # Mouth corners: spread/narrow based on viseme width
        corner_dx = (w_scale - 1.0) * 15 * self._config.warp_intensity
        for i in self._group_indices.get("mouth_corner", []):
            name = self._landmarks[i][0]
            if "cl" in name:  # left corner
                targets[i, 0] -= corner_dx
                targets[i, 1] += drop_px * 0.3
            else:  # right corner
                targets[i, 0] += corner_dx
                targets[i, 1] += drop_px * 0.3

        # Cheeks: slight pull down with jaw
        for i in self._group_indices.get("cheek", []):
            targets[i, 1] += drop_px * 0.1

        # --- Blinking ---
        blink = self._compute_blink(t)
        if blink > 0.05:
            for i in self._group_indices.get("eye_l", []) + self._group_indices.get("eye_r", []):
                name = self._landmarks[i][0]
                if "top" in name:
                    targets[i, 1] += blink * 10  # eyelid drops
                elif "bot" in name:
                    targets[i, 1] -= blink * 5   # lower lid rises

        # --- Idle head sway ---
        sway_scale = self._config.idle_head_sway
        sway_dx = math.sin(t * 0.3) * 4.0 * sway_scale
        sway_dy = math.sin(t * 0.2) * 2.0 * sway_scale
        # Breathing
        breath_dy = math.sin(t * 0.8) * 3.0 * sway_scale

        for i in range(len(targets)):
            if i not in self._boundary_indices:
                targets[i, 0] += sway_dx
                targets[i, 1] += sway_dy + breath_dy

        return targets

    def _compute_blink(self, t: float) -> float:
        """Compute blink intensity (0-1) with periodic blinking."""
        now = time.time()
        if not self._is_blinking:
            if (now - self._last_blink_time) > self._config.idle_blink_interval:
                self._is_blinking = True
                self._blink_progress = 0.0
                self._last_blink_time = now
            return 0.0

        self._blink_progress += 0.12
        if self._blink_progress >= 1.0:
            self._is_blinking = False
            self._blink_progress = 0.0
            return 0.0

        # Triangle wave
        if self._blink_progress < 0.5:
            return self._blink_progress * 2.0
        else:
            return (1.0 - self._blink_progress) * 2.0

    def _build_delaunay(self, pts: np.ndarray) -> List[Tuple[int, int, int]]:
        """Build Delaunay triangulation, return index triples."""
        rect = (0, 0, self._img_w, self._img_h)
        subdiv = cv2.Subdiv2D(rect)

        # Insert points, clamping to valid range
        for p in pts:
            px = float(max(0, min(p[0], self._img_w - 1)))
            py = float(max(0, min(p[1], self._img_h - 1)))
            subdiv.insert((px, py))

        triangles = []
        for t in subdiv.getTriangleList():
            p1 = np.array([t[0], t[1]])
            p2 = np.array([t[2], t[3]])
            p3 = np.array([t[4], t[5]])

            i1 = self._closest_idx(pts, p1)
            i2 = self._closest_idx(pts, p2)
            i3 = self._closest_idx(pts, p3)

            if i1 != i2 and i2 != i3 and i1 != i3:
                triangles.append((i1, i2, i3))

        return triangles

    @staticmethod
    def _closest_idx(pts: np.ndarray, target: np.ndarray) -> int:
        dists = np.sum((pts - target) ** 2, axis=1)
        return int(np.argmin(dists))

    def _warp_mesh(
        self, src: np.ndarray, src_pts: np.ndarray, dst_pts: np.ndarray,
    ) -> np.ndarray:
        """Warp image by deforming each Delaunay triangle."""
        output = src.copy()
        h, w = src.shape[:2]

        for tri_idx in self._triangles:
            i1, i2, i3 = tri_idx
            src_tri = np.float32([src_pts[i1], src_pts[i2], src_pts[i3]])
            dst_tri = np.float32([dst_pts[i1], dst_pts[i2], dst_pts[i3]])

            # Skip triangles with no movement
            if np.allclose(src_tri, dst_tri, atol=0.3):
                continue

            sr = cv2.boundingRect(src_tri)
            dr = cv2.boundingRect(dst_tri)

            sr = self._clamp_rect(sr, w, h)
            dr = self._clamp_rect(dr, w, h)

            if sr[2] <= 0 or sr[3] <= 0 or dr[2] <= 0 or dr[3] <= 0:
                continue

            src_tri_local = src_tri - np.float32([sr[0], sr[1]])
            dst_tri_local = dst_tri - np.float32([dr[0], dr[1]])

            # Affine warp this triangle
            warp_mat = cv2.getAffineTransform(src_tri_local, dst_tri_local)
            src_crop = src[sr[1]:sr[1]+sr[3], sr[0]:sr[0]+sr[2]]

            if src_crop.size == 0:
                continue

            warped_crop = cv2.warpAffine(
                src_crop, warp_mat, (dr[2], dr[3]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )

            # Triangle mask
            mask = np.zeros((dr[3], dr[2]), dtype=np.uint8)
            cv2.fillConvexPoly(mask, np.int32(dst_tri_local), 255)

            # Paste warped triangle into output
            dy1, dy2 = dr[1], dr[1] + dr[3]
            dx1, dx2 = dr[0], dr[0] + dr[2]
            region = output[dy1:dy2, dx1:dx2]

            if region.shape[:2] != warped_crop.shape[:2]:
                continue

            if region.ndim == 3:
                mask_3 = np.stack([mask] * region.shape[2], axis=-1)
            else:
                mask_3 = mask

            np.copyto(region, warped_crop, where=(mask_3 > 0))

        return output

    @staticmethod
    def _clamp_rect(rect, w, h):
        x, y, rw, rh = rect
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        rw = max(0, min(rw, w - x))
        rh = max(0, min(rh, h - y))
        return (x, y, rw, rh)


# ---------------------------------------------------------------------------
# LocalAnimationBackend (orchestrator)
# ---------------------------------------------------------------------------

class LocalAnimationBackend:
    """Orchestrates face detection, renderer selection, and frame output.

    Usage::

        config = LocalAnimationConfig(face_image_path="cyborg.jpg")
        backend = LocalAnimationBackend(config)
        await backend.initialize()

        frame = await backend.render_frame(avatar_state)
    """

    def __init__(self, config: LocalAnimationConfig):
        self._config = config
        self._detector = FaceRegionDetector(config)
        self._wav2lip: Optional[Wav2LipRenderer] = None
        self._face_rig: Optional[FaceRigAnimator] = None
        self._face_image: Optional[np.ndarray] = None
        self._mouth_region: Optional[MouthRegion] = None
        self._initialised = False
        self._use_wav2lip = False

    async def initialize(self, face_image_path: Optional[str] = None) -> bool:
        """Load image, detect face, initialise renderers"""
        if not HAS_CV2:
            logger.error("OpenCV not available - cannot initialise local animation")
            return False

        path = face_image_path or self._config.face_image_path
        if not path:
            logger.error("No face image path provided")
            return False

        img_path = Path(path)
        if not img_path.exists():
            logger.error(f"Face image not found: {path}")
            return False

        raw = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            logger.error(f"Failed to read image: {path}")
            return False

        if raw.ndim == 2:
            raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)

        self._face_image = cv2.resize(raw, (self._config.output_width, self._config.output_height))
        logger.info(f"Loaded face image: {path} -> {self._face_image.shape}")

        # Detect mouth region
        detect_img = self._face_image[:, :, :3] if self._face_image.shape[2] == 4 else self._face_image
        self._mouth_region = self._detector.detect(detect_img)

        # Try Wav2Lip renderer
        if self._config.wav2lip_model_path:
            self._wav2lip = Wav2LipRenderer(self._config.wav2lip_model_path)
            if self._wav2lip.load():
                self._use_wav2lip = True
                logger.info("Primary renderer: Wav2Lip (neural)")
            else:
                self._wav2lip = None

        # Set up face rig animator (tracking points + mesh warping)
        self._face_rig = FaceRigAnimator(self._config)
        anim_img = self._face_image[:, :, :3] if self._face_image.shape[2] == 4 else self._face_image
        if self._face_rig.setup(anim_img, self._mouth_region):
            if not self._use_wav2lip:
                logger.info("Primary renderer: FaceRig (tracking points + mesh warp)")
        else:
            if not self._use_wav2lip:
                logger.warning("Both renderers failed to initialise - static image only")

        self._initialised = True
        logger.info(f"LocalAnimationBackend initialised (mouth: {self._mouth_region.detection_method})")
        return True

    async def render_frame(
        self,
        state: AvatarState,
        audio_chunk: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """Render a single frame based on current avatar state."""
        if not self._initialised or self._face_image is None:
            return None

        frame: Optional[np.ndarray] = None

        # Primary: Wav2Lip neural (if available + audio present)
        if self._use_wav2lip and audio_chunk is not None and self._wav2lip is not None:
            face_bgr = self._face_image[:, :, :3] if self._face_image.shape[2] == 4 else self._face_image
            frame = self._wav2lip.render(face_bgr, audio_chunk, self._mouth_region)

        # Fallback: Face rig animator with tracking points
        if frame is None and self._face_rig is not None and self._face_rig.available:
            frame = self._face_rig.render(state)

        # Last resort: static image
        if frame is None:
            frame = self._face_image.copy()

        # Ensure 4 channels if base has alpha
        if self._face_image.shape[2] == 4 and frame.ndim == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

        return frame

    async def cleanup(self):
        """Release GPU memory, close MediaPipe, clear caches"""
        if self._wav2lip:
            self._wav2lip.cleanup()
            self._wav2lip = None
        self._detector.cleanup()
        self._face_image = None
        self._face_rig = None
        self._initialised = False
        logger.info("LocalAnimationBackend cleaned up")
