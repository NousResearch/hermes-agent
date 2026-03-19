"""
Avatar Controller - Multi-backend avatar rendering system
Supports: Live2D (live2d-py), VTube Studio (pyvts), Neural (MuseTalk), WebGL
"""

import asyncio
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
from pathlib import Path
import json
import time
from loguru import logger

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import live2d.v3 as live2d
    HAS_LIVE2D = True
except ImportError:
    HAS_LIVE2D = False

try:
    import pyvts
    HAS_PYVTS = True
except ImportError:
    HAS_PYVTS = False


class AvatarBackend(Enum):
    """Supported avatar rendering backends"""
    LIVE2D_PY = "live2d_py"        # Pure Python Live2D
    VTUBE_STUDIO = "vtube_studio"  # VTube Studio via WebSocket
    NEURAL = "neural"              # MuseTalk/StyleAvatar neural rendering
    WEBGL = "webgl"                # Three.js/WebGL (browser-based)
    IMAGE_SEQUENCE = "image_seq"   # Simple image-based (fallback)
    LOCAL_ANIM = "local_anim"      # Local Wav2Lip/OpenCV animation
    MUSETALK = "musetalk"          # MuseTalk neural lip sync (30+ FPS)
    SADTALKER = "sadtalker"        # SadTalker full face animation (D-ID quality)


@dataclass
class AvatarState:
    """Current state of the avatar"""
    # Mouth/Lip sync (0-1 ranges)
    mouth_open: float = 0.0
    mouth_form: float = 0.5  # 0=wide, 1=narrow

    # Eyes
    eye_left_open: float = 1.0
    eye_right_open: float = 1.0
    eye_x: float = 0.0  # -1 to 1 (left to right)
    eye_y: float = 0.0  # -1 to 1 (down to up)

    # Eyebrows
    brow_left_y: float = 0.0  # -1 to 1
    brow_right_y: float = 0.0

    # Head pose
    head_x: float = 0.0  # rotation
    head_y: float = 0.0
    head_z: float = 0.0

    # Body
    body_x: float = 0.0
    body_y: float = 0.0

    # Expression blend weights
    expression_weights: Dict[str, float] = field(default_factory=dict)

    # Current viseme for lip sync
    current_viseme: str = "sil"

    # Speaking state
    is_speaking: bool = False
    speaking_intensity: float = 0.0

    # Emotion state
    current_emotion: str = "neutral"
    emotion_intensity: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mouth_open": self.mouth_open,
            "mouth_form": self.mouth_form,
            "eye_left_open": self.eye_left_open,
            "eye_right_open": self.eye_right_open,
            "eye_x": self.eye_x,
            "eye_y": self.eye_y,
            "brow_left_y": self.brow_left_y,
            "brow_right_y": self.brow_right_y,
            "head_x": self.head_x,
            "head_y": self.head_y,
            "head_z": self.head_z,
            "body_x": self.body_x,
            "body_y": self.body_y,
            "current_viseme": self.current_viseme,
            "is_speaking": self.is_speaking,
            "current_emotion": self.current_emotion,
            "emotion_intensity": self.emotion_intensity,
        }


@dataclass
class AvatarConfig:
    """Configuration for avatar system"""
    backend: AvatarBackend = AvatarBackend.IMAGE_SEQUENCE
    model_path: Optional[str] = None

    # VTube Studio settings
    vts_host: str = "localhost"
    vts_port: int = 8001
    vts_plugin_name: str = "CosmosAI"
    vts_developer: str = "CosmosSwarm"

    # Rendering settings
    width: int = 1280
    height: int = 720
    fps: int = 30

    # Animation settings
    blink_interval: float = 4.0  # seconds
    blink_duration: float = 0.15
    idle_motion_scale: float = 0.3

    # Local animation settings
    local_anim_face_image: Optional[str] = None
    local_anim_manual_roi: Optional[Dict] = None
    local_anim_wav2lip_model: Optional[str] = None

    # MuseTalk settings
    musetalk_dir: Optional[str] = None
    musetalk_face_image: Optional[str] = None
    musetalk_version: str = "v15"
    musetalk_proxy_face: Optional[str] = None

    # SadTalker settings
    sadtalker_dir: Optional[str] = None
    sadtalker_face_image: Optional[str] = None
    sadtalker_size: int = 256

    # Expression mappings
    expression_map: Dict[str, str] = field(default_factory=dict)


class AvatarController:
    """
    Multi-backend avatar controller for Cosmos VTuber

    Handles:
    - Avatar model loading and rendering
    - Parameter control (mouth, eyes, expressions)
    - Idle animations and blinking
    - Expression transitions
    - Frame generation for streaming
    """

    def __init__(self, config: Optional[AvatarConfig] = None):
        self.config = config or AvatarConfig()
        self.state = AvatarState()
        self.backend = None
        self._running = False
        self._frame_callback: Optional[Callable] = None
        self._last_blink = time.time()
        self._blink_state = 0.0
        self._idle_offset = 0.0

        # Parameter smoothing
        self._target_state = AvatarState()
        self._smooth_factor = 0.3

        # VTube Studio connection
        self._vts_client = None
        self._vts_authenticated = False

        # Live2D model
        self._live2d_model = None

        # Local animation backend
        self._local_anim_backend = None

        # MuseTalk backend
        self._musetalk_backend = None

        # SadTalker backend
        self._sadtalker_backend = None

        # Image sequence fallback
        self._image_frames: Dict[str, np.ndarray] = {}
        self._base_image: Optional[np.ndarray] = None

        logger.info(f"AvatarController initialized with backend: {self.config.backend}")

    async def initialize(self) -> bool:
        """Initialize the avatar backend"""
        try:
            if self.config.backend == AvatarBackend.VTUBE_STUDIO:
                return await self._init_vtube_studio()
            elif self.config.backend == AvatarBackend.LIVE2D_PY:
                return await self._init_live2d()
            elif self.config.backend == AvatarBackend.IMAGE_SEQUENCE:
                return await self._init_image_sequence()
            elif self.config.backend == AvatarBackend.NEURAL:
                return await self._init_neural()
            elif self.config.backend == AvatarBackend.LOCAL_ANIM:
                return await self._init_local_animation()
            elif self.config.backend == AvatarBackend.MUSETALK:
                return await self._init_musetalk()
            elif self.config.backend == AvatarBackend.SADTALKER:
                return await self._init_sadtalker()
            else:
                logger.warning(f"Unknown backend: {self.config.backend}, using image sequence")
                return await self._init_image_sequence()
        except Exception as e:
            logger.error(f"Failed to initialize avatar backend: {e}")
            return False

    async def _init_vtube_studio(self) -> bool:
        """Initialize VTube Studio connection via pyvts"""
        if not HAS_PYVTS:
            logger.error("pyvts not installed. Install with: pip install pyvts")
            return False

        try:
            self._vts_client = pyvts.vts(
                plugin_info={
                    "plugin_name": self.config.vts_plugin_name,
                    "developer": self.config.vts_developer,
                    "authentication_token_path": "./vts_token.txt"
                },
                vts_api_info={
                    "host": self.config.vts_host,
                    "port": self.config.vts_port
                }
            )

            await self._vts_client.connect()
            await self._vts_client.request_authenticate_token()
            await self._vts_client.request_authenticate()

            self._vts_authenticated = True
            logger.info("VTube Studio connected and authenticated")
            return True

        except Exception as e:
            logger.error(f"VTube Studio connection failed: {e}")
            return False

    async def _init_live2d(self) -> bool:
        """Initialize Live2D model via live2d-py"""
        if not HAS_LIVE2D:
            logger.error("live2d-py not installed. Install with: pip install live2d-py")
            return False

        if not self.config.model_path:
            logger.error("No model path specified for Live2D")
            return False

        try:
            live2d.init()
            self._live2d_model = live2d.LAppModel()
            self._live2d_model.LoadModelJson(self.config.model_path)

            # Set up canvas
            live2d.glewInit()
            live2d.setGLProperties()

            self._live2d_model.Resize(self.config.width, self.config.height)

            logger.info(f"Live2D model loaded: {self.config.model_path}")
            return True

        except Exception as e:
            logger.error(f"Live2D initialization failed: {e}")
            return False

    async def _init_image_sequence(self) -> bool:
        """Initialize image-based avatar using generated Gemini images"""
        if not HAS_CV2:
            logger.error("OpenCV not installed. Install with: pip install opencv-python")
            return False

        # Try to load generated avatar images first
        avatar_dir = Path(__file__).parent / "avatars"

        if avatar_dir.exists():
            loaded = self._load_generated_avatars(avatar_dir)
            if loaded:
                logger.info(f"Loaded {len(self._image_frames)} generated avatar expressions")
                return True

        # Fallback to placeholder if no generated images
        logger.warning("No generated avatars found, using placeholder")
        self._base_image = self._create_placeholder_avatar()

        self._image_frames = {
            "neutral": self._base_image.copy(),
            "speaking_1": self._create_speaking_frame(0.3),
            "speaking_2": self._create_speaking_frame(0.6),
            "speaking_3": self._create_speaking_frame(1.0),
            "happy": self._create_expression_frame("happy"),
            "thinking": self._create_expression_frame("thinking"),
            "excited": self._create_expression_frame("excited"),
        }

        logger.info("Image sequence avatar initialized (placeholder)")
        return True

    def _load_generated_avatars(self, avatar_dir: Path) -> bool:
        """Load generated avatar images from directory"""
        self._image_frames = {}
        self._viseme_frames = {}  # Separate dict for viseme-specific frames

        # Expected avatar files
        expressions = [
            "base", "neutral", "happy", "excited", "thinking",
            "surprised", "speaking_1", "speaking_2", "speaking_3"
        ]

        for expr in expressions:
            img_path = avatar_dir / f"cosmos_{expr}.png"
            if img_path.exists():
                img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
                if img is not None:
                    # Resize to stream dimensions if needed
                    if img.shape[0] != self.config.height or img.shape[1] != self.config.width:
                        img = cv2.resize(img, (self.config.width, self.config.height))

                    # Ensure BGRA format
                    if len(img.shape) == 2:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
                    elif img.shape[2] == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

                    self._image_frames[expr] = img
                    logger.debug(f"Loaded avatar: {expr}")

        # Load viseme-specific frames (Rhubarb mouth shapes A-X)
        visemes_dir = avatar_dir / "visemes"
        if visemes_dir.exists():
            self._load_viseme_frames(visemes_dir)
        else:
            # Create default viseme mappings from speaking frames
            self._create_default_viseme_mapping()

        if not self._image_frames:
            return False

        # Set base image (can't use 'or' with numpy arrays)
        if "base" in self._image_frames:
            self._base_image = self._image_frames["base"]
        elif "neutral" in self._image_frames:
            self._base_image = self._image_frames["neutral"]
        else:
            # Use first available
            self._base_image = list(self._image_frames.values())[0]

        # Ensure we have speaking frames (duplicate if missing)
        if "speaking_1" not in self._image_frames:
            self._image_frames["speaking_1"] = self._base_image.copy()
        if "speaking_2" not in self._image_frames:
            self._image_frames["speaking_2"] = self._base_image.copy()
        if "speaking_3" not in self._image_frames:
            self._image_frames["speaking_3"] = self._base_image.copy()

        return True

    def _load_viseme_frames(self, visemes_dir: Path):
        """Load Rhubarb viseme-specific avatar frames"""
        # Rhubarb uses shapes A-X
        viseme_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'X']

        for viseme in viseme_names:
            # Try multiple naming patterns
            patterns = [
                f"cosmos_viseme_{viseme}.png",
                f"mouth_{viseme}.png",
                f"viseme_{viseme}.png",
            ]

            for pattern in patterns:
                img_path = visemes_dir / pattern
                if img_path.exists():
                    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
                    if img is not None:
                        if img.shape[0] != self.config.height or img.shape[1] != self.config.width:
                            img = cv2.resize(img, (self.config.width, self.config.height))
                        if len(img.shape) == 2:
                            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
                        elif img.shape[2] == 3:
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

                        self._viseme_frames[viseme] = img
                        logger.debug(f"Loaded viseme frame: {viseme}")
                        break

        if self._viseme_frames:
            logger.info(f"Loaded {len(self._viseme_frames)} viseme frames for lip-sync")
        else:
            self._create_default_viseme_mapping()

    def _create_default_viseme_mapping(self):
        """Create default viseme to expression mapping when no dedicated frames exist"""
        # Map Rhubarb visemes to available speaking frames
        # X = silence -> neutral
        # A = closed (M,B,P) -> neutral
        # B = slightly open -> speaking_1
        # C = open (E,AH) -> speaking_2
        # D = wide open (AA) -> speaking_3
        # E = round (OH) -> speaking_2
        # F = pucker (OO) -> speaking_1
        # G = teeth (F,V) -> speaking_1
        # H = tongue (L,TH) -> speaking_2

        base = self._base_image if self._base_image is not None else self._image_frames.get("neutral")

        if base is None:
            logger.warning("No base image for viseme mapping")
            return

        neutral = self._image_frames.get("neutral", base)
        speak1 = self._image_frames.get("speaking_1", base)
        speak2 = self._image_frames.get("speaking_2", base)
        speak3 = self._image_frames.get("speaking_3", base)

        self._viseme_frames = {
            'X': neutral,  # Silence
            'A': neutral,  # Closed lips (M, B, P)
            'B': speak1,   # Slightly open
            'C': speak2,   # Open (E, EH)
            'D': speak3,   # Wide open (AA)
            'E': speak2,   # Round (OH)
            'F': speak1,   # Pucker (OO)
            'G': speak1,   # Teeth on lip (F, V)
            'H': speak2,   # Tongue (L, TH)
        }

        logger.info("Created default viseme mapping from speaking frames")

    async def _init_neural(self) -> bool:
        """Initialize neural avatar (MuseTalk/StyleAvatar)"""
        logger.warning("Neural avatar backend not implemented, falling back to image sequence")
        return await self._init_image_sequence()

    async def _init_local_animation(self) -> bool:
        """Initialize local animation backend (Wav2Lip + OpenCV warper)"""
        try:
            from .local_animation import LocalAnimationBackend, LocalAnimationConfig

            la_config = LocalAnimationConfig(
                face_image_path=self.config.local_anim_face_image or "",
                output_width=self.config.width,
                output_height=self.config.height,
                wav2lip_model_path=self.config.local_anim_wav2lip_model,
                manual_mouth_roi=self.config.local_anim_manual_roi,
            )
            self._local_anim_backend = LocalAnimationBackend(la_config)
            success = await self._local_anim_backend.initialize()
            if success:
                logger.info("Local animation backend initialized")
            return success
        except Exception as e:
            logger.error(f"Local animation init failed: {e}")
            return False

    async def _init_musetalk(self) -> bool:
        """Initialize MuseTalk neural lip sync backend"""
        try:
            from .musetalk_backend import MuseTalkBackend, MuseTalkConfig

            mt_config = MuseTalkConfig(
                musetalk_dir=self.config.musetalk_dir or "/workspace/MuseTalk",
                model_version=self.config.musetalk_version,
                face_image_path=self.config.musetalk_face_image or "",
                proxy_face_path=self.config.musetalk_proxy_face or "",
                output_width=self.config.width,
                output_height=self.config.height,
                fps=self.config.fps,
            )
            self._musetalk_backend = MuseTalkBackend(mt_config)
            success = await self._musetalk_backend.initialize()
            if success:
                logger.info("MuseTalk backend initialized")
            return success
        except Exception as e:
            logger.error(f"MuseTalk init failed: {e}")
            return False

    async def _init_sadtalker(self) -> bool:
        """Initialize SadTalker full face animation backend"""
        try:
            from .sadtalker_backend import SadTalkerBackend, SadTalkerConfig

            st_config = SadTalkerConfig(
                sadtalker_dir=self.config.sadtalker_dir or "/workspace/SadTalker",
                face_image_path=self.config.sadtalker_face_image or "",
                output_width=self.config.width,
                output_height=self.config.height,
                fps=self.config.fps,
                size=self.config.sadtalker_size,
            )
            self._sadtalker_backend = SadTalkerBackend(st_config)
            success = await self._sadtalker_backend.initialize()
            if success:
                logger.info("SadTalker backend initialized")
            return success
        except Exception as e:
            logger.error(f"SadTalker init failed: {e}")
            return False

    def _create_placeholder_avatar(self) -> np.ndarray:
        """Create a placeholder Cosmos avatar image"""
        img = np.zeros((self.config.height, self.config.width, 4), dtype=np.uint8)

        # Dark background
        img[:, :] = [20, 20, 30, 255]

        if HAS_CV2:
            center_x = self.config.width // 2
            center_y = self.config.height // 2

            # Head (oval)
            cv2.ellipse(img, (center_x, center_y - 50), (120, 150), 0, 0, 360, (200, 180, 160, 255), -1)

            # Borg implant (half face metallic)
            pts = np.array([
                [center_x, center_y - 200],
                [center_x + 120, center_y - 50],
                [center_x + 100, center_y + 100],
                [center_x, center_y + 100]
            ], np.int32)
            cv2.fillPoly(img, [pts], (80, 80, 90, 255))

            # Red laser eye (right side - Borg)
            cv2.circle(img, (center_x + 50, center_y - 70), 20, (0, 0, 200, 255), -1)
            cv2.circle(img, (center_x + 50, center_y - 70), 10, (0, 0, 255, 255), -1)

            # Normal eye (left side)
            cv2.ellipse(img, (center_x - 50, center_y - 70), (25, 15), 0, 0, 360, (255, 255, 255, 255), -1)
            cv2.circle(img, (center_x - 50, center_y - 70), 8, (50, 50, 50, 255), -1)

            # Mouth (closed)
            cv2.ellipse(img, (center_x, center_y + 50), (40, 10), 0, 0, 360, (150, 100, 100, 255), -1)

            # White hair
            for i in range(-3, 4):
                x_offset = i * 30
                cv2.line(img, (center_x + x_offset, center_y - 180),
                        (center_x + x_offset + i*5, center_y - 220), (255, 255, 255, 255), 3)

            # Lab coat collar
            cv2.rectangle(img, (center_x - 100, center_y + 100),
                         (center_x + 100, center_y + 200), (240, 240, 240, 255), -1)

        return img

    def _create_speaking_frame(self, intensity: float) -> np.ndarray:
        """Create a speaking frame with mouth open"""
        img = self._base_image.copy()

        if HAS_CV2:
            center_x = self.config.width // 2
            center_y = self.config.height // 2

            # Open mouth based on intensity
            mouth_height = int(10 + intensity * 25)
            cv2.ellipse(img, (center_x, center_y + 50), (40, mouth_height), 0, 0, 360, (100, 50, 50, 255), -1)
            cv2.ellipse(img, (center_x, center_y + 50), (35, mouth_height - 5), 0, 0, 360, (50, 20, 20, 255), -1)

        return img

    def _create_expression_frame(self, expression: str) -> np.ndarray:
        """Create an expression frame"""
        img = self._base_image.copy()

        if HAS_CV2:
            center_x = self.config.width // 2
            center_y = self.config.height // 2

            if expression == "happy":
                # Smile
                cv2.ellipse(img, (center_x, center_y + 40), (50, 25), 0, 0, 180, (150, 100, 100, 255), -1)
                # Raised eyebrows
                cv2.line(img, (center_x - 80, center_y - 110), (center_x - 20, center_y - 115), (100, 80, 60, 255), 3)

            elif expression == "thinking":
                # Raised eyebrow on one side
                cv2.line(img, (center_x - 80, center_y - 115), (center_x - 20, center_y - 105), (100, 80, 60, 255), 3)
                # Slight frown
                cv2.ellipse(img, (center_x, center_y + 50), (30, 8), 0, 0, 360, (150, 100, 100, 255), -1)

            elif expression == "excited":
                # Wide eyes
                cv2.ellipse(img, (center_x - 50, center_y - 70), (30, 20), 0, 0, 360, (255, 255, 255, 255), -1)
                # Brighter laser eye
                cv2.circle(img, (center_x + 50, center_y - 70), 25, (0, 0, 255, 255), -1)
                # Open mouth smile
                cv2.ellipse(img, (center_x, center_y + 45), (50, 30), 0, 0, 180, (150, 100, 100, 255), -1)

        return img

    async def update_state(self, new_state: AvatarState, immediate: bool = False):
        """Update avatar state with optional smoothing"""
        if immediate:
            self.state = new_state
        else:
            self._target_state = new_state

    async def set_viseme(self, viseme: str, intensity: float = 1.0):
        """Set current viseme for lip sync"""
        self.state.current_viseme = viseme
        self.state.speaking_intensity = intensity

        # Map viseme to mouth parameters
        viseme_map = {
            "sil": (0.0, 0.5),      # Silent
            "PP": (0.1, 0.2),       # P, B, M
            "FF": (0.2, 0.3),       # F, V
            "TH": (0.3, 0.4),       # Th
            "DD": (0.4, 0.5),       # T, D, N
            "kk": (0.3, 0.6),       # K, G
            "CH": (0.4, 0.4),       # Ch, J, Sh
            "SS": (0.2, 0.3),       # S, Z
            "nn": (0.3, 0.5),       # N, L
            "RR": (0.4, 0.5),       # R
            "aa": (0.8, 0.7),       # A
            "E": (0.5, 0.4),        # E
            "ih": (0.4, 0.4),       # I
            "oh": (0.7, 0.3),       # O
            "ou": (0.6, 0.2),       # U
        }

        mouth_open, mouth_form = viseme_map.get(viseme, (0.0, 0.5))
        self.state.mouth_open = mouth_open * intensity
        self.state.mouth_form = mouth_form

    async def set_expression(self, emotion: str, intensity: float = 1.0):
        """Set avatar expression/emotion"""
        self.state.current_emotion = emotion
        self.state.emotion_intensity = intensity

        # Update expression weights
        self.state.expression_weights = {emotion: intensity}

        # Map emotions to facial parameters
        emotion_params = {
            "neutral": {"brow_left_y": 0, "brow_right_y": 0},
            "happy": {"brow_left_y": 0.2, "brow_right_y": 0.2},
            "sad": {"brow_left_y": -0.3, "brow_right_y": -0.3},
            "angry": {"brow_left_y": -0.5, "brow_right_y": -0.5},
            "surprised": {"brow_left_y": 0.5, "brow_right_y": 0.5, "eye_left_open": 1.2, "eye_right_open": 1.2},
            "thinking": {"brow_left_y": 0.3, "brow_right_y": -0.1, "eye_x": 0.3, "eye_y": 0.2},
            "excited": {"brow_left_y": 0.4, "brow_right_y": 0.4, "eye_left_open": 1.1, "eye_right_open": 1.1},
        }

        params = emotion_params.get(emotion, emotion_params["neutral"])
        for key, value in params.items():
            if hasattr(self.state, key):
                setattr(self.state, key, value * intensity)

    async def start_speaking(self):
        """Signal that avatar should start speaking animation"""
        self.state.is_speaking = True

    async def stop_speaking(self):
        """Signal that avatar should stop speaking"""
        self.state.is_speaking = False
        self.state.mouth_open = 0.0
        self.state.current_viseme = "sil"

    def _update_idle_animation(self, dt: float):
        """Update idle animations (blinking, subtle movement)"""
        current_time = time.time()

        # Blinking
        if current_time - self._last_blink > self.config.blink_interval:
            self._blink_state = 1.0
            self._last_blink = current_time

        if self._blink_state > 0:
            self._blink_state -= dt / self.config.blink_duration
            if self._blink_state < 0:
                self._blink_state = 0

            blink_value = 1.0 - self._blink_state
            self.state.eye_left_open = blink_value
            self.state.eye_right_open = blink_value

        # Subtle idle motion
        self._idle_offset += dt
        idle_scale = self.config.idle_motion_scale
        self.state.head_x = np.sin(self._idle_offset * 0.5) * idle_scale * 0.1
        self.state.head_y = np.sin(self._idle_offset * 0.3) * idle_scale * 0.05
        self.state.body_y = np.sin(self._idle_offset * 0.2) * idle_scale * 0.02

    def _smooth_state(self):
        """Smooth transition between current and target state"""
        for attr in ['mouth_open', 'mouth_form', 'eye_x', 'eye_y',
                     'brow_left_y', 'brow_right_y', 'head_x', 'head_y', 'head_z']:
            current = getattr(self.state, attr)
            target = getattr(self._target_state, attr)
            setattr(self.state, attr, current + (target - current) * self._smooth_factor)

    async def render_frame(self) -> Optional[np.ndarray]:
        """Render current frame based on avatar state"""
        try:
            if self.config.backend == AvatarBackend.VTUBE_STUDIO:
                return await self._render_vtube_studio()
            elif self.config.backend == AvatarBackend.LIVE2D_PY:
                return await self._render_live2d()
            elif self.config.backend == AvatarBackend.LOCAL_ANIM:
                return await self._render_local_animation()
            elif self.config.backend == AvatarBackend.MUSETALK:
                return await self._render_musetalk()
            elif self.config.backend == AvatarBackend.SADTALKER:
                return await self._render_sadtalker()
            else:
                return await self._render_image_sequence()
        except Exception as e:
            logger.error(f"Frame render error: {e}")
            return None

    async def _render_vtube_studio(self) -> Optional[np.ndarray]:
        """Send parameters to VTube Studio (returns None - VTS handles rendering)"""
        if not self._vts_authenticated:
            return None

        try:
            # Build parameter list
            params = [
                {"id": "MouthOpen", "value": self.state.mouth_open},
                {"id": "MouthForm", "value": self.state.mouth_form},
                {"id": "EyeOpenLeft", "value": self.state.eye_left_open},
                {"id": "EyeOpenRight", "value": self.state.eye_right_open},
                {"id": "EyeX", "value": self.state.eye_x},
                {"id": "EyeY", "value": self.state.eye_y},
                {"id": "BrowLeftY", "value": self.state.brow_left_y},
                {"id": "BrowRightY", "value": self.state.brow_right_y},
                {"id": "FaceAngleX", "value": self.state.head_x * 30},
                {"id": "FaceAngleY", "value": self.state.head_y * 30},
                {"id": "FaceAngleZ", "value": self.state.head_z * 30},
            ]

            # Send to VTube Studio
            for param in params:
                await self._vts_client.request(
                    self._vts_client.vts_request.requestSetParameterValue(
                        parameter=param["id"],
                        value=param["value"],
                        weight=1.0,
                        face_found=True
                    )
                )

            return None  # VTS handles its own rendering

        except Exception as e:
            logger.error(f"VTube Studio parameter update failed: {e}")
            return None

    async def _render_live2d(self) -> Optional[np.ndarray]:
        """Render Live2D model to frame"""
        if not self._live2d_model:
            return None

        try:
            # Update model parameters
            self._live2d_model.SetParameterValue("ParamMouthOpenY", self.state.mouth_open)
            self._live2d_model.SetParameterValue("ParamMouthForm", self.state.mouth_form)
            self._live2d_model.SetParameterValue("ParamEyeLOpen", self.state.eye_left_open)
            self._live2d_model.SetParameterValue("ParamEyeROpen", self.state.eye_right_open)
            self._live2d_model.SetParameterValue("ParamEyeBallX", self.state.eye_x)
            self._live2d_model.SetParameterValue("ParamEyeBallY", self.state.eye_y)
            self._live2d_model.SetParameterValue("ParamBrowLY", self.state.brow_left_y)
            self._live2d_model.SetParameterValue("ParamBrowRY", self.state.brow_right_y)
            self._live2d_model.SetParameterValue("ParamAngleX", self.state.head_x * 30)
            self._live2d_model.SetParameterValue("ParamAngleY", self.state.head_y * 30)
            self._live2d_model.SetParameterValue("ParamAngleZ", self.state.head_z * 30)
            self._live2d_model.SetParameterValue("ParamBodyAngleX", self.state.body_x * 10)
            self._live2d_model.SetParameterValue("ParamBodyAngleY", self.state.body_y * 10)

            # Render
            self._live2d_model.Update()
            self._live2d_model.Draw()

            # Return blank placeholder frame
            logger.debug("Frame capture not implemented for this backend, returning placeholder")
            try:
                import numpy as np
                return np.zeros((512, 512, 4), dtype=np.uint8)
            except ImportError:
                return None

        except Exception as e:
            logger.error(f"Live2D render failed: {e}")
            return None

    async def _render_local_animation(self) -> Optional[np.ndarray]:
        """Render frame using local animation backend"""
        if self._local_anim_backend is None:
            return None
        return await self._local_anim_backend.render_frame(self.state)

    async def _render_musetalk(self) -> Optional[np.ndarray]:
        """Render frame using MuseTalk backend (pops from generated queue)"""
        if self._musetalk_backend is None:
            return None
        return self._musetalk_backend.get_frame()

    async def _render_sadtalker(self) -> Optional[np.ndarray]:
        """Render frame using SadTalker backend (pops from generated queue)"""
        if self._sadtalker_backend is None:
            return None
        frame = self._sadtalker_backend.get_next_frame()
        if frame is not None:
            return frame
        return self._sadtalker_backend.idle_frame

    async def _render_image_sequence(self) -> Optional[np.ndarray]:
        """Render using pre-generated image frames with viseme support"""
        if self._base_image is None:
            return None

        # If speaking and we have viseme data, use viseme-based animation
        if self.state.is_speaking and self._viseme_frames:
            # Get current viseme from state
            viseme = self.state.current_viseme.upper() if self.state.current_viseme else 'X'

            # Map internal viseme names to Rhubarb shapes
            viseme_map = {
                'SIL': 'X', 'PP': 'A', 'FF': 'G', 'TH': 'H',
                'DD': 'B', 'KK': 'B', 'CH': 'C', 'SS': 'B',
                'NN': 'B', 'RR': 'B', 'AA': 'D', 'E': 'C',
                'IH': 'B', 'OH': 'E', 'OU': 'F',
                # Direct Rhubarb shapes pass through
                'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D',
                'E': 'E', 'F': 'F', 'G': 'G', 'H': 'H', 'X': 'X'
            }

            rhubarb_shape = viseme_map.get(viseme, 'X')

            if rhubarb_shape in self._viseme_frames:
                return self._viseme_frames[rhubarb_shape].copy()

            # Fallback to intensity-based if specific viseme not found
            intensity = self.state.mouth_open
            if intensity > 0.7:
                frame_key = "speaking_3"
            elif intensity > 0.4:
                frame_key = "speaking_2"
            elif intensity > 0.1:
                frame_key = "speaking_1"
            else:
                frame_key = "neutral"
        elif self.state.is_speaking:
            # No viseme frames, use intensity-based
            intensity = self.state.mouth_open
            if intensity > 0.7:
                frame_key = "speaking_3"
            elif intensity > 0.4:
                frame_key = "speaking_2"
            elif intensity > 0.1:
                frame_key = "speaking_1"
            else:
                frame_key = "neutral"
        else:
            # Not speaking, use emotion-based expression
            frame_key = self.state.current_emotion
            if frame_key not in self._image_frames:
                frame_key = "neutral"

        return self._image_frames.get(frame_key, self._base_image).copy()

    async def run_loop(self, frame_callback: Callable[[np.ndarray], None]):
        """Run the avatar update loop"""
        self._running = True
        self._frame_callback = frame_callback

        frame_time = 1.0 / self.config.fps
        last_time = time.time()

        while self._running:
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time

            # Update animations
            self._update_idle_animation(dt)
            self._smooth_state()

            # Render frame
            frame = await self.render_frame()

            if frame is not None and self._frame_callback:
                self._frame_callback(frame)

            # Maintain frame rate
            elapsed = time.time() - current_time
            if elapsed < frame_time:
                await asyncio.sleep(frame_time - elapsed)

    async def stop(self):
        """Stop the avatar controller"""
        self._running = False

        if self._vts_client:
            try:
                await self._vts_client.close()
            except Exception:
                pass

        if self._live2d_model:
            try:
                live2d.dispose()
            except Exception:
                pass

        if self._local_anim_backend:
            try:
                await self._local_anim_backend.cleanup()
            except Exception:
                pass

        if self._musetalk_backend:
            try:
                await self._musetalk_backend.cleanup()
            except Exception:
                pass

        if self._sadtalker_backend:
            try:
                await self._sadtalker_backend.cleanup()
            except Exception:
                pass

        logger.info("AvatarController stopped")
