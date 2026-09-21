"""
Silero VAD via ONNX — CPU-only, no torchaudio dependency.
Lightweight voice activity detection for the voice pipeline.

Model: snakers4/silero-vad mini (~0.5MB).
Window: 30ms frames at 16kHz mono.
Per-frame latency: ~2ms on ARM Neoverse.
"""
import logging
import os
import urllib.request

logger = logging.getLogger(__name__)

# Optional import — fail gracefully if numpy/onnxruntime aren't present
# at import time; downstream code will fall back to the simpler
# threshold-based detector.
try:
    import numpy as np
    import onnxruntime as ort
    _HAS_DEPS = True
except ImportError:  # pragma: no cover
    _HAS_DEPS = False
    np = None  # type: ignore[assignment]
    ort = None  # type: ignore[assignment]


ONNX_URL: str = (
    "https://github.com/snakers4/silero-vad/raw/refs/heads/master/models/silero_vad_mini.onnx"
)
WINDOW_SAMPLES: int = 480           # 30ms @ 16kHz
RNN_HIDDEN_DIM: int = 128
STATE_SHAPE = (2, 1, RNN_HIDDEN_DIM)  # derived from hidden dim
SPEECH_THRESHOLD_ON: float = 0.50   # start  speech
SPEECH_THRESHOLD_OFF: float = 0.35  # end    speech (hysteresis)
HISTORY_FRAMES: int = 10            # 300ms smoothing window
MAX_CONSECUTIVE_ERRORS: int = 5     # consecutive ONNX failures → degrade


def _cache_dir() -> str:
    path = os.path.expanduser("~/.hermes/var/silero-vad")
    os.makedirs(path, exist_ok=True)
    return path


def _ensure_model() -> str:
    """Return path to Silero VAD ONNX model.

    Priority:
    1. Bundled model from installed silero-vad package (no import needed).
    2. Cache directory of the official mini model (fallback download).
    """
    # --- Priority 1: locate bundled model without importing torchaudio-dependent package ---
    try:
        import importlib.util
        spec = importlib.util.find_spec("silero_vad")
        if spec and spec.origin:
            bundled = os.path.join(os.path.dirname(spec.origin), "data", "silero_vad_16k_op15.onnx")
            if os.path.exists(bundled):
                return bundled
    except Exception:
        pass

    # --- Priority 2: cache directory ---
    cache = _cache_dir()
    model_path = os.path.join(cache, "silero_vad_mini.onnx")
    if os.path.exists(model_path):
        # Validate it's actually a protobuf, not an HTML redirect
        try:
            with open(model_path, "rb") as f:
                header = f.read(8)
            if header[:4] == b'\x08\x00\x00\x00':
                return model_path
        except Exception:
            pass
    logger.info("Downloading Silero VAD model to %s", cache)
    try:
        urllib.request.urlretrieve(ONNX_URL, model_path)
        with open(model_path, "rb") as f:
            header = f.read(8)
        if header[:4] == b'\x08\x00\x00\x00':
            logger.info("Silero VAD model ready: %s", model_path)
            return model_path
        else:
            raise ValueError("Downloaded file is not a valid ONNX protobuf")
    except Exception as exc:
        logger.error("Silero VAD download failed: %s", exc)
        raise


class SileroVAD:
    """CPU-only VAD wrapper around the Silero mini ONNX model."""

    def __init__(self, model_path: str | None = None):
        if not _HAS_DEPS:
            raise ImportError(
                "Silero VAD requires numpy and onnxruntime; "
                "install them or set VAD mode to threshold"
            )
        self._model_path = model_path or _ensure_model()
        self._sess = ort.InferenceSession(  # type: ignore[union-attr]
            self._model_path,
            providers=["CPUExecutionProvider"],
        )
        self._reset_state()

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def _reset_state(self) -> None:
        self._state = np.zeros(STATE_SHAPE, dtype=np.float32)  # type: ignore[union-attr]
        self._sr = np.array(16000, dtype=np.int64)  # sample rate for ONNX feed
        self._history: list[float] = []
        self._is_speech = False
        self._degraded = False
        self._error_streak = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_speech(self) -> bool:
        """Current speech state after the most recent frame(s).
        Returns True when degraded (ONNX errors) to pass through audio unchanged."""
        return True if self._degraded else self._is_speech

    def reset(self) -> None:
        """Reset internal RNN state. Call on new utterance / reconnect."""
        self._reset_state()

    def feed(self, pcm) -> bool:
        """
        Feed raw 16kHz mono PCM (int16 or float32 numpy array).
        Returns the current speech state (True = speech active).
        Slices internally into 30-ms ONNX frames.
        """
        if pcm is None or len(pcm) == 0:
            return self._is_speech

        buf = self._normalize_pcm(pcm)
        if len(buf) == 0:
            return self._is_speech

        for start in range(0, len(buf), WINDOW_SAMPLES):
            frame = buf[start:start + WINDOW_SAMPLES]
            if len(frame) < WINDOW_SAMPLES:
                frame = np.concatenate([frame, np.zeros(WINDOW_SAMPLES - len(frame), dtype=np.float32)])  # type: ignore[union-attr]
            prob = self._infer_frame(frame)
            self._update(prob)

        return self._is_speech

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_pcm(pcm) -> "np.ndarray":
        """Return float32 [-1,1] numpy array."""
        if pcm.dtype == np.int16:
            return pcm.astype(np.float32) / 32768.0
        return pcm.astype(np.float32)

    def _infer_frame(self, frame) -> float:
        """Run one 30ms frame through ONNX. Returns speech probability [0,1]."""
        assert _HAS_DEPS and np is not None and ort is not None
        inp = frame.reshape(1, -1)
        try:
            out = self._sess.run(
                None,
                {
                    "input": inp,
                    "state": self._state,
                    "sr": self._sr,
                },
            )
            # out[0] = probability, out[1] = next state
            prob = float(out[0].item())
            self._state = out[1] if len(out) > 1 else self._state
            self._error_streak = 0  # reset on success
            return prob
        except Exception as exc:
            self._error_streak += 1
            if self._error_streak >= MAX_CONSECUTIVE_ERRORS and not self._degraded:
                logger.warning(
                    "Silero VAD degraded after %d consecutive errors; "
                    "passing through audio to threshold-based detection. Last error: %s",
                    self._error_streak, exc,
                )
                self._degraded = True
            else:
                logger.warning("Silero VAD inference error: %s", exc)
            return 0.0

    def _update(self, prob: float) -> None:
        """Hysteresis + smoothing over recent frame history."""
        self._history.append(prob)
        if len(self._history) > HISTORY_FRAMES:
            self._history.pop(0)

        max_prob = max(self._history) if self._history else 0.0

        if not self._is_speech and max_prob >= SPEECH_THRESHOLD_ON:
            self._is_speech = True
        elif self._is_speech and max_prob < SPEECH_THRESHOLD_OFF:
            self._is_speech = False
