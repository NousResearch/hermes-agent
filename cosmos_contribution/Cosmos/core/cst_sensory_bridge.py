"""
cosmos CST Sensory Bridge - Physics-Based Emotional Intelligence API

Cosmic Synapse Theory (CST) Integration Module

This module replaces standard ML "approximations" with verified physics-based
emotional analysis. Instead of converting audio to text or detecting facial
expressions as labels, we measure:

1. Frequency Energy (Audio Mass) - The vibrational energy in voice
2. Geometric Phase (Facial Tension) - Angular displacement from stable state
3. Truth Probability - Derived from phase stability and audio/visual coherence

Research Sources:
- CST 12D Cosmic Synapse Theory (September 2024)
- Phi-Stability and Phase Mapping protocols
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Union, Tuple
from enum import Enum

from loguru import logger

# Attempt numpy import for FFT operations
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False
    logger.warning("NumPy not available - FrequencyAnalyzer will use fallback")


class DetectedIntent(Enum):
    """Detected emotional intent categories based on CST physics."""
    HONEST = "Honest"
    DECEPTION = "Deception"
    UNCERTAIN = "Uncertain"
    STRESSED = "Stressed"
    CALM = "Calm"


@dataclass
class CSTState:
    """
    Physics-based emotional state from CST analysis.
    
    This replaces traditional emotion labels with measurable physical quantities:
    - emotional_mass: Energy derived from voice frequency analysis
    - geometric_phase: Angular tension from facial geometry
    - truth_probability: Coherence between audio and visual signals
    - detected_intent: Categorical inference from physics state
    """
    emotional_mass: float = 0.0        # 0.0 - 1.0, from voice frequency energy
    geometric_phase: float = 0.0       # Radians, 0 = stable, >π/4 = unstable
    truth_probability: float = 0.5     # 0.0 - 1.0, derived from phi-stability
    detected_intent: DetectedIntent = DetectedIntent.UNCERTAIN
    
    # Raw measurements for debugging/analysis
    raw_frequencies: list[float] = field(default_factory=list)
    raw_amplitudes: list[float] = field(default_factory=list)
    facial_landmarks_used: int = 0
    
    # Timestamps
    audio_duration_ms: float = 0.0
    processing_time_ms: float = 0.0
    
    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dictionary."""
        return {
            "cst_state": {
                "emotional_mass": round(self.emotional_mass, 4),
                "geometric_phase": round(self.geometric_phase, 4),
                "truth_probability": round(self.truth_probability, 4),
                "detected_intent": self.detected_intent.value,
            },
            "metadata": {
                "audio_duration_ms": self.audio_duration_ms,
                "processing_time_ms": self.processing_time_ms,
                "facial_landmarks_used": self.facial_landmarks_used,
            }
        }
    
    def is_coherent(self, threshold: float = 0.3) -> bool:
        """
        Check if audio and visual signals are coherent.
        
        Incoherence (high emotional_mass but low geometric_phase, or vice versa)
        may indicate deception or emotional masking.
        """
        audio_intensity = self.emotional_mass
        visual_intensity = min(1.0, self.geometric_phase / (math.pi / 4))
        
        return abs(audio_intensity - visual_intensity) < threshold


class FrequencyAnalyzer:
    """
    Audio Mass Calculator using Fast Fourier Transform.
    
    CST Priority Method: generate_frequency_matrix (Line 9, Sept 2024)
    
    Instead of converting Audio → Text, we convert Audio → Sine Wave Energy.
    The formula captures the "tremble" in a voice that text transcription misses:
    
        E_audio = ∫(Amplitude × Frequency) dt
    
    This energy value ("emotional mass") represents the vibrational intensity
    of the voice signal.
    """
    
    # Human voice frequency ranges (Hz)
    VOICE_LOW = 85      # Low male voice
    VOICE_HIGH = 3000   # High harmonics
    
    # Stress indicator frequencies
    STRESS_TREMOR_LOW = 8     # Voice tremor range
    STRESS_TREMOR_HIGH = 12
    
    def __init__(
        self,
        sample_rate: int = 16000,
        window_size_ms: int = 25,
        hop_size_ms: int = 10,
    ):
        """
        Initialize the frequency analyzer.
        
        Args:
            sample_rate: Audio sample rate in Hz
            window_size_ms: FFT window size in milliseconds
            hop_size_ms: Hop size between windows in milliseconds
        """
        self.sample_rate = sample_rate
        self.window_size = int(sample_rate * window_size_ms / 1000)
        self.hop_size = int(sample_rate * hop_size_ms / 1000)
        
        logger.debug(f"FrequencyAnalyzer initialized: sr={sample_rate}, window={self.window_size}")
    
    def analyze(
        self,
        audio_buffer: Union[bytes[float], "np.ndarray"],
        normalize: bool = True,
    ) -> Tuple[float, list[float], list[float]]:
        """
        Analyze audio buffer and return emotional mass.
        
        Args:
            audio_buffer: Raw PCM audio data (16-bit signed or float)
            normalize: Whether to normalize energy to 0-1 range
            
        Returns:
            Tuple of (emotional_mass, frequencies, amplitudes)
        """
        import time
        start_time = time.time()
        
        if not HAS_NUMPY:
            return self._analyze_fallback(audio_buffer, normalize)
        
        # Convert to numpy array if needed
        if isinstance(audio_buffer, bytes):
            audio = np.frombuffer(audio_buffer, dtype=np.int16).astype(np.float32)
            audio = audio / 32768.0  # Normalize from int16
        elif isinstance(audio_buffer):
            audio = np.array(audio_buffer, dtype=np.float32)
        else:
            audio = audio_buffer.astype(np.float32)
        
        if len(audio) < self.window_size:
            logger.warning(f"Audio buffer too short: {len(audio)} < {self.window_size}")
            return (0.0, [], [])
        
        # Calculate energy using FFT
        emotional_mass, frequencies, amplitudes = self._calculate_energy_fft(audio)
        
        if normalize:
            # Normalize to 0-1 range using sigmoid-like function
            emotional_mass = 2.0 / (1.0 + math.exp(-emotional_mass * 0.1)) - 1.0
            emotional_mass = max(0.0, min(1.0, emotional_mass))
        
        processing_time = (time.time() - start_time) * 1000
        logger.debug(f"FrequencyAnalyzer: mass={emotional_mass:.4f}, time={processing_time:.2f}ms")
        
        return (emotional_mass, frequencies, amplitudes)
    
    def _calculate_energy_fft(
        self, 
        audio: "np.ndarray"
    ) -> Tuple[float, list[float], list[float]]:
        """
        Calculate emotional energy using CST formula: E = ∫(Amp × Freq) dt
        
        This integrates amplitude-weighted frequency over the audio duration.
        """
        # Apply windowing to reduce spectral leakage
        window = np.hanning(self.window_size)
        
        total_energy = 0.0
        frame_count = 0
        all_frequencies = []
        all_amplitudes = []
        
        # Process in overlapping windows
        for start in range(0, len(audio) - self.window_size, self.hop_size):
            frame = audio[start:start + self.window_size] * window
            
            # FFT
            spectrum = np.abs(np.fft.rfft(frame))
            freqs = np.fft.rfftfreq(len(frame), 1.0 / self.sample_rate)
            
            # Filter to voice frequency range
            voice_mask = (freqs >= self.VOICE_LOW) & (freqs <= self.VOICE_HIGH)
            voice_freqs = freqs[voice_mask]
            voice_amps = spectrum[voice_mask]
            
            if len(voice_amps) > 0:
                # CST Energy Formula: E = Σ(Amplitude × Frequency)
                frame_energy = np.sum(voice_amps * voice_freqs)
                total_energy += frame_energy
                frame_count += 1
                
                # Store peak frequencies
                top_indices = np.argsort(voice_amps)[-5:]
                all_frequencies.extend(voice_freqs[top_indices].tolist())
                all_amplitudes.extend(voice_amps[top_indices].tolist())
        
        # Average energy per frame
        if frame_count > 0:
            avg_energy = total_energy / frame_count
        else:
            avg_energy = 0.0
        
        return (avg_energy, all_frequencies, all_amplitudes)
    
    def _analyze_fallback(
        self,
        audio_buffer: Union[list[float]],
        normalize: bool,
    ) -> Tuple[float, list[float], list[float]]:
        """
        Fallback analysis without NumPy - uses simple RMS energy.
        """
        if isinstance(audio_buffer, bytes):
            # Simple RMS calculation from bytes
            samples = []
            for i in range(0, len(audio_buffer) - 1, 2):
                sample = int.from_bytes(audio_buffer[i:i+2], 'little', signed=True)
                samples.append(sample / 32768.0)
        else:
            samples = list(audio_buffer)
        
        if not samples:
            return (0.0, [], [])
        
        # RMS energy as fallback
        rms = math.sqrt(sum(s * s for s in samples) / len(samples))
        
        if normalize:
            emotional_mass = min(1.0, rms * 3.0)  # Scale factor
        else:
            emotional_mass = rms
        
        return (emotional_mass, [], [])
    
    def detect_tremor(self, audio_buffer: Union[list[float]]) -> float:
        """
        Detect voice tremor as a stress indicator.
        
        Returns tremor intensity 0.0 - 1.0
        """
        if not HAS_NUMPY:
            return 0.0
        
        # Convert to numpy
        if isinstance(audio_buffer, bytes):
            audio = np.frombuffer(audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            audio = np.array(audio_buffer, dtype=np.float32)
        
        if len(audio) < self.sample_rate:  # Need at least 1 second
            return 0.0
        
        # Compute envelope using Hilbert transform approximation
        # Simple approach: use sliding RMS
        envelope = []
        window = self.sample_rate // 20  # 50ms window
        for i in range(0, len(audio) - window, window // 2):
            chunk = audio[i:i+window]
            rms = np.sqrt(np.mean(chunk ** 2))
            envelope.append(rms)
        
        if len(envelope) < 10:
            return 0.0
        
        envelope = np.array(envelope)
        
        # FFT of envelope to find tremor frequency
        env_spectrum = np.abs(np.fft.rfft(envelope - np.mean(envelope)))
        env_freqs = np.fft.rfftfreq(len(envelope), 0.025)  # 25ms steps
        
        # Look for energy in tremor band (8-12 Hz)
        tremor_mask = (env_freqs >= self.STRESS_TREMOR_LOW) & (env_freqs <= self.STRESS_TREMOR_HIGH)
        tremor_energy = np.sum(env_spectrum[tremor_mask])
        total_energy = np.sum(env_spectrum) + 1e-10
        
        tremor_ratio = tremor_energy / total_energy
        return min(1.0, tremor_ratio * 5.0)  # Scale factor


class GeometricPhaseMapper:
    """
    Facial Geometry to Phase Angle Mapper.
    
    CST Priority Method: Angular Phase Mapping (theta, phi)
    
    Instead of detecting "I see a smile" or "frown", we measure Geometric Tension.
    Micro-expressions are mapped to an angle φ (phi):
    
        Relaxed Face: φ ≈ 0 (Stable Particle)
        Stressed Face: φ > 45° = π/4 radians (High Velocity Particle)
    
    This treats the face as a geometric system where tension creates angular
    displacement from the stable equilibrium state.
    """
    
    # Facial landmark indices for key features (68-point model)
    # These define the geometric frame of reference
    BROW_LEFT = [17, 18, 19, 20, 21]
    BROW_RIGHT = [22, 23, 24, 25, 26]
    EYE_LEFT = [36, 37, 38, 39, 40, 41]
    EYE_RIGHT = [42, 43, 44, 45, 46, 47]
    NOSE = [27, 28, 29, 30, 31, 32, 33, 34, 35]
    MOUTH_OUTER = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]
    MOUTH_INNER = [60, 61, 62, 63, 64, 65, 66, 67]
    JAW = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    
    # Baseline ratios for relaxed face (calibration targets)
    RELAXED_BROW_HEIGHT = 0.15  # Relative to face height
    RELAXED_EYE_OPENNESS = 0.08
    RELAXED_MOUTH_OPENNESS = 0.02
    
    def __init__(self, baseline_landmarks: Optional[list[Tuple[float, float]]] = None):
        """
        Initialize the geometric phase mapper.
        
        Args:
            baseline_landmarks: Optional baseline for personalized calibration
        """
        self.baseline = baseline_landmarks
        self._calibrated = baseline_landmarks is not None
        
        logger.debug(f"GeometricPhaseMapper initialized, calibrated={self._calibrated}")
    
    def calculate_phase(
        self,
        landmarks: list[Tuple[float, float]],
        use_baseline: bool = True,
    ) -> Tuple[float]:
        """
        Calculate geometric phase from facial landmarks.
        
        Args:
            landmarks: list of (x, y) coordinates for facial landmarks
            use_baseline: Whether to use personalized baseline if available
            
        Returns:
            Tuple of (geometric_phase_radians, feature_tensions)
        """
        import time
        start_time = time.time()
        
        if len(landmarks) < 68:
            logger.warning(f"Insufficient landmarks: {len(landmarks)} < 68")
            return (0.0, {})
        
        # Calculate feature-specific tensions
        tensions = {}
        
        # 1. Brow tension (raised = stress)
        tensions['brow'] = self._calculate_brow_tension(landmarks)
        
        # 2. Eye tension (squinting, widening)
        tensions['eye'] = self._calculate_eye_tension(landmarks)
        
        # 3. Mouth tension (tightness, asymmetry)
        tensions['mouth'] = self._calculate_mouth_tension(landmarks)
        
        # 4. Jaw tension (clenching)
        tensions['jaw'] = self._calculate_jaw_tension(landmarks)
        
        # Combine tensions into phase angle
        # Weight factors based on CST micro-expression mapping
        weights = {'brow': 0.25, 'eye': 0.25, 'mouth': 0.3, 'jaw': 0.2}
        
        total_tension = sum(tensions[k] * weights[k] for k in tensions)
        
        # Map tension (0-1) to phase angle (0 - π/2)
        geometric_phase = total_tension * (math.pi / 2)
        
        processing_time = (time.time() - start_time) * 1000
        logger.debug(f"GeometricPhaseMapper: phase={geometric_phase:.4f} rad, time={processing_time:.2f}ms")
        
        return (geometric_phase, tensions)
    
    def _calculate_brow_tension(self, landmarks: list[Tuple[float, float]]) -> float:
        """Calculate brow tension from landmark positions."""
        # Get brow center heights
        left_brow_y = sum(landmarks[i][1] for i in self.BROW_LEFT) / len(self.BROW_LEFT)
        right_brow_y = sum(landmarks[i][1] for i in self.BROW_RIGHT) / len(self.BROW_RIGHT)
        
        # Get eye center as reference
        left_eye_y = sum(landmarks[i][1] for i in self.EYE_LEFT) / len(self.EYE_LEFT)
        right_eye_y = sum(landmarks[i][1] for i in self.EYE_RIGHT) / len(self.EYE_RIGHT)
        
        # Brow-eye distance (normalized)
        # Note: In image coordinates, Y increases downward
        left_height = left_eye_y - left_brow_y
        right_height = right_eye_y - right_brow_y
        
        # Calculate deviation from baseline
        avg_height = (left_height + right_height) / 2
        
        if self._calibrated and self.baseline:
            baseline_height = self._get_baseline_brow_height()
            deviation = abs(avg_height - baseline_height) / (baseline_height + 1e-6)
        else:
            # Use standard relaxed ratio
            face_height = self._get_face_height(landmarks)
            expected = self.RELAXED_BROW_HEIGHT * face_height
            deviation = abs(avg_height - expected) / (expected + 1e-6)
        
        # Also check asymmetry (stress indicator)
        asymmetry = abs(left_height - right_height) / (avg_height + 1e-6)
        
        tension = min(1.0, deviation * 0.5 + asymmetry * 0.5)
        return tension
    
    def _calculate_eye_tension(self, landmarks: list[Tuple[float, float]]) -> float:
        """Calculate eye tension from openness and squinting."""
        # Eye aspect ratio (height / width)
        left_ear = self._eye_aspect_ratio(landmarks, self.EYE_LEFT)
        right_ear = self._eye_aspect_ratio(landmarks, self.EYE_RIGHT)
        
        avg_ear = (left_ear + right_ear) / 2
        
        # Deviation from relaxed openness
        expected_ear = 0.25  # Typical relaxed EAR
        deviation = abs(avg_ear - expected_ear) / expected_ear
        
        # Asymmetry
        asymmetry = abs(left_ear - right_ear) / (avg_ear + 1e-6)
        
        tension = min(1.0, deviation * 0.6 + asymmetry * 0.4)
        return tension
    
    def _calculate_mouth_tension(self, landmarks: list[Tuple[float, float]]) -> float:
        """Calculate mouth tension from tightness and asymmetry."""
        # Mouth aspect ratio (height / width)
        outer_width = self._distance(landmarks[48], landmarks[54])
        outer_height = self._distance(
            ((landmarks[51][0] + landmarks[52][0]) / 2, (landmarks[51][1] + landmarks[52][1]) / 2),
            ((landmarks[57][0] + landmarks[58][0]) / 2, (landmarks[57][1] + landmarks[58][1]) / 2)
        )
        
        mouth_ratio = outer_height / (outer_width + 1e-6)
        
        # Tight mouth = low ratio, open mouth = high ratio
        # Both extremes indicate tension
        expected_ratio = 0.1  # Relaxed mouth ratio
        deviation = abs(mouth_ratio - expected_ratio) / expected_ratio
        
        # Check corner asymmetry (one side raised/lowered)
        left_corner = landmarks[48][1]
        right_corner = landmarks[54][1]
        center = (landmarks[51][1] + landmarks[57][1]) / 2
        
        left_angle = left_corner - center
        right_angle = right_corner - center
        asymmetry = abs(left_angle - right_angle) / (outer_height + 1e-6)
        
        tension = min(1.0, deviation * 0.5 + asymmetry * 0.5)
        return tension
    
    def _calculate_jaw_tension(self, landmarks: list[Tuple[float, float]]) -> float:
        """Calculate jaw tension from clenching indicators."""
        # Jaw width at different heights
        top_width = self._distance(landmarks[2], landmarks[14])
        mid_width = self._distance(landmarks[4], landmarks[12])
        bottom_width = self._distance(landmarks[6], landmarks[10])
        
        # Clenched jaw tends to widen at mid-point
        width_ratio = mid_width / ((top_width + bottom_width) / 2 + 1e-6)
        
        # Deviation from relaxed ratio (typically around 0.95)
        expected_ratio = 0.95
        deviation = abs(width_ratio - expected_ratio) / expected_ratio
        
        tension = min(1.0, deviation * 2.0)
        return tension
    
    def _eye_aspect_ratio(self, landmarks: list[Tuple[float, float]], eye_indices: list[int]) -> float:
        """Calculate Eye Aspect Ratio (EAR) for openness detection."""
        # Vertical distances
        v1 = self._distance(landmarks[eye_indices[1]], landmarks[eye_indices[5]])
        v2 = self._distance(landmarks[eye_indices[2]], landmarks[eye_indices[4]])
        
        # Horizontal distance
        h = self._distance(landmarks[eye_indices[0]], landmarks[eye_indices[3]])
        
        ear = (v1 + v2) / (2.0 * h + 1e-6)
        return ear
    
    def _distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Euclidean distance between two points."""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def _get_face_height(self, landmarks: list[Tuple[float, float]]) -> float:
        """Get approximate face height from landmarks."""
        top = min(landmarks[i][1] for i in self.BROW_LEFT + self.BROW_RIGHT)
        bottom = landmarks[8][1]  # Chin
        return bottom - top
    
    def _get_baseline_brow_height(self) -> float:
        """Get brow height from calibration baseline."""
        if not self.baseline:
            return 20.0  # Default
        
        left_brow_y = sum(self.baseline[i][1] for i in self.BROW_LEFT) / len(self.BROW_LEFT)
        left_eye_y = sum(self.baseline[i][1] for i in self.EYE_LEFT) / len(self.EYE_LEFT)
        return left_eye_y - left_brow_y
    
    def calibrate(self, landmarks: list[Tuple[float, float]]):
        """Set baseline from a relaxed face sample."""
        self.baseline = landmarks.copy()
        self._calibrated = True
        logger.info("GeometricPhaseMapper calibrated with baseline landmarks")


class CSTSensoryBridge:
    """
    Main CST Sensory Bridge - Unified Physics-Based Emotional Analysis.
    
    Combines FrequencyAnalyzer (audio) and GeometricPhaseMapper (visual) into
    a coherent physics state assessment.
    
    Usage:
        bridge = CSTSensoryBridge()
        
        # Analyze audio only
        state = bridge.analyze_audio(audio_buffer)
        
        # Analyze facial landmarks only
        state = bridge.analyze_visual(landmarks)
        
        # Combined analysis (highest accuracy)
        state = bridge.analyze(audio_buffer, landmarks)
    """
    
    # Thresholds for intent detection
    STRESS_PHASE_THRESHOLD = math.pi / 4  # 45 degrees
    HIGH_ENERGY_THRESHOLD = 0.7
    LOW_ENERGY_THRESHOLD = 0.3
    COHERENCE_THRESHOLD = 0.3
    
    def __init__(
        self,
        sample_rate: int = 16000,
        calibration_landmarks: Optional[list[Tuple[float, float]]] = None,
    ):
        """
        Initialize the CST Sensory Bridge.
        
        Args:
            sample_rate: Audio sample rate in Hz
            calibration_landmarks: Optional baseline facial landmarks
        """
        self.frequency_analyzer = FrequencyAnalyzer(sample_rate=sample_rate)
        self.phase_mapper = GeometricPhaseMapper(baseline_landmarks=calibration_landmarks)
        
        logger.info("CSTSensoryBridge initialized")
    
    def analyze_audio(self, audio_buffer: Union[list[float]]) -> CSTState:
        """
        Analyze audio-only input.
        
        Args:
            audio_buffer: Raw PCM audio data
            
        Returns:
            CSTState with emotional_mass populated
        """
        import time
        start_time = time.time()
        
        emotional_mass, frequencies, amplitudes = self.frequency_analyzer.analyze(audio_buffer)
        tremor = self.frequency_analyzer.detect_tremor(audio_buffer)
        
        # Infer state from audio alone
        state = CSTState(
            emotional_mass=emotional_mass,
            geometric_phase=0.0,  # Unknown without visual
            truth_probability=0.5,  # Neutral without visual correlation
            raw_frequencies=frequencies[:10],  # Top 10
            raw_amplitudes=amplitudes[:10],
            audio_duration_ms=len(audio_buffer) / 32.0 if isinstance(audio_buffer, bytes) else 0,
            processing_time_ms=(time.time() - start_time) * 1000,
        )
        
        # Adjust based on tremor (stress indicator)
        if tremor > 0.5:
            state.detected_intent = DetectedIntent.STRESSED
        elif emotional_mass > self.HIGH_ENERGY_THRESHOLD:
            state.detected_intent = DetectedIntent.STRESSED
        elif emotional_mass < self.LOW_ENERGY_THRESHOLD:
            state.detected_intent = DetectedIntent.CALM
        else:
            state.detected_intent = DetectedIntent.UNCERTAIN
        
        return state
    
    def analyze_visual(self, landmarks: list[Tuple[float, float]]) -> CSTState:
        """
        Analyze visual-only input (facial landmarks).
        
        Args:
            landmarks: list of (x, y) facial landmark coordinates
            
        Returns:
            CSTState with geometric_phase populated
        """
        import time
        start_time = time.time()
        
        geometric_phase, tensions = self.phase_mapper.calculate_phase(landmarks)
        
        state = CSTState(
            emotional_mass=0.0,  # Unknown without audio
            geometric_phase=geometric_phase,
            truth_probability=0.5,  # Neutral without audio correlation
            facial_landmarks_used=len(landmarks),
            processing_time_ms=(time.time() - start_time) * 1000,
        )
        
        # Infer state from visual alone
        if geometric_phase > self.STRESS_PHASE_THRESHOLD:
            state.detected_intent = DetectedIntent.STRESSED
        elif geometric_phase < self.STRESS_PHASE_THRESHOLD / 2:
            state.detected_intent = DetectedIntent.CALM
        else:
            state.detected_intent = DetectedIntent.UNCERTAIN
        
        return state
    
    def analyze(
        self,
        audio_buffer: Optional[Union[list[float]]] = None,
        landmarks: Optional[list[Tuple[float, float]]] = None,
    ) -> CSTState:
        """
        Combined audio + visual analysis.
        
        This is the most accurate mode, allowing cross-correlation between
        voice energy and facial tension to detect coherence/deception.
        
        Args:
            audio_buffer: Raw PCM audio data
            landmarks: Facial landmark coordinates
            
        Returns:
            CSTState with full analysis
        """
        import time
        start_time = time.time()
        
        # Analyze individual modalities
        audio_state = self.analyze_audio(audio_buffer) if audio_buffer else None
        visual_state = self.analyze_visual(landmarks) if landmarks else None
        
        # Combine results
        state = CSTState(
            emotional_mass=audio_state.emotional_mass if audio_state else 0.0,
            geometric_phase=visual_state.geometric_phase if visual_state else 0.0,
            raw_frequencies=audio_state.raw_frequencies if audio_state else [],
            raw_amplitudes=audio_state.raw_amplitudes if audio_state else [],
            facial_landmarks_used=visual_state.facial_landmarks_used if visual_state else 0,
            audio_duration_ms=audio_state.audio_duration_ms if audio_state else 0.0,
            processing_time_ms=(time.time() - start_time) * 1000,
        )
        
        # Calculate truth probability from coherence
        if audio_state and visual_state:
            state.truth_probability = self._calculate_truth_probability(
                state.emotional_mass, 
                state.geometric_phase
            )
            state.detected_intent = self._infer_intent(state)
        elif audio_state:
            state.truth_probability = 0.5
            state.detected_intent = audio_state.detected_intent
        elif visual_state:
            state.truth_probability = 0.5
            state.detected_intent = visual_state.detected_intent
        else:
            state.detected_intent = DetectedIntent.UNCERTAIN
        
        return state
    
    def _calculate_truth_probability(
        self, 
        emotional_mass: float, 
        geometric_phase: float
    ) -> float:
        """
        Calculate truth probability from audio/visual coherence.
        
        CST principle: If voice energy and facial tension are coherent
        (both high or both low), the signal is more likely truthful.
        Mismatches (calm voice + tense face, or vice versa) suggest deception.
        """
        # Normalize phase to 0-1 range for comparison
        visual_intensity = min(1.0, geometric_phase / (math.pi / 2))
        audio_intensity = emotional_mass
        
        # Coherence = how similar the two signals are
        difference = abs(audio_intensity - visual_intensity)
        coherence = 1.0 - difference
        
        # Truth probability is high when coherent
        # Apply sigmoid to smooth the transition
        truth_prob = 1.0 / (1.0 + math.exp(-5.0 * (coherence - 0.5)))
        
        return truth_prob
    
    def _infer_intent(self, state: CSTState) -> DetectedIntent:
        """
        Infer emotional intent from combined state.
        """
        is_coherent = state.is_coherent(self.COHERENCE_THRESHOLD)
        is_stressed_audio = state.emotional_mass > self.HIGH_ENERGY_THRESHOLD
        is_stressed_visual = state.geometric_phase > self.STRESS_PHASE_THRESHOLD
        
        if not is_coherent:
            # Mismatch between audio and visual = potential deception
            return DetectedIntent.DECEPTION
        
        if is_stressed_audio or is_stressed_visual:
            return DetectedIntent.STRESSED
        
        if state.emotional_mass < self.LOW_ENERGY_THRESHOLD and \
           state.geometric_phase < self.STRESS_PHASE_THRESHOLD / 2:
            return DetectedIntent.CALM
        
        if state.truth_probability > 0.7:
            return DetectedIntent.HONEST
        
        return DetectedIntent.UNCERTAIN
    
    def calibrate_visual(self, landmarks: list[Tuple[float, float]]):
        """Calibrate the visual mapper with a relaxed face baseline."""
        self.phase_mapper.calibrate(landmarks)
    
    def get_stats(self) -> dict:
        """Get bridge statistics."""
        return {
            "has_numpy": HAS_NUMPY,
            "sample_rate": self.frequency_analyzer.sample_rate,
            "visual_calibrated": self.phase_mapper._calibrated,
        }
