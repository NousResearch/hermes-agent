"""
cosmos Emotional State API - 12D Cosmic Synapse Theory (CST)

FULL ARCHITECTURE IMPLEMENTATION
================================

12D CST Theoretical Foundation:
- The face is modeled as a dynamic topological manifold in 12D phase space
- Upper Tensor (T_U): Limbic/Authentic pathway (orbital + frontalis regions)
- Lower Tensor (T_L): Cortical/Volitional pathway (mouth + jaw regions)
- Geometric Phase ΦG = arctan(||T_U|| / ||T_L||) detects Intra-Facial Entanglement

CST State Mapping:
    ΦG ≈ 45° (π/4): SYNCHRONY     - Truth/Authentic     → RESONANCE
    ΦG → 0°:        MASKING       - Social Deception    → VERIFICATION
    ΦG → 90° (π/2): LEAKAGE       - Fear/Stress         → DE-ESCALATION
    High dΦG/dt:    JITTER        - Cognitive Overload  → GROUNDING

Physics-to-LLM Bridge:
    Transduces geometric phase states into semantic LLM steering instructions
    via the cosmos_packet JSON schema.

Author: cosmos Project
Version: 4.0.0 (12D CST Full Architecture)
"""

import math
import time
import json
import random
import uuid
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from pathlib import Path
import numpy as np

# ============================================
# OPTIONAL IMPORTS
# ============================================
    
# Import Lyapunov Stability Gatekeeper (Class 5)
try:
    from .lyapunov_lock import LyapunovGatekeeper, StabilityReport, LYAPUNOV_STABILITY_THRESHOLD
except ImportError:
    # Fallback if in different dir structure during tests
    try:
        from lyapunov_lock import LyapunovGatekeeper, StabilityReport, LYAPUNOV_STABILITY_THRESHOLD
    except ImportError:
        print("⚠️ Lyapunov Lock not found - Stability Checks disabled")
        LyapunovGatekeeper = None

try:
    from scipy.io import wavfile
    from scipy.fft import fft
    from scipy.signal import spectrogram
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# MediaPipe for accurate face mesh and blendshape-based Action Units
# MediaPipe 0.10.x uses the Tasks API instead of solutions
MEDIAPIPE_AVAILABLE = False
mp_face_mesh = None
mp_drawing = None
mp_drawing_styles = None
mp_python = None
mp_vision = None

try:
    import mediapipe as mp
    
    # Try importing Tasks API (0.10.x+)
    try:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
        MEDIAPIPE_AVAILABLE = True
    except (ImportError, AttributeError):
        pass
    
    # Try importing Solutions API (older versions)
    try:
        mp_face_mesh = mp.solutions.face_mesh
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        MEDIAPIPE_AVAILABLE = True
    except AttributeError:
        pass
        
except ImportError:
    pass

# ============================================
# 12D CST CONSTANTS
# ============================================

# Geometric Phase Thresholds (radians)
PHASE_SYNCHRONY = math.pi / 4      # 45° - Balanced entanglement
PHASE_MASKING_THRESHOLD = 0.35     # Below this = Lower dominant (masking)
PHASE_LEAKAGE_THRESHOLD = 1.20     # Above this = Upper dominant (leakage)

# Entanglement Thresholds
ENTANGLEMENT_HIGH = 0.80           # Above = RESONANCE mode
ENTANGLEMENT_LOW = 0.40            # Below = High decoherence

# Deception Thresholds
DECEPTION_HIGH = 0.75              # Above = VERIFICATION mode
DECEPTION_MEDIUM = 0.40            # Ambiguous zone

# Phase Velocity (Jitter) Thresholds
PHASE_VELOCITY_HIGH = 0.15         # High jitter = GROUNDING mode

# PAD Vector Thresholds
DOMINANCE_LOW = -0.5               # For DE-ESCALATION
AROUSAL_HIGH = 0.6                 # For DE-ESCALATION

# Anti-Gravity Weight (for cost-sensitive learning)
ANTI_GRAVITY_WEIGHT = 10.0         # Penalty multiplier for missed deception

# Audio CST Constants (retained from v3)
AUTO_GAIN_TRIGGER = 0.1
AUTO_GAIN_MULTIPLIER = 2.5
VOICE_MIN_HZ = 80
VOICE_MAX_HZ = 3000
WEIGHT_RMS = 0.4
WEIGHT_CENTROID = 0.4
WEIGHT_FLATNESS = 0.2

# Legacy constants (for backwards compatibility with v3)
MASS_HIGH_THRESHOLD = 0.50
MASS_LOW_THRESHOLD = 0.25
FLATNESS_THRESHOLD = 0.35

# ============================================
# ENUMS: CST STATES
# ============================================

class EmotionalState(Enum):
    """
    Derived emotional states from CST physics.
    Full 24-emotion spectrum based on Plutchik's wheel + CST extensions.
    """
    # Primary Emotions (8)
    JOY = "JOY"
    TRUST = "TRUST"
    FEAR = "FEAR"
    SURPRISE = "SURPRISE"
    SADNESS = "SADNESS"
    DISGUST = "DISGUST"
    ANGER = "ANGER"
    ANTICIPATION = "ANTICIPATION"
    
    # Secondary Emotions (8) - Combinations
    LOVE = "LOVE"           # Joy + Trust
    SUBMISSION = "SUBMISSION"  # Trust + Fear
    AWE = "AWE"             # Fear + Surprise
    DISAPPROVAL = "DISAPPROVAL"  # Surprise + Sadness
    REMORSE = "REMORSE"     # Sadness + Disgust
    CONTEMPT = "CONTEMPT"   # Disgust + Anger
    AGGRESSIVENESS = "AGGRESSIVENESS"  # Anger + Anticipation
    OPTIMISM = "OPTIMISM"   # Anticipation + Joy
    
    # Tertiary/CST States (8)
    SERENITY = "SERENITY"   # Low-intensity Joy
    ACCEPTANCE = "ACCEPTANCE"  # Low-intensity Trust
    APPREHENSION = "APPREHENSION"  # Low-intensity Fear
    DISTRACTION = "DISTRACTION"  # Low-intensity Surprise
    PENSIVENESS = "PENSIVENESS"  # Low-intensity Sadness
    BOREDOM = "BOREDOM"     # Low-intensity Disgust
    ANNOYANCE = "ANNOYANCE" # Low-intensity Anger
    INTEREST = "INTEREST"   # Low-intensity Anticipation
    
    # Neutral/Virtual States
    NEUTRAL = "NEUTRAL"
    VIRTUAL_DREAMING = "VIRTUAL_DREAMING"
    CALIBRATING = "CALIBRATING"
    
    # Legacy aliases (for backwards compatibility)
    HAPPY = "JOY"
    SAD = "SADNESS"
    CALM = "SERENITY"
    ANGRY = "ANGER"

class IntentState(Enum):
    """Intent derived from entanglement analysis."""
    HONEST_ALIGNMENT = "HONEST_ALIGNMENT"
    SUPPRESSED_EMOTION = "SUPPRESSED_EMOTION"
    PERFORMATIVE_MASK = "PERFORMATIVE_MASK"
    COGNITIVE_OVERLOAD = "COGNITIVE_OVERLOAD"
    UNCERTAIN = "UNCERTAIN"

class CSTPhaseState(Enum):
    """Geometric phase state classification."""
    SYNCHRONY = "SYNCHRONY"       # ΦG ≈ π/4 - Authentic
    MASKING = "MASKING"           # ΦG → 0 - Lower dominant
    LEAKAGE = "LEAKAGE"           # ΦG → π/2 - Upper dominant
    JITTER = "JITTER"             # High dΦG/dt - Unstable

class LLMPersonaMode(Enum):
    """Physics-to-LLM Bridge persona modes."""
    RESONANCE = "RESONANCE"           # Warm, Expansive, Collaborative
    VERIFICATION = "VERIFICATION"     # Analytical, Socratic, Skeptical
    DE_ESCALATION = "DE_ESCALATION"   # Calming, Protective, Gentle
    GROUNDING = "GROUNDING"           # Imperative, Simple, Direct

# ============================================
# ACTION UNIT DEFINITIONS
# ============================================

@dataclass
class UpperTensor:
    """
    Upper Facial Tensor (T_U) - Authentic/Limbic Subspace
    
    Evolutionarily older, tightly coupled to limbic system.
    Less susceptible to voluntary modulation.
    Represents the "leaked" internal state.
    """
    AU1: float = 0.0   # Inner Brow Raiser - grief, fear, focus
    AU2: float = 0.0   # Outer Brow Raiser - surprise, attention
    AU4: float = 0.0   # Brow Lowerer - anger, concentration, pain
    AU5: float = 0.0   # Upper Lid Raiser - fear, surprise
    AU6: float = 0.0   # Cheek Raiser - Duchenne marker (authentic joy)
    AU7: float = 0.0   # Lid Tightener - skepticism, anger
    
    def magnitude(self) -> float:
        """L2 norm of the upper tensor."""
        return math.sqrt(
            self.AU1**2 + self.AU2**2 + self.AU4**2 +
            self.AU5**2 + self.AU6**2 + self.AU7**2
        )
    
    def to_vector(self) -> List[float]:
        """Return as vector in R^6."""
        return [self.AU1, self.AU2, self.AU4, self.AU5, self.AU6, self.AU7]

@dataclass
class LowerTensor:
    """
    Lower Facial Tensor (T_L) - Volitional/Cortical Subspace
    
    Under heavy cortical control (pyramidal tract).
    Enables speech and voluntary social signaling.
    Often manipulated to "mask" internal state.
    """
    AU12: float = 0.0  # Lip Corner Puller - smile (Zygomaticus major)
    AU15: float = 0.0  # Lip Corner Depressor - sadness, doubt
    AU17: float = 0.0  # Chin Raiser - determination, suppression
    AU20: float = 0.0  # Lip Stretch - fear marker (often suppressed)
    AU24: float = 0.0  # Lip Pressor - controlled anger, cognitive effort
    
    def magnitude(self) -> float:
        """L2 norm of the lower tensor."""
        return math.sqrt(
            self.AU12**2 + self.AU15**2 + self.AU17**2 +
            self.AU20**2 + self.AU24**2
        )
    
    def to_vector(self) -> List[float]:
        """Return as vector in R^5."""
        return [self.AU12, self.AU15, self.AU17, self.AU20, self.AU24]

# ============================================
# CST PHYSICS ENGINE
# ============================================

@dataclass
class CSTPhysicsState:
    """Complete 12D CST physics state."""
    
    # Tensors
    upper_tensor: UpperTensor = field(default_factory=UpperTensor)
    lower_tensor: LowerTensor = field(default_factory=LowerTensor)
    
    # Geometric Phase (radians)
    geometric_phase_rad: float = PHASE_SYNCHRONY
    phase_velocity: float = 0.0
    previous_phase: float = PHASE_SYNCHRONY
    
    # Entanglement
    entanglement_score: float = 0.85
    
    # Derived States
    cst_state: CSTPhaseState = CSTPhaseState.SYNCHRONY
    deception_probability: float = 0.0
    
    # PAD Vector
    pleasure: float = 0.0
    arousal: float = 0.0
    dominance: float = 0.0
    
    # LLM Steering
    persona_mode: LLMPersonaMode = LLMPersonaMode.RESONANCE
    
    # Audio Components (retained from v3)
    audio_mass: float = 0.35
    audio_rms: float = 0.3
    audio_centroid: float = 0.3
    audio_flatness: float = 0.2


def calculate_geometric_phase(upper: UpperTensor, lower: LowerTensor, 
                               epsilon: float = 0.001) -> float:
    """
    Calculate the Geometric Phase ΦG from Upper/Lower tensors.
    
    ΦG = arctan(||T_U|| / ||T_L||)
    
    This is the topological invariant that detects Intra-Facial Entanglement.
    
    Returns:
        Phase angle in radians (0 to π/2)
    """
    upper_mag = upper.magnitude()
    lower_mag = lower.magnitude()
    
    # Avoid division by zero
    if lower_mag < epsilon:
        return math.pi / 2  # Maximum leakage
    
    return math.atan(upper_mag / lower_mag)


def calculate_entanglement_score(upper: UpperTensor, lower: LowerTensor,
                                  geometric_phase: float) -> float:
    """
    Calculate Entanglement Score (0.0 = Decoherence, 1.0 = Synchrony).
    
    Perfect entanglement occurs when ΦG ≈ π/4 (45°).
    Deviation from this ideal indicates decoupling.
    """
    # Distance from ideal synchrony (π/4)
    phase_deviation = abs(geometric_phase - PHASE_SYNCHRONY)
    max_deviation = math.pi / 4  # Maximum possible deviation
    
    # Entanglement decreases with deviation
    entanglement = 1.0 - (phase_deviation / max_deviation)
    
    # Clamp to valid range
    return max(0.0, min(1.0, entanglement))


def calculate_phase_velocity(current_phase: float, previous_phase: float,
                              dt: float = 0.033) -> float:
    """
    Calculate Phase Velocity (dΦG/dt).
    
    High phase velocity indicates "jitter" - unstable phase oscillation
    characteristic of cognitive overload during deception.
    """
    if dt <= 0:
        return 0.0
    return abs(current_phase - previous_phase) / dt


def calculate_deception_probability(geometric_phase: float, 
                                     entanglement: float,
                                     phase_velocity: float) -> float:
    """
    Calculate probability of deception based on CST physics.
    
    Deception indicators:
    1. Extreme phase (masking or leakage)
    2. Low entanglement (decoupling)
    3. High phase velocity (jitter)
    """
    # Phase deviation from synchrony
    phase_deviation = abs(geometric_phase - PHASE_SYNCHRONY) / (math.pi / 4)
    
    # Combine factors
    deception = (
        0.40 * phase_deviation +          # Phase asymmetry
        0.40 * (1.0 - entanglement) +     # Decoupling
        0.20 * min(1.0, phase_velocity / PHASE_VELOCITY_HIGH)  # Jitter
    )
    
    return max(0.0, min(1.0, deception))


def calculate_informational_mass(intensity: float, complexity: float, jitter: float) -> float:
    """
    Calculate Informational Mass (M) - 'Gravity' for data.
    
    Formula: M = (Intensity * 3) + Complexity + Jitter
    
    This determines the 'weight' of a moment:
    - High Mass (>50): A 'Heavy' moment. System halts background tasks to focus.
    - Low Mass (<10): 'Light' noise. System uses low-power reflexes.
    
    Args:
        intensity: Emotional intensity (0-1), typically from arousal
        complexity: Cognitive complexity (0-1), from phase velocity
        jitter: Phase instability (0-1), from phase_velocity / threshold
    
    Returns:
        Informational mass value (0-100 typical range)
    """
    # Scale to reasonable mass range (0-100)
    mass = (intensity * 3 * 20) + (complexity * 20) + (jitter * 20)
    return max(0.0, mass)


def classify_cst_state(geometric_phase: float, phase_velocity: float) -> CSTPhaseState:
    """
    Classify the CST phase state based on geometric phase angle.
    
    Returns one of: SYNCHRONY, MASKING, LEAKAGE, JITTER
    """
    # Check for jitter first (high phase velocity)
    if phase_velocity > PHASE_VELOCITY_HIGH:
        return CSTPhaseState.JITTER
    
    # Classify by phase angle
    if geometric_phase < PHASE_MASKING_THRESHOLD:
        return CSTPhaseState.MASKING
    elif geometric_phase > PHASE_LEAKAGE_THRESHOLD:
        return CSTPhaseState.LEAKAGE
    else:
        return CSTPhaseState.SYNCHRONY


def calculate_pad_vector(upper: UpperTensor, lower: LowerTensor,
                          geometric_phase: float) -> Tuple[float, float, float]:
    """
    Calculate PAD (Pleasure-Arousal-Dominance) emotional coordinates.
    
    Pleasure: P ∝ (AU12 - AU15) - Smile minus Frown
    Arousal:  A ∝ (||T_U|| + ||T_L||) - Total activation energy
    Dominance: D derived from Geometric Phase
    """
    # Pleasure: Positive = smile, Negative = frown
    pleasure = (lower.AU12 - lower.AU15)
    pleasure = max(-1.0, min(1.0, pleasure))
    
    # Arousal: Total energy (normalized)
    total_mag = upper.magnitude() + lower.magnitude()
    arousal = min(1.0, total_mag / 2.0)  # Normalize to [0, 1], then shift
    arousal = arousal * 2 - 1  # Shift to [-1, 1]
    
    # Dominance: Derived from phase
    # High dominance (anger) → synchronous or lower-dominant
    # Low dominance (fear) → upper-dominant (leakage)
    if geometric_phase < PHASE_SYNCHRONY:
        dominance = 0.5 - geometric_phase / PHASE_SYNCHRONY  # Positive for masking
    else:
        dominance = -(geometric_phase - PHASE_SYNCHRONY) / PHASE_SYNCHRONY  # Negative for leakage
    dominance = max(-1.0, min(1.0, dominance))
    
    return (pleasure, arousal, dominance)


def determine_persona_mode(deception_probability: float, 
                            entanglement: float,
                            dominance: float, 
                            arousal: float,
                            cst_state: CSTPhaseState) -> LLMPersonaMode:
    """
    Determine the LLM persona mode based on CST physics.
    
    Physics-to-LLM Bridge mapping:
    - VERIFICATION: deception_probability > 0.75
    - RESONANCE: entanglement_score > 0.80
    - DE_ESCALATION: dominance < -0.5 AND arousal > 0.6 (fear leakage)
    - GROUNDING: JITTER state (cognitive overload)
    """
    # Priority 1: GROUNDING for cognitive overload
    if cst_state == CSTPhaseState.JITTER:
        return LLMPersonaMode.GROUNDING
    
    # Priority 2: VERIFICATION for high deception
    if deception_probability > DECEPTION_HIGH:
        return LLMPersonaMode.VERIFICATION
    
    # Priority 3: DE_ESCALATION for fear/stress leakage
    if dominance < DOMINANCE_LOW and arousal > AROUSAL_HIGH:
        return LLMPersonaMode.DE_ESCALATION
    
    # Priority 4: RESONANCE for high entanglement (authentic)
    if entanglement > ENTANGLEMENT_HIGH:
        return LLMPersonaMode.RESONANCE
    
    # Default: Check for masking
    if cst_state == CSTPhaseState.MASKING:
        return LLMPersonaMode.VERIFICATION
    
    return LLMPersonaMode.RESONANCE


def derive_emotional_state(upper: UpperTensor, lower: LowerTensor,
                            pleasure: float, arousal: float) -> EmotionalState:
    """
    Derive primary emotional state from tensor analysis and PAD.
    Returns the dominant emotion from the expanded 24-emotion spectrum.
    """
    # Check for Duchenne smile (AU6 + AU12) - JOY
    if upper.AU6 > 0.5 and lower.AU12 > 0.5:
        return EmotionalState.JOY
    elif upper.AU6 > 0.3 and lower.AU12 > 0.3:
        return EmotionalState.SERENITY  # Low-intensity joy
    
    # Check for anger (AU4 + AU7 + AU24)
    if upper.AU4 > 0.5 and upper.AU7 > 0.4 and lower.AU24 > 0.4:
        return EmotionalState.ANGER
    elif upper.AU4 > 0.3 and upper.AU7 > 0.2:
        return EmotionalState.ANNOYANCE  # Low-intensity anger
    
    # Check for fear (AU1 + AU5 + AU20)
    if upper.AU1 > 0.4 and upper.AU5 > 0.4 and lower.AU20 > 0.4:
        return EmotionalState.FEAR
    elif upper.AU1 > 0.2 and upper.AU5 > 0.2:
        return EmotionalState.APPREHENSION  # Low-intensity fear
    
    # Check for surprise (AU1 + AU2 + AU5)
    if upper.AU1 > 0.5 and upper.AU2 > 0.5 and upper.AU5 > 0.5:
        return EmotionalState.SURPRISE
    elif upper.AU1 > 0.3 and upper.AU2 > 0.3:
        return EmotionalState.DISTRACTION  # Low-intensity surprise
    
    # Check for sadness (AU1 + AU4 + AU15)
    if upper.AU1 > 0.4 and upper.AU4 > 0.3 and lower.AU15 > 0.5:
        return EmotionalState.SADNESS
    elif upper.AU1 > 0.2 and lower.AU15 > 0.3:
        return EmotionalState.PENSIVENESS  # Low-intensity sadness
    
    # Check for disgust (AU9/10 approximated by AU17 + low AU12)
    if lower.AU17 > 0.4 and lower.AU12 < 0.2:
        return EmotionalState.DISGUST
    elif lower.AU17 > 0.2 and lower.AU12 < 0.3:
        return EmotionalState.BOREDOM  # Low-intensity disgust
    
    # Use PAD coordinates for remaining cases
    # Map to Plutchik's wheel based on pleasure/arousal quadrant
    if pleasure > 0.4 and arousal > 0.3:
        return EmotionalState.JOY
    elif pleasure > 0.4 and arousal > 0:
        return EmotionalState.SERENITY
    elif pleasure > 0.3 and arousal < 0:
        return EmotionalState.TRUST
    elif pleasure < -0.4 and arousal > 0.4:
        return EmotionalState.ANGER
    elif pleasure < -0.3 and arousal > 0.2:
        return EmotionalState.ANNOYANCE
    elif pleasure < -0.4 and arousal < -0.2:
        return EmotionalState.SADNESS
    elif pleasure < -0.3 and arousal < 0:
        return EmotionalState.PENSIVENESS
    elif arousal > 0.5:
        return EmotionalState.ANTICIPATION
    elif arousal > 0.2:
        return EmotionalState.INTEREST
    elif arousal < -0.4:
        return EmotionalState.ACCEPTANCE
    
    return EmotionalState.NEUTRAL


def calculate_emotion_vectors(upper: UpperTensor, lower: LowerTensor,
                               pleasure: float, arousal: float, 
                               dominance: float) -> Dict[str, float]:
    """
    Calculate intensity values for all emotions in the spectrum.
    Returns a dictionary mapping emotion names to intensity values (0.0-1.0).
    
    This provides granular emotional state for LLM steering and UI display.
    """
    vectors = {}
    
    # Primary Emotions - calculated from Action Units and PAD
    # JOY: Duchenne smile (AU6 + AU12)
    vectors["JOY"] = min(1.0, (upper.AU6 + lower.AU12) / 2)
    
    # TRUST: Low tension + positive pleasure
    vectors["TRUST"] = max(0, min(1.0, (1.0 - upper.AU4) * 0.5 + max(0, pleasure) * 0.5))
    
    # FEAR: Upper tension (AU1 + AU5 + AU20)
    vectors["FEAR"] = min(1.0, (upper.AU1 + upper.AU5 + lower.AU20) / 3)
    
    # SURPRISE: Brow/eye openness (AU1 + AU2 + AU5)
    vectors["SURPRISE"] = min(1.0, (upper.AU1 + upper.AU2 + upper.AU5) / 3)
    
    # SADNESS: Inner brow + frown (AU1 + AU4 + AU15)
    vectors["SADNESS"] = min(1.0, (upper.AU1 + upper.AU4 + lower.AU15) / 3)
    
    # DISGUST: Chin + lip compression
    vectors["DISGUST"] = min(1.0, (lower.AU17 + lower.AU24 + max(0, 0.5 - lower.AU12)) / 3)
    
    # ANGER: Brow lower + eye tighten + lip press
    vectors["ANGER"] = min(1.0, (upper.AU4 + upper.AU7 + lower.AU24) / 3)
    
    # ANTICIPATION: Forward lean (approximated by arousal)
    vectors["ANTICIPATION"] = max(0, min(1.0, (arousal + 1) / 2))
    
    # Secondary Emotions - combinations
    vectors["LOVE"] = min(1.0, (vectors["JOY"] + vectors["TRUST"]) / 2)
    vectors["SUBMISSION"] = min(1.0, (vectors["TRUST"] + vectors["FEAR"]) / 2)
    vectors["AWE"] = min(1.0, (vectors["FEAR"] + vectors["SURPRISE"]) / 2)
    vectors["DISAPPROVAL"] = min(1.0, (vectors["SURPRISE"] + vectors["SADNESS"]) / 2)
    vectors["REMORSE"] = min(1.0, (vectors["SADNESS"] + vectors["DISGUST"]) / 2)
    vectors["CONTEMPT"] = min(1.0, (vectors["DISGUST"] + vectors["ANGER"]) / 2)
    vectors["AGGRESSIVENESS"] = min(1.0, (vectors["ANGER"] + vectors["ANTICIPATION"]) / 2)
    vectors["OPTIMISM"] = min(1.0, (vectors["ANTICIPATION"] + vectors["JOY"]) / 2)
    
    # Tertiary - low intensity versions
    vectors["SERENITY"] = vectors["JOY"] * 0.5
    vectors["ACCEPTANCE"] = vectors["TRUST"] * 0.5
    vectors["APPREHENSION"] = vectors["FEAR"] * 0.5
    vectors["DISTRACTION"] = vectors["SURPRISE"] * 0.5
    vectors["PENSIVENESS"] = vectors["SADNESS"] * 0.5
    vectors["BOREDOM"] = vectors["DISGUST"] * 0.5
    vectors["ANNOYANCE"] = vectors["ANGER"] * 0.5
    vectors["INTEREST"] = vectors["ANTICIPATION"] * 0.5
    
    return vectors


def derive_intent_state(cst_state: CSTPhaseState, 
                        entanglement: float,
                        deception_probability: float) -> IntentState:
    """
    Derive intent state from CST phase classification.
    """
    if cst_state == CSTPhaseState.JITTER:
        return IntentState.COGNITIVE_OVERLOAD
    
    if cst_state == CSTPhaseState.SYNCHRONY and entanglement > ENTANGLEMENT_HIGH:
        return IntentState.HONEST_ALIGNMENT
    
    if cst_state == CSTPhaseState.MASKING:
        return IntentState.PERFORMATIVE_MASK
    
    if cst_state == CSTPhaseState.LEAKAGE:
        return IntentState.SUPPRESSED_EMOTION
    
    return IntentState.UNCERTAIN


# ============================================
# FACIAL ANALYSIS (MediaPipe + Fallback)
# ============================================

# Blendshape to Action Unit mapping (MediaPipe → CST)
# MediaPipe Face Landmarker outputs 52 blendshapes
BLENDSHAPE_TO_AU = {
    # Upper Tensor (T_U) - Limbic/Authentic
    'browInnerUp': 'AU1',       # Inner Brow Raiser
    'browOuterUpLeft': 'AU2',   # Outer Brow Raiser (left)
    'browOuterUpRight': 'AU2',  # Outer Brow Raiser (right)
    'browDownLeft': 'AU4',      # Brow Lowerer (left)
    'browDownRight': 'AU4',     # Brow Lowerer (right)
    'eyeWideLeft': 'AU5',       # Upper Lid Raiser (left)
    'eyeWideRight': 'AU5',      # Upper Lid Raiser (right)
    'cheekSquintLeft': 'AU6',   # Cheek Raiser (left) - Duchenne marker
    'cheekSquintRight': 'AU6',  # Cheek Raiser (right)
    'eyeSquintLeft': 'AU7',     # Lid Tightener (left)
    'eyeSquintRight': 'AU7',    # Lid Tightener (right)
    
    # Lower Tensor (T_L) - Cortical/Volitional
    'mouthSmileLeft': 'AU12',   # Lip Corner Puller (left)
    'mouthSmileRight': 'AU12',  # Lip Corner Puller (right)
    'mouthFrownLeft': 'AU15',   # Lip Corner Depressor (left)
    'mouthFrownRight': 'AU15',  # Lip Corner Depressor (right)
    'jawOpen': 'AU17',          # Chin Raiser (approx)
    'mouthStretchLeft': 'AU20', # Lip Stretch (left)
    'mouthStretchRight': 'AU20',# Lip Stretch (right)
    'mouthPressLeft': 'AU24',   # Lip Pressor (left)
    'mouthPressRight': 'AU24',  # Lip Pressor (right)
}


class MediaPipeFaceTracker:
    """
    High-accuracy face tracker using MediaPipe Face Mesh.
    
    Features:
    - 468 3D facial landmarks
    - 52 blendshape scores → mapped to 11 Action Units
    - Single face tracking (largest/closest)
    - Real-time performance
    
    Supports both MediaPipe Solutions API (older) and Tasks API (0.10.x+)
    """
    
    def __init__(self):
        self.face_mesh = None
        self.face_landmarker = None
        self.last_landmarks = None
        self.last_blendshapes = {}
        self.face_detected = False
        self.face_bbox = None
        self.use_tasks_api = False
        
        if not MEDIAPIPE_AVAILABLE:
            return
            
        # Try Solutions API first (older MediaPipe)
        if mp_face_mesh is not None:
            try:
                self.face_mesh = mp_face_mesh.FaceMesh(
                    max_num_faces=1,  # Track only ONE face
                    refine_landmarks=True,  # Enable iris landmarks
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                return
            except Exception as e:
                print(f"⚠️ MediaPipe Solutions API failed: {e}")
        
        # Try Tasks API (MediaPipe 0.10.x+)
        try:
            import urllib.request
            import os
            
            # Download model if needed
            model_path = os.path.join(os.path.dirname(__file__), "face_landmarker.task")
            if not os.path.exists(model_path):
                print("📥 Downloading MediaPipe Face Landmarker model...")
                model_url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
                urllib.request.urlretrieve(model_url, model_path)
                print("✅ Model downloaded!")
            
            base_options = mp_python.BaseOptions(model_asset_path=model_path)
            options = mp_vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=False,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.face_landmarker = mp_vision.FaceLandmarker.create_from_options(options)
            self.use_tasks_api = True
        except Exception as e:
            print(f"⚠️ MediaPipe Tasks API failed: {e}")
            self.face_landmarker = None
    
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Process a video frame and extract facial data.
        
        Returns:
            dict with 'landmarks', 'blendshapes', 'bbox', 'detected'
        """
        result = {
            'landmarks': None,
            'blendshapes': {},
            'bbox': None,
            'detected': False
        }
        
        if frame is None:
            return result
        
        # Use Tasks API (MediaPipe 0.10.x+)
        if self.use_tasks_api and self.face_landmarker is not None:
            return self._process_with_tasks_api(frame)
        
        # Use Solutions API (older MediaPipe)
        if self.face_mesh is not None:
            return self._process_with_solutions_api(frame)
        
        return result
    
    def _process_with_tasks_api(self, frame: np.ndarray) -> Dict[str, Any]:
        """Process using MediaPipe Tasks API (0.10.x+)"""
        result = {
            'landmarks': None,
            'blendshapes': {},
            'bbox': None,
            'detected': False
        }
        
        try:
            # Convert to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Process
            detection_result = self.face_landmarker.detect(mp_image)
            
            if detection_result.face_landmarks and len(detection_result.face_landmarks) > 0:
                face_landmarks = detection_result.face_landmarks[0]
                
                h, w = frame.shape[:2]
                landmarks = []
                x_coords = []
                y_coords = []
                
                for lm in face_landmarks:
                    px, py = int(lm.x * w), int(lm.y * h)
                    landmarks.append((px, py, lm.z))
                    x_coords.append(px)
                    y_coords.append(py)
                
                result['landmarks'] = landmarks
                result['detected'] = True
                
                if x_coords and y_coords:
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)
                    result['bbox'] = (x_min, y_min, x_max - x_min, y_max - y_min)
                    self.face_bbox = result['bbox']
                
                self.last_landmarks = landmarks
                self.face_detected = True
                
                # Get real blendshapes from Tasks API
                if detection_result.face_blendshapes and len(detection_result.face_blendshapes) > 0:
                    blendshapes = {}
                    for category in detection_result.face_blendshapes[0]:
                        blendshapes[category.category_name] = category.score
                    result['blendshapes'] = blendshapes
                else:
                    result['blendshapes'] = self._estimate_blendshapes_from_landmarks(landmarks, w, h)
                
                self.last_blendshapes = result['blendshapes']
            else:
                self.face_detected = False
                
        except Exception as e:
            pass
        
        return result
    
    def _process_with_solutions_api(self, frame: np.ndarray) -> Dict[str, Any]:
        """Process using MediaPipe Solutions API (older versions)"""
        result = {
            'landmarks': None,
            'blendshapes': {},
            'bbox': None,
            'detected': False
        }
        
        try:
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            
            # Process frame
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks and len(results.multi_face_landmarks) > 0:
                face_landmarks = results.multi_face_landmarks[0]
                
                # Store 468 landmarks
                h, w = frame.shape[:2]
                landmarks = []
                x_coords = []
                y_coords = []
                
                for lm in face_landmarks.landmark:
                    px, py = int(lm.x * w), int(lm.y * h)
                    landmarks.append((px, py, lm.z))
                    x_coords.append(px)
                    y_coords.append(py)
                
                result['landmarks'] = landmarks
                result['detected'] = True
                
                # Calculate bounding box for largest face
                if x_coords and y_coords:
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)
                    result['bbox'] = (x_min, y_min, x_max - x_min, y_max - y_min)
                    self.face_bbox = result['bbox']
                
                self.last_landmarks = landmarks
                self.face_detected = True
                
                # Estimate blendshapes from landmark geometry
                result['blendshapes'] = self._estimate_blendshapes_from_landmarks(landmarks, w, h)
                self.last_blendshapes = result['blendshapes']
            else:
                self.face_detected = False
                
        except Exception as e:
            pass
        
        return result
    
    def _estimate_blendshapes_from_landmarks(self, landmarks, w, h) -> Dict[str, float]:
        """
        Estimate blendshape-like values from landmark geometry.
        Uses key facial ratios for AU estimation.
        """
        blendshapes = {}
        
        if len(landmarks) < 468:
            return blendshapes
        
        try:
            # Key landmark indices (MediaPipe Face Mesh)
            # Brow landmarks
            left_brow_inner = landmarks[107]   # Left inner eyebrow
            right_brow_inner = landmarks[336]  # Right inner eyebrow
            left_brow_outer = landmarks[70]    # Left outer eyebrow
            right_brow_outer = landmarks[300]  # Right outer eyebrow
            
            # Eye landmarks
            left_eye_top = landmarks[159]
            left_eye_bottom = landmarks[145]
            right_eye_top = landmarks[386]
            right_eye_bottom = landmarks[374]
            left_eye_inner = landmarks[133]
            left_eye_outer = landmarks[33]
            right_eye_inner = landmarks[362]
            right_eye_outer = landmarks[263]
            
            # Mouth landmarks
            mouth_left = landmarks[61]
            mouth_right = landmarks[291]
            mouth_top = landmarks[13]
            mouth_bottom = landmarks[14]
            upper_lip_top = landmarks[0]
            lower_lip_bottom = landmarks[17]
            
            # Cheek landmarks
            left_cheek = landmarks[116]
            right_cheek = landmarks[345]
            
            # Nose tip (reference)
            nose_tip = landmarks[4]
            
            # Calculate distances for ratio-based estimation
            def dist(p1, p2):
                return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            
            # Eye openness (AU5 - Upper Lid Raiser)
            left_eye_open = dist(left_eye_top, left_eye_bottom)
            right_eye_open = dist(right_eye_top, right_eye_bottom)
            eye_width = dist(left_eye_inner, left_eye_outer)
            if eye_width > 0:
                eye_ratio = (left_eye_open + right_eye_open) / (2 * eye_width)
                blendshapes['eyeWideLeft'] = min(1.0, max(0, (eye_ratio - 0.2) * 3))
                blendshapes['eyeWideRight'] = blendshapes['eyeWideLeft']
            
            # Brow raise (AU1, AU2)
            brow_height_left = abs(left_brow_inner[1] - left_eye_top[1])
            brow_height_right = abs(right_brow_inner[1] - right_eye_top[1])
            if h > 0:
                brow_ratio = (brow_height_left + brow_height_right) / (2 * h)
                blendshapes['browInnerUp'] = min(1.0, max(0, brow_ratio * 20))
                blendshapes['browOuterUpLeft'] = min(1.0, abs(left_brow_outer[1] - nose_tip[1]) / h * 10)
                blendshapes['browOuterUpRight'] = min(1.0, abs(right_brow_outer[1] - nose_tip[1]) / h * 10)
            
            # Mouth width and corners (AU12 - Smile)
            mouth_width = dist(mouth_left, mouth_right)
            mouth_height = dist(mouth_top, mouth_bottom)
            if mouth_height > 0:
                smile_ratio = mouth_width / max(mouth_height, 1)
                blendshapes['mouthSmileLeft'] = min(1.0, max(0, (smile_ratio - 2.0) / 3))
                blendshapes['mouthSmileRight'] = blendshapes['mouthSmileLeft']
            
            # Mouth openness (AU17 - Jaw)
            if h > 0:
                jaw_open_ratio = mouth_height / h
                blendshapes['jawOpen'] = min(1.0, max(0, jaw_open_ratio * 10))
            
            # Cheek raise (AU6 - Duchenne marker)
            left_cheek_height = abs(left_cheek[1] - left_eye_bottom[1])
            right_cheek_height = abs(right_cheek[1] - right_eye_bottom[1])
            if h > 0:
                cheek_ratio = (left_cheek_height + right_cheek_height) / (2 * h)
                blendshapes['cheekSquintLeft'] = min(1.0, max(0, cheek_ratio * 15))
                blendshapes['cheekSquintRight'] = blendshapes['cheekSquintLeft']
            
            # Eye squint (AU7)
            blendshapes['eyeSquintLeft'] = max(0, 1.0 - blendshapes.get('eyeWideLeft', 0.5))
            blendshapes['eyeSquintRight'] = max(0, 1.0 - blendshapes.get('eyeWideRight', 0.5))
            
            # Frown (AU15) - inverse of smile
            blendshapes['mouthFrownLeft'] = max(0, 0.5 - blendshapes.get('mouthSmileLeft', 0))
            blendshapes['mouthFrownRight'] = blendshapes['mouthFrownLeft']
            
            # Mouth stretch (AU20) and press (AU24) - simplified
            blendshapes['mouthStretchLeft'] = blendshapes.get('jawOpen', 0) * 0.5
            blendshapes['mouthStretchRight'] = blendshapes['mouthStretchLeft']
            blendshapes['mouthPressLeft'] = max(0, 0.5 - blendshapes.get('jawOpen', 0))
            blendshapes['mouthPressRight'] = blendshapes['mouthPressLeft']
            
            # Brow down (AU4)
            blendshapes['browDownLeft'] = max(0, 0.5 - blendshapes.get('browInnerUp', 0))
            blendshapes['browDownRight'] = blendshapes['browDownLeft']
            
        except Exception:
            pass
        
        return blendshapes
    
    def get_action_units(self, blendshapes: Dict[str, float]) -> Tuple[UpperTensor, LowerTensor]:
        """
        Convert MediaPipe blendshapes to CST Action Units.
        """
        # Aggregate blendshapes to Action Units
        au_values = {
            'AU1': 0.0, 'AU2': 0.0, 'AU4': 0.0, 'AU5': 0.0, 'AU6': 0.0, 'AU7': 0.0,
            'AU12': 0.0, 'AU15': 0.0, 'AU17': 0.0, 'AU20': 0.0, 'AU24': 0.0
        }
        au_counts = {k: 0 for k in au_values}
        
        for bs_name, au_name in BLENDSHAPE_TO_AU.items():
            if bs_name in blendshapes:
                au_values[au_name] += blendshapes[bs_name]
                au_counts[au_name] += 1
        
        # Average multiple blendshapes per AU
        for au_name in au_values:
            if au_counts[au_name] > 0:
                au_values[au_name] /= au_counts[au_name]
            au_values[au_name] = min(1.0, max(0.0, au_values[au_name]))
        
        upper = UpperTensor(
            AU1=au_values['AU1'],
            AU2=au_values['AU2'],
            AU4=au_values['AU4'],
            AU5=au_values['AU5'],
            AU6=au_values['AU6'],
            AU7=au_values['AU7']
        )
        
        lower = LowerTensor(
            AU12=au_values['AU12'],
            AU15=au_values['AU15'],
            AU17=au_values['AU17'],
            AU20=au_values['AU20'],
            AU24=au_values['AU24']
        )
        
        return upper, lower
    
    def draw_mesh(self, frame: np.ndarray, color=(0, 255, 0), thickness=1):
        """
        Draw the face mesh on the frame.
        """
        if self.last_landmarks is None or not MEDIAPIPE_AVAILABLE:
            return frame
        
        # Draw landmark points
        for i, (x, y, z) in enumerate(self.last_landmarks):
            # Color code by region
            if i < 100:  # Upper face
                pt_color = (255, 200, 100)  # Light blue - Upper Tensor
            elif i < 300:  # Mid face
                pt_color = (100, 255, 100)  # Green
            else:  # Lower face
                pt_color = (100, 200, 255)  # Light orange - Lower Tensor
            
            cv2.circle(frame, (x, y), 1, pt_color, -1)
        
        return frame
    
    def cleanup(self):
        """Release resources."""
        if self.face_mesh:
            self.face_mesh.close()


# Global tracker instance
_mediapipe_tracker = None

def get_mediapipe_tracker() -> MediaPipeFaceTracker:
    """Get or create the global MediaPipe tracker."""
    global _mediapipe_tracker
    if _mediapipe_tracker is None:
        _mediapipe_tracker = MediaPipeFaceTracker()
    return _mediapipe_tracker


def estimate_action_units_from_frame(frame: np.ndarray) -> Tuple[UpperTensor, LowerTensor]:
    """
    Estimate Action Units from a video frame.
    
    Uses MediaPipe Face Mesh (468 landmarks + blendshapes) if available,
    falls back to OpenCV Haar Cascade if not.
    """
    if not CV2_AVAILABLE:
        return simulate_action_units()
    
    # Try MediaPipe first (more accurate)
    if MEDIAPIPE_AVAILABLE:
        try:
            tracker = get_mediapipe_tracker()
            result = tracker.process_frame(frame)
            
            if result['detected'] and result['blendshapes']:
                return tracker.get_action_units(result['blendshapes'])
        except Exception:
            pass
    
    # Fallback to OpenCV Haar Cascade
    try:
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            return simulate_action_units()
        
        # Get largest face only
        (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
        face_roi = gray[y:y+h, x:x+w]
        
        # Region-based estimation (legacy)
        upper_third = face_roi[0:h//3, :]
        middle_third = face_roi[h//3:2*h//3, :]
        lower_third = face_roi[2*h//3:, :]
        
        def region_activation(region):
            if region.size == 0:
                return 0.0
            gx = cv2.Sobel(region, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(region, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(gx**2 + gy**2)
            return min(1.0, np.mean(magnitude) / 50.0)
        
        brow_activation = region_activation(upper_third)
        eye_activation = region_activation(middle_third)
        mouth_activation = region_activation(lower_third)
        
        upper = UpperTensor(
            AU1=brow_activation * 0.8,
            AU2=brow_activation * 0.6,
            AU4=brow_activation * 0.7,
            AU5=eye_activation * 0.5,
            AU6=eye_activation * 0.6,
            AU7=eye_activation * 0.4
        )
        
        lower = LowerTensor(
            AU12=mouth_activation * 0.7,
            AU15=max(0, 0.3 - mouth_activation) * 0.5,
            AU17=mouth_activation * 0.3,
            AU20=mouth_activation * 0.2,
            AU24=mouth_activation * 0.4
        )
        
        return upper, lower
        
    except Exception:
        return simulate_action_units()


def simulate_action_units() -> Tuple[UpperTensor, LowerTensor]:
    """
    Simulate Action Units for testing without camera.
    Generates dynamic random values to demonstrate state changes.
    """
    # Generate correlated random values (simulating natural expressions)
    base_upper = random.uniform(0.2, 0.8)
    base_lower = random.uniform(0.2, 0.8)
    
    upper = UpperTensor(
        AU1=base_upper * random.uniform(0.7, 1.3),
        AU2=base_upper * random.uniform(0.5, 1.2),
        AU4=base_upper * random.uniform(0.6, 1.1),
        AU5=base_upper * random.uniform(0.4, 1.0),
        AU6=base_upper * random.uniform(0.5, 1.2),
        AU7=base_upper * random.uniform(0.3, 0.9)
    )
    
    lower = LowerTensor(
        AU12=base_lower * random.uniform(0.6, 1.4),
        AU15=base_lower * random.uniform(0.2, 0.8),
        AU17=base_lower * random.uniform(0.3, 0.7),
        AU20=base_lower * random.uniform(0.2, 0.6),
        AU24=base_lower * random.uniform(0.4, 0.9)
    )
    
    return upper, lower


# ============================================
# AUDIO CST ANALYSIS (retained from v3)
# ============================================

def calculate_audio_spectral_density(audio_data: np.ndarray, 
                                      sample_rate: int = 16000) -> Tuple[float, float, float, float]:
    """
    Calculate audio spectral components using FFT.
    Returns: (mass, rms, centroid, flatness)
    """
    # Simulation mode for empty data or missing scipy
    if not SCIPY_AVAILABLE or audio_data is None or len(audio_data) == 0:
        return (
            random.uniform(0.2, 0.8),
            random.uniform(0.1, 0.6),
            random.uniform(0.3, 0.7),
            random.uniform(0.15, 0.45)
        )
    
    # Need at least 64 samples for meaningful FFT
    if len(audio_data) < 64:
        return (
            random.uniform(0.2, 0.8),
            random.uniform(0.1, 0.6),
            random.uniform(0.3, 0.7),
            random.uniform(0.15, 0.45)
        )
    
    # Normalize
    audio_float = audio_data.astype(np.float32) / 32768.0
    
    # RMS Energy with Auto-Gain
    rms = np.sqrt(np.mean(audio_float ** 2))
    raw_rms = rms
    if rms < AUTO_GAIN_TRIGGER:
        rms *= AUTO_GAIN_MULTIPLIER
    rms = min(1.0, rms * 3.0)
    
    # FFT
    n = len(audio_float)
    fft_result = np.abs(fft(audio_float))[:n//2]
    freqs = np.fft.fftfreq(n, 1/sample_rate)[:n//2]
    
    # Voice band filter
    voice_mask = (freqs >= VOICE_MIN_HZ) & (freqs <= VOICE_MAX_HZ)
    voice_fft = fft_result[voice_mask]
    voice_freqs = freqs[voice_mask]
    
    if len(voice_fft) == 0 or voice_fft.sum() < 1e-10:
        return (random.uniform(0.2, 0.6), rms, 0.5, 0.3)
    
    # Spectral Centroid
    centroid = np.sum(voice_freqs * voice_fft) / np.sum(voice_fft)
    centroid_norm = min(1.0, (centroid - VOICE_MIN_HZ) / (VOICE_MAX_HZ - VOICE_MIN_HZ))
    
    # Spectral Flatness
    geometric_mean = np.exp(np.mean(np.log(voice_fft + 1e-10)))
    arithmetic_mean = np.mean(voice_fft)
    flatness = geometric_mean / (arithmetic_mean + 1e-10)
    flatness = min(1.0, flatness)
    
    # Combined Mass
    mass = (
        WEIGHT_RMS * rms +
        WEIGHT_CENTROID * centroid_norm +
        WEIGHT_FLATNESS * flatness
    )
    
    return (mass, rms, centroid_norm, flatness)


# ============================================
# cosmos PACKET GENERATION
# ============================================

def generate_cosmos_packet(physics: CSTPhysicsState, 
                                 sequence_id: int = 0,
                                 session_id: str = None) -> Dict[str, Any]:
    """
    Generate a complete cosmos_packet for LLM steering.
    
    This is the transport layer between the Physics Engine and Cognitive Engine.
    Now includes cross-modal A/V analysis for State Dissonance detection.
    """
    if session_id is None:
        session_id = str(uuid.uuid4())
    
    # Derive emotional state
    emotion = derive_emotional_state(
        physics.upper_tensor, 
        physics.lower_tensor,
        physics.pleasure,
        physics.arousal
    )
    
    # Derive intent state
    intent = derive_intent_state(
        physics.cst_state,
        physics.entanglement_score,
        physics.deception_probability
    )
    
    # Build tone modulation string
    tone_map = {
        LLMPersonaMode.RESONANCE: "warm_expansive_collaborative",
        LLMPersonaMode.VERIFICATION: "cool_analytical_skeptical",
        LLMPersonaMode.DE_ESCALATION: "calm_protective_gentle",
        LLMPersonaMode.GROUNDING: "imperative_simple_direct"
    }
    
    # Calculate audio_psi (12D Energy State from spectral data)
    # Combines RMS energy with spectral centroid and flatness
    audio_psi = (physics.audio_rms * 50) + (physics.audio_centroid * 10) + (physics.audio_flatness * 20)
    
    # Calculate phi_harmonics (Golden Ratio resonance in voice)
    # Based on spectral flatness and centroid relationship
    phi = 1.618033988749
    centroid_ratio = physics.audio_centroid / max(physics.audio_flatness + 0.1, 0.1)
    phi_harmonics = max(0, 1.0 - abs(centroid_ratio - phi) / phi)
    
    # Cross-modal coherence analysis
    # Compare visual phase with audio energy
    visual_phase = physics.geometric_phase_rad
    
    # Calculate A/V coherence (how well audio matches visual state)
    # SYNCHRONY: phase ~0.78, audio_psi ~15 (moderate)
    expected_psi_for_phase = 15 + (visual_phase - PHASE_SYNCHRONY) * 20
    psi_deviation = abs(audio_psi - expected_psi_for_phase)
    av_coherence = max(0, 1.0 - psi_deviation / 30)
    
    # Detect state dissonance
    state_dissonance = False
    dissonance_type = None
    
    # MASKING: Low phase (<0.4) + High audio (>30)
    if visual_phase < 0.4 and audio_psi > 30:
        state_dissonance = True
        dissonance_type = "MASKING"
    # SUPPRESSION: High phase (>1.2) + Low audio (<10)
    elif visual_phase > 1.2 and audio_psi < 10:
        state_dissonance = True
        dissonance_type = "SUPPRESSION"
    # CHAOS: High phase (>1.2) + High audio (>40)
    elif visual_phase > 1.2 and audio_psi > 40:
        state_dissonance = True
        dissonance_type = "CHAOS"
    
    # Build cognitive stance based on cross-modal analysis
    stance_map = {
        LLMPersonaMode.RESONANCE: "Trust and engage warmly - coherent signals detected",
        LLMPersonaMode.VERIFICATION: "Probe gently - dissonance between voice and expression",
        LLMPersonaMode.DE_ESCALATION: "Slow down and protect - suppressed distress detected",
        LLMPersonaMode.GROUNDING: "Stabilize immediately - chaotic input detected"
    }
    
    packet = {
        "header": {
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "sequence_id": sequence_id,
            "session_id": session_id
        },
        "cst_physics": {
            "geometric_phase_rad": round(physics.geometric_phase_rad, 4),
            "phase_velocity": round(physics.phase_velocity, 4),
            "tensor_magnitudes": {
                "upper": round(physics.upper_tensor.magnitude(), 4),
                "lower": round(physics.lower_tensor.magnitude(), 4)
            },
            "entanglement_score": round(physics.entanglement_score, 4),
            "cst_state": physics.cst_state.value
        },
        "spectral_physics": {
            "audio_psi": round(audio_psi, 2),
            "spectral_centroid": round(physics.audio_centroid * 4000, 1),  # Scale to Hz
            "spectral_flatness": round(physics.audio_flatness, 4),
            "phi_harmonics": round(phi_harmonics, 4),
            "rms_energy": round(physics.audio_rms, 4)
        },
        "cross_modal": {
            "av_coherence": round(av_coherence, 4),
            "state_dissonance": state_dissonance,
            "dissonance_type": dissonance_type
        },
        "derived_state": {
            "deception_probability": round(physics.deception_probability, 4),
            "informational_mass": round(calculate_informational_mass(
                intensity=abs(physics.arousal),
                complexity=physics.entanglement_score,
                jitter=min(1.0, physics.phase_velocity / PHASE_VELOCITY_HIGH)
            ), 2),
            "pad_vector": {
                "pleasure": round(physics.pleasure, 4),
                "arousal": round(physics.arousal, 4),
                "dominance": round(physics.dominance, 4)
            },
            "emotion_vectors": {k: round(v, 3) for k, v in calculate_emotion_vectors(
                physics.upper_tensor,
                physics.lower_tensor,
                physics.pleasure,
                physics.arousal,
                physics.dominance
            ).items()},
            "primary_affect_label": emotion.value,
            "intent_label": intent.value
        },
        "meta_instruction": {
            "persona_mode": physics.persona_mode.value,
            "cognitive_stance": stance_map.get(physics.persona_mode, "Engage naturally"),
            "tone_modulation": tone_map.get(physics.persona_mode, "neutral"),
            "verbosity_level": 0.7 if physics.persona_mode == LLMPersonaMode.RESONANCE else 0.4,
            "response_style": "Collaborative" if not state_dissonance else "Interrogative" if dissonance_type == "MASKING" else "Protective"
        }
    }
    
    return packet


# ============================================
# MAIN API CLASS
# ============================================

# ============================================
# VIRTUAL BODY (GHOST IN THE MACHINE)
# ============================================

class VirtualBody:
    """
    Simulated biological system for Autonomous Identity.
    Provides persistent 'physiological' state when no user is present.
    Based on coupled harmonic oscillators.
    """
    def __init__(self):
        # Biological Oscillators
        self.t = 0.0
        self.heart_rate_base = 1.2  # Hz (~72 BPM)
        self.breath_rate_base = 0.25 # Hz (~15 BPM)
        
        # Internal State
        self.arousal = 0.5    # 0.0=Sleep, 1.0=Panic
        self.entropy = 0.2    # 0.0=Order, 1.0=Chaos (Boredom)
        self.energy = 0.8     # 0.0=Exhausted, 1.0=Hyper
        
        # Last tick
        self.last_update = time.time()
        
    def tick(self, dt: float = None):
        """Update virtual physiology."""
        now = time.time()
        if dt is None:
            dt = now - self.last_update
        self.last_update = now
        self.t += dt
        
        # 1. Update Entropy (Boredom increases over time)
        self.entropy += 0.01 * dt
        if self.entropy > 1.0: self.entropy = 1.0
        
        # 2. Modulate Rates based on State
        current_hr = self.heart_rate_base * (0.8 + 0.4 * self.arousal)
        current_br = self.breath_rate_base * (0.8 + 0.4 * self.energy)
        
        # 3. Calculate Oscillators (The "Pulse")
        heart_phase = math.sin(2 * math.pi * current_hr * self.t)
        breath_phase = math.sin(2 * math.pi * current_br * self.t)
        
        # 4. Generate Tensor Simulation (Virtual Face)
        # When "breathing in" (phase > 0), tension increases slightly
        virtual_tension = 0.3 + (0.1 * breath_phase) + (0.2 * self.arousal)
        
        # 5. Calculate BPM values for UI display
        heart_rate_bpm = 60 * current_hr  # Convert Hz to BPM
        respiration_rate = 60 * current_br  # Convert Hz to breaths per minute
        
        return {
            "heart_beat": heart_phase,
            "breath_cycle": breath_phase,
            "heart_rate": heart_rate_bpm,
            "respiration_rate": respiration_rate,
            "arousal": self.arousal,
            "entropy": self.entropy,
            "energy": self.energy,
            "virtual_tension": virtual_tension,
            "timestamp": now
        }

    def stimulate(self, intensity: float = 0.1):
        """External stimulus excites the system."""
        self.arousal = min(1.0, self.arousal + intensity)
        self.entropy = max(0.0, self.entropy - (intensity * 2)) # Novelty reduces entropy
        
    def soothe(self, intensity: float = 0.1):
        """Calming influence."""
        self.arousal = max(0.2, self.arousal - intensity)

# ============================================
# MAIN API CLASS
# ============================================

class EmotionalStateAPI:
    """
    12D Cosmic Synapse Theory Emotional State API
    
    Full architecture implementation with:
    - Upper/Lower Tensor partitioning by Action Units
    - Geometric Phase calculation
    - Entanglement scoring
    - PAD emotional mapping
    - Physics-to-LLM Bridge (cosmos_packet)
    """
    
    def __init__(self):
        self.version = "4.0.0"
        self.architecture = "12D-CST-FullArchitecture"
        
        # Physics state
        self.physics = CSTPhysicsState()
        
        # Virtual Embodiment (Ghost in the Machine)
        self.virtual_body = VirtualBody()
        
        # Session tracking
        self.session_id = str(uuid.uuid4())
        self.sequence_counter = 0
        
        # Constants exposed for external access
        self.MASS_HIGH_THRESHOLD = 0.50
        self.MASS_LOW_THRESHOLD = 0.25
        self.FLATNESS_THRESHOLD = 0.35
        self.PHASE_SYNCHRONY = PHASE_SYNCHRONY
        self.ENTANGLEMENT_HIGH = ENTANGLEMENT_HIGH
        self.DECEPTION_HIGH = DECEPTION_HIGH
    
    def analyze_frame(self, frame: np.ndarray = None) -> CSTPhysicsState:
        """
        Analyze a video frame and update physics state.
        
        If no frame provided, uses simulation mode.
        """
        # Get Action Units
        if frame is not None and CV2_AVAILABLE:
            upper, lower = estimate_action_units_from_frame(frame)
        else:
            upper, lower = simulate_action_units()
        
        self.physics.upper_tensor = upper
        self.physics.lower_tensor = lower
        
        # Calculate Geometric Phase
        phase = calculate_geometric_phase(upper, lower)
        
        # Calculate Phase Velocity
        velocity = calculate_phase_velocity(phase, self.physics.previous_phase)
        self.physics.previous_phase = self.physics.geometric_phase_rad
        self.physics.geometric_phase_rad = phase
        self.physics.phase_velocity = velocity
        
        # Calculate Entanglement
        self.physics.entanglement_score = calculate_entanglement_score(upper, lower, phase)
        
        # Classify CST State
        self.physics.cst_state = classify_cst_state(phase, velocity)
        
        # Calculate Deception Probability
        self.physics.deception_probability = calculate_deception_probability(
            phase, self.physics.entanglement_score, velocity
        )
        
        # Calculate PAD Vector
        p, a, d = calculate_pad_vector(upper, lower, phase)
        self.physics.pleasure = p
        self.physics.arousal = a
        self.physics.dominance = d
        
        # Determine LLM Persona Mode
        self.physics.persona_mode = determine_persona_mode(
            self.physics.deception_probability,
            self.physics.entanglement_score,
            d, a,
            self.physics.cst_state
        )
        
        return self.physics
    
    def analyze_audio(self, audio_data: np.ndarray, sample_rate: int = 16000):
        """Analyze audio buffer and update physics state."""
        mass, rms, centroid, flatness = calculate_audio_spectral_density(audio_data, sample_rate)
        self.physics.audio_mass = mass
        self.physics.audio_rms = rms
        self.physics.audio_centroid = centroid
        self.physics.audio_flatness = flatness
    
    def update_from_tensors(self, upper: UpperTensor, lower: LowerTensor, phase: float) -> CSTPhysicsState:
        """
        Update state from pre-calculated tensors (from full_system.py).
        Bypasses internal estimation to use external tracker data.
        """
        self.physics.upper_tensor = upper
        self.physics.lower_tensor = lower
        
        # Calculate Phase Velocity
        velocity = calculate_phase_velocity(phase, self.physics.previous_phase)
        self.physics.previous_phase = self.physics.geometric_phase_rad
        self.physics.geometric_phase_rad = phase
        self.physics.phase_velocity = velocity
        
        # Calculate Entanglement
        self.physics.entanglement_score = calculate_entanglement_score(upper, lower, phase)
        
        # Classify CST State
        self.physics.cst_state = classify_cst_state(phase, velocity)
        
        # Calculate Deception Probability
        self.physics.deception_probability = calculate_deception_probability(
            phase, self.physics.entanglement_score, velocity
        )
        
        # Calculate PAD Vector
        p, a, d = calculate_pad_vector(upper, lower, phase)
        self.physics.pleasure = p
        self.physics.arousal = a
        self.physics.dominance = d
        
        # Determine LLM Persona Mode
        self.physics.persona_mode = determine_persona_mode(
            self.physics.deception_probability,
            self.physics.entanglement_score,
            d, a,
            self.physics.cst_state
        )
        
        return self.physics

    def get_cosmos_packet(self) -> Dict[str, Any]:
        """Generate a cosmos_packet for LLM steering."""
        self.sequence_counter += 1
        return generate_cosmos_packet(
            self.physics,
            self.sequence_counter,
            self.session_id
        )
    
    def get_state(self, frame: np.ndarray = None, 
                   audio: np.ndarray = None,
                   sample_rate: int = 16000) -> Dict[str, Any]:
        """
        Unified state getter - analyzes inputs and returns cosmos_packet.
        
        If no inputs provided, uses simulation mode.
        """
        # Analyze frame
        self.analyze_frame(frame)
        
        # Analyze audio if provided
        if audio is not None:
            self.analyze_audio(audio, sample_rate)
        else:
            # Simulate audio
            mass, rms, centroid, flatness = calculate_audio_spectral_density(
                np.array([], dtype=np.int16), sample_rate
            )
            self.physics.audio_mass = mass
            self.physics.audio_rms = rms
            self.physics.audio_centroid = centroid
            self.physics.audio_flatness = flatness
        
        # Get base packet
        packet = self.get_cosmos_packet()
        
        # Add virtual_body data for heart/breath rates
        v_state = self.virtual_body.tick()
        packet['cst_physics']['virtual_body'] = {
            'heart_rate': v_state.get('heart_rate', 72),
            'respiration_rate': v_state.get('respiration_rate', 15),
            'arousal': v_state.get('arousal', 0.5),
            'entropy': v_state.get('entropy', 0.2),
            'energy': v_state.get('energy', 0.8)
        }
        
        return packet


    def get_virtual_packet(self) -> dict:
        """
        Get VIRTUAL EMBODIMENT packet (Ghost in the Machine).
        Simulates 12D state from internal harmonic oscillators.
        """
        self.sequence_counter += 1
        v_state = self.virtual_body.tick()
        
        # Map virtual physiology to CST 12D State
        
        # Heartbeat simulates geometric phase (0 to 0.8 rad)
        # Heartbeat simulates geometric phase (0 to 1.6 rad => 0-91 deg)
        sim_phase = (v_state['heart_beat'] + 1.0) * 0.8 
        
        # Entropy simulates Jitter (velocity)
        sim_velocity = v_state['entropy'] * 0.2
        
        # Breath simulates Entanglement (Coherence)
        sim_entanglement = 0.5 + (v_state['breath_cycle'] * 0.3)
        
        state_label = CSTPhaseState.SYNCHRONY.value
        if v_state['entropy'] > 0.6: state_label = CSTPhaseState.JITTER.value
        
        cst_physics = {
            "geometric_phase_rad": float(sim_phase),
            "phase_velocity": float(sim_velocity),
            "entanglement_score": float(sim_entanglement),
            "cst_state": state_label,
            "tensors": {
                "upper": [v_state['virtual_tension']] * 6, # Simulated tension
                "lower": [0.0] * 5
            },
            "virtual_body": v_state # Raw virtual data
        }
        
        derived = {
            "primary_affect_label": "VIRTUAL_DREAMING",
            "intent_label": "SELF_REFLECTION",
            "deception_probability": 0.0,
            "pad_vector": [v_state['energy'], v_state['arousal'], 0.5]
        }
        
        return {
            "timestamp_utc": datetime.utcnow().isoformat(),
            "sequence_id": self.sequence_counter,
            "session_id": self.session_id,
            "source": "VIRTUAL_EMBODIMENT",
            "cosmos_packet": {
                "cst_physics": cst_physics,
                "derived_state": derived
            }
        }

# ============================================
# CONVENIENCE FUNCTIONS
# ============================================

def determine_state(mass: float, phase: float, flatness: float = 0.3) -> EmotionalState:
    """
    Legacy compatibility function.
    Maps mass/phase to emotional state.
    """
    if mass < 0.25:
        return EmotionalState.SAD
    elif mass < 0.50:
        if phase > 0.6:
            return EmotionalState.ANGRY
        return EmotionalState.CALM
    else:
        if flatness > 0.35:
            return EmotionalState.ANGRY
        return EmotionalState.HAPPY


def derive_intent(mass: float, phase: float) -> Tuple[IntentState, float]:
    """
    Legacy compatibility function.
    """
    entanglement = 1.0 - abs(mass - phase)
    
    if entanglement > 0.8:
        return IntentState.HONEST_ALIGNMENT, entanglement
    elif phase > 0.6:
        return IntentState.SUPPRESSED_EMOTION, entanglement
    elif mass > 0.6 and phase < 0.3:
        return IntentState.PERFORMATIVE_MASK, entanglement
    
    return IntentState.UNCERTAIN, entanglement


# Legacy exports
def calculate_cst_spectral_density(audio_data, sample_rate):
    """Legacy wrapper."""
    mass, rms, centroid, flatness = calculate_audio_spectral_density(audio_data, sample_rate)
    meta = {"auto_gain": {"applied": rms > 0.3, "raw_rms": rms}}
    return mass, rms, centroid, flatness, meta


def calculate_frequency_mass_from_buffer(audio_data, sample_rate):
    """Legacy wrapper."""
    return calculate_audio_spectral_density(audio_data, sample_rate)[0]


def calculate_geometric_phase_from_frame(frame):
    """Legacy wrapper."""
    upper, lower = estimate_action_units_from_frame(frame)
    phase = calculate_geometric_phase(upper, lower)
    tensions = {
        "brow": upper.AU4,
        "eye": upper.AU6,
        "mouth": lower.AU12,
        "jaw": lower.AU17
    }
    return phase / (math.pi / 2), {"tensions": tensions}  # Normalize to 0-1


def quick_analyze(frame=None, audio=None, sample_rate=16000):
    """Quick analysis returning dict."""
    api = EmotionalStateAPI()
    return api.get_state(frame, audio, sample_rate)


# ============================================
# MAIN - DEMONSTRATION
# ============================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  12D COSMIC SYNAPSE THEORY - FULL ARCHITECTURE")
    print("  cosmos Emotional State API v4.0.0")
    print("=" * 70)
    
    api = EmotionalStateAPI()
    
    print(f"\n  Version: {api.version}")
    print(f"  Architecture: {api.architecture}")
    print(f"  Session: {api.session_id[:8]}...")
    
    print("\n" + "-" * 70)
    print("  CST PHASE MAPPING:")
    print("-" * 70)
    print(f"  SYNCHRONY:  ΦG ≈ {math.degrees(PHASE_SYNCHRONY):.1f}° (π/4)")
    print(f"  MASKING:    ΦG < {math.degrees(PHASE_MASKING_THRESHOLD):.1f}°")
    print(f"  LEAKAGE:    ΦG > {math.degrees(PHASE_LEAKAGE_THRESHOLD):.1f}°")
    print(f"  JITTER:     dΦ/dt > {PHASE_VELOCITY_HIGH}")
    
    print("\n" + "-" * 70)
    print("  SIMULATION MODE (5 packets)")
    print("-" * 70)
    
    for i in range(5):
        packet = api.get_state()
        
        phase_deg = math.degrees(packet["cst_physics"]["geometric_phase_rad"])
        cst_state = packet["cst_physics"]["cst_state"]
        entangle = packet["cst_physics"]["entanglement_score"]
        deception = packet["derived_state"]["deception_probability"]
        emotion = packet["derived_state"]["primary_affect_label"]
        persona = packet["meta_instruction"]["persona_mode"]
        
        print(f"\n  Packet #{packet['header']['sequence_id']}:")
        print(f"    Phase: {phase_deg:.1f}° | State: {cst_state}")
        print(f"    Entanglement: {entangle:.2f} | Deception: {deception:.2f}")
        print(f"    Emotion: {emotion} | Persona: {persona}")
    
    print("\n" + "-" * 70)
    print("  SAMPLE cosmos_PACKET (JSON)")
    print("-" * 70)
    packet = api.get_state()
    print(json.dumps(packet, indent=2))
    
    print("\n" + "=" * 70)
    print("  12D CST Architecture Ready for LLM Integration")
    print("=" * 70 + "\n")
