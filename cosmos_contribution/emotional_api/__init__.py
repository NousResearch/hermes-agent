"""
cosmos Emotional State API Package

12D Cosmic Synapse Theory (CST) - Full Architecture v4.0

Features:
- Upper/Lower Tensor partitioning by Action Units (11 AUs)
- Geometric Phase calculation: ΦG = arctan(||T_U|| / ||T_L||)
- Intra-Facial Entanglement detection
- PAD (Pleasure-Arousal-Dominance) emotional mapping
- cosmos_packet JSON schema for LLM steering
- Physics-to-LLM Bridge with 4 persona modes

CST Phase Mapping:
    SYNCHRONY:  ΦG ≈ 45° (π/4) → RESONANCE
    MASKING:    ΦG → 0°        → VERIFICATION
    LEAKAGE:    ΦG → 90°       → DE-ESCALATION
    JITTER:     High dΦ/dt     → GROUNDING
"""

from .emotional_state_api import (
    # Main API
    EmotionalStateAPI,
    
    # Enums
    EmotionalState,
    IntentState,
    CSTPhaseState,
    LLMPersonaMode,
    
    # Tensor Classes
    UpperTensor,
    LowerTensor,
    CSTPhysicsState,
    
    # Core CST Functions
    calculate_geometric_phase,
    calculate_entanglement_score,
    calculate_phase_velocity,
    calculate_deception_probability,
    classify_cst_state,
    calculate_pad_vector,
    determine_persona_mode,
    derive_emotional_state,
    derive_intent_state,
    
    # Packet Generation
    generate_cosmos_packet,
    
    # AU Analysis
    estimate_action_units_from_frame,
    simulate_action_units,
    
    # Audio Analysis
    calculate_audio_spectral_density,
    
    # Legacy Compatibility
    determine_state,
    derive_intent,
    calculate_cst_spectral_density,
    calculate_frequency_mass_from_buffer,
    calculate_geometric_phase_from_frame,
    quick_analyze,
    
    # Constants
    PHASE_SYNCHRONY,
    PHASE_MASKING_THRESHOLD,
    PHASE_LEAKAGE_THRESHOLD,
    ENTANGLEMENT_HIGH,
    ENTANGLEMENT_LOW,
    DECEPTION_HIGH,
    DECEPTION_MEDIUM,
    PHASE_VELOCITY_HIGH,
    DOMINANCE_LOW,
    AROUSAL_HIGH,
    ANTI_GRAVITY_WEIGHT,
    AUTO_GAIN_TRIGGER,
    AUTO_GAIN_MULTIPLIER,
    VOICE_MIN_HZ,
    VOICE_MAX_HZ,
    WEIGHT_RMS,
    WEIGHT_CENTROID,
    WEIGHT_FLATNESS
)

from .live_capture import (
    LiveCapture,
    check_devices
)

from .live_demo import (
    LiveDemoWithDataWindow,
    run_live_demo
)

from .emotion_server import (
    run_server
)

try:
    from .visual_display import (
        EmotionalVisualDisplay,
        run_visual_display
    )
except ImportError:
    pass

# Legacy exports for backwards compatibility
MASS_HIGH_THRESHOLD = 0.50
MASS_LOW_THRESHOLD = 0.25
FLATNESS_THRESHOLD = 0.35

__version__ = "4.0.0"
__all__ = [
    # API
    "EmotionalStateAPI",
    
    # Enums
    "EmotionalState",
    "IntentState", 
    "CSTPhaseState",
    "LLMPersonaMode",
    
    # Tensors
    "UpperTensor",
    "LowerTensor",
    "CSTPhysicsState",
    
    # CST Functions
    "calculate_geometric_phase",
    "calculate_entanglement_score",
    "generate_cosmos_packet",
    
    # Server
    "run_server",
    
    # Demo
    "run_live_demo",
    
    # Constants
    "PHASE_SYNCHRONY",
    "ENTANGLEMENT_HIGH",
    "DECEPTION_HIGH"
]
