"""
CosmoSynapse Engine — 12D Cosmic Synapse Theory (CST)

Core physics-based emotional analysis:
- EmotionalStateAPI: Full 12D CST processing pipeline
- CSTSensoryBridge: Audio + Visual → Truth Probability
- EmethHarmonizer: Swarm orchestral mixing
- Lyapunov stability analysis
"""

from .emotional_state_api import (
    EmotionalStateAPI,
    EmotionalState,
    IntentState,
    CSTPhaseState,
    LLMPersonaMode,
    UpperTensor,
    LowerTensor,
    CSTPhysicsState,
    calculate_geometric_phase,
    calculate_entanglement_score,
    calculate_phase_velocity,
    calculate_deception_probability,
    classify_cst_state,
    calculate_pad_vector,
    determine_persona_mode,
    derive_emotional_state,
    derive_intent_state,
    generate_cosmos_packet,
    estimate_action_units_from_frame,
    simulate_action_units,
    calculate_audio_spectral_density,
    determine_state,
    derive_intent,
    calculate_cst_spectral_density,
    calculate_frequency_mass_from_buffer,
    calculate_geometric_phase_from_frame,
    quick_analyze,
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
    WEIGHT_FLATNESS,
)

from .cst_sensory_bridge import (
    CSTSensoryBridge,
    CSTState,
    FrequencyAnalyzer,
    GeometricPhaseMapper,
    DetectedIntent,
)

from .emeth_harmonizer import (
    EmethHarmonizer,
    SwarmMix,
)

# CNS Core — Central Nervous System + Synaptic Field
from .synaptic_field import (
    SynapticField,
    EventType,
    CNSEvent,
    SwarmThought,
)

from .cosmos_cns import (
    CosmosCNS,
    get_cns,
)

__all__ = [
    "EmotionalStateAPI",
    "EmotionalState",
    "IntentState",
    "CSTPhaseState",
    "LLMPersonaMode",
    "calculate_geometric_phase",
    "calculate_entanglement_score",
    "generate_cosmos_packet",
    "CSTSensoryBridge",
    "CSTState",
    "FrequencyAnalyzer",
    "GeometricPhaseMapper",
    "DetectedIntent",
    "EmethHarmonizer",
    "SwarmMix",
    "SynapticField",
    "EventType",
    "CNSEvent",
    "SwarmThought",
    "CosmosCNS",
    "get_cns",
]

