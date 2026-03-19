"""
cosmos Multimodal Sensory Cortex
====================================
Integrates Audio, Visual, and Text analysis into a unified 12D state.
"""

from .multimodal_fusion import UnifiedMultimodalSystem, MultimodalLightToken
from .affective_engine import EmotionalState
from .visual_engine import VisualLightToken
from .audio_engine import MusicalLightToken

_shared_system = None

def get_multimodal_system():
    """Get or create the shared UnifiedMultimodalSystem instance."""
    global _shared_system
    if _shared_system is None:
        try:
            _shared_system = UnifiedMultimodalSystem()
        except Exception as e:
            print(f"Failed to initialize Multimodal System: {e}")
            return None
    return _shared_system

