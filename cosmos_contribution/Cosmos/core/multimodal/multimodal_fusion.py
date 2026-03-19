"""
MULTIMODAL FUSION
=================
Unifies audio, visual, and text into single 12D representation.
"""

import numpy as np
import sys
from pathlib import Path

# Import subsystems
sys.path.append(str(Path(__file__).parent))
from visual_engine import VisualLightToken
from affective_engine import EmotionalState, detect_emotion_from_audio, detect_emotion_from_vision, detect_emotion_from_text

#  Audio imports (assumes audio_engine.py exists)
try:
    from audio_engine import MusicalLightToken
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("Warning: audio_engine not found. Audio fusion disabled.")

# ============================================================================
# MULTIMODAL LIGHT TOKEN
# ============================================================================

class MultimodalLightToken:
    """Unified token for audio + vision + text"""
    
    def __init__(self, audio=None, image=None, text=None, sample_rate=44100):
        self.has_audio = audio is not None
        self.has_vision = image is not None
        self.has_text = text is not None
        
        # Create modality-specific tokens
        if self.has_audio and AUDIO_AVAILABLE:
            self.audio_token = MusicalLightToken(audio, sample_rate)
        else:
            self.audio_token = None
        
        if self.has_vision:
            self.visual_token = VisualLightToken(image)
        else:
            self.visual_token = None
        
        self.text = text
        
        # Fused 12D embedding
        self.embedding_12d = self._fuse_embeddings()
        
        # Cross-modal resonance
        self.resonance = self.cross_modal_resonance()
    
    def _fuse_embeddings(self):
        """Fuse modalities into unified 12D representation"""
        embeddings = []
        
        if self.audio_token:
            embeddings.append(self.audio_token.embedding.to_vector())
        
        if self.visual_token:
            embeddings.append(self.visual_token.embedding.to_vector())
        
        if len(embeddings) == 0:
            # Text-only (create zero embedding)
            return np.zeros(12)
        elif len(embeddings) == 1:
            return embeddings[0]
        else:
            # Average multiple modalities
            return np.mean(embeddings, axis=0)
    
    def cross_modal_resonance(self):
        """
        Compute resonance between audio and visual frequencies
        
        Checks if audio frequency matches visual color frequency
        """
        if not (self.audio_token and self.visual_token):
            return 0.0
        
        # Audio fundamental
        audio_freq = self.audio_token.fundamental_hz if self.audio_token.fundamental_hz else 0
        
        # Visual color frequency
        visual_freq = self.visual_token.embedding.color_frequency
        
        if audio_freq == 0 or visual_freq == 0:
            return 0.0
        
        # Scale visual to audio range (visible light is ~10^14 Hz, audio is ~10^2 Hz)
        # Map 430-770 THz → 20-800 Hz
        visual_freq_scaled = (visual_freq - 430e12) / (770e12 - 430e12) * 780 + 20
        
        # Check harmonic relationship
        ratio = visual_freq_scaled / audio_freq
        
        for n in range(1, 9):
            if abs(ratio - n) < 0.1 or abs(ratio - 1/n) < 0.1:
                return 1.0 / n  # Stronger for lower harmonics
        
        return 0.0

# ============================================================================
# UNIFIED MULTIMODAL SYSTEM
# ============================================================================

class UnifiedMultimodalSystem:
    """Complete system integrating all modalities"""
    
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.memory = []  # Store all multimodal tokens
        
        # Current emotional state
        self.emotional_state = EmotionalState(valence=0.0, arousal=0.5, dominance=0.5)
        
        # Cognitive processor
        from affective_engine import CognitiveProcessor
        self.cognitive = CognitiveProcessor()
    
    def process_multimodal_input(self, audio=None, image=None, text=None):
        """
        Process dict combination of modalities
        
        Returns:
            token: MultimodalLightToken
            emotion: EmotionalState
            thought: str or None
        """
        # Create multimodal token
        token = MultimodalLightToken(audio, image, text, self.sample_rate)
        
        # Detect emotions from each modality
        emotions = []
        
        if token.audio_token:
            emotions.append(detect_emotion_from_audio(token.audio_token))
        
        if token.visual_token:
            emotions.append(detect_emotion_from_vision(token.visual_token))
        
        if text:
            emotions.append(detect_emotion_from_text(text))
        
        # Update emotional state (blend)
        if emotions:
            avg_valence = np.mean([e.valence for e in emotions])
            avg_arousal = np.mean([e.arousal for e in emotions])
            avg_dominance = np.mean([e.dominance for e in emotions])
            
            # Smooth update
            alpha = 0.3
            self.emotional_state.valence = (
                alpha * avg_valence + (1 - alpha) * self.emotional_state.valence
            )
            self.emotional_state.arousal = (
                alpha * avg_arousal + (1 - alpha) * self.emotional_state.arousal
            )
            self.emotional_state.dominance = (
                alpha * avg_dominance + (1 - alpha) * self.emotional_state.dominance
            )
        
        # Generate thought
        thought = None
        if len(self.memory) > 0:
            old_state = self.memory[-1].embedding_12d
            new_state = token.embedding_12d
            
            context = []
            if audio is not None: context.append("audio")
            if image is not None: context.append("visual")
            if text: context.append("text")
            context_str = "+".join(context) if context else "input"
            
            thought = self.cognitive.observe_state_transition(old_state, new_state, context_str)
        
        # Store in memory
        self.memory.append(token)
        
        return token, self.emotional_state, thought
    
    def get_modulated_parameters(self):
        """Get learning parameters modulated by emotion"""
        modulation = self.emotional_state.to_12d_modulation()
        
        return {
            'k': 1.0 * modulation['k'],
            'gamma': 0.005 / modulation['k'],
            'sigma': 0.5 * modulation['omega_scale'],
            'x12_bias': modulation['x12_target'] * 0.1
        }
