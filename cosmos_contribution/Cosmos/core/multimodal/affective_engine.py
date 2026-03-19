"""
AFFECTIVE ENGINE (EMOTIONAL LAYER)
===================================
Maps emotions to 12D state space and modulates learning.
Emotions are configurations in the 12D state space.
"""

import numpy as np

# ============================================================================
# EMOTIONAL STATE
# ============================================================================

class EmotionalState:
    """Emotional state in 12D framework"""
    
    def __init__(self, valence=0.0, arousal=0.5, dominance=0.5):
        """
        Args:
            valence: -1 (negative) to +1 (positive)
            arousal: 0 (calm) to 1 (excited)
            dominance: 0 (submissive) to 1 (dominant)
        """
        self.valence = np.clip(valence, -1, 1)
        self.arousal = np.clip(arousal, 0, 1)
        self.dominance = np.clip(dominance, 0, 1)
    
    def to_12d_modulation(self):
        """
        Convert emotion to 12D state space modulation
        
        Returns parameters that modify system dynamics
        """
        # Valence affects x₁₂ directly
        x12_target = self.valence
        
        # Arousal affects rate of change (learning rate k)
        k_modulation = 0.5 + self.arousal * 1.5  # Range: [0.5, 2.0]
        
        # Arousal also affects chaos (λ)
        lambda_modulation = 0.2 + self.arousal * 0.8  # Range: [0.2, 1.0]
        
        # Dominance affects connectivity strength
        omega_modulation = 0.3 + self.dominance * 0.7  # Range: [0.3, 1.0]
        
        return {
            'x12_target': x12_target,
            'k': k_modulation,
            'lambda': lambda_modulation,
            'omega_scale': omega_modulation
        }
    
    def classify_emotion(self):
        """Map continuous state to discrete emotion labels"""
        if self.arousal > 0.6:
            if self.valence > 0.3:
                return "excited" if self.dominance > 0.5 else "happy"
            elif self.valence < -0.3:
                return "angry" if self.dominance > 0.5 else "anxious"
            else:
                return "surprised"
        else:  # Low arousal
            if self.valence > 0.3:
                return "content" if self.dominance > 0.5 else "relaxed"
            elif self.valence < -0.3:
                return "sad" if self.dominance < 0.5 else "bored"
            else:
                return "neutral"

# ============================================================================
# EMOTION DETECTION
# ============================================================================

def detect_emotion_from_audio(audio_token):
    """
    Infer emotion from audio features
    
    Based on psychoacoustics:
    - High energy + high pitch → excitement
    - Low energy + low pitch → sadness
    - High entropy → stress/anxiety
    """
    # Energy → arousal
    energy = audio_token.embedding.D1_energy
    arousal = np.clip(energy * 2, 0, 1)
    
    # Pitch → valence
    if audio_token.fundamental_hz:
        pitch_deviation = (audio_token.fundamental_hz - 200) / 200
        valence = np.tanh(pitch_deviation)
    else:
        valence = 0.0
    
    # Loudness → dominance
    loudness_db = audio_token.loudness_db
    dominance = np.clip((loudness_db + 30) / 60, 0, 1)
    
    return EmotionalState(valence, arousal, dominance)

def detect_emotion_from_vision(visual_token):
    """
    Infer emotion from visual features
    
    Based on image properties:
    - Bright, saturated → positive valence
    - Dark, desaturated → negative valence
    - High spatial frequency → high arousal
    """
    image = visual_token.image
    
    # Brightness → valence
    brightness = np.mean(image) / 255.0
    valence = (brightness - 0.5) * 2  # Map [0, 1] → [-1, 1]
    
    # Saturation → arousal (if RGB)
    if len(image.shape) == 3:
        # Simple saturation estimate
        r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]
        max_rgb = np.maximum(np.maximum(r, g), b)
        min_rgb = np.minimum(np.minimum(r, g), b)
        saturation = np.mean((max_rgb - min_rgb) / (max_rgb + 1e-10))
        arousal = saturation
    else:
        arousal = 0.5
    
    # Spatial frequency → arousal contribution
    spatial_freq = visual_token.embedding.D11_frequency
    arousal = (arousal + spatial_freq) / 2
    
    # Contrast → dominance
    contrast = np.std(image) / 128.0
    dominance = np.clip(contrast, 0, 1)
    
    return EmotionalState(valence, arousal, dominance)

def detect_emotion_from_text(text):
    """
    Simple sentiment analysis for emotion detection
    """
    positive_words = {'happy', 'joy', 'love', 'great', 'excellent', 'wonderful', 'amazing', 'fantastic'}
    negative_words = {'sad', 'hate', 'terrible', 'awful', 'angry', 'depressed', 'bad', 'horrible'}
    arousal_words = {'excited', 'thrilled', 'shocked', 'furious', 'terrified', 'energetic'}
    
    words = set(text.lower().split())
    
    positive_count = len(words & positive_words)
    negative_count = len(words & negative_words)
    arousal_count = len(words & arousal_words)
    
    # Valence
    if positive_count + negative_count > 0:
        valence = (positive_count - negative_count) / (positive_count + negative_count)
    else:
        valence = 0.0
    
    # Arousal
    arousal = min(arousal_count / 10, 1.0)
    
    # Dominance (from exclamation marks, caps)
    exclamations = text.count('!')
    caps_ratio = sum(1 for c in text if c.isupper()) / (len(text) + 1)
    dominance = np.clip((exclamations / 5 + caps_ratio) / 2, 0, 1)
    
    return EmotionalState(valence, arousal, dominance)

# ============================================================================
# COGNITIVE PROCESSOR (THOUGHT GENERATION)
# ============================================================================

class CognitiveProcessor:
    """Generate thoughts from state space dynamics"""
    
    def __init__(self):
        self.state_history = []
        self.thought_buffer = []
        
    def observe_state_transition(self, old_state, new_state, input_description):
        """
        Observe a state transition and generate thought
        
        Thoughts are interpretations of ΔΨ
        """
        # Compute state change
        delta_psi = new_state - old_state
        
        # Magnitude of change
        delta_magnitude = np.linalg.norm(delta_psi)
        
        # Threshold for noticing
        if delta_magnitude < 0.1:
            return None
        
        # Direction of change
        dimension_changes = np.abs(delta_psi)
        dominant_dim = np.argmax(dimension_changes)
        
        dimension_names = [
            "energy", "mass", "harmony", "chaos",
            "motion_x", "motion_y", "motion_z", "connectivity",
            "brightness", "complexity", "frequency", "internal_state"
        ]
        
        thought = self._generate_thought(
            dimension_names[dominant_dim],
            delta_psi[dominant_dim],
            delta_magnitude,
            input_description
        )
        
        self.thought_buffer.append({
            'thought': thought,
            'magnitude': delta_magnitude,
            'dimension': dimension_names[dominant_dim]
        })
        
        return thought
    
    def _generate_thought(self, dimension, change, magnitude, context):
        """Map dimension change to natural language thought"""
        thoughts = {
            'energy': {
                'increase': f"Increased intensity in {context}",
                'decrease': f"Energy fading in {context}"
            },
            'chaos': {
                'increase': "This is becoming complex and unpredictable",
                'decrease': "Finding patterns and order"
            },
            'connectivity': {
                'increase': "This resonates with past experiences",
                'decrease': "This feels novel and unfamiliar"
            },
            'internal_state': {
                'increase': "Becoming more activated",
                'decrease': "Settling into calmness"
            },
            'frequency': {
                'increase': "Pitch/speed is rising",
                'decrease': "Slowing down, becoming lower"
            },
            'brightness': {
                'increase': "Visuals becoming brighter",
                'decrease': "Darkness or muting"
            }
        }
        
        if dimension in thoughts:
            direction = 'increase' if change > 0 else 'decrease'
            return thoughts[dimension][direction]
        else:
            return f"Shift in {dimension}"
    
    def get_stream_of_consciousness(self):
        """Return recent thoughts as internal monologue"""
        recent = self.thought_buffer[-10:]
        return [t['thought'] for t in recent if t['thought']]
