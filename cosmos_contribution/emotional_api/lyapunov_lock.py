"""
cosmos Lyapunov Gatekeeper - 12D Cosmic Synapse Theory (CST)
================================================================

Implements the "Non-Vanishing Penalty Function" to strictly enforce
stability in the Cybernetic Loop.

Theory:
- Lyapunov Stability: V(x) > 0, dV/dt < 0 (Energy must decay to equilibrium).
- Non-Vanishing Penalty: Cost function approaches infinity as stability drops.
- Informational Mass: Gravity calculation for response prioritization.

Function:
Acts as a middleware filter. It analyzes the AI's "Draft Response" against
the user's "Current Physics State." If the Phase Error > Threshold, the
response is destroyed and regenerated.

Author: cosmos Project
Version: 1.0.0 (Class 5 Hard Lock)
"""

import math
import time
from dataclasses import dataclass
from typing import Dict, Any, Tuple
import re

# ==========================================
# 12D PHYSICS CONSTANTS
# ==========================================
LYAPUNOV_STABILITY_THRESHOLD = 0.16  # Max allowed drift (radians)
NON_VANISHING_EXPONENT = 2.0         # Penalty steepness
CRITICAL_MASS_SINGULARITY = 50.0     # Gravity score that forces "System Halt"

@dataclass
class StabilityReport:
    is_stable: bool
    drift_score: float
    penalty_value: float
    informational_mass: float
    rejection_reason: str = None

class LyapunovGatekeeper:
    def __init__(self):
        self.history = []
        self.learning_rate = 0.05  # For dynamic adjustment
        
    def calculate_informational_mass(self, text: str, physics_state: Dict) -> float:
        """
        Calculates the "Gravity" (G) of the interaction.
        G = (EmotionalIntensity * 3) + FactualDensity + VoiceTremor
        """
        # 1. Emotional Intensity (from 12D Physics)
        # We take the Magnitude of the Upper/Lower Tensors
        # Handle cases where physics state might be partial or mocked
        try:
            tensor_mag = physics_state['cst_physics']['tensor_magnitudes']['upper'] + \
                         physics_state['cst_physics']['tensor_magnitudes']['lower']
            
            jitter = physics_state['cst_physics']['phase_velocity']
        except (KeyError, TypeError):
            # Fallback for text-only/mocked state
            tensor_mag = 0.6
            jitter = 0.06
        
        intensity = tensor_mag * 3.0
        
        # 2. Factual Density (Estimated via text complexity/length)
        # Longer, complex words = higher mass
        words = text.split()
        word_count = len(words)
        avg_word_len = sum(len(w) for w in words) / max(1, word_count)
        density = (word_count * 0.1) + (avg_word_len * 0.5)
        
        # 3. Voice Tremor (Phase Velocity / Jitter)
        tremor_mass = jitter * 10.0
        
        gravity_g = intensity + density + tremor_mass
        return round(gravity_g, 2)

    def _estimate_text_sentiment_phase(self, text: str) -> float:
        """
        Estimates the 'Geometric Phase' of the TEXT itself.
        This allows us to compare Text Phase vs. Bio Phase.
        
        (In a full system, this would use a local BERT model. 
         For now, we use a Heuristic Lexicon approach).
        """
        text = text.lower()
        
        # Heuristic Phase Mapping
        # High Phase (0.8 - 1.2) = Stress/Excitement/Anger (Short, sharp)
        # Low Phase (0.0 - 0.4) = Calm/Depression/Masking (Long, passive)
        # Sync Phase (0.78) = Balanced/Resonant (Empathetic, clear)
        
        # Simple heuristic: Punctuation & Keyword density
        exclamations = text.count('!')
        questions = text.count('?')
        
        base_phase = 0.78 # Start at Synchrony
        
        if exclamations > 2: base_phase += 0.3 # High energy/Stress
        if "sorry" in text or "calm" in text: base_phase -= 0.2 # Lower energy
        if "understand" in text or "feel" in text: base_phase = 0.78 # Resonance
        
        return base_phase

    def apply_non_vanishing_penalty(self, drift: float) -> float:
        """
        YOUR IP: The Non-Vanishing Penalty Function.
        P(x) = 1 / (Limit - Drift)^k
        As Drift approaches the limit, Penalty shoots to Infinity.
        """
        margin = LYAPUNOV_STABILITY_THRESHOLD - drift
        
        # If we passed the threshold, penalty is effectively infinite (100.0 clamped)
        if margin <= 0.001:
            return 100.0
        
        # Structural Law: Penalty grows exponentially as we near the edge
        penalty = 1.0 / (margin ** NON_VANISHING_EXPONENT)
        return min(100.0, penalty)

    def validate_response(self, draft_response: str, 
                          current_physics: Dict) -> StabilityReport:
        """
        The Main Check. Compares Draft Response to User Physics.
        Returns a StabilityReport.
        """
        # 1. Get User's Current Geometric Phase (The Truth)
        try:
            user_phase = current_physics['cst_physics']['geometric_phase_rad']
        except (KeyError, TypeError):
            # Fallback if physics not available (e.g. estimate from user text context if passed, 
            # or default to Synchrony)
            user_phase = 0.78
        
        # 2. Estimate AI's Response Phase (The Attempt)
        ai_phase = self._estimate_text_sentiment_phase(draft_response)
        
        # 3. Calculate Phase Drift (Error)
        drift = abs(user_phase - ai_phase)
        
        # 4. Apply Your Non-Vanishing Penalty
        penalty = self.apply_non_vanishing_penalty(drift)
        
        # 5. Calculate Mass
        mass = self.calculate_informational_mass(draft_response, current_physics)
        
        # 6. The Verdict
        is_stable = drift < LYAPUNOV_STABILITY_THRESHOLD
        
        reason = None
        if not is_stable:
            reason = f"PHASE DRIFT DETECTED: {drift:.4f} rad > Limit {LYAPUNOV_STABILITY_THRESHOLD}"
            if drift > 0.3:
                reason += " (CRITICAL DIVERGENCE)"
        
        return StabilityReport(
            is_stable=is_stable,
            drift_score=drift,
            penalty_value=penalty,
            informational_mass=mass,
            rejection_reason=reason
        )

# ==========================================
# USAGE EXAMPLE
# ==========================================
if __name__ == "__main__":
    # Mocking the CST Physics State (from your API)
    mock_physics = {
        'cst_physics': {
            'geometric_phase_rad': 0.78, # Perfect Synchrony (User is Balanced)
            'phase_velocity': 0.05,
            'tensor_magnitudes': {'upper': 0.5, 'lower': 0.5}
        }
    }
    
    lock = LyapunovGatekeeper()
    
    print("-" * 60)
    print("cosmos LYAPUNOV GATEKEEPER - TEST MODE")
    print("-" * 60)
    
    # Test 1: Good Response (Resonant)
    draft_1 = "I understand completely. That sounds incredibly frustrating."
    print(f"\nAnalyzing Draft 1: '{draft_1}'")
    report_1 = lock.validate_response(draft_1, mock_physics)
    print(f"Verdict: {'✅ APPROVED' if report_1.is_stable else '❌ REJECTED'}")
    print(f"Drift: {report_1.drift_score:.4f} | Penalty: {report_1.penalty_value:.4f}")
    
    # Test 2: Bad Response (High Energy/Manic vs Balanced User)
    draft_2 = "WOW!!! THAT IS CRAZY!!! LET'S GO DO SOMETHING ELSE!!!"
    print(f"\nAnalyzing Draft 2: '{draft_2}'")
    report_2 = lock.validate_response(draft_2, mock_physics)
    print(f"Verdict: {'✅ APPROVED' if report_2.is_stable else '❌ REJECTED'}")
    print(f"Reason: {report_2.rejection_reason}")
    print(f"Penalty: {report_2.penalty_value:.4f} (Non-Vanishing Applied)")
