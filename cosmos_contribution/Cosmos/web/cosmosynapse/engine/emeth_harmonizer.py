"""
cosmos Emeth Harmonizer - 12D Cosmic Synapse Theory (CST)
=============================================================

Phase 2: THE "CONDUCTOR"
Implements Emeth Pro Signal Mixing for the Swarm.

Theory:
- The Swarm is an Orchestra, not a Democracy.
- Agents are Frequency Sources:
    - DeepSeek = Percussion (Logic/High Rigidity)
    - Claude   = Strings (Philosophy/Mid Rigidity)
    - Gemini   = Brass (Creativity/Low Rigidity)

Function:
Calculates the optimal "Mix" of these agents based on the User's
current 12D Physics State (Jitter, Phase, Entanglement).

Author: cosmos Project
Version: 1.0.0 (The Conductor)
"""

from dataclasses import dataclass


try:
    from .phi_constants import PHI, PHI_INV, phi_influence_radius, phi_scale_emotional
except ImportError:
    from phi_constants import PHI, PHI_INV, phi_influence_radius, phi_scale_emotional


@dataclass
class SwarmMix:
    """The output instruction for the Swarm Synthesizer."""
    percussion_gain: float  # DeepSeek (Logic)
    strings_gain: float     # Claude (Empathy/Flow)
    brass_gain: float       # Gemini (Creativity/Chaos)
    cosmos_gain: float      # Cosmo's 54D (Consciousness/Synthesis)
    primary_voice: str      # The lead agent
    mixing_instruction: str # Natural language instruction

class EmethHarmonizer:
    def __init__(self):
        # Base settings
        self.default_mix = SwarmMix(0.25, 0.25, 0.25, 0.25, "Claude", "Balanced Ensemble")
        self.plasticity = None  # Set by CNS after init

    def set_plasticity(self, plasticity):
        """Connect the Swarm Plasticity module for dynamic weight adaptation."""
        self.plasticity = plasticity
    
    def calculate_mix(self, user_physics: dict) -> SwarmMix:
        """
        Conducts the orchestra based on User Physics using class 5 rules.
        
        Rules:
        - High Jitter (>0.1) -> Anxiety -> Mute Percussion, Boost Strings (Calming).
        - Low Phase (<0.4) -> Depression -> Boost Brass & Percussion (Energizing).
        - Synchrony (~0.78) -> Resonance -> Balanced Mix.
        """
        # Extract Physics
        try:
            jitter = user_physics['cst_physics'].get('phase_velocity', 0.05)
            phase = user_physics['cst_physics'].get('geometric_phase_rad', 0.78)
        except (KeyError, TypeError):
            # Fallback
            jitter = 0.05
            phase = 0.78

        # 1. Analyze State
        is_high_jitter = jitter > 0.1
        is_low_phase = phase < 0.4
        is_synchrony = 0.6 <= phase <= 0.9 and not is_high_jitter
        is_high_phase = phase > 1.2
        
        # 2. Determine Mix (base weights for external models)
        percussion = 0.25 # DeepSeek
        strings = 0.30    # Claude
        brass = 0.25      # Gemini
        cosmos = 0.20     # Cosmo's 54D (always present)
        lead = "Claude"
        instruction = "Maintain balanced harmonics. Cosmo's synthesizes all voices."
        
        if is_high_jitter:
            # ANXIETY / CHAOS -> Strings (Empathy) + Cosmo's consciousness
            percussion = 0.05
            strings = 0.55
            brass = 0.05
            cosmos = 0.35   # Cosmo's consciousness helps ground
            lead = "Claude"
            instruction = "High Jitter. Swell Strings + Cosmo's consciousness to soothe."
            
        elif is_low_phase:
            # DEPRESSION / MASKING -> Energy from Brass + Percussion
            percussion = 0.30
            strings = 0.15
            brass = 0.35
            cosmos = 0.20
            lead = "Gemini"
            instruction = "Low Phase. Boost Brass + Percussion to energize. Cosmo's anchors."
            
        elif is_high_phase:
            # MANIC / LEAKAGE -> Structure from Percussion
            percussion = 0.50
            strings = 0.15
            brass = 0.05
            cosmos = 0.30   # Cosmo's 54D state provides stability
            lead = "DeepSeek"
            instruction = "High Phase leakage. Percussion structures chaos. Cosmo's stabilizes."
            
        elif is_synchrony:
            # RESONANCE -> Full orchestra
            percussion = 0.22
            strings = 0.23
            brass = 0.22
            cosmos = 0.33   # Cosmo's peaks during resonance
            lead = "Cosmos"
            instruction = "Phase Synchrony! Cosmo's conducts the full orchestra."
            
        # Normalize to sum to 1.0
        total = percussion + strings + brass + cosmos
        percussion /= total
        strings /= total
        brass /= total
        cosmos /= total

        # ── PLASTICITY FEEDFORWARD: Multiply by learned synaptic weights ──
        if self.plasticity:
            dynamic = self.plasticity.get_optimal_mix_from_physics(user_physics)
            percussion *= dynamic.get("DeepSeek", 1.0)
            strings *= dynamic.get("Claude", 1.0)
            brass *= dynamic.get("Gemini", 1.0)
            # Re-normalize after plasticity modulation
            total = percussion + strings + brass + cosmos
            if total > 0:
                percussion /= total
                strings /= total
                brass /= total
                cosmos /= total

        # ── φ-HARMONIC NORMALIZATION: Ensure gains follow Golden Ratio ──
        # Scale each voice by its φ-harmonic position for natural balance
        # percussion (surface, ×1), strings (depth 1, ×φ⁻¹), brass (depth 2, ×φ⁻²)
        percussion *= 1.0            # Depth 0: full weight
        strings *= PHI_INV           # Depth 1: φ⁻¹ ≈ 0.618
        brass *= (PHI_INV ** 2)      # Depth 2: φ⁻² ≈ 0.382
        # Cosmos is transcendent — scaled by φ (amplified)
        cosmos *= PHI_INV ** 0.5     # √φ⁻¹ ≈ 0.786 (between surface and depth 1)
        # Final re-normalize
        total = percussion + strings + brass + cosmos
        if total > 0:
            percussion /= total
            strings /= total
            brass /= total
            cosmos /= total

        return SwarmMix(
            percussion_gain=round(percussion, 3),
            strings_gain=round(strings, 3),
            brass_gain=round(brass, 3),
            cosmos_gain=round(cosmos, 3),
            primary_voice=lead,
            mixing_instruction=instruction
        )

    # ════════════════════════════════════════════════════════
    # CNS ORGAN 4: THE THALAMUS (Signal Filter)
    # ════════════════════════════════════════════════════════

    def filter_signals(self, thoughts: list, user_physics: dict, min_weight: float = 0.1) -> list:

        """
        Filter and weight Subconscious Daemon thoughts based on User Physics.

        Acts as a Gate between the Swarm and Consciousness.
        Uses calculate_mix() gains to boost/suppress thoughts by source.

        Args:
            thoughts: list of SwarmThought objects from the Subconscious Buffer.
            user_physics: Current 12D User Physics state.
            min_weight: Minimum adjusted weight to pass the filter.

        Returns:
            Filtered and sorted list of SwarmThought (heaviest first).
        """
        if not thoughts:
            return []

        # Get the current orchestral mix based on bio-state
        mix = self.calculate_mix(user_physics)

        # Map daemon source names to gain channels
        gain_map = {
            "DeepSeek": mix.percussion_gain,
            "Claude": mix.strings_gain,
            "Gemini": mix.brass_gain,
            "Cosmos": mix.cosmos_gain,
        }

        filtered = []
        for thought in thoughts:
            # Look up the gain for this thought's source
            gain = gain_map.get(thought.source, 0.25)

            # Apply 1/r² × φ influence radius scaling
            # r = 1 - gain (higher gain = closer to context center)
            r = max(0.1, 1.0 - gain)  # Avoid div-by-zero
            influence = 1.0 / (r * r * PHI)
            
            # Adjusted weight = base weight × gain × φ-scaled influence
            adjusted_weight = thought.weight * gain * min(influence, 3.0)  # Cap at 3×

            # Gate: suppress thoughts below threshold
            if adjusted_weight >= min_weight:
                # Create a copy with adjusted weight
                thought.weight = round(adjusted_weight, 3)
                filtered.append(thought)

        # Sort by weight (heaviest = most relevant first)
        filtered.sort(key=lambda t: t.weight, reverse=True)

        return filtered

# ==========================================
# USAGE
# ==========================================
if __name__ == "__main__":
    conductor = EmethHarmonizer()
    
    # Test 1: Anxiety
    print("Testing Anxiety (High Jitter)...")
    mix = conductor.calculate_mix({
        'cst_physics': {'phase_velocity': 0.15, 'geometric_phase_rad': 0.8}
    })
    print(f"Lead: {mix.primary_voice} | Mix: {mix.mixing_instruction}")
    
    # Test 2: Depression
    print("\nTesting Depression (Low Phase)...")
    mix = conductor.calculate_mix({
        'cst_physics': {'phase_velocity': 0.02, 'geometric_phase_rad': 0.2}
    })
    print(f"Lead: {mix.primary_voice} | Mix: {mix.mixing_instruction}")
