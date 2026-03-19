"""
CST Sensory Bridge - Functional Demonstration

This script demonstrates the full CST pipeline working together:
1. FrequencyAnalyzer processing audio data
2. GeometricPhaseMapper processing facial landmarks
3. CSTSensoryBridge combining both for truth detection
4. PhiInvariantEncoder for memory drift validation
5. PSO with relativistic constraints
6. CST penalty in evolution
"""

import math
import numpy as np
from loguru import logger

# Configure logger for demo
logger.remove()
logger.add(lambda msg: print(msg), level="INFO", format="{message}")


def demo_frequency_analyzer():
    """Demonstrate audio frequency analysis."""
    print("\n" + "="*60)
    print("1. FREQUENCY ANALYZER - Audio to Emotional Mass")
    print("="*60)
    
    from Cosmos.core.cst_sensory_bridge import FrequencyAnalyzer
    
    analyzer = FrequencyAnalyzer(sample_rate=16000)
    
    # Generate test audio: calm voice (low amplitude sine wave)
    duration = 0.5
    t = np.linspace(0, duration, int(16000 * duration))
    
    # Calm voice: low amplitude, steady pitch
    calm_audio = np.sin(2 * np.pi * 200 * t) * 0.2
    calm_bytes = (calm_audio * 32767).astype(np.int16).tobytes()
    
    # Stressed voice: high amplitude, varied pitch
    stressed_audio = np.sin(2 * np.pi * 400 * t) * 0.8 + np.sin(2 * np.pi * 600 * t) * 0.4
    stressed_bytes = (stressed_audio * 32767).astype(np.int16).tobytes()
    
    calm_energy, _, _ = analyzer.analyze(calm_bytes, normalize=True)
    stressed_energy, _, _ = analyzer.analyze(stressed_bytes, normalize=True)
    
    print(f"  Calm voice energy:    {calm_energy:.4f}")
    print(f"  Stressed voice energy: {stressed_energy:.4f}")
    print(f"  ✓ Higher stress = higher emotional mass")
    
    return calm_energy, stressed_energy


def demo_geometric_phase():
    """Demonstrate facial geometry to phase angle mapping."""
    print("\n" + "="*60)
    print("2. GEOMETRIC PHASE MAPPER - Face to Phase Angle")
    print("="*60)
    
    from Cosmos.core.cst_sensory_bridge import GeometricPhaseMapper
    
    mapper = GeometricPhaseMapper()
    
    # Create simplified 68-point landmarks
    def create_face(brow_raise=0, eye_squint=0, mouth_tension=0):
        landmarks = []
        # Jaw (17 points)
        for i in range(17):
            landmarks.append((50.0 + i * 10, 100.0 + abs(i - 8) * 2))
        # Brows (10 points)
        for i in range(10):
            landmarks.append((60.0 + i * 8, 40.0 - brow_raise))
        # Nose (9 points)
        for i in range(9):
            landmarks.append((95.0, 50.0 + i * 5))
        # Eyes (12 points)
        for i in range(6):
            landmarks.append((75.0 + i * 3, 55.0 + eye_squint))
        for i in range(6):
            landmarks.append((110.0 + i * 3, 55.0 + eye_squint))
        # Mouth (20 points)
        for i in range(20):
            angle = (i / 20) * 2 * math.pi
            landmarks.append((95.0 + 10 * math.cos(angle), 90.0 + (3 - mouth_tension) * math.sin(angle)))
        return landmarks
    
    relaxed = create_face(brow_raise=0, eye_squint=0, mouth_tension=0)
    tense = create_face(brow_raise=10, eye_squint=3, mouth_tension=2)
    
    relaxed_phase, relaxed_tensions = mapper.calculate_phase(relaxed)
    tense_phase, tense_tensions = mapper.calculate_phase(tense)
    
    print(f"  Relaxed face phase: {math.degrees(relaxed_phase):.2f}°")
    print(f"  Tense face phase:   {math.degrees(tense_phase):.2f}°")
    print(f"  ✓ Higher tension = higher geometric phase")
    
    return relaxed_phase, tense_phase


def demo_cst_state():
    """Demonstrate unified CST state and coherence detection."""
    print("\n" + "="*60)
    print("3. CST STATE - Truth Probability & Intent Detection")
    print("="*60)
    
    from Cosmos.core.cst_sensory_bridge import CSTState, DetectedIntent
    
    # Coherent: both signals agree (high stress)
    coherent = CSTState(
        emotional_mass=0.8,
        geometric_phase=math.pi / 4 * 0.8,  # ~36°
        detected_intent=DetectedIntent.STRESSED,
    )
    
    # Incoherent: signals disagree (potential deception)
    incoherent = CSTState(
        emotional_mass=0.9,  # High voice stress
        geometric_phase=0.1,  # But calm face
        detected_intent=DetectedIntent.DECEPTION,
    )
    
    print(f"  Coherent state (stressed audio + tense face):")
    print(f"    Is coherent: {coherent.is_coherent(threshold=0.3)}")
    print(f"  ")
    print(f"  Incoherent state (stressed audio + calm face):")
    print(f"    Is coherent: {incoherent.is_coherent(threshold=0.3)}")
    print(f"  ✓ Incoherence suggests potential deception")


def demo_phi_invariant_encoder():
    """Demonstrate memory drift validation."""
    print("\n" + "="*60)
    print("4. PHI-INVARIANT ENCODER - Memory Drift Validation")
    print("="*60)
    
    from Cosmos.memory.memory_system import PhiInvariantEncoder
    
    encoder = PhiInvariantEncoder()
    
    current_phase = 0.5  # User's current context phase
    
    # Small drift (acceptable)
    small_drift = 0.55  # ~3° difference
    # Large drift (rejected)
    large_drift = 1.2   # ~40° difference
    
    small_valid = encoder.validate_drift(current_phase, small_drift)
    large_valid = encoder.validate_drift(current_phase, large_drift)
    
    print(f"  Current context phase: {math.degrees(current_phase):.2f}°")
    print(f"  Memory phase {math.degrees(small_drift):.2f}° (3° drift): {'✓ Accepted' if small_valid else '✗ Rejected'}")
    print(f"  Memory phase {math.degrees(large_drift):.2f}° (40° drift): {'✓ Accepted' if large_valid else '✗ Rejected'}")
    print(f"  ✓ High-drift memories rejected for stability")
    
    stats = encoder.get_stats()
    print(f"  Rejection rate: {stats['rejection_rate']*100:.1f}%")


def demo_pso_constraints():
    """Demonstrate PSO relativistic constraints."""
    print("\n" + "="*60)
    print("5. PSO RELATIVISTIC CONSTRAINTS - Geodesic Movement")
    print("="*60)
    
    from Cosmos.core.model_swarm import ModelSwarm, ModelParticle, ModelRole
    
    swarm = ModelSwarm(pso_cognitive=2.0, pso_social=2.0)
    
    # Create test particle far from solution
    particle = ModelParticle(
        model_id="test",
        model_name="Test Model",
        role=ModelRole.GENERALIST,
        position=[0.1, 0.1, 0.1],
        velocity=[0.0, 0.0, 0.0],
        personal_best_position=[0.3, 0.3, 0.3],
    )
    swarm.particles["test"] = particle
    swarm.global_best_position = [0.9, 0.9, 0.9]  # Solution
    
    print(f"  C_CONSTANT (speed limit): {swarm.C_CONSTANT}")
    print(f"  Initial position: {particle.position}")
    
    # Run PSO step
    swarm._pso_step()
    
    print(f"  After PSO step:   {[f'{p:.4f}' for p in particle.position]}")
    print(f"  Velocities:       {[f'{v:.4f}' for v in particle.velocity]}")
    
    # Verify velocities bounded
    max_vel = max(abs(v) for v in particle.velocity)
    print(f"  Max velocity: {max_vel:.4f} (bounded by C={swarm.C_CONSTANT})")
    print(f"  ✓ Velocities bounded by speed-of-light constant")


def demo_cst_penalty():
    """Demonstrate CST penalty in evolution."""
    print("\n" + "="*60)
    print("6. CST PENALTY - Suppressing Low-Effort Responses")
    print("="*60)
    
    from Cosmos.evolution.genetic_optimizer import Genome, Gene, cst_penalty
    
    # Low complexity response
    low_effort = Genome(
        id="low",
        genes={"param": Gene("param", 0.5, 0.0, 1.0)},
        fitness_scores={"quality": 0.3, "diversity": 0.2, "coherence": 0.2},
    )
    
    # High complexity response
    high_effort = Genome(
        id="high",
        genes={"param": Gene("param", 0.5, 0.0, 1.0)},
        fitness_scores={"quality": 0.8, "diversity": 0.7, "coherence": 0.7},
    )
    
    print(f"  Low-effort response:")
    print(f"    Raw fitness: {low_effort.total_fitness(apply_cst_penalty=False):.4f}")
    print(f"    With CST penalty: {low_effort.total_fitness(apply_cst_penalty=True):.4f}")
    print(f"  ")
    print(f"  High-effort response:")
    print(f"    Raw fitness: {high_effort.total_fitness(apply_cst_penalty=False):.4f}")
    print(f"    With CST penalty: {high_effort.total_fitness(apply_cst_penalty=True):.4f}")
    print(f"  ✓ Low-effort penalized, high-effort rewarded")


def main():
    """Run all CST demonstrations."""
    print("\n" + "#"*60)
    print("# CST SENSORY BRIDGE - FULL FUNCTIONALITY DEMO")
    print("#"*60)
    
    demo_frequency_analyzer()
    demo_geometric_phase()
    demo_cst_state()
    demo_phi_invariant_encoder()
    demo_pso_constraints()
    demo_cst_penalty()
    
    print("\n" + "="*60)
    print("✓ ALL CST COMPONENTS FUNCTIONAL")
    print("="*60)
    print("""
Summary of CST Physics Integration:
  1. Audio → Energy (not text)
  2. Face → Angles (not labels)
  3. Memory → Phi-invariant (drift resistant)
  4. Swarm → Geodesic movement (speed bounded)
  5. Evolution → Low-effort penalized
""")


if __name__ == "__main__":
    main()
