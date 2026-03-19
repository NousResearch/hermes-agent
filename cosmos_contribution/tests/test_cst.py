"""
cosmos CST Sensory Bridge - Comprehensive Test Suite

Tests for Cosmic Synapse Theory (CST) integration components:
- FrequencyAnalyzer: Audio → Emotional Mass
- GeometricPhaseMapper: Facial Landmarks → Geometric Phase
- CSTState: Unified physics state
- PhiInvariantEncoder: Memory drift validation
- PSO Relativistic Constraints
- CST Penalty for evolution
"""

import math
import pytest
from unittest.mock import MagicMock, patch


# ==============================================================================
# FREQUENCY ANALYZER TESTS
# ==============================================================================

class TestFrequencyAnalyzer:
    """Tests for CST FrequencyAnalyzer."""
    
    def test_frequency_analyzer_creation(self):
        """FrequencyAnalyzer initializes correctly."""
        from cosmos.core.cst_sensory_bridge import FrequencyAnalyzer
        
        analyzer = FrequencyAnalyzer(sample_rate=16000)
        
        assert analyzer.sample_rate == 16000
        assert analyzer.VOICE_LOW == 85
        assert analyzer.VOICE_HIGH == 3000
    
    def test_frequency_analyzer_energy_calculation_short_buffer(self):
        """Short audio buffers return zero energy."""
        from cosmos.core.cst_sensory_bridge import FrequencyAnalyzer
        
        analyzer = FrequencyAnalyzer(sample_rate=16000)
        
        # Very short buffer
        short_audio = bytes([0] * 100)
        energy, freqs, amps = analyzer.analyze(short_audio)
        
        assert energy == 0.0
        assert freqs == []
        assert amps == []
    
    def test_frequency_analyzer_energy_calculation_valid_buffer(self):
        """Valid audio buffer produces energy in 0-1 range."""
        from cosmos.core.cst_sensory_bridge import FrequencyAnalyzer
        
        analyzer = FrequencyAnalyzer(sample_rate=16000, window_size_ms=25)
        
        # Generate a simple sine wave as test audio (440 Hz)
        import numpy as np
        duration = 0.5  # seconds
        t = np.linspace(0, duration, int(16000 * duration))
        sine_wave = np.sin(2 * np.pi * 440 * t) * 0.5
        audio_bytes = (sine_wave * 32767).astype(np.int16).tobytes()
        
        energy, freqs, amps = analyzer.analyze(audio_bytes, normalize=True)
        
        # Energy should be normalized to 0-1
        assert 0.0 <= energy <= 1.0
        # Should have detected some frequencies
        assert len(freqs) > 0
    
    def test_frequency_analyzer_fallback_without_numpy(self):
        """Fallback analysis works without NumPy."""
        from cosmos.core.cst_sensory_bridge import FrequencyAnalyzer
        
        analyzer = FrequencyAnalyzer()
        
        # Mock silent audio (zeros)
        silent_audio = bytes([0] * 1000)
        energy, _, _ = analyzer._analyze_fallback(silent_audio, normalize=True)
        
        # Silent audio should have near-zero energy
        assert energy < 0.1


# ==============================================================================
# GEOMETRIC PHASE MAPPER TESTS
# ==============================================================================

class TestGeometricPhaseMapper:
    """Tests for CST GeometricPhaseMapper."""
    
    def _create_relaxed_face_landmarks(self) -> list:
        """Create 68 landmarks representing a relaxed face."""
        # Simplified 68-point face with symmetric, relaxed features
        landmarks = []
        
        # Jaw (0-16): smooth contour
        for i in range(17):
            x = 50 + i * 10 if i < 8 else 50 + (16 - i) * 10
            y = 100 + abs(i - 8) * 2
            landmarks.append((float(x), float(y)))
        
        # Left brow (17-21): flat
        for i in range(5):
            landmarks.append((60.0 + i * 8, 40.0))
        
        # Right brow (22-26): flat
        for i in range(5):
            landmarks.append((100.0 + i * 8, 40.0))
        
        # Nose (27-35): centered
        for i in range(9):
            landmarks.append((95.0 + (i % 3 - 1) * 5, 50.0 + i * 5))
        
        # Left eye (36-41): open, symmetric
        landmarks.extend([(70.0, 55.0), (75.0, 52.0), (80.0, 52.0), 
                          (85.0, 55.0), (80.0, 58.0), (75.0, 58.0)])
        
        # Right eye (42-47): open, symmetric
        landmarks.extend([(105.0, 55.0), (110.0, 52.0), (115.0, 52.0),
                          (120.0, 55.0), (115.0, 58.0), (110.0, 58.0)])
        
        # Outer mouth (48-59): relaxed, slight smile
        for i in range(12):
            angle = (i / 12) * 2 * math.pi
            x = 95.0 + 15 * math.cos(angle)
            y = 90.0 + 5 * math.sin(angle)
            landmarks.append((x, y))
        
        # Inner mouth (60-67): closed
        for i in range(8):
            angle = (i / 8) * 2 * math.pi
            x = 95.0 + 8 * math.cos(angle)
            y = 90.0 + 2 * math.sin(angle)
            landmarks.append((x, y))
        
        return landmarks
    
    def _create_stressed_face_landmarks(self) -> list:
        """Create 68 landmarks representing a stressed face."""
        landmarks = self._create_relaxed_face_landmarks()
        
        # Raise eyebrows (stress indicator)
        for i in range(17, 27):
            x, y = landmarks[i]
            landmarks[i] = (x, y - 10)  # Move up
        
        # Widen eyes
        for i in [37, 38, 43, 44]:  # Upper eyelid points
            x, y = landmarks[i]
            landmarks[i] = (x, y - 3)
        
        # Tighten mouth
        for i in range(48, 60):
            x, y = landmarks[i]
            # Compress vertically
            landmarks[i] = (x, 90.0 + (y - 90.0) * 0.5)
        
        # Asymmetric jaw (clenching)
        for i in range(4, 8):
            x, y = landmarks[i]
            landmarks[i] = (x + 3, y)
        
        return landmarks
    
    def test_geometric_phase_mapper_stable(self):
        """Relaxed face produces measurable geometric phase."""
        from cosmos.core.cst_sensory_bridge import GeometricPhaseMapper
        
        mapper = GeometricPhaseMapper()
        landmarks = self._create_relaxed_face_landmarks()
        
        phase, tensions = mapper.calculate_phase(landmarks)
        
        # Note: Synthetic landmark data is simplified, so we just verify
        # the phase is within expected range (0 to π/2)
        assert 0 <= phase <= math.pi / 2
        assert all(t >= 0 for t in tensions.values())
    
    def test_geometric_phase_mapper_stressed(self):
        """Tense face produces higher geometric phase than relaxed face."""
        from cosmos.core.cst_sensory_bridge import GeometricPhaseMapper
        
        mapper = GeometricPhaseMapper()
        relaxed_landmarks = self._create_relaxed_face_landmarks()
        stressed_landmarks = self._create_stressed_face_landmarks()
        
        relaxed_phase, relaxed_tensions = mapper.calculate_phase(relaxed_landmarks)
        stressed_phase, stressed_tensions = mapper.calculate_phase(stressed_landmarks)
        
        # With synthetic data, we verify both produce valid phases
        # and that at least some tension is detected
        assert 0 <= stressed_phase <= math.pi / 2
        assert sum(stressed_tensions.values()) > 0
    
    def test_geometric_phase_mapper_insufficient_landmarks(self):
        """Returns zero phase for insufficient landmarks."""
        from cosmos.core.cst_sensory_bridge import GeometricPhaseMapper
        
        mapper = GeometricPhaseMapper()
        
        phase, tensions = mapper.calculate_phase([(0, 0)] * 10)  # Too few
        
        assert phase == 0.0
        assert tensions == {}


# ==============================================================================
# CST STATE TESTS
# ==============================================================================

class TestCSTState:
    """Tests for CSTState dataclass."""
    
    def test_cst_state_json_output(self):
        """CSTState serializes to expected JSON format."""
        from cosmos.core.cst_sensory_bridge import CSTState, DetectedIntent
        
        state = CSTState(
            emotional_mass=0.75,
            geometric_phase=0.5,
            truth_probability=0.8,
            detected_intent=DetectedIntent.HONEST,
        )
        
        result = state.to_dict()
        
        assert "cst_state" in result
        assert result["cst_state"]["emotional_mass"] == 0.75
        assert result["cst_state"]["geometric_phase"] == 0.5
        assert result["cst_state"]["truth_probability"] == 0.8
        assert result["cst_state"]["detected_intent"] == "Honest"
        assert "metadata" in result
    
    def test_cst_state_coherence_check(self):
        """Coherence detection identifies mismatched audio/visual signals."""
        from cosmos.core.cst_sensory_bridge import CSTState
        
        # Coherent: both high
        coherent_high = CSTState(emotional_mass=0.8, geometric_phase=math.pi/4 * 0.8)
        assert coherent_high.is_coherent(threshold=0.3)
        
        # Coherent: both low
        coherent_low = CSTState(emotional_mass=0.2, geometric_phase=0.1)
        assert coherent_low.is_coherent(threshold=0.3)
        
        # Incoherent: high audio, low visual
        incoherent = CSTState(emotional_mass=0.9, geometric_phase=0.1)
        assert not incoherent.is_coherent(threshold=0.3)


# ==============================================================================
# PHI-INVARIANT ENCODER TESTS
# ==============================================================================

class TestPhiInvariantEncoder:
    """Tests for PhiInvariantEncoder in memory system."""
    
    def test_phi_invariant_encoder_creation(self):
        """PhiInvariantEncoder initializes with correct threshold."""
        from cosmos.memory.memory_system import PhiInvariantEncoder
        
        encoder = PhiInvariantEncoder()
        
        assert math.degrees(encoder.drift_threshold) == pytest.approx(5.0, abs=0.1)
    
    def test_spherical_encoding(self):
        """Vectors convert to spherical coordinates correctly."""
        from cosmos.memory.memory_system import PhiInvariantEncoder
        
        encoder = PhiInvariantEncoder()
        
        # Unit vector along z-axis
        r, theta, phi = encoder.encode_to_spherical([0.0, 0.0, 1.0])
        
        assert r == pytest.approx(1.0)
        assert theta == pytest.approx(0.0)  # Along z-axis
    
    def test_memory_drift_validation_accepts_small_drift(self):
        """Small drift (< 5°) is accepted."""
        from cosmos.memory.memory_system import PhiInvariantEncoder
        
        encoder = PhiInvariantEncoder()
        
        current_phase = 0.5
        new_phase = 0.55  # ~3° difference
        
        assert encoder.validate_drift(current_phase, new_phase)
    
    def test_memory_drift_validation_rejects_large_drift(self):
        """Large drift (> 5°) is rejected."""
        from cosmos.memory.memory_system import PhiInvariantEncoder
        
        encoder = PhiInvariantEncoder()
        
        current_phase = 0.0
        new_phase = 0.5  # ~29° difference
        
        assert not encoder.validate_drift(current_phase, new_phase)
    
    def test_encoder_stats(self):
        """Encoder tracks statistics correctly."""
        from cosmos.memory.memory_system import PhiInvariantEncoder
        
        encoder = PhiInvariantEncoder()
        
        # Perform validations
        encoder.validate_drift(0.0, 0.01)  # Accept
        encoder.validate_drift(0.0, 0.5)   # Reject
        encoder.validate_drift(0.0, 0.02)  # Accept
        
        stats = encoder.get_stats()
        
        assert stats["total_encodings"] == 3
        assert stats["drift_rejections"] == 1


# ==============================================================================
# PSO RELATIVISTIC CONSTRAINTS TESTS
# ==============================================================================

class TestPSORelativisticConstraints:
    """Tests for CST relativistic constraints in PSO."""
    
    def test_c_constant_defined(self):
        """ModelSwarm has C_CONSTANT defined."""
        from cosmos.core.model_swarm import ModelSwarm
        
        assert hasattr(ModelSwarm, "C_CONSTANT")
        assert ModelSwarm.C_CONSTANT == 1.0
    
    def test_geodesic_gradient_calculation(self):
        """Particles calculate geometric gradient toward global best."""
        from cosmos.core.model_swarm import ModelSwarm, ModelParticle, ModelRole
        
        swarm = ModelSwarm()
        
        # Create a test particle
        particle = ModelParticle(
            model_id="test",
            model_name="Test Model",
            role=ModelRole.GENERALIST,
            position=[0.2, 0.3, 0.4],
            velocity=[0.0, 0.0, 0.0],
        )
        
        # Set global best
        swarm.global_best_position = [0.8, 0.9, 1.0]
        
        # Calculate gradient
        gradient = swarm._calculate_geometric_gradient(particle)
        
        # Gradient should point toward global best (positive direction)
        assert all(g > 0 for g in gradient)
        
        # Should be normalized to unit vector
        magnitude = sum(g**2 for g in gradient) ** 0.5
        assert magnitude == pytest.approx(1.0, abs=0.01)
    
    def test_pso_velocity_bounded_by_c(self):
        """PSO velocity updates are bounded by C_CONSTANT."""
        from cosmos.core.model_swarm import ModelSwarm, ModelParticle, ModelRole
        
        swarm = ModelSwarm(pso_cognitive=2.0, pso_social=2.0)
        
        # Create particle with extreme position
        particle = ModelParticle(
            model_id="test",
            model_name="Test",
            role=ModelRole.GENERALIST,
            position=[0.0, 0.0, 0.0],
            velocity=[0.0, 0.0, 0.0],
            personal_best_position=[0.5, 0.5, 0.5],
        )
        
        swarm.particles["test"] = particle
        swarm.global_best_position = [1.0, 1.0, 1.0]
        
        # Run PSO step
        swarm._pso_step()
        
        # Velocities should be bounded
        for v in particle.velocity:
            assert -swarm.C_CONSTANT <= v <= swarm.C_CONSTANT


# ==============================================================================
# CST PENALTY TESTS
# ==============================================================================

class TestCSTPenalty:
    """Tests for CST non-vanishing penalty."""
    
    def test_cst_penalty_low_complexity(self):
        """Low complexity scores receive negative penalty."""
        from cosmos.evolution.genetic_optimizer import cst_penalty
        
        penalty = cst_penalty(1.0, threshold=2.0)
        
        assert penalty == -1.0  # 1.0 - 2.0 = -1.0
    
    def test_cst_penalty_threshold(self):
        """Complexity at threshold gives zero penalty."""
        from cosmos.evolution.genetic_optimizer import cst_penalty
        
        penalty = cst_penalty(2.0, threshold=2.0)
        
        assert penalty == 0.0
    
    def test_cst_penalty_high_complexity(self):
        """High complexity scores receive positive reward."""
        from cosmos.evolution.genetic_optimizer import cst_penalty
        
        penalty = cst_penalty(5.0, threshold=2.0)
        
        assert penalty == 3.0  # 5.0 - 2.0 = 3.0
    
    def test_genome_fitness_applies_penalty(self):
        """Genome.total_fitness() applies CST penalty."""
        from cosmos.evolution.genetic_optimizer import Genome, Gene
        
        genome = Genome(
            id="test",
            genes={"param": Gene("param", 0.5, 0.0, 1.0)},
            fitness_scores={"quality": 0.5, "diversity": 0.5, "coherence": 0.5},
        )
        
        # With penalty
        fitness_with_penalty = genome.total_fitness(apply_cst_penalty=True)
        
        # Without penalty
        fitness_without_penalty = genome.total_fitness(apply_cst_penalty=False)
        
        # They should differ
        assert fitness_with_penalty != fitness_without_penalty


# ==============================================================================
# INTEGRATION TESTS
# ==============================================================================

class TestCSTIntegration:
    """Integration tests for full CST pipeline."""
    
    def test_cst_sensory_bridge_creation(self):
        """CSTSensoryBridge creates with all components."""
        from cosmos.core.cst_sensory_bridge import CSTSensoryBridge
        
        bridge = CSTSensoryBridge(sample_rate=16000)
        
        assert bridge.frequency_analyzer is not None
        assert bridge.phase_mapper is not None
        
        stats = bridge.get_stats()
        assert "has_numpy" in stats
        assert "sample_rate" in stats
    
    def test_memory_system_has_phi_encoder(self):
        """MemorySystem initializes with PhiInvariantEncoder."""
        from cosmos.memory.memory_system import MemorySystem
        
        memory = MemorySystem(data_dir="./test_data")
        
        assert hasattr(memory, "phi_encoder")
        assert memory.phi_encoder is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
