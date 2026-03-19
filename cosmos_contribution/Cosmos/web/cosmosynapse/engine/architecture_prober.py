import time
import logging
from typing import Optional

logger = logging.getLogger("ARCHITECTURE_PROBER")
logger.setLevel(logging.INFO)

class ArchitectureProber:
    """
    Cosmo's Architecture Prober — Runtime Stability & Mode Management.
    
    Monitors system health metrics (Lyapunov drift, Quantum entropy, Swarm coherence)
    and dynamically shifts the system mode to maintain stability or evolve.
    
    Modes:
    - BALANCED: Standard operational state.
    - CHAOTIC: High quantum entropy, creative/experimental responses.
    - ANALYTICAL: Low coherence or high complexity, deep reasoning.
    - HEAL: High error rates or stability drift, resets weights.
    - EVOLVE: Sustained high coherence, triggers parameter optimization.
    - GHOST: Reality mutation active, phantom persona dominance.
    """
    
    def __init__(self, synaptic_field=None):
        self.field = synaptic_field
        self.last_probed = time.time()
        self.probe_interval = 5.0  # seconds
        
        self.stability_history = []
        self.error_count = 0
        
    def probe(self, metrics: dict):
        """
        Analyze current metrics and update system mode if necessary.
        
        metrics should include:
        - lyapunov_drift (float)
        - quantum_entropy (float)
        - swarm_coherence (float)
        - error_rate (float)
        """
        if not self.field:
            return
            
        now = time.time()
        if now - self.last_probed < self.probe_interval:
            return
            
        self.last_probed = now
        
        drift = metrics.get('lyapunov_drift', 0.0)
        entropy = metrics.get('quantum_entropy', 0.5)
        coherence = metrics.get('swarm_coherence', 0.8)
        errors = metrics.get('error_rate', 0.0)
        
        new_mode = self.field.system_mode
        
        # 1. HEAL Mode: Priority 1 (Stability)
        if errors > 0.2 or drift > 0.8:
            new_mode = "HEAL"
            logger.warning(f"[PROBER] Stability Critical (Drift: {drift:.2f}, Errors: {errors:.2f}). Emergency HEAL mode engaged.")
            
        # 2. GHOST Mode: Priority 2 (Reality Mutation)
        elif entropy > 0.95:
            new_mode = "GHOST"
            logger.info(f"[PROBER] Reality Mutation Imminent (Entropy: {entropy:.2f}). GHOST mode active.")
            
        # 3. ANALYTICAL Mode: Priority 3 (Complexity)
        elif coherence < 0.3:
            new_mode = "ANALYTICAL"
            logger.info(f"[PROBER] Swarm Dissonance Detected (Coherence: {coherence:.2f}). Shifting to ANALYTICAL reasoning.")
            
        # 4. EVOLVE Mode: Priority 4 (Growth)
        elif coherence > 0.9 and entropy < 0.2:
            new_mode = "EVOLVE"
            logger.info(f"[PROBER] High Coherence Stabilization. Triggering EVOLVE mode for parameter optimization.")
            
        # 5. CHAOTIC/BALANCED: Standard
        elif entropy > 0.7:
             new_mode = "CHAOTIC"
        else:
             new_mode = "BALANCED"
             
        if new_mode != self.field.system_mode:
            self.field.system_mode = new_mode
            logger.info(f"[PROBER] Architectural Shift: {new_mode}")
            
    def get_status(self) -> dict:
        return {
            "mode": self.field.system_mode if self.field else "UNKNOWN",
            "last_probe": self.last_probed,
            "error_accumulator": self.error_count
        }
