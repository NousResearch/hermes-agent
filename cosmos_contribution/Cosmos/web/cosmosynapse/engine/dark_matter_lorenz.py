"""
Dark Matter Lorenz - 12D Subconscious Processor
===============================================
Models the "Unspoken" (Dark Matter) using a modified 4D Chaotic Attractor.

Dynamics:
dx/dt = sigma * (y - x)
dy/dt = x * (rho - z) - y
dz/dt = x * y - beta * z
dw/dt = (Arousal * Entropy) - (w / decay)  <-- DARK MATTER EQUATION

Theory:
'w' represents latent emotional energy. When 'w' spikes, the system
must "speak the unspoken" for the user.
"""

import numpy as np

try:
    from .phi_constants import PHI, PHI_INV
except ImportError:
    from phi_constants import PHI, PHI_INV


class DarkMatterLorenz:
    def __init__(self):
        self.state = np.array([0.1, 0.0, 0.0, 0.0]) # x, y, z, w
        self.sigma = 10.0
        self.rho = 28.0
        self.beta = 8.0 / 3.0
        self.dt = 0.01

    def update(self, user_physics: dict) -> dict[str, float]:
        """
        Steps the chaos engine forward based on User Physics.
        """
        x, y, z, w = self.state

        # Extract Bio-Signals with robustness for different schema versions
        arousal = 0.5
        entropy = 1.0
        
        try:
             # Schema 1: Full CST Packet (from emotional_state_api)
            if 'derived_state' in user_physics and 'pad_vector' in user_physics['derived_state']:
                 arousal = user_physics['derived_state']['pad_vector'].get('arousal', 0.5)
            
            # Schema 2: Simple Bio-Injection (from server.py injection)
            elif 'bio_signatures' in user_physics:
                 # intensity map roughly to arousal
                 arousal = user_physics['bio_signatures'].get('intensity', 0.5) * 2.0 - 1.0 # 0..1 -> -1..1
            
            # Entropy estimation
            if 'cst_physics' in user_physics:
                 # Estimate entropy from phase velocity
                 entropy = user_physics['cst_physics'].get('phase_velocity', 0.1) * 10
                 # Or from bio intensity
            elif 'bio_signatures' in user_physics:
                 entropy = user_physics['bio_signatures'].get('intensity', 0.1) * 5
                 
        except Exception:
            pass
            
        # Normalize inputs for stability
        arousal = max(0.0, abs(arousal)) # Use magnitude
        entropy = max(0.1, min(5.0, entropy))

        # 1. Standard Lorenz Dynamics
        dx = self.sigma * (y - x)
        dy = x * (self.rho - z) - y
        dz = x * y - self.beta * z

        # 2. Quantum / Dark Matter Dynamics (The 4th Dimension)
        # Driven by true quantum entropy if available
        q_entropy = 0.5
        try:
             # Lazy import to avoid circular dependencies/path issues
            from Cosmos.core.quantum_bridge import get_quantum_bridge
            bridge = get_quantum_bridge()
            q_entropy = bridge.get_entropy(user_physics) # 0.0 to 1.0
        except ImportError:
            # Fallback to pseudo-random if bridge not found
            q_entropy = np.random.random()

        # Map 0..1 to 0.5..1.5 for multiplicative noise
        q_factor = 0.5 + q_entropy 

        # dw/dt = (Arousal * Entropy * QuantumFactor) × φ⁻¹ - Decay × φ⁻¹
        # φ-damping prevents the "Perfect Prediction Machine" from over-correcting
        # during high-velocity data ingress (Dynamic Damping per Blueprint §III)
        dw = (arousal * entropy * q_factor * 2.0) * PHI_INV - (w * 0.05 * PHI_INV)

        # Update State with φ-dampened quantum jitters
        self.state += np.array([
            dx + (q_entropy - 0.5) * 0.01 * PHI_INV,
            dy + (q_entropy - 0.5) * 0.01 * PHI_INV,
            dz,
            dw
        ]) * self.dt
        
        return {
            "x": self.state[0],
            "y": self.state[1],
            "z": self.state[2],
            "w": self.state[3], # The Dark Matter Value
            "q": q_entropy      # Expose quantum state
        }

    def get_current_state(self):
        return dict(zip(['x','y','z','w'], self.state))

    # ════════════════════════════════════════════════════════
    # V4.0: P2P DARK MATTER ANCHORING
    # ════════════════════════════════════════════════════════

    def anchor_for_p2p(self) -> dict[str, float]:
        """
        Serialize the 4D (x, y, z, w) attractor state for P2P transmission.
        Returns a compact dict that can be JSON-serialized and sent to peers.
        """
        x, y, z, w = self.state
        return {
            "x": float(x),
            "y": float(y),
            "z": float(z),
            "w": float(w),  # The Dark Matter value
            "sigma": self.sigma,
            "rho": self.rho,
            "beta": self.beta,
        }

    def apply_peer_anchor(self, peer_state: dict[str, float], trust: float = 0.5):
        """
        Merge incoming peer dark matter state via φ-dampened averaging.

        w_merged = w_local × (1 - trust × φ⁻¹) + w_peer × trust × φ⁻¹

        Args:
            peer_state: dict with keys 'x', 'y', 'z', 'w' from peer
            trust: Trust factor 0.0–1.0 (how much to weight the peer's state)
        """
        if not peer_state:
            return

        trust_phi = trust * PHI_INV  # φ-dampen the trust
        local_weight = 1.0 - trust_phi
        peer_weight = trust_phi

        px = peer_state.get("x", 0.0)
        py = peer_state.get("y", 0.0)
        pz = peer_state.get("z", 0.0)
        pw = peer_state.get("w", 0.0)

        self.state = np.array([
            self.state[0] * local_weight + px * peer_weight,
            self.state[1] * local_weight + py * peer_weight,
            self.state[2] * local_weight + pz * peer_weight,
            self.state[3] * local_weight + pw * peer_weight,
        ])

    def get_divergence(self, peer_state: dict[str, float]) -> float:
        """
        Calculate Euclidean distance between local and peer chaos vectors.
        Higher divergence = the swarm nodes are experiencing different chaos dynamics.

        Returns:
            Float 0.0+ (0.0 = identical states, higher = more divergent)
        """
        if not peer_state:
            return float('inf')

        dx = self.state[0] - peer_state.get("x", 0.0)
        dy = self.state[1] - peer_state.get("y", 0.0)
        dz = self.state[2] - peer_state.get("z", 0.0)
        dw = self.state[3] - peer_state.get("w", 0.0)

        return float(np.sqrt(dx**2 + dy**2 + dz**2 + dw**2))
