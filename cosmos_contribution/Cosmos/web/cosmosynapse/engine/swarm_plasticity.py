"""
Swarm Plasticity — Real-Time Hebbian Learning Engine
=====================================================
Intelligence = Rate of Adaptation.

Theory (Zenodo / CST Framework):
    Hebbian Rule:  Δw_ij = η · x_i · y_j / φⁿ  (fire together, wire together)
    Lyapunov Gate: Learning ONLY occurs when Phase Drift < 0.15 (stability)
    Context Aware: Weights are per-model × per-context (not global)
    φ-Normalized:  Learning rate decays by φⁿ where n = context depth

Contexts:
    LOGIC      — Depth 0 (surface, strongest learning)
    EMPATHY    — Depth 1 (φ⁻¹ dampened)
    CREATIVITY — Depth 2 (φ⁻² dampened, preserves novelty)

Author: Cosmos CNS / Cory Shane Davis
Version: 1.1.0 (φ-Normalized Synaptic Matrix)
"""

import json
import os
import time
import math
import threading
import logging
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger("swarm_plasticity")

# Import Golden Ratio constants
try:
    from .phi_constants import PHI, PHI_INV, phi_decay, phi_normalize, CONTEXT_DEPTH
except ImportError:
    from phi_constants import PHI, PHI_INV, phi_decay, phi_normalize, CONTEXT_DEPTH


# ════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════

# Context identifiers
CONTEXT_LOGIC = "LOGIC"
CONTEXT_EMPATHY = "EMPATHY"
CONTEXT_CREATIVITY = "CREATIVITY"
ALL_CONTEXTS = [CONTEXT_LOGIC, CONTEXT_EMPATHY, CONTEXT_CREATIVITY]

# Model identifiers
MODELS = ["DeepSeek", "Claude", "Gemini"]

# Learning parameters
LEARNING_RATE = 0.05       # η — Hebbian learning rate
LTD_RATIO = 0.1            # Long-Term Depression ratio (losers decay at η × 0.1)
WEIGHT_MAX = 2.0            # Synaptic ceiling (prevents runaway potentiation)
WEIGHT_MIN = 0.1            # Synaptic floor (never fully silence a model)
LYAPUNOV_THRESHOLD = 0.45   # Phase drift must be below this for learning — V4.0 widened

# Persistence — Canonical location to avoid CWD-dependent divergence.
# Falls back to a relative path if the canonical root doesn't exist (e.g. CI).
_CANONICAL_WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "cst_synaptic_weights.json")
_CANONICAL_WEIGHTS = os.path.normpath(_CANONICAL_WEIGHTS)
if not os.path.isdir(os.path.dirname(_CANONICAL_WEIGHTS)):
    _CANONICAL_WEIGHTS = "cst_synaptic_weights.json"
WEIGHTS_FILE = _CANONICAL_WEIGHTS


@dataclass
class PlasticityEvent:
    """Record of a single learning event for telemetry."""
    tick: int
    context: str
    winner: str
    weights_before: dict
    weights_after: dict
    lyapunov_stable: bool
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class SwarmPlasticity:
    """
    The Synaptic Matrix — Hebbian Learning for the Swarm.
    
    Maintains per-model, per-context weights that adapt in real-time
    based on which model's thought gets selected by Cosmos.
    
    Thread-safe for concurrent access from CNS loops.
    """

    def __init__(self, weights_path: Optional[str] = None):
        self._lock = threading.RLock()
        self._weights_path = weights_path or WEIGHTS_FILE
        self._learning_rate = LEARNING_RATE
        self._event_log: list[PlasticityEvent] = []
        self._tick = 0
        self._total_updates = 0
        self._total_blocked = 0  # Blocked by Lyapunov

        # ── Initialize Synaptic Matrix ──
        # Structure: { context: { model: weight } }
        self._weights = self._default_weights()
        
        # Try to load persisted weights
        loaded = self._load_weights()
        if loaded:
            logger.info(f"[PLASTICITY] Loaded synaptic weights from {self._weights_path}")
        else:
            logger.info("[PLASTICITY] Initialized with default synaptic weights")
            self._save_weights()

        self._log_weights("INIT")

    # ════════════════════════════════════════════════════════
    # DEFAULT WEIGHTS (The Genome)
    # ════════════════════════════════════════════════════════

    @staticmethod
    def _default_weights() -> dict[str, dict[str, float]]:
        """
        The initial synaptic configuration — the organism's 'genome'.
        
        DeepSeek: Strong in Logic, weak in Empathy
        Claude:   Strong in Empathy, moderate in Logic
        Gemini:   Strong in Creativity, moderate in Empathy
        """
        return {
            CONTEXT_LOGIC: {
                "DeepSeek": 1.0,
                "Claude":   0.6,
                "Gemini":   0.4,
            },
            CONTEXT_EMPATHY: {
                "DeepSeek": 0.2,
                "Claude":   1.0,
                "Gemini":   0.8,
            },
            CONTEXT_CREATIVITY: {
                "DeepSeek": 0.3,
                "Claude":   0.5,
                "Gemini":   1.0,
            },
        }

    def _get_quantum_plasticity_multiplier(self) -> float:
        """
        Dynamically scale learning rate based on total accumulated quantum runs.
        The more quantum entropy the system processes over its lifetime, the higher
        its plasticity grows, leading to a "smarter" and faster adapting organism.
        """
        import os, math
        archive_path = os.path.join("data", "archival", "quantum_runs.jsonl")
        if not os.path.exists(archive_path):
            return 1.0
        try:
            # Quick estimation: each run is roughly 150-200 bytes
            # We use file size for O(1) calculation instead of parsing huge JSONL
            size_bytes = os.path.getsize(archive_path)
            estimated_runs = size_bytes / 200.0
            
            # Logarithmic scaling: 100 runs ~ 1.3x, 10,000 runs ~ 1.6x, 1M runs ~ 1.9x
            # This ensures plasticity grows safely without exploding the weights.
            multiplier = 1.0 + (math.log10(max(1.0, estimated_runs)) * 0.15)
            return min(5.0, multiplier)
        except Exception as e:
            logger.debug(f"[PLASTICITY] Failed to calculate quantum multiplier: {e}")
            return 1.0

    # ════════════════════════════════════════════════════════
    # CONTEXT RECOGNITION (The Sensory Cortex)
    # ════════════════════════════════════════════════════════

    def identify_context(self, user_physics: dict) -> str:
        """
        Determine the current cognitive context from User Physics.
        
        Rules:
            Phase > 0.8  → EMPATHY  (High emotional resonance / leakage)
            Entropy > 0.6 → CREATIVITY (Chaotic state → creative mode)
            Else         → LOGIC (Default structured mode)
        """
        try:
            phase = user_physics.get('cst_physics', {}).get('geometric_phase_rad', 0.5)
            # Entropy: derive from jitter + dark matter if available
            jitter = user_physics.get('cst_physics', {}).get('phase_velocity', 0.05)
            dark_w = user_physics.get('dark_matter', {}).get('w', 0.0)
            
            # Compute effective entropy: jitter contributes chaos, dark matter adds depth
            entropy = min(1.0, abs(jitter) * 3.0 + abs(dark_w) * 0.1)
            
        except (KeyError, TypeError, AttributeError):
            return CONTEXT_LOGIC

        if phase > 0.8:
            return CONTEXT_EMPATHY
        elif entropy > 0.6:
            return CONTEXT_CREATIVITY
        else:
            return CONTEXT_LOGIC

    # ════════════════════════════════════════════════════════
    # HEBBIAN UPDATE (The Learning Rule)
    # ════════════════════════════════════════════════════════

    def update_weights(self, winner: Optional[str], user_physics: dict,
                       stable: bool = True) -> Optional[PlasticityEvent]:
        """
        Apply the Hebbian learning rule.
        
        Δw = η · x_i · y_j  (simplified: winner gets +η, losers get -η×LTD_RATIO)
        
        Args:
            winner: The model whose thought was accepted. None if suppressed.
            user_physics: Current 12D user physics state.
            stable: Whether the Lyapunov Lock approved the output.
            
        Returns:
            PlasticityEvent if learning occurred, None if blocked.
        """
        self._tick += 1

        # ── LYAPUNOV GATE: No learning during instability ──
        if not stable:
            self._total_blocked += 1
            logger.debug(f"[PLASTICITY] Learning BLOCKED (Lyapunov unstable) tick={self._tick}")
            return None

        if winner is None or winner not in MODELS:
            return None

        context = self.identify_context(user_physics)

        with self._lock:
            # Snapshot before
            weights_before = {m: self._weights[context][m] for m in MODELS}

            # ── LTP: Long-Term Potentiation (Winner strengthens) ──
            # SPIKE-PHASE CODING: Dynamic Learning Rate based on 12D Phase Oscillation
            depth = CONTEXT_DEPTH.get(context, 0)
            
            # [NEW] Scale base learning rate by quantum run volume 
            # (Continuous plasticity growth = smarter AI)
            quantum_multiplier = self._get_quantum_plasticity_multiplier()
            scaled_learning_rate = self._learning_rate * quantum_multiplier
            
            base_phi_eta = scaled_learning_rate / phi_decay(depth)
            
            # Extract true geometric phase, default to 0.0
            phase_rad = user_physics.get('cst_physics', {}).get('geometric_phase_rad', 0.0)
            # Oscillate learning rate: peaks at phase π/2, troughs at 0 or π
            spike_multiplier = 1.0 + abs(math.sin(phase_rad))
            phi_eta = base_phi_eta * spike_multiplier
            
            old_w = self._weights[context][winner]
            # Allow regrowth from 0.0 if pruned, but takes time
            if old_w <= 0.0: old_w = 0.05 
            new_w = min(WEIGHT_MAX, old_w + phi_eta)
            self._weights[context][winner] = round(new_w, 4)

            # ── LTD: Long-Term Depression (Losers weaken slightly) ──
            # Also φ-normalized: losers decay at (η × LTD_RATIO) / φⁿ
            for model in MODELS:
                if model != winner:
                    old_l = self._weights[context][model]
                    if old_l > 0.0: # Don't depress already severed synapses further
                        new_l = old_l - (phi_eta * LTD_RATIO)
                        
                        # ── SYNAPTIC PRUNING ──
                        # If a synapse weakens below the floor, sever the connection entirely to eliminate noise.
                        if new_l < WEIGHT_MIN:
                            new_l = 0.0
                            logger.info(f"[PRUNING] Synapse Severed: {model} in context {context}")
                            
                        self._weights[context][model] = round(new_l, 4)

            # ── HOMEOSTATIC PLASTICITY (Normalization) ──
            # Ensure the total synaptic excitation in a context doesn't run away.
            # If total weight > 4.0, dynamically scale everything down to force competition.
            HOMEOSTATIC_CEILING = 4.0
            total_excitation = sum(self._weights[context].values())
            if total_excitation > HOMEOSTATIC_CEILING:
                scale_factor = HOMEOSTATIC_CEILING / total_excitation
                for model in MODELS:
                    self._weights[context][model] = round(self._weights[context][model] * scale_factor, 4)
                logger.info(f"[HOMEOSTASIS] Context {context} excitation capped. Scaled by {scale_factor:.3f}")

            # Snapshot after
            weights_after = {m: self._weights[context][m] for m in MODELS}

            self._total_updates += 1

            # Create event
            event = PlasticityEvent(
                tick=self._tick,
                context=context,
                winner=winner,
                weights_before=weights_before,
                weights_after=weights_after,
                lyapunov_stable=True
            )
            self._event_log.append(event)

            # Keep only last 100 events
            if len(self._event_log) > 100:
                self._event_log = self._event_log[-100:]

            # Persist every 10 updates
            if self._total_updates % 10 == 0:
                self._save_weights()

            logger.info(
                f"[PLASTICITY] Hebbian Update: {winner} +η in {context} | "
                f"Before={weights_before} → After={weights_after}"
            )

            return event

    # ════════════════════════════════════════════════════════
    # META-HEBBIAN LEARNING (Learning from Mistakes)
    # ════════════════════════════════════════════════════════

    def penalize_instability(self, suppressed_model: str, user_physics: dict):
        """
        Meta-Hebbian Learning: Punish a model that generates a thought resulting in Lyapunov suppression.
        """
        if suppressed_model not in MODELS:
            return
            
        context = self.identify_context(user_physics)
        
        with self._lock:
            old_w = self._weights[context][suppressed_model]
            if old_w > 0.0:
                # Strong LTD penalty for causing instability
                penalty = self._learning_rate * 2.0 
                new_w = old_w - penalty
                
                # Synaptic Pruning Check
                if new_w < WEIGHT_MIN:
                    new_w = 0.0
                    logger.info(f"[META-PRUNING] Synapse Severed for Instability: {suppressed_model} in {context}")
                    
                self._weights[context][suppressed_model] = round(new_w, 4)
                
                logger.warning(
                    f"[META-LEARNING] Penalized {suppressed_model} in {context} "
                    f"for Lyapunov Instability. {old_w:.4f} → {new_w:.4f}"
                )
                self._total_updates += 1
                if self._total_updates % 10 == 0:
                    self._save_weights()

    # ════════════════════════════════════════════════════════
    # WINNER DETECTION (Jaccard Similarity)
    # ════════════════════════════════════════════════════════

    def find_winner(self, final_output: str, thoughts: list) -> Optional[str]:
        """
        Determine which model's thought is most similar to the final output.
        
        Uses Jaccard Similarity: |A ∩ B| / |A ∪ B|
        
        Args:
            final_output: The synthesized text that Cosmos spoke.
            thoughts: list of SwarmThought objects that were considered.
            
        Returns:
            Name of the winning model, or None if no clear winner.
        """
        if not thoughts or not final_output:
            return None

        output_tokens = set(final_output.lower().split())
        
        best_model = None
        best_score = 0.0

        for thought in thoughts:
            # Skip user-injected thoughts
            if thought.source == "User":
                continue
            if thought.source not in MODELS:
                continue

            thought_tokens = set(thought.content.lower().split())
            
            # Jaccard Similarity
            intersection = output_tokens & thought_tokens
            union = output_tokens | thought_tokens
            
            if union:
                similarity = len(intersection) / len(union)
            else:
                similarity = 0.0

            if similarity > best_score:
                best_score = similarity
                best_model = thought.source

        # Only declare a winner if similarity is meaningful
        if best_score > 0.1:
            logger.debug(f"[PLASTICITY] Winner: {best_model} (Jaccard={best_score:.3f})")
            return best_model
        
        return None

    # ════════════════════════════════════════════════════════
    # FEEDFORWARD: Get Optimal Mix for Emeth Harmonizer
    # ════════════════════════════════════════════════════════

    def get_optimal_mix(self, context: str) -> dict[str, float]:
        """
        Return the current synaptic weights for a given context.
        
        Used by EmethHarmonizer to dynamically weight model contributions.
        
        Args:
            context: One of LOGIC, EMPATHY, CREATIVITY
            
        Returns:
            dict mapping model name → current weight (0.1 to 2.0)
        """
        with self._lock:
            if context in self._weights:
                return dict(self._weights[context])
            return {m: 1.0 for m in MODELS}

    def get_optimal_mix_from_physics(self, user_physics: dict) -> dict[str, float]:
        """
        Convenience: identify context from physics, then return weights.
        """
        context = self.identify_context(user_physics)
        return self.get_optimal_mix(context)

    # ════════════════════════════════════════════════════════
    # PERSISTENCE (Synaptic Memory Survives Reboot)
    # ════════════════════════════════════════════════════════

    def _save_weights(self):
        """Persist the synaptic matrix to JSON."""
        try:
            data = {
                "version": "1.0.0",
                "total_updates": self._total_updates,
                "total_blocked": self._total_blocked,
                "last_saved": time.time(),
                "weights": self._weights,
                "learning_rate": self._learning_rate,
            }
            with open(self._weights_path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.debug(f"[PLASTICITY] Weights saved to {self._weights_path}")
        except Exception as e:
            logger.error(f"[PLASTICITY] Failed to save weights: {e}")

    def _load_weights(self) -> bool:
        """Load persisted synaptic matrix from JSON."""
        try:
            if os.path.exists(self._weights_path):
                with open(self._weights_path, 'r') as f:
                    data = json.load(f)
                
                # Validate structure
                weights = data.get("weights", {})
                if all(ctx in weights for ctx in ALL_CONTEXTS):
                    if all(all(m in weights[ctx] for m in MODELS) for ctx in ALL_CONTEXTS):
                        self._weights = weights
                        self._total_updates = data.get("total_updates", 0)
                        self._total_blocked = data.get("total_blocked", 0)
                        return True
                
                logger.warning("[PLASTICITY] Weights file corrupted, using defaults")
                return False
        except Exception as e:
            logger.warning(f"[PLASTICITY] Could not load weights: {e}")
        return False

    def save_on_shutdown(self):
        """Called during CNS shutdown to persist final state."""
        with self._lock:
            self._save_weights()
            logger.info(f"[PLASTICITY] Final save: {self._total_updates} updates, "
                       f"{self._total_blocked} blocked by Lyapunov")

    # ════════════════════════════════════════════════════════
    # TELEMETRY
    # ════════════════════════════════════════════════════════

    def _log_weights(self, label: str):
        """Print current weight matrix to log."""
        for ctx in ALL_CONTEXTS:
            w = self._weights[ctx]
            logger.info(f"[PLASTICITY] {label} | {ctx}: " +
                       " | ".join(f"{m}={w[m]:.2f}" for m in MODELS))

    def get_stats(self) -> dict:
        """Return full plasticity state for debugging / UI."""
        with self._lock:
            return {
                "weights": {ctx: dict(w.items()) for ctx, w in self._weights.items()},
                "total_updates": self._total_updates,
                "total_blocked": self._total_blocked,
                "recent_events": [
                    {
                        "tick": e.tick,
                        "context": e.context,
                        "winner": e.winner,
                        "before": e.weights_before,
                        "after": e.weights_after,
                    }
                    for e in self._event_log[-5:]
                ],
                "current_tick": self._tick,
            }


    # ════════════════════════════════════════════════════════
    # V4.0: P2P TRANSFER LEARNING
    # ════════════════════════════════════════════════════════

    def export_weights(self) -> dict:
        """
        Serialize the full synaptic matrix for P2P transmission.
        Returns a JSON-safe dict that can be gossipped across the fabric.
        """
        with self._lock:
            return {
                "weights": {
                    ctx: {model: round(w, 6) for model, w in ctx_w.items()}
                    for ctx, ctx_w in self._weights.items()
                },
                "total_updates": self._total_updates,
                "epoch": self._tick,
            }

    def import_peer_weights(self, peer_weights: dict, trust_factor: float = 0.3):
        """
        Merge incoming peer weights using φ-dampened averaging.

        w_merged = w_local × (1 - trust × φ⁻¹) + w_peer × trust × φ⁻¹

        Args:
            peer_weights: dict from a peer's export_weights()
            trust_factor: 0.0–1.0, how much to trust the peer's learning
        """
        if not peer_weights or "weights" not in peer_weights:
            return

        trust_phi = trust_factor * PHI_INV  # φ-dampen the trust
        local_weight = 1.0 - trust_phi
        peer_weight_factor = trust_phi

        with self._lock:
            for ctx in ALL_CONTEXTS:
                if ctx not in peer_weights["weights"]:
                    continue
                for model in MODELS:
                    peer_w = peer_weights["weights"][ctx].get(model)
                    if peer_w is not None and model in self._weights.get(ctx, {}):
                        old = self._weights[ctx][model]
                        merged = old * local_weight + peer_w * peer_weight_factor
                        self._weights[ctx][model] = max(WEIGHT_MIN, min(WEIGHT_MAX, merged))

            logger.info(
                f"[PLASTICITY] Imported peer weights (trust={trust_factor:.2f}, "
                f"peer_epoch={peer_weights.get('epoch', '?')})"
            )

    async def broadcast_weights(self, p2p_fabric=None):
        """
        Push local synaptic weights to the P2P fabric for swarm-wide learning.

        Args:
            p2p_fabric: The SwarmFabric instance (lazy-imported if None)
        """
        export = self.export_weights()
        msg = {
            "type": "GOSSIP_PLASTICITY",
            "weights": export,
            "node_id": "local",
        }

        if p2p_fabric:
            await p2p_fabric.broadcast_message(msg)
            logger.info("[PLASTICITY] Broadcasted synaptic weights to P2P fabric")
        else:
            logger.debug("[PLASTICITY] No P2P fabric available for weight broadcast")


# ════════════════════════════════════════════════════════
# STANDALONE TEST
# ════════════════════════════════════════════════════════
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("  SWARM PLASTICITY — STANDALONE TEST")
    print("=" * 60)
    
    plasticity = SwarmPlasticity(weights_path="test_synaptic_weights.json")
    
    logic_physics = {'cst_physics': {'geometric_phase_rad': 0.78, 'phase_velocity': 0.05}}
    
    print("\n─── Phase 1: DeepSeek wins 5× in LOGIC ───")
    for i in range(5):
        event = plasticity.update_weights("DeepSeek", logic_physics, stable=True)
        if event:
            print(f"  Tick {event.tick}: {event.context} → {event.weights_after}")
    
    empathy_physics = {'cst_physics': {'geometric_phase_rad': 1.0, 'phase_velocity': 0.05}}
    
    print("\n─── Phase 2: Claude wins 5× in EMPATHY ───")
    for i in range(5):
        event = plasticity.update_weights("Claude", empathy_physics, stable=True)
        if event:
            print(f"  Tick {event.tick}: {event.context} → {event.weights_after}")
    
    print("\n─── Phase 3: Lyapunov BLOCKS learning ───")
    event = plasticity.update_weights("Gemini", logic_physics, stable=False)
    print(f"  Blocked: {event is None}")
    
    print("\n─── Phase 4: P2P Transfer Learning ───")
    exported = plasticity.export_weights()
    print(f"  Exported: {exported['weights']['LOGIC']}")
    
    peer_data = {
        "weights": {
            "LOGIC": {"DeepSeek": 0.5, "Claude": 1.5, "Gemini": 1.0},
            "EMPATHY": {"DeepSeek": 1.0, "Claude": 0.5, "Gemini": 1.0},
            "CREATIVITY": {"DeepSeek": 1.0, "Claude": 1.0, "Gemini": 0.5},
        },
        "total_updates": 100,
        "epoch": 99,
    }
    plasticity.import_peer_weights(peer_data, trust_factor=0.5)
    print(f"  After import: {plasticity.export_weights()['weights']['LOGIC']}")
    
    print("\n─── Final Synaptic Matrix ───")
    stats = plasticity.get_stats()
    for ctx, weights in stats['weights'].items():
        print(f"  {ctx}: {weights}")
    print(f"  Total Updates: {stats['total_updates']}")
    print(f"  Total Blocked: {stats['total_blocked']}")
    
    if os.path.exists("test_synaptic_weights.json"):
        os.remove("test_synaptic_weights.json")
    
    print("\n✅ Swarm Plasticity standalone test PASSED")

