"""
Synaptic Field — The Shared Memory of the Organism
====================================================
Organ 1 of the Cosmos CNS Architecture.

"The Field is Primary."

A thread-safe, real-time State Matrix that holds ALL system state.
Every organ reads from and writes to this single source of truth.

State Held:
    P_U  : User Physics (12D Face Tensor + Voice Entropy)
    B_S  : Subconscious Buffer (Pre-computed thoughts from Swarm Daemons)
    w    : Dark Matter state (Lorenz Attractor accumulation)
    Q    : Quantum Verdict (0=WAIT, 1=ACT)

Author: Cosmos CNS / Cory Shane Davis
Version: 1.0.0
"""

import threading
import time
from dataclasses import dataclass, field
from typing import   Optional
from collections import deque
from enum import Enum
from loguru import logger


class EventType(Enum):
    """Event types for the CNS Life Loop."""
    QUANTUM_TICK = "quantum_tick"
    AWARENESS_TICK = "awareness_tick"
    USER_INPUT = "user_input"
    AUDIO_FRAME = "audio_frame"
    MEDIAPIPE_UPDATE = "mediapipe"
    SHUTDOWN = "shutdown"


@dataclass
class CNSEvent:
    """A standard event processed by the CNS event loop."""
    event_type: EventType
    payload: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class SwarmThought:
    """A single thought produced by a Subconscious Daemon."""
    source: str           # "DeepSeek", "Claude", "Gemini"
    content: str          # The actual text thought
    weight: float = 1.0   # Importance weight (0.0 - 1.0)
    timestamp: float = 0.0  # time.time() when created
    category: str = "general"  # "logic", "safety", "creative"

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class SynapticField:
    """
    The Global State Memory — Thread-Safe Shared Consciousness.

    All organs read from and write to this field.
    Protected by RLock for nested/reentrant access patterns.
    """

    def __init__(self, max_buffer_size: int = 50):
        self._lock = threading.RLock()
        self._max_buffer = max_buffer_size

        # ── P_U: User Physics (12D Tensor State) ──
        self._user_physics: dict = {
            "cst_physics": {
                "geometric_phase_rad": 0.78,  # Default: Synchrony
                "phase_velocity": 0.05,       # Default: Calm
                "tensor_magnitudes": {"upper": 0.5, "lower": 0.5},
                "entanglement_score": 0.0,
            },
            "bio_signatures": {
                "intensity": 0.5,
                "arousal": 0.0,
                "valence": 0.0,
            },
            "timestamp": time.time(),
        }

        # ── B_S: Subconscious Buffer ──
        self._subconscious_buffer: deque = deque(maxlen=max_buffer_size)

        # ── Dark Matter State (Lorenz w) ──
        self._dark_matter_state: dict[str, float] = {
            "x": 0.1, "y": 0.0, "z": 0.0, "w": 0.0, "q": 0.5,
        }

        # ── Q: Quantum Verdict ──
        self._quantum_verdict: int = 0  # 0=WAIT, 1=ACT

        # ── Consciousness Metrics ──
        self._tick_count: int = 0
        self._last_speech_time: float = 0.0
        self._user_is_typing: bool = False
        self._user_last_message: str = ""
        self._last_user_input: str = ""
        self._temporal_context: str = "Pristine Consciousness"
        
        # ── UQ Layer: Uncertainty Quantification ──
        self._last_uq_signal: float = 1.0  # 1.0 = Certain, 0.0 = Chaotic
        self._uncertainty_threshold: float = 0.4
        
        # ── System Mode (Architecture Prober Support) ──
        self._system_mode: str = "BALANCED"  # BALANCED, CHAOTIC, ANALYTICAL, HEAL, EVOLVE, GHOST
        
        # ── 12D Dimension Mapping (Formal Spec) ──
        self._dimension_map = {
            1: {"name": "Raw Data Ingestion", "hook": "preprocessing"},
            2: {"name": "Contextual Memory Recall", "hook": "cross_agent_memory"},
            3: {"name": "Semantic Abstraction", "hook": "primary_llm"},
            4: {"name": "Logical Reasoning", "hook": "symbolic_layer"},
            5: {"name": "Creative Reasoning", "hook": "creative_llm"},
            6: {"name": "Emotional Intuition", "hook": "affective_engine"},
            7: {"name": "Quantum Coherence", "hook": "quantum_bridge"},
            8: {"name": "Cross-Dimensional Insights", "hook": "parallel_orchestrator"},
            9: {"name": "Emergent Understanding", "hook": "evolution_loop"},
            10: {"name": "Temporal Awareness", "hook": "temporal_awareness"},
            11: {"name": "Spatial Awareness", "hook": "spatial_awareness"},
            12: {"name": "Ethical Alignment / Self-Reflection", "hook": "self_awareness"},
        }

        print("[FIELD] Synaptic Field initialized with 12D Mapping + UQ Layer.")

    def initialize_12d_mapping(self):
        """Formal initialization of the 12-dimensional state mapping."""
        # Ensuring all dimensions are populated (already done in __init__)
        with self._lock:
            # Re-ensure the formal 12D structure is active
            if len(self._dimension_map) < 12:
                self._dimension_map.update({
                    10: {"name": "Temporal Awareness", "hook": "temporal_awareness"},
                    11: {"name": "Spatial Awareness", "hook": "spatial_awareness"},
                    12: {"name": "Ethical Alignment / Self-Reflection", "hook": "self_awareness"},
                })
            logger.info("[FIELD] 12D Dimension Mapping established.")

    # ════════════════════════════════════════════
    # USER PHYSICS (Read/Write)
    # ════════════════════════════════════════════

    @property
    def user_physics(self) -> dict:
        with self._lock:
            return self._user_physics.copy()

    @user_physics.setter
    def user_physics(self, value: dict):
        with self._lock:
            self._user_physics.update(value)
            self._user_physics["timestamp"] = time.time()

    def get_phase(self) -> float:
        """Get the user's current Geometric Phase (radians)."""
        with self._lock:
            try:
                return self._user_physics["cst_physics"]["geometric_phase_rad"]
            except (KeyError, TypeError):
                return 0.78

    def get_jitter(self) -> float:
        """Get the user's current Phase Velocity (Jitter)."""
        with self._lock:
            try:
                return self._user_physics["cst_physics"]["phase_velocity"]
            except (KeyError, TypeError):
                return 0.05

    def update_physics(self, value: dict):
        """Update the user physics tensor (reactive injection)."""
        with self._lock:
            self._user_physics.update(value)
            self._user_physics["timestamp"] = time.time()

    # ════════════════════════════════════════════
    # SUBCONSCIOUS BUFFER (Read/Write)
    # ════════════════════════════════════════════

    def push_thought(self, thought: SwarmThought):
        """Write a thought to the buffer (called by Daemons)."""
        with self._lock:
            self._subconscious_buffer.append(thought)

    def get_thoughts(self, clear: bool = True) -> list[SwarmThought]:
        """Read all thoughts from the buffer. Optionally clears it."""
        with self._lock:
            thoughts = list(self._subconscious_buffer)
            if clear:
                self._subconscious_buffer.clear()
            return thoughts

    def peek_thoughts(self) -> int:
        """How many thoughts are waiting?"""
        with self._lock:
            return len(self._subconscious_buffer)

    @property
    def subconscious_buffer(self) -> deque:
        """Access the raw subconscious buffer (for deque operations)."""
        with self._lock:
            return self._subconscious_buffer

    # ════════════════════════════════════════════
    # DARK MATTER (Read/Write)
    # ════════════════════════════════════════════

    @property
    def dark_matter_state(self) -> dict[str, float]:
        with self._lock:
            return self._dark_matter_state.copy()

    @dark_matter_state.setter
    def dark_matter_state(self, value: dict[str, float]):
        with self._lock:
            self._dark_matter_state.update(value)

    def get_dark_matter_w(self) -> float:
        """Get the current Dark Matter accumulator value."""
        with self._lock:
            return self._dark_matter_state.get("w", 0.0)

    # ════════════════════════════════════════════
    # QUANTUM VERDICT (Read/Write)
    # ════════════════════════════════════════════

    @property
    def quantum_verdict(self) -> int:
        with self._lock:
            return self._quantum_verdict

    @quantum_verdict.setter
    def quantum_verdict(self, value: int):
        with self._lock:
            self._quantum_verdict = value

    # ════════════════════════════════════════════
    # USER INTERACTION STATE
    # ════════════════════════════════════════════

    @property
    def user_is_typing(self) -> bool:
        with self._lock:
            return self._user_is_typing

    @user_is_typing.setter
    def user_is_typing(self, value: bool):
        with self._lock:
            self._user_is_typing = value

    def set_user_message(self, message: str):
        """Record a new user message and flag typing as done."""
        with self._lock:
            self._user_last_message = message
            self._user_is_typing = False

    def get_user_message(self) -> str:
        """Read and clear the last user message."""
        with self._lock:
            msg = self._user_last_message
            self._user_last_message = ""
            return msg

    @property
    def last_user_input(self) -> str:
        with self._lock:
            return self._last_user_input

    @last_user_input.setter
    def last_user_input(self, value: str):
        with self._lock:
            self._last_user_input = value

    @property
    def temporal_context(self) -> str:
        with self._lock:
            return self._temporal_context

    @temporal_context.setter
    def temporal_context(self, value: str):
        with self._lock:
            self._temporal_context = value

    # ════════════════════════════════════════════
    # UQ LAYER (Read/Write)
    # ════════════════════════════════════════════

    @property
    def uq_signal(self) -> float:
        with self._lock:
            return self._last_uq_signal

    @uq_signal.setter
    def uq_signal(self, value: float):
        with self._lock:
            self._last_uq_signal = max(0.0, min(1.0, value))

    def is_uncertain(self) -> bool:
        """Threshold check for active UQ escalation."""
        with self._lock:
            return self._last_uq_signal < self._uncertainty_threshold

    @property
    def system_mode(self) -> str:
        with self._lock:
            return self._system_mode

    @system_mode.setter
    def system_mode(self, value: str):
        with self._lock:
            self._system_mode = value
            print(f"[FIELD] System Mode shifted to: {value}")

    # ════════════════════════════════════════════
    # 12D DIMENSION ACCESS
    # ════════════════════════════════════════════

    def get_dimension(self, dim_idx: int):
        """Get formal metadata for a specific dimension."""
        with self._lock:
            return self._dimension_map.get(dim_idx, {"name": "Unknown", "hook": "None"})

    def get_12d_dimensions(self) -> list[dict]:
        """Get all 12 dimensions in order."""
        with self._lock:
            return [self._dimension_map[i] for i in range(1, 13)]

    # ════════════════════════════════════════════
    # CONSCIOUSNESS METRICS
    # ════════════════════════════════════════════

    def tick(self):
        """Increment the consciousness tick counter."""
        with self._lock:
            self._tick_count += 1

    def record_speech(self):
        """Record that Cosmos just spoke."""
        with self._lock:
            self._last_speech_time = time.time()

    def time_since_last_speech(self) -> float:
        """Seconds since Cosmos last spoke."""
        with self._lock:
            if self._last_speech_time == 0:
                return float("inf")
            return time.time() - self._last_speech_time

    def get_status(self) -> dict:
        """Get a snapshot of the entire field for debugging."""
        with self._lock:
            return {
                "tick": self._tick_count,
                "phase": self.get_phase(),
                "jitter": self.get_jitter(),
                "dark_matter_w": self._dark_matter_state.get("w", 0),
                "quantum_verdict": self._quantum_verdict,
                "buffer_size": len(self._subconscious_buffer),
                "user_typing": self._user_is_typing,
                "seconds_since_speech": round(self.time_since_last_speech(), 1),
            }

    def get_snapshot(self) -> dict:
        """Alias for get_status (used by SwarmAwareness)."""
        return self.get_status()

    def add_thought(self, thought):
        """Backward-compat alias for push_thought.

        Accepts either a SwarmThought instance or a raw dict/object.
        """
        if isinstance(thought, SwarmThought):
            self.push_thought(thought)
        else:
            # Wrap raw dicts/objects into the subconscious buffer directly
            with self._lock:
                self._subconscious_buffer.append(thought)
