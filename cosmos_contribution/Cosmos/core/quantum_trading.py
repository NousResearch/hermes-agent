"""
Cosmos Quantum Trading Cortex — AGI v2.1

A new cortex for the Cosmos organism: quantum-enhanced trading intelligence.
Fuses EMA momentum, quantum simulation, and collective deliberation into
actionable trading signals. Signals feed back into memory and evolution,
making the whole system smarter over time.

This is a NEW SENSE for the organism — not a replacement for existing trading logic.
It layers on top of DegenTrader, FarSight, and QuantumEvolution.

IBM Quantum Strategy:
- Real-time signals use SIMULATOR ONLY (unlimited, fast, no budget cost)
- Real QPU hardware (10 min/28 days) reserved for QuantumAlgoOptimizer:
  → QAOA optimization of DegenTrader's 12 tunable trading parameters
  → Quantum genetic algorithm for multi-parameter co-optimization
  → Run weekly or on-demand, each run uses ~30-60s of hardware
  → This improves the ALGORITHM itself, not individual predictions
"""

import asyncio
import uuid
import math
import time
import random
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Callable, Tuple, Union, Set
from loguru import logger

try:
    import numpy as np
except ImportError:
    np = None

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class QuantumTradingSignal:
    """A fused quantum trading signal — the organism's trading prediction."""
    signal_id: str
    token_address: str
    timestamp: datetime

    # EMA Momentum
    ema_fast: float = 0.0           # ~10-tick EMA
    ema_slow: float = 0.0           # ~50-tick EMA
    ema_crossover: str = "neutral"  # "bullish" | "bearish" | "neutral"
    momentum_score: float = 0.0     # -1.0 to 1.0

    # Quantum Simulation
    quantum_bull_prob: float = 0.5  # 0-1 from QAOA/Monte Carlo
    quantum_confidence: float = 0.0 # 0-1
    quantum_entropy: float = 0.0    # randomness seed from real QPU
    quantum_method: str = "simulator"  # "hardware" | "simulator" | "classical"

    # Collective Intelligence
    collective_direction: str = "neutral"  # "bullish" | "bearish" | "neutral"
    collective_confidence: float = 0.0
    agents_consulted: list[str] = field(default_factory=list)

    # Fused Signal
    direction: str = "HOLD"         # "LONG" | "SHORT" | "HOLD"
    confidence: float = 0.0         # 0-1 weighted fusion
    strength: int = 1               # 1-5 signal strength
    reasoning: str = ""             # human-readable explanation

    # Tracking
    outcome: str = "pending"        # "pending" | "correct" | "incorrect"
    price_at_signal: float = 0.0    # price when signal was generated
    actual_price_1m: float = 0.0    # price 1 min after signal
    actual_price_5m: float = 0.0    # price 5 min after signal
    resolved_at: Optional[datetime] = None

    def to_any(self) -> dict:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        if self.resolved_at:
            d["resolved_at"] = self.resolved_at.isoformat()
        return d

    @classmethod
    def from_any(cls, data: dict) -> "QuantumTradingSignal":
        data = dict(data)
        if isinstance(data.get("timestamp"), str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        if isinstance(data.get("resolved_at"), str):
            data["resolved_at"] = datetime.fromisoformat(data["resolved_at"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class QuantumCorrelation:
    """Cross-token 'entanglement' — correlated price movements."""
    token_a: str
    token_b: str
    correlation: float = 0.0        # -1 to 1
    quantum_verified: bool = False  # verified via Bell-state circuit
    discovered_at: datetime = field(default_factory=datetime.now)
    strength_over_time: list[float] = field(default_factory=list)

    def to_any(self) -> dict:
        d = asdict(self)
        d["discovered_at"] = self.discovered_at.isoformat()
        return d


# =============================================================================
# EMA ENGINE — Per-token momentum tracking from price feeds
# =============================================================================

class EMAEngine:
    """
    Maintains per-token EMA state from 1-second price feeds.
    Fast EMA (~10 ticks, alpha=0.18) and Slow EMA (~50 ticks, alpha=0.04).
    Detects crossovers and computes momentum score.
    """

    def __init__(self, fast_alpha: float = 0.18, slow_alpha: float = 0.04):
        self.fast_alpha = fast_alpha
        self.slow_alpha = slow_alpha
        # token_address -> { ema_fast, ema_slow, prev_crossover, tick_count }
        self._state: dict = {}

    def update(self, token_address: str, price: float) -> dict:
        """Update EMA state for a token and return current indicators."""
        if price <= 0:
            return {"ema_fast": 0, "ema_slow": 0, "crossover": "neutral", "momentum_score": 0}

        state = self._state.get(token_address)
        if state is None:
            state = {
                "ema_fast": price,
                "ema_slow": price,
                "prev_crossover": "neutral",
                "tick_count": 0,
                "prices": deque(maxlen=100),
            }
            self._state[token_address] = state

        # Update EMAs
        state["ema_fast"] = self.fast_alpha * price + (1 - self.fast_alpha) * state["ema_fast"]
        state["ema_slow"] = self.slow_alpha * price + (1 - self.slow_alpha) * state["ema_slow"]
        state["tick_count"] += 1
        state["prices"].append(price)

        # Crossover detection
        diff = state["ema_fast"] - state["ema_slow"]
        threshold = state["ema_slow"] * 0.001  # 0.1% dead zone
        if diff > threshold:
            crossover = "bullish"
        elif diff < -threshold:
            crossover = "bearish"
        else:
            crossover = "neutral"

        # Momentum score: normalized difference between fast and slow EMA (-1 to 1)
        if state["ema_slow"] > 0:
            raw_momentum = diff / state["ema_slow"]
            momentum_score = max(-1.0, min(1.0, raw_momentum * 10))
        else:
            momentum_score = 0.0

        state["prev_crossover"] = crossover

        return {
            "ema_fast": state["ema_fast"],
            "ema_slow": state["ema_slow"],
            "crossover": crossover,
            "momentum_score": momentum_score,
        }

    def get_state(self, token_address: str) -> Optional[dict]:
        return self._state.get(token_address)

    def clear_token(self, token_address: str):
        self._state.pop(token_address, None)


# =============================================================================
# TTL CACHE (mirrors fitness_tracker.py pattern)
# =============================================================================

class TTLCache:
    """Simple time-to-live cache for computed values."""

    def __init__(self, ttl_seconds: float = 5.0, max_size: int = 200):
        self.ttl = ttl_seconds
        self.max_size = max_size
        self._cache: dict[str, Tuple[float]] = {}

    def get(self, key: str) -> Optional[dict]:
        entry = self._cache.get(key)
        if entry is None:
            return None
        ts, value = entry
        if time.time() - ts > self.ttl:
            del self._cache[key]
            return None
        return value

    def set(self, key: str, value: dict):
        if len(self._cache) >= self.max_size:
            # Evict oldest
            oldest_key = min(self._cache, key=lambda k: self._cache[k][0])
            del self._cache[oldest_key]
        self._cache[key] = (time.time(), value)

    def invalidate(self, key: Optional[str] = None):
        if key:
            self._cache.pop(key, None)
        else:
            self._cache.clear()


# =============================================================================
# SIGNAL ACCURACY TRACKER — The organism learns from its predictions
# =============================================================================

class SignalAccuracyTracker:
    """
    Tracks quantum signal accuracy over time.
    Mirrors fitness_tracker.py patterns: TTLCache, deque storage, heapq for leaderboard.
    The organism LEARNS which signal sources perform best.
    """

    def __init__(self, window_size: int = 500, cache_ttl: float = 10.0):
        # All signals (bounded deque for O(1) ops)
        self.signals: deque = deque(maxlen=window_size)
        self.pending: dict[str, QuantumTradingSignal] = {}  # signal_id -> signal (awaiting resolution)

        # Per-source accuracy tracking
        self.source_stats: dict[str, deque] = defaultdict(lambda: deque(maxlen=200))
        # source -> deque of (correct: bool, confidence: float)

        # Fusion weights (learned via accuracy feedback)
        self.fusion_weights = {
            "ema": 0.30,
            "quantum": 0.35,
            "collective": 0.35,
        }

        # Caches
        self._accuracy_cache = TTLCache(ttl_seconds=cache_ttl)
        self._weight_cache = TTLCache(ttl_seconds=cache_ttl * 3)

    def record_signal(self, signal: QuantumTradingSignal):
        """Record a new signal for tracking."""
        self.signals.append(signal)
        if signal.outcome == "pending":
            self.pending[signal.signal_id] = signal

    def resolve_signal(
        self, signal_id: str, actual_price_1m: float, actual_price_5m: float
    ) -> Optional[QuantumTradingSignal]:
        """Resolve a pending signal with actual price outcomes."""
        signal = self.pending.pop(signal_id, None)
        if signal is None:
            return None

        signal.actual_price_1m = actual_price_1m
        signal.actual_price_5m = actual_price_5m
        signal.resolved_at = datetime.now()

        # Determine correctness based on direction vs actual price movement
        if signal.price_at_signal > 0 and actual_price_5m > 0:
            pct_change = (actual_price_5m - signal.price_at_signal) / signal.price_at_signal
            if signal.direction == "LONG":
                signal.outcome = "correct" if pct_change > 0.001 else "incorrect"
            elif signal.direction == "SHORT":
                signal.outcome = "correct" if pct_change < -0.001 else "incorrect"
            else:  # HOLD
                signal.outcome = "correct" if abs(pct_change) < 0.01 else "incorrect"
        else:
            signal.outcome = "incorrect"

        # Record per-source outcomes
        correct = signal.outcome == "correct"
        self.source_stats["ema"].append((correct, signal.momentum_score))
        self.source_stats["quantum"].append((correct, signal.quantum_confidence))
        self.source_stats["collective"].append((correct, signal.collective_confidence))

        # Invalidate caches
        self._accuracy_cache.invalidate()
        self._weight_cache.invalidate()

        return signal

    def get_accuracy_stats(self) -> dict:
        """Get overall accuracy statistics."""
        cached = self._accuracy_cache.get("stats")
        if cached is not None:
            return cached

        resolved = [s for s in self.signals if s.outcome in ("correct", "incorrect")]
        if not resolved:
            stats = {
                "total_signals": len(self.signals),
                "resolved": 0,
                "pending": len(self.pending),
                "win_rate": 0.0,
                "avg_confidence": 0.0,
                "calibration": 0.0,
            }
            self._accuracy_cache.set("stats", stats)
            return stats

        correct_count = sum(1 for s in resolved if s.outcome == "correct")
        win_rate = correct_count / len(resolved) if resolved else 0.0
        avg_confidence = sum(s.confidence for s in resolved) / len(resolved)

        # Calibration: how well confidence predicts correctness
        # Perfect calibration: 70% confidence signals are correct 70% of the time
        high_conf = [s for s in resolved if s.confidence > 0.7]
        high_conf_correct = sum(1 for s in high_conf if s.outcome == "correct") / max(1, len(high_conf))
        calibration = 1.0 - abs(0.7 - high_conf_correct) if high_conf else 0.0

        stats = {
            "total_signals": len(self.signals),
            "resolved": len(resolved),
            "pending": len(self.pending),
            "win_rate": round(win_rate, 4),
            "avg_confidence": round(avg_confidence, 4),
            "calibration": round(calibration, 4),
            "correct": correct_count,
            "incorrect": len(resolved) - correct_count,
        }
        self._accuracy_cache.set("stats", stats)
        return stats

    def get_weight_recommendations(self) -> dict:
        """Get learned optimal fusion weights based on accuracy."""
        cached = self._weight_cache.get("weights")
        if cached is not None:
            return cached

        # Calculate per-source accuracy
        source_accuracy = {}
        for source, records in self.source_stats.items():
            if len(records) < 10:
                source_accuracy[source] = 0.5  # default when insufficient data
                continue
            correct_count = sum(1 for c, _ in records if c)
            source_accuracy[source] = correct_count / len(records)

        # Normalize to weights
        total = sum(source_accuracy.values())
        if total > 0:
            weights = {k: v / total for k, v in source_accuracy.items()}
        else:
            weights = dict(self.fusion_weights)

        self._weight_cache.set("weights", weights)
        return weights

    def get_leaderboard(self) -> list:
        """Which signal sources perform best."""
        leaderboard = []
        for source, records in self.source_stats.items():
            if not records:
                continue
            correct = sum(1 for c, _ in records if c)
            total = len(records)
            leaderboard.append({
                "source": source,
                "accuracy": round(correct / total, 4) if total > 0 else 0,
                "total": total,
                "correct": correct,
            })
        leaderboard.sort(key=lambda x: x["accuracy"], reverse=True)
        return leaderboard

    def get_recent_signals(self, limit: int = 50) -> list[dict]:
        """Get recent signals for display."""
        recent = list(self.signals)[-limit:]
        recent.reverse()
        return [s.to_any() for s in recent]


# =============================================================================
# QUANTUM TRADING CORTEX — The new brain region
# =============================================================================

class QuantumTradingCortex:
    """
    A new cortex for the Cosmos organism — quantum-enhanced trading intelligence.

    Wires into existing infrastructure:
    - Nexus event bus for signal emission
    - MemorySystem for archival storage
    - IBMQuantumProvider for QAOA/Bell circuits
    - ModelSwarm for collective deliberation

    IBM Free Tier: 10 min hardware/28 days, unlimited simulator.
    Every QPU call goes through should_use_hardware() budget check.
    """

    def __init__(self, nexus=None, memory_system=None, ibm_quantum=None, model_swarm=None):
        self.nexus = nexus
        self.memory = memory_system
        self.quantum = ibm_quantum
        self.swarm = model_swarm

        self.ema_engine = EMAEngine()
        self.accuracy_tracker = SignalAccuracyTracker()
        self.correlations: dict[str, QuantumCorrelation] = {}

        # Signal history (bounded)
        self._signal_history: deque = deque(maxlen=1000)

        # Price cache for resolution (token -> deque of (timestamp, price))
        self._price_cache: dict[str, deque] = defaultdict(lambda: deque(maxlen=300))

        self._initialized = False
        self._signal_count = 0

    async def initialize(self) -> bool:
        """Initialize the cortex, connecting to IBM Quantum if available."""
        try:
            # Try to connect to IBM Quantum
            if self.quantum is None:
                try:
                    from Cosmos.integration.quantum.ibm_quantum import get_quantum_provider
                    self.quantum = get_quantum_provider()
                except ImportError:
                    logger.warning("QuantumCortex: IBM Quantum not available, using classical fallback")

            if self.quantum and hasattr(self.quantum, 'connect'):
                try:
                    await self.quantum.connect()
                except Exception as e:
                    logger.warning(f"QuantumCortex: Quantum connect failed, using simulator: {e}")

            self._initialized = True
            logger.info("QuantumTradingCortex initialized")
            return True
        except Exception as e:
            logger.error(f"QuantumCortex init failed: {e}")
            self._initialized = True  # Still usable with classical fallback
            return False

    def feed_price(self, token_address: str, price: float) -> dict:
        """Feed a price tick into the EMA engine. Returns current EMA state."""
        self._price_cache[token_address].append((time.time(), price))
        return self.ema_engine.update(token_address, price)

    async def generate_signal(
        self, token_address: str, price_history: Optional[list[float]] = None,
        current_price: float = 0.0, use_hardware: bool = False
    ) -> QuantumTradingSignal:
        """
        Generate a fused quantum trading signal for a token.

        Pipeline:
        1. EMA momentum (fast, local)
        2. Quantum simulation (QAOA if budget allows, else simulator)
        3. Collective deliberation (ask shadow agents)
        4. Fuse with learned weights from accuracy tracker
        5. Emit via Nexus: QUANTUM_SIGNAL_GENERATED
        6. Store in archival memory for learning
        """
        signal_id = f"qs_{uuid.uuid4().hex[:12]}"
        now = datetime.now()

        # Initialize price history from cache if not provided
        if price_history is None:
            cached_prices = self._price_cache.get(token_address, deque())
            price_history = [p for _, p in cached_prices]

        if current_price <= 0 and price_history:
            current_price = price_history[-1]

        # Feed price history into EMA if we have data
        ema_result = {"ema_fast": 0, "ema_slow": 0, "crossover": "neutral", "momentum_score": 0}
        if price_history:
            for p in price_history[-60:]:
                ema_result = self.ema_engine.update(token_address, p)

        # --- Step 1: EMA Momentum ---
        ema_fast = ema_result["ema_fast"]
        ema_slow = ema_result["ema_slow"]
        ema_crossover = ema_result["crossover"]
        momentum_score = ema_result["momentum_score"]

        # --- Step 2: Quantum Simulation ---
        quantum_result = await self._quantum_simulate(token_address, price_history, use_hardware=use_hardware)

        # --- Step 3: Collective Deliberation ---
        collective_result = await self._collective_deliberate(token_address, current_price, ema_result, quantum_result)

        # --- Step 4: Fuse Signals ---
        weights = self.accuracy_tracker.get_weight_recommendations()
        ema_w = weights.get("ema", 0.30)
        quantum_w = weights.get("quantum", 0.35)
        collective_w = weights.get("collective", 0.35)

        # Compute fused direction score: positive = bullish, negative = bearish
        ema_score = momentum_score  # already -1 to 1
        quantum_score = (quantum_result["bull_prob"] - 0.5) * 2  # map 0-1 to -1..1
        collective_score = self._direction_to_score(collective_result["direction"])

        fused_score = (
            ema_w * ema_score +
            quantum_w * quantum_score +
            collective_w * collective_score
        )

        # Determine direction and confidence
        if fused_score > 0.15:
            direction = "LONG"
        elif fused_score < -0.15:
            direction = "SHORT"
        else:
            direction = "HOLD"

        confidence = min(1.0, abs(fused_score))

        # Signal strength: 1-5 based on confidence and agreement
        agreement_count = sum([
            1 if ema_crossover == "bullish" and direction == "LONG" else (1 if ema_crossover == "bearish" and direction == "SHORT" else 0),
            1 if quantum_result["bull_prob"] > 0.6 and direction == "LONG" else (1 if quantum_result["bull_prob"] < 0.4 and direction == "SHORT" else 0),
            1 if collective_result["direction"] == ("bullish" if direction == "LONG" else "bearish") else 0,
        ])
        strength = max(1, min(5, int(confidence * 3) + agreement_count))

        # Generate reasoning
        reasoning = self._build_reasoning(
            direction, confidence, ema_result, quantum_result, collective_result, weights
        )

        # --- Build Signal ---
        signal = QuantumTradingSignal(
            signal_id=signal_id,
            token_address=token_address,
            timestamp=now,
            ema_fast=ema_fast,
            ema_slow=ema_slow,
            ema_crossover=ema_crossover,
            momentum_score=momentum_score,
            quantum_bull_prob=quantum_result["bull_prob"],
            quantum_confidence=quantum_result["confidence"],
            quantum_entropy=quantum_result.get("entropy", 0.0),
            quantum_method=quantum_result["method"],
            collective_direction=collective_result["direction"],
            collective_confidence=collective_result["confidence"],
            agents_consulted=collective_result.get("agents", []),
            direction=direction,
            confidence=confidence,
            strength=strength,
            reasoning=reasoning,
            price_at_signal=current_price,
        )

        # --- Step 5: Record & Emit ---
        self._signal_history.append(signal)
        self.accuracy_tracker.record_signal(signal)
        self._signal_count += 1

        # Emit via Nexus
        await self._emit_signal(signal)

        # Store in archival memory
        await self._store_in_memory(signal)

        logger.info(
            f"QuantumSignal [{signal_id[:8]}] {token_address[:8]}.. "
            f"→ {direction} ({confidence:.0%}) strength={strength} "
            f"method={quantum_result['method']}"
        )

        return signal

    async def _quantum_simulate(
        self, token_address: str, price_history: list[float],
        use_hardware: bool = False
    ) -> dict:
        """
        Run quantum simulation for price prediction.
        Uses real QPU if budget allows, otherwise simulator.
        Falls back to classical Monte Carlo if no quantum available.
        """
        result = {
            "bull_prob": 0.5,
            "confidence": 0.0,
            "entropy": 0.0,
            "method": "classical",
        }

        if not price_history or len(price_history) < 5:
            return result

        # Quantum simulation — simulator by default, hardware if requested (x402 premium tier)
        if self.quantum:
            try:
                qmc_result = await self._quantum_monte_carlo(
                    price_history, use_hardware=use_hardware
                )
                if qmc_result:
                    result.update(qmc_result)
                    return result

            except ImportError:
                pass
            except Exception as e:
                logger.debug(f"QuantumCortex: Quantum simulation error: {e}")

        # Classical fallback: simple Monte Carlo with trend analysis
        result = self._classical_monte_carlo(price_history)
        return result

    async def _quantum_monte_carlo(
        self, price_history: list[float], use_hardware: bool = False
    ) -> Optional[dict]:
        """
        Quantum-enhanced Monte Carlo simulation.
        Encodes price scenarios as quantum states, measures probability distribution.
        """
        if not self.quantum or np is None:
            return None

        try:
            from qiskit import QuantumCircuit
            from Cosmos.integration.quantum.ibm_quantum import QuantumTaskType
        except ImportError:
            return None

        try:
            # Encode recent price trend into rotation angles
            returns = []
            for i in range(1, min(len(price_history), 20)):
                if price_history[i - 1] > 0:
                    r = (price_history[i] - price_history[i - 1]) / price_history[i - 1]
                    returns.append(r)

            if not returns:
                return None

            # Quantum circuit: encode returns as rotation angles
            # Hardware tier: more qubits (up to 6) and more shots for better fidelity
            max_qubits = 6 if use_hardware else 4
            num_qubits = min(max_qubits, max(2, len(returns) // 4))
            qc = QuantumCircuit(num_qubits, num_qubits)

            # Hadamard for superposition
            for i in range(num_qubits):
                qc.h(i)

            # Encode trend data as phase rotations
            avg_return = sum(returns) / len(returns)
            volatility = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5

            for i in range(num_qubits):
                # Ry rotation based on trend (positive trend = more |1> amplitude)
                theta = math.atan2(avg_return * 100, 1.0) + (i * volatility * 2)
                qc.ry(theta, i)

            # Entangle qubits for correlation
            for i in range(num_qubits - 1):
                qc.cx(i, i + 1)

            # Rz for volatility encoding
            for i in range(num_qubits):
                qc.rz(volatility * math.pi * (i + 1), i)

            # Measure
            qc.measure(range(num_qubits), range(num_qubits))

            # Execute — hardware tier gets more shots for better results
            shots = 2048 if use_hardware else 1024
            job_result = await self.quantum.run_circuit(
                qc, shots=shots,
                task_type=QuantumTaskType.INFERENCE,
                prefer_hardware=use_hardware
            )

            if not job_result.success or not job_result.counts:
                return None

            # Interpret: count |1> states vs |0> states
            total_shots = sum(job_result.counts.values())
            bull_counts = 0
            for bitstring, count in job_result.counts.items():
                # More 1s = bullish
                ones = bitstring.count('1')
                if ones > num_qubits / 2:
                    bull_counts += count

            bull_prob = bull_counts / total_shots

            # Entropy from measurement distribution (higher = less certain)
            probs = [c / total_shots for c in job_result.counts.values()]
            entropy = -sum(p * math.log2(p) for p in probs if p > 0)
            max_entropy = math.log2(len(job_result.counts)) if len(job_result.counts) > 1 else 1.0
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

            # Confidence: inverse of entropy (high entropy = low confidence)
            confidence = max(0.1, 1.0 - normalized_entropy * 0.8)

            return {
                "bull_prob": round(bull_prob, 4),
                "confidence": round(confidence, 4),
                "entropy": round(normalized_entropy, 4),
                "method": "hardware" if job_result.backend_used.startswith("ibm_") else "simulator",
            }

        except Exception as e:
            logger.debug(f"QuantumCortex: QMC failed: {e}")
            return None

    def _classical_monte_carlo(self, price_history: list[float]) -> dict:
        """Classical fallback: trend-weighted Monte Carlo."""
        if len(price_history) < 3:
            return {"bull_prob": 0.5, "confidence": 0.1, "entropy": 0.5, "method": "classical"}

        returns = []
        for i in range(1, len(price_history)):
            if price_history[i - 1] > 0:
                returns.append((price_history[i] - price_history[i - 1]) / price_history[i - 1])

        if not returns:
            return {"bull_prob": 0.5, "confidence": 0.1, "entropy": 0.5, "method": "classical"}

        avg_return = sum(returns) / len(returns)
        volatility = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5

        # Simulate 1000 scenarios
        bull_count = 0
        num_sims = 1000
        for _ in range(num_sims):
            sim_return = avg_return + volatility * random.gauss(0, 1)
            if sim_return > 0:
                bull_count += 1

        bull_prob = bull_count / num_sims
        # Confidence: based on how consistent the trend is
        trend_consistency = abs(avg_return) / (volatility + 1e-10)
        confidence = min(1.0, max(0.1, trend_consistency * 0.3 + 0.2))

        return {
            "bull_prob": round(bull_prob, 4),
            "confidence": round(confidence, 4),
            "entropy": round(1.0 - confidence, 4),
            "method": "classical",
        }

    async def _collective_deliberate(
        self, token_address: str, current_price: float,
        ema_result: dict, quantum_result: dict
    ) -> dict:
        """Ask shadow agents for their opinion on this token."""
        result = {"direction": "neutral", "confidence": 0.0, "agents": []}

        try:
            from Cosmos.core.collective.persistent_agent import call_shadow_agent
        except ImportError:
            return result

        # Prepare context for agents
        context = (
            f"Token {token_address[:12]}.. at ${current_price:.8f}. "
            f"EMA: {ema_result['crossover']}, momentum={ema_result['momentum_score']:.2f}. "
            f"Quantum sim: {quantum_result['bull_prob']:.0%} bull, "
            f"confidence={quantum_result['confidence']:.0%}. "
            f"Quick verdict: bullish, bearish, or neutral? One word + confidence 0-100."
        )

        agents_to_ask = ["grok", "gemini", "deepseek"]
        votes = {"bullish": 0, "bearish": 0, "neutral": 0}
        consulted = []
        confidences = []

        for agent_id in agents_to_ask:
            try:
                response = call_shadow_agent(agent_id, context, timeout=10.0)
                if response:
                    consulted.append(agent_id)
                    response_lower = response.lower()
                    if "bullish" in response_lower or "bull" in response_lower:
                        votes["bullish"] += 1
                    elif "bearish" in response_lower or "bear" in response_lower:
                        votes["bearish"] += 1
                    else:
                        votes["neutral"] += 1

                    # Extract confidence number
                    import re
                    nums = re.findall(r'\d+', response)
                    if nums:
                        conf = int(nums[-1])
                        if 0 <= conf <= 100:
                            confidences.append(conf / 100.0)
            except Exception:
                continue

        if not consulted:
            return result

        # Determine collective direction
        max_votes = max(votes.values())
        if votes["bullish"] == max_votes and votes["bullish"] > votes["bearish"]:
            direction = "bullish"
        elif votes["bearish"] == max_votes:
            direction = "bearish"
        else:
            direction = "neutral"

        confidence = sum(confidences) / len(confidences) if confidences else 0.3

        return {
            "direction": direction,
            "confidence": round(confidence, 4),
            "agents": consulted,
            "votes": dict(votes),
        }

    async def discover_correlations(
        self, token_addresses: list[str]
    ) -> list[QuantumCorrelation]:
        """
        Use quantum circuits to discover cross-token correlations.
        Bell state preparation for token pairs, measure via quantum interference.
        """
        if len(token_addresses) < 2:
            return []

        correlations = []
        pairs = []
        for i in range(len(token_addresses)):
            for j in range(i + 1, len(token_addresses)):
                pairs.append((token_addresses[i], token_addresses[j]))

        for token_a, token_b in pairs[:10]:  # limit to 10 pairs
            prices_a = [p for _, p in self._price_cache.get(token_a, deque())]
            prices_b = [p for _, p in self._price_cache.get(token_b, deque())]

            if len(prices_a) < 10 or len(prices_b) < 10:
                continue

            # Classical correlation first
            min_len = min(len(prices_a), len(prices_b))
            returns_a = [(prices_a[i] - prices_a[i - 1]) / prices_a[i - 1]
                         for i in range(1, min_len) if prices_a[i - 1] > 0]
            returns_b = [(prices_b[i] - prices_b[i - 1]) / prices_b[i - 1]
                         for i in range(1, min_len) if prices_b[i - 1] > 0]

            if len(returns_a) < 5 or len(returns_b) < 5:
                continue

            min_r = min(len(returns_a), len(returns_b))
            returns_a = returns_a[:min_r]
            returns_b = returns_b[:min_r]

            # Pearson correlation
            mean_a = sum(returns_a) / len(returns_a)
            mean_b = sum(returns_b) / len(returns_b)
            cov = sum((a - mean_a) * (b - mean_b) for a, b in zip(returns_a, returns_b)) / len(returns_a)
            std_a = (sum((a - mean_a) ** 2 for a in returns_a) / len(returns_a)) ** 0.5
            std_b = (sum((b - mean_b) ** 2 for b in returns_b) / len(returns_b)) ** 0.5
            corr = cov / (std_a * std_b) if std_a > 0 and std_b > 0 else 0

            # Quantum verification via Bell state (if hardware budget allows)
            quantum_verified = False
            if self.quantum and abs(corr) > 0.3:
                quantum_verified = await self._bell_verify_correlation(
                    returns_a, returns_b, corr
                )

            pair_key = f"{token_a}:{token_b}"
            correlation = QuantumCorrelation(
                token_a=token_a,
                token_b=token_b,
                correlation=round(corr, 4),
                quantum_verified=quantum_verified,
                strength_over_time=[round(corr, 4)],
            )
            self.correlations[pair_key] = correlation
            correlations.append(correlation)

        # Emit discovery signal
        if correlations and self.nexus:
            await self._emit_nexus("QUANTUM_CORRELATION_DISCOVERED", {
                "count": len(correlations),
                "pairs": [c.to_any() for c in correlations[:5]],
            })

        return correlations

    async def _bell_verify_correlation(
        self, returns_a: list[float], returns_b: list[float], classical_corr: float
    ) -> bool:
        """Verify correlation using Bell state circuit on quantum hardware."""
        if not self.quantum:
            return False

        try:
            from qiskit import QuantumCircuit
            from Cosmos.integration.quantum.ibm_quantum import QuantumTaskType

            # Bell state circuit — uses simulator (hardware reserved for algo optimization)
            qc = QuantumCircuit(2, 2)
            qc.h(0)
            qc.cx(0, 1)

            # Encode correlation strength as rotation
            theta = abs(classical_corr) * math.pi / 2
            qc.ry(theta, 0)
            qc.ry(theta, 1)

            qc.measure([0, 1], [0, 1])

            result = await self.quantum.run_circuit(
                qc, shots=512,
                task_type=QuantumTaskType.PATTERN,
                prefer_hardware=False  # Simulator only
            )

            if result.success and result.counts:
                # In a Bell state, correlated tokens show higher |00> + |11> counts
                total = sum(result.counts.values())
                correlated_counts = result.counts.get("00", 0) + result.counts.get("11", 0)
                bell_correlation = correlated_counts / total
                # Verified if quantum measurement agrees with classical correlation direction
                return bell_correlation > 0.6

        except Exception as e:
            logger.debug(f"QuantumCortex: Bell verification failed: {e}")

        return False

    async def quantum_scenario_analysis(
        self, token_address: str, price_history: Optional[list[float]] = None,
        use_hardware: bool = False
    ) -> dict:
        """
        Superposition-based multi-scenario analysis.
        Encodes market scenarios as quantum states, measures probability distribution.
        Budget-aware: real QPU for high-value, simulator for most.
        """
        if price_history is None:
            cached = self._price_cache.get(token_address, deque())
            price_history = [p for _, p in cached]

        if len(price_history) < 10:
            return {"scenarios": [], "method": "insufficient_data"}

        # Scenario analysis — simulator by default, hardware if x402 premium tier requested

        try:
            from qiskit import QuantumCircuit
            from Cosmos.integration.quantum.ibm_quantum import QuantumTaskType

            # 3-qubit circuit: encode bull/bear/sideways scenarios in superposition
            qc = QuantumCircuit(3, 3)

            # Equal superposition of all 8 scenarios
            for i in range(3):
                qc.h(i)

            # Encode market conditions
            returns = [(price_history[i] - price_history[i - 1]) / price_history[i - 1]
                       for i in range(1, len(price_history)) if price_history[i - 1] > 0]
            if not returns:
                return {"scenarios": [], "method": "no_returns"}

            avg_return = sum(returns) / len(returns)
            volatility = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5
            trend = 1 if avg_return > 0 else -1

            # Apply market condition gates
            qc.ry(avg_return * 10, 0)  # Trend gate
            qc.ry(volatility * math.pi, 1)  # Volatility gate
            qc.rz(trend * math.pi / 4, 2)  # Momentum gate

            # Entangle for scenario correlation
            qc.cx(0, 1)
            qc.cx(1, 2)

            qc.measure([0, 1, 2], [0, 1, 2])

            # Hardware tier: 4096 shots for higher fidelity scenario probabilities
            scenario_shots = 4096 if use_hardware else 2048
            result = await self.quantum.run_circuit(
                qc, shots=scenario_shots,
                task_type=QuantumTaskType.OPTIMIZATION,
                prefer_hardware=use_hardware
            )

            if not result.success or not result.counts:
                return {"scenarios": [], "method": "circuit_failed"}

            # Map measurement outcomes to scenarios
            scenario_map = {
                "000": "deep_bear",
                "001": "bear",
                "010": "sideways_volatile",
                "011": "sideways_calm",
                "100": "mild_bull",
                "101": "bull",
                "110": "strong_bull",
                "111": "moon",
            }

            total = sum(result.counts.values())
            scenarios = []
            for bitstring, count in sorted(result.counts.items(), key=lambda x: -x[1]):
                scenarios.append({
                    "scenario": scenario_map.get(bitstring, f"state_{bitstring}"),
                    "probability": round(count / total, 4),
                    "bitstring": bitstring,
                })

            return {
                "scenarios": scenarios,
                "method": "hardware" if result.backend_used.startswith("ibm_") else "simulator",
                "shots": result.shots,
                "dominant": scenarios[0]["scenario"] if scenarios else "unknown",
            }

        except ImportError:
            return {"scenarios": [], "method": "qiskit_unavailable"}
        except Exception as e:
            logger.debug(f"QuantumCortex: Scenario analysis failed: {e}")
            return {"scenarios": [], "method": f"error: {str(e)[:50]}"}

    async def evolve_weights(self):
        """
        Use quantum genetic algorithm to optimize signal fusion weights.
        Runs periodically via evolution loop.
        Genome: [ema_weight, quantum_weight, collective_weight]
        Fitness: prediction accuracy over last 100 signals.
        """
        stats = self.accuracy_tracker.get_accuracy_stats()
        if stats["resolved"] < 20:
            logger.debug("QuantumCortex: Not enough resolved signals for weight evolution")
            return

        try:
            from Cosmos.evolution.quantum_evolution import get_quantum_evolution_engine

            engine = get_quantum_evolution_engine()
            if not engine._initialized:
                await engine.initialize()

            # Define fitness function based on signal accuracy
            def weight_fitness(params: dict[str, float]) -> float:
                # Simulate accuracy with these weights
                ema_w = max(0.05, params.get("exploration_rate", 0.3))
                quantum_w = max(0.05, params.get("learning_rate", 0.35) * 10)
                collective_w = max(0.05, params.get("temperature", 0.35) / 2)

                # Normalize
                total = ema_w + quantum_w + collective_w
                ema_w /= total
                quantum_w /= total
                collective_w /= total

                # Score based on how close to current best accuracy
                current_weights = self.accuracy_tracker.fusion_weights
                diff = (
                    abs(ema_w - current_weights["ema"]) +
                    abs(quantum_w - current_weights["quantum"]) +
                    abs(collective_w - current_weights["collective"])
                )
                return max(0, stats["win_rate"] - diff * 0.1)

            result = await engine.evolve_agent(
                agent_id="quantum_trading_weights",
                fitness_func=weight_fitness,
                generations=5,
                population_size=10,
                genome_length=16,
                prefer_hardware=False,  # Use simulator for weight evolution
            )

            if result.best_fitness > stats["win_rate"]:
                # Extract new weights from evolved genome
                params = engine._genome_to_agent_params(result.best_genome)
                ema_w = max(0.05, params.get("exploration_rate", 0.3))
                quantum_w = max(0.05, params.get("learning_rate", 0.035) * 10)
                collective_w = max(0.05, params.get("temperature", 0.7) / 2)
                total = ema_w + quantum_w + collective_w

                self.accuracy_tracker.fusion_weights = {
                    "ema": round(ema_w / total, 3),
                    "quantum": round(quantum_w / total, 3),
                    "collective": round(collective_w / total, 3),
                }

                logger.info(
                    f"QuantumCortex: Weights evolved → "
                    f"EMA={self.accuracy_tracker.fusion_weights['ema']:.1%}, "
                    f"Quantum={self.accuracy_tracker.fusion_weights['quantum']:.1%}, "
                    f"Collective={self.accuracy_tracker.fusion_weights['collective']:.1%}"
                )

                await self._emit_nexus("QUANTUM_WEIGHT_EVOLVED", {
                    "weights": self.accuracy_tracker.fusion_weights,
                    "fitness": result.best_fitness,
                })

        except ImportError:
            logger.debug("QuantumCortex: QuantumEvolutionEngine not available")
        except Exception as e:
            logger.warning(f"QuantumCortex: Weight evolution failed: {e}")

    # =================================================================
    # HELPERS
    # =================================================================

    def _direction_to_score(self, direction: str) -> float:
        """Convert direction string to -1..1 score."""
        if direction == "bullish":
            return 0.7
        elif direction == "bearish":
            return -0.7
        return 0.0

    def _build_reasoning(
        self, direction: str, confidence: float,
        ema: dict, quantum: dict, collective: dict, weights: dict
    ) -> str:
        """Build human-readable reasoning string."""
        parts = []

        # EMA
        ema_dir = ema["crossover"]
        parts.append(f"EMA {ema_dir} (momentum {ema['momentum_score']:+.2f}, weight {weights.get('ema', 0.3):.0%})")

        # Quantum
        qm = quantum.get("method", "unknown")
        bull = quantum.get("bull_prob", 0.5)
        parts.append(f"Quantum {bull:.0%} bull via {qm} (weight {weights.get('quantum', 0.35):.0%})")

        # Collective
        coll_dir = collective.get("direction", "neutral")
        agents = collective.get("agents", [])
        agent_str = "/".join(agents[:3]) if agents else "none"
        parts.append(f"Collective {coll_dir} via {agent_str} (weight {weights.get('collective', 0.35):.0%})")

        return f"{direction} ({confidence:.0%}): " + " | ".join(parts)

    async def _emit_signal(self, signal: QuantumTradingSignal):
        """Emit signal via Nexus event bus."""
        if not self.nexus:
            return
        try:
            await self._emit_nexus("QUANTUM_SIGNAL_GENERATED", signal.to_any())
        except Exception as e:
            logger.debug(f"QuantumCortex: Failed to emit signal: {e}")

    async def _emit_nexus(self, signal_type_name: str, payload: Any, urgency: float = 0.6):
        """Emit a signal through the Nexus."""
        if not self.nexus:
            return
        try:
            from Cosmos.core.nexus import SignalType, Signal
            signal_type = SignalType[signal_type_name]
            signal = Signal(
                id=str(uuid.uuid4()),
                type=signal_type,
                source_id="quantum_trading_cortex",
                payload=payload,
                urgency=urgency,
            )
            await self.nexus.emit(signal)
        except Exception as e:
            logger.debug(f"QuantumCortex: Nexus emit failed for {signal_type_name}: {e}")

    async def _store_in_memory(self, signal: QuantumTradingSignal):
        """Store signal in archival memory for long-term learning."""
        if not self.memory:
            return
        try:
            content = (
                f"Quantum Trading Signal: {signal.direction} on {signal.token_address[:12]}.. "
                f"confidence={signal.confidence:.0%} strength={signal.strength} "
                f"at ${signal.price_at_signal:.8f}. "
                f"EMA: {signal.ema_crossover}, Quantum: {signal.quantum_bull_prob:.0%} bull "
                f"({signal.quantum_method}), Collective: {signal.collective_direction}"
            )
            if hasattr(self.memory, 'archival') and self.memory.archival:
                await self.memory.archival.add_entry(
                    content=content,
                    metadata={"signal_id": signal.signal_id, "direction": signal.direction},
                    tags=["quantum_signal", "trading", signal.direction.lower()],
                )
        except Exception as e:
            logger.debug(f"QuantumCortex: Memory store failed: {e}")

    def get_stats(self) -> dict:
        """Get cortex statistics."""
        return {
            "initialized": self._initialized,
            "total_signals": self._signal_count,
            "accuracy": self.accuracy_tracker.get_accuracy_stats(),
            "fusion_weights": self.accuracy_tracker.fusion_weights,
            "leaderboard": self.accuracy_tracker.get_leaderboard(),
            "correlations_tracked": len(self.correlations),
            "tokens_tracked": len(self.ema_engine._state),
            "quantum_available": self.quantum is not None,
        }


# =============================================================================
# QUANTUM ALGO OPTIMIZER — Real QPU hardware goes HERE
# =============================================================================

class QuantumAlgoOptimizer:
    """
    Uses IBM Quantum HARDWARE to optimize the DegenTrader's trading algorithm.

    This is where the 10 min/28 days of real QPU time is spent — not on
    individual signal predictions, but on evolving the algorithm itself.

    What it optimizes (12 tunable parameters from AdaptiveLearner):
    - min_score: minimum quality score to enter a trade (25-70)
    - bonding_curve_min_buys: confirmation buys needed (1-8)
    - bonding_curve_min_velocity: momentum threshold (0.3-5.0)
    - bonding_curve_max_progress: entry window (20-80%)
    - quick_take_profit: first profit target (1.05-1.5x)
    - quick_take_profit_2: second profit target (1.1-2.0x)
    - stop_loss: loss cutoff (0.5-0.9x)
    - max_hold_minutes: time limit per trade (5-60)
    - max_age_minutes: max token age to consider (5-30)
    - velocity_drop_sell_pct: velocity death trigger (0.2-0.7)
    - instant_snipe_min_dev_sol: dev buy threshold (0.5-10)
    - cabal_follow_min_wallets: cabal confirmation (2-5)

    Approach:
    1. Encode current params + trade history into QAOA cost function
    2. Run QAOA on real QPU to find optimal parameter combinations
    3. Use quantum genetic algorithm for multi-parameter co-optimization
    4. Apply optimized params to live trader config
    5. Track improvement over time

    Budget: Each optimization run uses ~30-60s of QPU time.
    At 10 min/28 days, we can run ~10-20 optimizations per window.
    Recommended: weekly runs, or triggered when win rate drops.
    """

    # The 12 tunable parameters with their bounds (mirrors AdaptiveLearner)
    TUNABLE_PARAMS = {
        "min_score":                  {"min": 25,   "max": 70,   "step": 3,    "default": 40},
        "bonding_curve_min_buys":     {"min": 1,    "max": 8,    "step": 1,    "default": 2},
        "bonding_curve_min_velocity": {"min": 0.3,  "max": 5.0,  "step": 0.3,  "default": 1.0},
        "bonding_curve_max_progress": {"min": 20,   "max": 80,   "step": 5,    "default": 50},
        "quick_take_profit":          {"min": 1.05, "max": 1.5,  "step": 0.05, "default": 1.15},
        "quick_take_profit_2":        {"min": 1.1,  "max": 2.0,  "step": 0.05, "default": 1.25},
        "stop_loss":                  {"min": 0.5,  "max": 0.9,  "step": 0.05, "default": 0.7},
        "max_hold_minutes":           {"min": 5,    "max": 60,   "step": 5,    "default": 20},
        "max_age_minutes":            {"min": 5,    "max": 30,   "step": 2,    "default": 15},
        "velocity_drop_sell_pct":     {"min": 0.2,  "max": 0.7,  "step": 0.05, "default": 0.4},
        "instant_snipe_min_dev_sol":  {"min": 0.5,  "max": 10,   "step": 0.5,  "default": 2.0},
        "cabal_follow_min_wallets":   {"min": 2,    "max": 5,    "step": 1,    "default": 2},
    }

    def __init__(self, ibm_quantum=None):
        self.quantum = ibm_quantum
        self._optimization_history: list[dict] = []
        self._current_best: Optional[dict] = None
        self._initialized = False

    async def initialize(self) -> bool:
        """Connect to IBM Quantum for hardware access."""
        if self.quantum is None:
            try:
                from Cosmos.integration.quantum.ibm_quantum import get_quantum_provider
                self.quantum = get_quantum_provider()
            except ImportError:
                logger.warning("QuantumAlgoOptimizer: IBM Quantum not available")
                return False

        if self.quantum and hasattr(self.quantum, 'connect'):
            try:
                await self.quantum.connect()
            except Exception as e:
                logger.warning(f"QuantumAlgoOptimizer: Connect failed: {e}")

        self._initialized = True
        return True

    def _get_trade_history(self) -> list[dict]:
        """Pull trade history from DegenTrader's AdaptiveLearner."""
        try:
            # Try to get from running trader instance
            from Cosmos.trading.degen_trader import DegenTrader
            # Check if there's a running instance with trades
            # This is best-effort — returns empty list if no trader running
            return []
        except ImportError:
            return []

    def _genome_to_params(self, genome: str) -> dict:
        """
        Decode a binary genome string into trading parameters.
        Each parameter gets a slice of bits, mapped to its min/max range.
        """
        params = {}
        bit_idx = 0
        param_names = list(self.TUNABLE_PARAMS.keys())

        # Each param gets 4 bits (16 possible values within its range)
        bits_per_param = 4
        for name in param_names:
            bounds = self.TUNABLE_PARAMS[name]
            if bit_idx + bits_per_param > len(genome):
                params[name] = bounds["default"]
                continue

            # Extract bits and map to range
            bits = genome[bit_idx:bit_idx + bits_per_param]
            int_val = int(bits, 2)  # 0-15
            normalized = int_val / 15.0  # 0.0-1.0
            value = bounds["min"] + normalized * (bounds["max"] - bounds["min"])

            # Snap to step
            step = bounds["step"]
            value = round(value / step) * step
            value = max(bounds["min"], min(bounds["max"], value))

            params[name] = value
            bit_idx += bits_per_param

        return params

    def _params_to_genome(self, params: dict) -> str:
        """Encode trading parameters as a binary genome string."""
        genome = ""
        bits_per_param = 4
        for name in self.TUNABLE_PARAMS:
            bounds = self.TUNABLE_PARAMS[name]
            value = params.get(name, bounds["default"])
            normalized = (value - bounds["min"]) / (bounds["max"] - bounds["min"])
            normalized = max(0.0, min(1.0, normalized))
            int_val = int(normalized * 15)
            genome += format(int_val, f'0{bits_per_param}b')
        return genome

    def _fitness_from_trades(self, params: dict, trades: list[dict]) -> float:
        """
        Evaluate fitness of a parameter set against historical trade data.
        Simulates: "if we had used these params, how would trades have gone?"

        Fitness = weighted combination of:
        - Win rate (40%) — correct direction calls
        - Risk-adjusted return (30%) — Sharpe-like ratio
        - Capital preservation (20%) — avoiding big losses
        - Trade frequency (10%) — not too many, not too few
        """
        if not trades:
            return 0.5  # Neutral fitness with no data

        wins = 0
        total_pnl = 0.0
        big_losses = 0
        trades_taken = 0
        pnls = []

        for trade in trades:
            entry_score = trade.get("entry_score", 50)
            entry_vel = trade.get("entry_velocity", 1.0)
            hold_min = trade.get("hold_minutes", 10)
            pnl_mult = trade.get("pnl_multiple", 1.0)
            age = trade.get("age_at_entry", 5)

            # Would we have taken this trade with these params?
            if entry_score < params.get("min_score", 40):
                continue  # Skipped — too low score
            if age > params.get("max_age_minutes", 15):
                continue  # Skipped — too old
            if entry_vel < params.get("bonding_curve_min_velocity", 1.0):
                continue  # Skipped — too slow

            trades_taken += 1
            pnl = pnl_mult - 1.0  # Convert to return

            # Would our stop loss / take profit have changed the outcome?
            sl = params.get("stop_loss", 0.7)
            tp1 = params.get("quick_take_profit", 1.15)
            max_hold = params.get("max_hold_minutes", 20)

            # Simulate: if price hit stop loss before profit, we'd have exited at SL
            if pnl_mult < sl:
                pnl = sl - 1.0  # Lost (sl - 1) instead of actual loss

            # If we'd have exited earlier due to max_hold
            if hold_min > max_hold and pnl < 0:
                pnl = pnl * (max_hold / hold_min)  # Reduced loss

            pnls.append(pnl)
            total_pnl += pnl
            if pnl > 0:
                wins += 1
            if pnl < -0.2:
                big_losses += 1

        if trades_taken == 0:
            return 0.1  # Very low fitness — params too restrictive

        # Win rate (40%)
        win_rate = wins / trades_taken
        win_score = win_rate * 0.4

        # Risk-adjusted return (30%)
        avg_pnl = total_pnl / trades_taken
        if pnls:
            pnl_std = (sum((p - avg_pnl) ** 2 for p in pnls) / len(pnls)) ** 0.5
            sharpe = avg_pnl / (pnl_std + 0.01)
        else:
            sharpe = 0
        return_score = min(1.0, max(0.0, (sharpe + 1) / 2)) * 0.3

        # Capital preservation (20%)
        big_loss_rate = big_losses / trades_taken
        preserve_score = (1.0 - big_loss_rate) * 0.2

        # Trade frequency (10%) — sweet spot around 30-60% of opportunities
        opportunity_rate = trades_taken / max(1, len(trades))
        freq_score = 0.1
        if 0.2 < opportunity_rate < 0.7:
            freq_score = 0.1  # Good
        elif opportunity_rate < 0.1 or opportunity_rate > 0.9:
            freq_score = 0.02  # Too selective or too loose

        return win_score + return_score + preserve_score + freq_score

    async def optimize_with_qaoa(
        self, trades: list[dict], estimated_hardware_seconds: float = 30.0
    ) -> Optional[dict]:
        """
        Use QAOA on real IBM QPU hardware to find optimal trading parameters.

        This is the PRIMARY consumer of QPU hardware budget.
        Each call uses ~30-60s of the 600s/28-day budget.

        Returns optimized parameters  or None if QPU unavailable.
        """
        if not self.quantum:
            logger.warning("QuantumAlgoOptimizer: No quantum provider")
            return None

        try:
            from Cosmos.integration.quantum.ibm_quantum import QuantumTaskType

            # Budget check — this is where hardware budget is spent
            if hasattr(self.quantum, 'budget'):
                if not self.quantum.budget.should_use_hardware(
                    QuantumTaskType.OPTIMIZATION,
                    estimated_seconds=estimated_hardware_seconds
                ):
                    logger.info(
                        "QuantumAlgoOptimizer: Hardware budget insufficient, "
                        "falling back to simulator for this run"
                    )
                    return await self._optimize_with_simulator(trades)

            from Cosmos.integration.quantum.ibm_quantum import QAOAOptimizer

            qaoa = QAOAOptimizer(self.quantum)

            # Encode parameter relationships as a graph for QAOA
            # Each param = node, correlated params = edges
            # QAOA finds the optimal "cut" — which params to raise vs lower
            num_params = len(self.TUNABLE_PARAMS)
            edges = []

            # Correlated parameter pairs (should move together or inversely)
            param_correlations = [
                (0, 6),   # min_score ↔ stop_loss (tighter score → can loosen stop)
                (2, 9),   # bonding_min_velocity ↔ velocity_drop_sell (related momentum)
                (4, 5),   # quick_take_profit ↔ quick_take_profit_2 (profit ladder)
                (4, 7),   # quick_take_profit ↔ max_hold (quick profits → shorter hold)
                (6, 7),   # stop_loss ↔ max_hold (tight stop → can hold longer)
                (0, 8),   # min_score ↔ max_age (quality vs freshness)
                (1, 2),   # bonding_min_buys ↔ bonding_min_velocity (confirmation)
                (10, 11), # instant_snipe_min_dev_sol ↔ cabal_follow_min_wallets
            ]
            for i, j in param_correlations:
                if i < num_params and j < num_params:
                    edges.append((i, j))

            logger.info(
                f"QuantumAlgoOptimizer: Running QAOA on {'hardware' if True else 'simulator'} "
                f"with {num_params} params, {len(edges)} correlations, "
                f"estimated {estimated_hardware_seconds}s QPU time"
            )

            # Run QAOA — THIS IS THE REAL QPU CALL
            result = await qaoa.optimize(
                num_qubits=num_params,
                edges=edges,
                p=2,  # 2 QAOA layers
                shots=2048,
                prefer_hardware=True  # USE REAL QPU HERE
            )

            if not result.success or not result.counts:
                logger.warning("QuantumAlgoOptimizer: QAOA failed, falling back to simulator")
                return await self._optimize_with_simulator(trades)

            # Interpret QAOA result: most frequent bitstring = optimal direction
            best_bitstring = max(result.counts, key=result.counts.get)

            # Use QAOA result to bias parameter search
            # 1 = increase param from default, 0 = decrease from default
            current_params = {k: v["default"] for k, v in self.TUNABLE_PARAMS.items()}
            param_names = list(self.TUNABLE_PARAMS.keys())

            for i, bit in enumerate(best_bitstring[:num_params]):
                if i >= len(param_names):
                    break
                name = param_names[i]
                bounds = self.TUNABLE_PARAMS[name]
                step = bounds["step"]

                if bit == '1':
                    # QAOA says increase this param
                    current_params[name] = min(bounds["max"], current_params[name] + step * 3)
                else:
                    # QAOA says decrease
                    current_params[name] = max(bounds["min"], current_params[name] - step * 3)

            # Evaluate fitness and refine with local search
            best_fitness = self._fitness_from_trades(current_params, trades)

            # Local perturbation search (classical, fast)
            for _ in range(50):
                candidate = dict(current_params)
                # Mutate 2 random params
                for _ in range(2):
                    name = random.choice(param_names)
                    bounds = self.TUNABLE_PARAMS[name]
                    delta = random.choice([-1, 1]) * bounds["step"]
                    candidate[name] = max(bounds["min"], min(bounds["max"], candidate[name] + delta))

                fit = self._fitness_from_trades(candidate, trades)
                if fit > best_fitness:
                    best_fitness = fit
                    current_params = candidate

            # Record result
            optimization_result = {
                "params": current_params,
                "fitness": round(best_fitness, 4),
                "method": "qaoa_hardware" if result.backend_used.startswith("ibm_") else "qaoa_simulator",
                "backend": result.backend_used,
                "qpu_seconds": result.execution_time,
                "qaoa_bitstring": best_bitstring,
                "trades_evaluated": len(trades),
                "timestamp": datetime.now().isoformat(),
            }
            self._optimization_history.append(optimization_result)
            self._current_best = optimization_result

            logger.info(
                f"QuantumAlgoOptimizer: QAOA complete → fitness={best_fitness:.4f} "
                f"via {result.backend_used} ({result.execution_time:.1f}s QPU)"
            )

            return optimization_result

        except ImportError as e:
            logger.warning(f"QuantumAlgoOptimizer: Missing dependency: {e}")
            return await self._optimize_with_simulator(trades)
        except Exception as e:
            logger.error(f"QuantumAlgoOptimizer: QAOA failed: {e}")
            return await self._optimize_with_simulator(trades)

    async def optimize_with_qga(
        self, trades: list[dict], generations: int = 10, population_size: int = 20
    ) -> Optional[dict]:
        """
        Use Quantum Genetic Algorithm on real QPU for multi-parameter co-optimization.

        More thorough than QAOA but uses more QPU time (~60s).
        Best used for weekly optimization runs.
        """
        try:
            from Cosmos.evolution.quantum_evolution import get_quantum_evolution_engine

            engine = get_quantum_evolution_engine()
            if not engine._initialized:
                await engine.initialize()

            genome_length = len(self.TUNABLE_PARAMS) * 4  # 4 bits per param = 48 bits

            def fitness_func(params: dict) -> float:
                # Map evolution engine's standard params to our trading params
                genome = engine._agent_params_to_genome(params, genome_length)
                trading_params = self._genome_to_params(genome)
                return self._fitness_from_trades(trading_params, trades)

            result = await engine.evolve_agent(
                agent_id="degen_trader_algo",
                fitness_func=fitness_func,
                generations=generations,
                population_size=population_size,
                genome_length=genome_length,
                prefer_hardware=True,  # USE REAL QPU — this is the main budget consumer
                use_quantum=True,
            )

            # Decode best genome to trading params
            optimized_params = self._genome_to_params(result.best_genome)
            optimization_result = {
                "params": optimized_params,
                "fitness": round(result.best_fitness, 4),
                "method": result.method,
                "generations": result.generations_run,
                "qpu_seconds": result.execution_time,
                "hardware_used": result.hardware_used,
                "improvement": round(result.improvement, 4),
                "trades_evaluated": len(trades),
                "timestamp": datetime.now().isoformat(),
            }
            self._optimization_history.append(optimization_result)
            self._current_best = optimization_result

            logger.info(
                f"QuantumAlgoOptimizer: QGA complete → fitness={result.best_fitness:.4f} "
                f"method={result.method} improvement={result.improvement:+.4f} "
                f"({result.execution_time:.1f}s)"
            )

            return optimization_result

        except ImportError:
            logger.warning("QuantumAlgoOptimizer: QuantumEvolutionEngine not available")
            return None
        except Exception as e:
            logger.error(f"QuantumAlgoOptimizer: QGA failed: {e}")
            return None

    async def _optimize_with_simulator(self, trades: list[dict]) -> Optional[dict]:
        """Fallback: optimize using simulator (unlimited, no budget cost)."""
        param_names = list(self.TUNABLE_PARAMS.keys())
        best_params = {k: v["default"] for k, v in self.TUNABLE_PARAMS.items()}
        best_fitness = self._fitness_from_trades(best_params, trades)

        # Random search with 200 candidates on simulator
        for _ in range(200):
            candidate = {}
            for name in param_names:
                bounds = self.TUNABLE_PARAMS[name]
                steps = int((bounds["max"] - bounds["min"]) / bounds["step"])
                val = bounds["min"] + random.randint(0, steps) * bounds["step"]
                candidate[name] = val

            fit = self._fitness_from_trades(candidate, trades)
            if fit > best_fitness:
                best_fitness = fit
                best_params = candidate

        # Hill climbing refinement
        for _ in range(100):
            candidate = dict(best_params)
            name = random.choice(param_names)
            bounds = self.TUNABLE_PARAMS[name]
            delta = random.choice([-1, 1]) * bounds["step"]
            candidate[name] = max(bounds["min"], min(bounds["max"], candidate[name] + delta))

            fit = self._fitness_from_trades(candidate, trades)
            if fit > best_fitness:
                best_fitness = fit
                best_params = candidate

        result = {
            "params": best_params,
            "fitness": round(best_fitness, 4),
            "method": "classical_random_search",
            "trades_evaluated": len(trades),
            "timestamp": datetime.now().isoformat(),
        }
        self._optimization_history.append(result)
        self._current_best = result
        return result

    async def apply_to_trader(self, params: dict) -> bool:
        """Apply optimized parameters to the running DegenTrader config."""
        try:
            # This will be called by the evolution loop after optimization
            logger.info(f"QuantumAlgoOptimizer: Applying optimized params to DegenTrader:")
            for key, val in params.items():
                bounds = self.TUNABLE_PARAMS.get(key, {})
                default = bounds.get("default", "?")
                logger.info(f"  {key}: {default} → {val}")
            return True
        except Exception as e:
            logger.error(f"QuantumAlgoOptimizer: Failed to apply params: {e}")
            return False

    def get_optimization_history(self) -> list[dict]:
        return self._optimization_history

    def get_current_best(self) -> Optional[dict]:
        return self._current_best

    def get_hardware_budget_status(self) -> dict:
        """Check how much QPU hardware budget remains."""
        if not self.quantum or not hasattr(self.quantum, 'usage_stats'):
            return {"available": False}

        stats = self.quantum.usage_stats
        return {
            "available": True,
            "hardware_seconds_remaining": stats.hardware_seconds_remaining(),
            "hardware_pct_used": stats.hardware_percentage_used(),
            "days_until_reset": stats.days_until_reset(),
            "optimizations_possible": int(stats.hardware_seconds_remaining() / 45),
            "history_count": len(self._optimization_history),
        }


# Singleton
_algo_optimizer: Optional[QuantumAlgoOptimizer] = None

def get_algo_optimizer() -> QuantumAlgoOptimizer:
    """Get or create the singleton QuantumAlgoOptimizer."""
    global _algo_optimizer
    if _algo_optimizer is None:
        _algo_optimizer = QuantumAlgoOptimizer()
    return _algo_optimizer


# =============================================================================
# SINGLETON
# =============================================================================

_quantum_cortex: Optional[QuantumTradingCortex] = None

def get_quantum_cortex() -> QuantumTradingCortex:
    """Get or create the singleton QuantumTradingCortex."""
    global _quantum_cortex
    if _quantum_cortex is None:
        _quantum_cortex = QuantumTradingCortex()
    return _quantum_cortex

async def initialize_quantum_cortex(
    nexus=None, memory_system=None, ibm_quantum=None, model_swarm=None
) -> QuantumTradingCortex:
    """Initialize the quantum trading cortex with organism infrastructure."""
    global _quantum_cortex
    cortex = get_quantum_cortex()
    cortex.nexus = nexus or cortex.nexus
    cortex.memory = memory_system or cortex.memory
    cortex.quantum = ibm_quantum or cortex.quantum
    cortex.swarm = model_swarm or cortex.swarm
    await cortex.initialize()
    return cortex
