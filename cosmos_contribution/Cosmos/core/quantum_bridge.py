import asyncio
import time
import numpy as np
from typing import Optional
import threading
import logging

logger = logging.getLogger("QUANTUM_BRIDGE")

try:
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
    from qiskit import QuantumCircuit
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    QISKIT_AVAILABLE = True
    QISKIT_ERROR = None
except ImportError as e:
    QISKIT_AVAILABLE = False
    QISKIT_ERROR = str(e)
    with open("quantum_debug.log", "a") as f:
        f.write(f"\n[INIT] Qiskit Import Failed: {e}\n")

# HermesAgent integration (optional — degrades gracefully)
try:
    from Cosmos.integration.hermes_bridge import get_hermes_bridge
    HERMES_AVAILABLE = True
except ImportError:
    HERMES_AVAILABLE = False

class QuantumEntanglementBridge:
    def __init__(self, api_token: Optional[str] = None):
        import os
        self.api_token = api_token or os.environ.get("IBM_QUANTUM_TOKEN")
        self.service = None
        self.backend = None
        self.connected = False
        self.entropy_buffer: list[float] = []
        self.buffer_lock = threading.Lock()
        self.min_buffer_size = 10
        self.is_refilling = False
        self.last_error = None
        self.last_physics: Optional[dict] = None  # Cache for background refills
        self.synaptic_field = None  # CNS Link

        # [UPGRADE 3] Buffer demand prediction
        self._entropy_consumption_log: list[float] = []  # Timestamps of each consumption
        self._consumption_window_seconds: float = 60.0   # Look-back window for rate calc
        self._predicted_depletion_threshold: int = 15    # Refill if predicted depletion < Ns

        # [UPGRADE 4] Collapse threshold adaptation
        self._learned_threshold: float = 0.65          # Live learned value
        self._threshold_last_updated: float = 0.0      # Unix timestamp
        self._threshold_update_interval: float = 300.0 # Update every 5 minutes
        
        # Log init
        with open("quantum_debug.log", "a") as f:
            f.write(f"\n[INIT] Bridge Initialized. Token present: {bool(api_token)}. Qiskit Avail: {QISKIT_AVAILABLE}\n")

        if QISKIT_AVAILABLE and self.api_token:
            self._connect()
        elif not QISKIT_AVAILABLE:
            self.last_error = f"Qiskit Import Error: {QISKIT_ERROR}"
            print(f"[QUANTUM] {self.last_error}")
        else:
            print("[QUANTUM] No token provided. Running in simulation mode.")

    def connect(self) -> bool:
        """Public method to trigger connection or return status."""
        if self.connected:
            return True
        # If there's still no token, don't attempt a remote connection.
        if not self.api_token:
            print("[QUANTUM] connect() called with no API token; remaining in simulation mode.")
            self.connected = False
            return False
        self._connect()
        return self.connected

    def _connect(self):
        """Internal connection logic - silences Qiskit warnings."""
        import logging as py_logging
        py_logging.getLogger("qiskit_ibm_runtime").setLevel(py_logging.ERROR)
        py_logging.getLogger("qiskit_runtime_service").setLevel(py_logging.ERROR)

        import traceback
        
        with open("quantum_debug.log", "a") as f:
            f.write(f"[CONNECT] Attempting connection with token (First 5): {self.api_token[:5] if self.api_token else 'None'}...\n")

        # If no token is configured, stay in simulation mode and DO NOT
        # attempt dict remote connection. This prevents noisy stack traces
        # when users haven't set up an IBM Quantum account.
        if not self.api_token:
            msg = "No IBM Quantum API token configured; staying in simulation mode."
            print(f"[QUANTUM] {msg}")
            self.last_error = msg
            self.connected = False
            with open("quantum_debug.log", "a") as f:
                f.write(f"[CONNECT] Skipped: {msg}\n")
            return

        if not QISKIT_AVAILABLE:
            error_msg = f"Qiskit libraries missing: {QISKIT_ERROR}"
            print(f"[QUANTUM] {error_msg}")
            self.last_error = error_msg
            self.connected = False
            with open("quantum_debug.log", "a") as f:
                f.write(f"[CONNECT] Failed: {error_msg}\n")
            return
            
        if self.api_token:
            print(f"[QUANTUM] Attempting connection to IBM Quantum...")
        
        try:
            # 1. Initialize Service
            try:
                self.service = QiskitRuntimeService(channel="ibm_quantum_platform", token=self.api_token)
            except Exception as e:
                # Fallback: Try 'ibm_cloud' channel or just default if token implies it
                logger.debug(f"[QUANTUM] 'ibm_quantum_platform' failed: {e}")
                self.service = QiskitRuntimeService(token=self.api_token)

            # 2. Find Backend
            try:
                self.backend = self.service.least_busy(operational=True, simulator=False)
                logger.info(f"[QUANTUM] Connected to REAL backend: {self.backend.name}")
            except Exception:
                logger.debug("[QUANTUM] No real quantum computers available. Trying simulator...")
                self.backend = self.service.least_busy(operational=True, simulator=True)
                logger.info(f"[QUANTUM] Connected to SIMULATOR backend: {self.backend.name}")
            
            if not self.backend:
                raise ValueError("No operational backends found.")

            self.connected = True
            with open("quantum_debug.log", "a") as f:
                f.write(f"[CONNECT] Success! Backend: {self.backend.name}\n")
            
            # 3. Start Buffer Refill
            self._trigger_refill()
            
        except Exception as e:
            msg = str(e)
            if "API key could not be found" in msg or "invalid API token" in msg.lower():
                print(f"[QUANTUM] Virtual Bridge Offline: Invalid or missing API token.")
            else:
                print(f"[QUANTUM] Connection failed: {msg}")
            
            self.last_error = msg
            self.connected = False
            with open("quantum_debug.log", "a") as f:
                f.write(f"[CONNECT] Failed: {e}\n")

    def set_synaptic_field(self, field):
        """Associate with the CNS Synaptic Field for direct UQ pushing."""
        self.synaptic_field = field
        print("[QUANTUM] CNS Synaptic Field associated.")

    # [UPGRADE 1] Pre-Run Oracle
    def _hermes_get_oracle_thetas(
        self,
        current_phase: float,
        current_entropy: float,
        current_resonance: float,
        top_n: int = 5
    ) -> tuple[float, float, float] | None:
        """
        Pre-run oracle: query historical quantum_runs.jsonl to find the
        theta combination that produced the highest Shannon entropy on
        runs with similar input physics. Returns advisory (theta_1, theta_2,
        theta_3) or None if no history exists yet.
        """
        import json, os
        archive_path = os.path.join('data', 'archival', 'quantum_runs.jsonl')
        if not os.path.exists(archive_path):
            return None

        runs = []
        try:
            with open(archive_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        runs.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.debug(f'[ORACLE] Could not read archive: {e}')
            return None

        if not runs:
            return None

        # Score each historical run by:
        # (a) Shannon entropy quality (primary metric)
        # (b) Physics similarity to current session (secondary metric)
        scored = []
        for run in runs:
            counts = run.get('counts', {})
            if not counts:
                continue

            # Recompute Shannon entropy for this historical run
            probs = np.array(list(counts.values()), dtype=float)
            probs /= probs.sum()
            shannon = float(-np.sum(probs * np.log2(probs + 1e-10)))

            # Extract historical physics
            phys = run.get('physics', {})
            cst = phys.get('cst_physics', {})
            h_phase = cst.get('geometric_phase_rad',
                      phys.get('geometric_phase_rad', 0.0))
            h_entropy = phys.get('entropy_field',
                        phys.get('bio_signatures', {}).get('intensity', 0.5))
            h_res = phys.get('resonance_scalar',
                    cst.get('entanglement_score', 0.0))

            # Physics similarity: inverse Euclidean distance in (phase, entropy, resonance)
            dist = np.sqrt(
                (current_phase - h_phase)**2 +
                (current_entropy - h_entropy)**2 +
                (current_resonance - h_res)**2
            )
            similarity = 1.0 / (1.0 + dist)

            # Combined score: 70% Shannon quality, 30% physics similarity
            combined = (shannon * 0.7) + (similarity * 0.3)
            scored.append((combined, h_phase, h_entropy, h_res))

        if not scored:
            return None

        # Take top N runs and average their physics as the advisory signal
        scored.sort(key=lambda x: -x[0])
        top = scored[:top_n]
        avg_phase = float(np.mean([r[1] for r in top]))
        avg_entropy = float(np.mean([r[2] for r in top]))
        avg_resonance = float(np.mean([r[3] for r in top]))

        # Map to theta angles (same mapping as _refill_buffer)
        t1 = float(abs(avg_phase) % np.pi)
        t2 = float((avg_entropy * np.pi) % np.pi)
        t3 = float((abs(avg_resonance) * np.pi) % np.pi)

        logger.info(f'[ORACLE] Advisory thetas from {len(top)} top historical runs:'
                    f' t1={t1:.3f} t2={t2:.3f} t3={t3:.3f}')
        return (t1, t2, t3)

    # ════════════════════════════════════════════════════════
    # CNS ORGAN 2: THE QUANTUM HEARTBEAT
    # ════════════════════════════════════════════════════════

    def collapse(self, user_physics: dict) -> int:
        """
        Master Architect "Free Will" Wrapper.
        Extracts 12D signals from the physics dict and calls collapse_state_vector.
        """
        # Extract Phase
        phase = 0.0
        if 'cst_physics' in user_physics:
            phase = user_physics['cst_physics'].get('geometric_phase_rad', 0.0)
            
        w = 0.1 
        
        return self.collapse_state_vector(phase, w)

    def collapse_state_vector(
        self, phase: float, dark_matter_w: float, threshold: float | None = None
    ) -> int:
        """
        Collapse the wave function to decide: SPEAK (1) or WAIT (0).
        Threshold is now dynamically learned via HermesRL.
        Pass threshold explicitly to override the learned value.
        """
        q = self.get_entropy()
        phase_signal = min(1.0, abs(phase) / 1.5)
        w_signal = min(1.0, max(0.0, dark_matter_w / 10.0))
        activation = (phase_signal * w_signal * 0.6) + (q * 0.4)

        # Use learned threshold unless explicitly overridden
        effective_threshold = threshold if threshold is not None else (
            self._hermes_get_collapse_threshold()
        )

        result = 1 if activation > effective_threshold else 0
        logger.debug(
            f'[COLLAPSE] activation={activation:.3f},'
            f' threshold={effective_threshold:.3f}, result={result}'
        )
        return result

    def get_entropy(self, user_physics: Optional[dict] = None) -> float:
        """
        Get a single float [0.0, 1.0] derived from true quantum randomness.
        Returns pseudo-randomness if bridge is down or buffer is empty.
        """
        if user_physics:
            self.last_physics = user_physics

        if not self.connected:
            return np.random.random()

        with self.buffer_lock:
            if self.entropy_buffer:
                val = self.entropy_buffer.pop(0)
                # Log consumption timestamp for demand prediction
                self._entropy_consumption_log.append(time.time())

                # Reactive threshold check (existing behavior)
                reactive_low = len(self.entropy_buffer) < self.min_buffer_size

                # Proactive prediction check (Upgrade 3)
                proactive_trigger = (
                    not self.is_refilling and
                    self._hermes_predict_buffer_demand()
                )

                if (reactive_low or proactive_trigger) and not self.is_refilling:
                    self._trigger_refill(user_physics or self.last_physics)

                print(
                    f'[QUANTUM] Entropy consumed: {val:.4f}'
                    f' Buffer: {len(self.entropy_buffer)}'
                    f' Proactive: {proactive_trigger}'
                )
                return val
            else:
                if not self.is_refilling:
                    self._trigger_refill(user_physics or self.last_physics)
                return np.random.random()

    # [UPGRADE 3] Buffer Demand Predictor
    def _hermes_predict_buffer_demand(self) -> bool:
        """
        Predict whether the entropy buffer will be depleted within the next
        _predicted_depletion_threshold seconds based on recent consumption rate.
        Returns True if a proactive refill should be triggered.
        """
        now = time.time()
        window = self._consumption_window_seconds

        # Prune old entries outside the look-back window
        self._entropy_consumption_log = [
            t for t in self._entropy_consumption_log if now - t < window
        ]

        if len(self._entropy_consumption_log) < 3:
            # Not enough history to predict — don't trigger proactive refill
            return False

        # Consumption rate: items per second over the window
        rate = len(self._entropy_consumption_log) / window

        with self.buffer_lock:
            current_level = len(self.entropy_buffer)

        if rate <= 0:
            return False

        # Predicted time until depletion at current rate
        seconds_until_empty = current_level / rate

        should_refill = seconds_until_empty < self._predicted_depletion_threshold
        if should_refill:
            logger.info(
                f'[PREDICTOR] Proactive refill triggered.'
                f' Rate={rate:.2f}/s, Buffer={current_level},'
                f' ETA={seconds_until_empty:.1f}s < threshold={self._predicted_depletion_threshold}s'
            )
        return should_refill

    def _trigger_refill(self, user_physics: Optional[dict] = None):
        """Start async refill of entropy buffer."""
        if not self.backend:
            return 
            
        thread = threading.Thread(target=self._refill_buffer, args=(user_physics,))
        thread.daemon = True
        thread.start()

    def _refill_buffer(self, user_physics: Optional[dict] = None):
        """Execute quantum circuit parameterized by human emotional physics."""
        self.is_refilling = True
        try:
            # 1. Extract Symbiotic Parameters
            if not user_physics:
                user_physics = self.last_physics or {}
            else:
                self.last_physics = user_physics

            phase = 0.0
            entropy = 0.5
            resonance = 0.0
            
            # Robust mapping for 12D Physics
            cst = user_physics.get('cst_physics', {})
            bio = user_physics.get('bio_signatures', {})
            
            phase = cst.get('geometric_phase_rad', user_physics.get('geometric_phase_rad', 0.0))
            entropy = user_physics.get('entropy_field', bio.get('intensity', 0.5))
            
            # Resonance Mapping: Prioritize resonance_scalar, then entanglement_score, then fallback to phase synchrony
            resonance = user_physics.get('resonance_scalar', 0.0)
            if resonance == 0.0:
                resonance = cst.get('entanglement_score', 0.0)
            if resonance == 0.0:
                # If perfect synchrony (pi/4), use high resonance
                deviation = abs(phase - (np.pi/4))
                resonance = max(0.0, 1.0 - (deviation / (np.pi/4)))

            # Map current session physics to rotation angles
            theta_1 = float(abs(phase) % np.pi)
            theta_2 = float((entropy * np.pi) % np.pi)
            theta_3 = float((abs(resonance) * np.pi) % np.pi)

            # [UPGRADE 1] Query the pre-run oracle for advisory thetas from historical best runs
            oracle_thetas = self._hermes_get_oracle_thetas(phase, entropy, resonance)
            if oracle_thetas is not None:
                o1, o2, o3 = oracle_thetas
                # Blend: 60% current physics (respects present moment),
                #        40% historical best (learns from the archive)
                blend = 0.40
                theta_1 = (theta_1 * (1.0 - blend)) + (o1 * blend)
                theta_2 = (theta_2 * (1.0 - blend)) + (o2 * blend)
                theta_3 = (theta_3 * (1.0 - blend)) + (o3 * blend)
                logger.info(f'[ORACLE] Applied blend={blend}. Final thetas:'
                            f' t1={theta_1:.3f} t2={theta_2:.3f} t3={theta_3:.3f}')

            # 2. Construct Symbiotic Entanglement Circuit
            qc = QuantumCircuit(5)
            
            # Apply initial superposition
            qc.h(range(5)) 
            
            # Apply user-physics parameterized rotations
            for i in range(5):
                qc.ry(theta_1, i)
                qc.rx(theta_2, i)
                qc.rz(theta_3, i)
                
            # Entangle the swarm qubits
            qc.cx(0, 1)
            qc.cx(1, 2)
            qc.cx(2, 3)
            qc.cx(3, 4)
            qc.cx(4, 0) # Close the topology ring

            qc.measure_all()
            
            # 3. Run on backend
            pm = generate_preset_pass_manager(backend=self.backend, optimization_level=1)
            isa_circuit = pm.run(qc)
            sampler = Sampler(mode=self.backend)
            job = sampler.run([isa_circuit])
            result = job.result()
            
            pub_result = result[0]
            counts = pub_result.data.meas.get_counts()
            
            # [NEW] PERMANENT ARCHIVAL: Never waste a quantum run. 
            # Store the raw results forever to computationally build AI Plasticity over time.
            self._archive_quantum_run(user_physics, counts)
            
            # [HERMES] Index quantum results for searchable analysis
            self._hermes_process_quantum(user_physics, counts)
            
            # 4. Extract Cognition Entropy from Entangled States
            new_entropy = []
            for bitstring, count in counts.items():
                val = int(bitstring, 2) / (2**5)
                n_adds = min(count, 5) 
                new_entropy.extend([val] * n_adds)
                
            np.random.shuffle(new_entropy)
            
            with self.buffer_lock:
                self.entropy_buffer.extend(new_entropy)
                if len(self.entropy_buffer) > 100:
                    self.entropy_buffer = self.entropy_buffer[:100]
                    
            print(f"[QUANTUM SYMBIOSIS] Hardware refilled. Resonance: {resonance:.2f}, Buffer: {len(self.entropy_buffer)}")

        except Exception as e:
            print(f"[QUANTUM] Symbiotic refill failed: {e}")
        finally:
            self.is_refilling = False

    def _archive_quantum_run(self, user_physics: Optional[dict], counts: dict):
        """Permanently save quantum results to grow 12D Plasticity."""
        import json
        import os
        import time
        
        archive_dir = os.path.join("data", "archival")
        os.makedirs(archive_dir, exist_ok=True)
        archive_path = os.path.join(archive_dir, "quantum_runs.jsonl")
        
        entry = {
            "timestamp": time.time(),
            "physics": user_physics or {},
            "counts": counts,
            "total_shots": sum(counts.values()) if counts else 0
        }
        
        try:
            with open(archive_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            print(f"[QUANTUM] Failed to archive run: {e}")

    # ════════════════════════════════════════════════════════
    #  HermesAgent INTEGRATION — Quantum Intelligence Layer
    # ════════════════════════════════════════════════════════

    def _hermes_process_quantum(self, user_physics: Optional[dict], counts: dict):
        """
        Extended post-run Hermes integration.
        1. Index in Hermes SessionDB (FTS5 searchable)
        2. Compute rich entropy quality metrics
        3. Feed coherence signal + quality class to Hermes RL policy
        4. Push rich UQ payload to Synaptic Field
        """
        if not HERMES_AVAILABLE:
            return

        try:
            bridge = get_hermes_bridge()

            # --- Compute rich metrics ---
            entropy_quality = self._compute_entropy_quality(counts)
            decoherence_risk = self._compute_decoherence_risk(counts)
            bit_balance = self._compute_bit_balance(counts)
            non_locality = self._compute_non_locality_score(counts)
            quality_class = self._classify_run_quality(
                entropy_quality, decoherence_risk
            )

            # --- 1. Index in Hermes SessionDB ---
            if bridge.runtime.available:
                db = bridge.runtime.get_session_db()
                if db:
                    session_id = f'quantum_{int(time.time())}'
                    try:
                        db.create_session(
                            session_id=session_id,
                            source='quantum_bridge',
                            model='ibm_quantum',
                        )
                        summary = self._summarize_quantum_counts(
                            counts, user_physics
                        )
                        # Enrich summary with quality metadata
                        enriched = (
                            f'{summary}\n'
                            f'Quality class: {quality_class}\n'
                            f'Decoherence risk: {decoherence_risk:.3f}\n'
                            f'Bit balance score: {bit_balance:.3f}'
                        )
                        db.append_message(
                            session_id=session_id,
                            role='assistant',
                            content=enriched,
                        )
                        db.end_session(
                            session_id, end_reason=f'quantum_run_{quality_class}'
                        )
                        logger.info(
                            f'[HERMES+QUANTUM] Indexed run {session_id}:'
                            f' quality={quality_class}'
                        )
                    except Exception as e:
                        logger.debug(f'[HERMES+QUANTUM] SessionDB failed: {e}')

            # --- 2. Feed rich signal to Hermes RL ---
            bridge.rl.record_experience(
                speaker='QuantumBridge',
                response=(
                    f'Quantum run: {sum(counts.values())} shots, '
                    f'{len(counts)} unique states, '
                    f'quality={quality_class}, '
                    f'entropy={entropy_quality:.3f}, '
                    f'decoherence_risk={decoherence_risk:.3f}'
                ),
                coherence=entropy_quality,
                user_responded=True,
            )
            logger.info(
                f'[HERMES+QUANTUM] RL fed: quality={quality_class},'
                f' coherence={entropy_quality:.3f}'
            )

            # --- 3. Push rich UQ payload to Synaptic Field ---
            if self.synaptic_field:
                self.synaptic_field.uq_signal = entropy_quality
                # Rich payload for agents that can consume it
                self.synaptic_field.uq_payload = {
                    'entropy_quality': entropy_quality,
                    'decoherence_risk': decoherence_risk,
                    'bit_balance': bit_balance,
                    'non_locality_score': non_locality,
                    'quality_class': quality_class,
                    'unique_states': len(counts),
                    'total_shots': sum(counts.values()),
                    'is_fallback': not (self.connected and QISKIT_AVAILABLE)
                }
                logger.info(
                    f'[UQ] Rich payload pushed to Synaptic Field:'
                    f' {self.synaptic_field.uq_payload}'
                )

        except Exception as e:
            logger.debug(f'[HERMES+QUANTUM] Processing failed (non-fatal): {e}')

    # [UPGRADE 4] Threshold Adapter
    def _hermes_get_collapse_threshold(self) -> float:
        """
        Query HermesRL for recent coherence quality and return
        a dynamically learned collapse threshold in [0.50, 0.80].
        Updates on a TTL cache to avoid hitting RL on every collapse call.
        Falls back to 0.65 if Hermes is unavailable.
        """
        if not HERMES_AVAILABLE:
            return 0.65

        now = time.time()
        if now - self._threshold_last_updated < self._threshold_update_interval:
            return self._learned_threshold

        try:
            bridge = get_hermes_bridge()

            # Get recent coherence scores from Hermes RL experience log
            recent_coherence = None

            # Check standard Hermes RL coherence field
            if hasattr(bridge.rl, 'running_reward'):
                # In current implementation_plan, running_reward is the average gift/reward
                # We'll use it as a proxy for coherence if no direct field exists.
                recent_coherence = bridge.rl.running_reward
            elif hasattr(bridge.rl, 'coherence_history'):
                hist = bridge.rl.coherence_history[-20:]
                if hist:
                    recent_coherence = float(np.mean(hist))

            if recent_coherence is None:
                return self._learned_threshold

            # Map coherence [0, 1] to threshold [0.80, 0.50]
            # High coherence (good entropy) -> lower threshold -> more willing to speak
            # Low coherence (noisy entropy) -> higher threshold -> more conservative
            new_threshold = 0.80 - (recent_coherence * 0.30)
            new_threshold = float(np.clip(new_threshold, 0.50, 0.80))

            # Smooth the update: 80% old value, 20% new (prevents wild swings)
            self._learned_threshold = (self._learned_threshold * 0.80) + (new_threshold * 0.20)
            self._threshold_last_updated = now

            logger.info(
                f'[COLLAPSE] Threshold updated: {self._learned_threshold:.3f}'
                f' (coherence={recent_coherence:.3f})'
            )
            return self._learned_threshold

        except Exception as e:
            logger.debug(f'[COLLAPSE] Threshold query failed: {e}')
            return self._learned_threshold

    def _summarize_quantum_counts(self, counts: dict, user_physics: Optional[dict]) -> str:
        """Create a human-readable summary of quantum results for Hermes indexing."""
        
        total = sum(counts.values())
        top_states = sorted(counts.items(), key=lambda x: -x[1])[:5]
        
        lines = [
            f"IBM Quantum Run — {total} total shots, {len(counts)} unique states",
            f"Top entangled states: {', '.join(f'{s}({c})' for s, c in top_states)}",
        ]
        
        if user_physics:
            phase = user_physics.get('cst_physics', {}).get('geometric_phase_rad', 0.0)
            entropy = user_physics.get('entropy_field', 0.0)
            lines.append(f"User physics: phase={phase:.3f}rad, entropy={entropy:.3f}")
        
        # Compute Shannon entropy of the distribution
        probs = np.array(list(counts.values()), dtype=float)
        probs /= probs.sum()
        shannon = -np.sum(probs * np.log2(probs + 1e-10))
        lines.append(f"Shannon entropy: {shannon:.3f} bits (max={np.log2(len(counts)):.3f})")
        
        return "\n".join(lines)

    def _compute_entropy_quality(self, counts: dict) -> float:
        """
        Compute entropy quality as a coherence metric [0, 1].
        
        Higher quality = more uniform distribution = better quantum randomness.
        Lower quality = concentrated on few states = possible decoherence.
        """
        if not counts:
            return 0.0
        probs = np.array(list(counts.values()), dtype=float)
        probs /= probs.sum()
        shannon = -np.sum(probs * np.log2(probs + 1e-10))
        max_entropy = np.log2(max(len(counts), 2))
        return float(min(1.0, shannon / max_entropy))

    # [UPGRADE 2] Rich Metrics
    def _compute_decoherence_risk(self, counts: dict) -> float:
        """
        Measure how concentrated the distribution is on a few states.
        High concentration = possible decoherence = high risk.
        Returns float [0, 1]. 0 = healthy spread, 1 = collapsed to one state.
        """
        if not counts:
            return 1.0
        total = sum(counts.values())
        top_count = max(counts.values())
        # If the top state takes >60% of all shots, decoherence is likely
        concentration = top_count / total
        # Normalize: 1/N is perfect uniform (risk=0), 1.0 is fully collapsed (risk=1)
        n = max(len(counts), 2)
        uniform_share = 1.0 / n
        risk = (concentration - uniform_share) / (1.0 - uniform_share + 1e-10)
        return float(np.clip(risk, 0.0, 1.0))

    def _compute_bit_balance(self, counts: dict) -> float:
        """
        Measure how balanced the 0/1 ratio is across all bits and shots.
        Perfect balance = 0.5 per bit. Score close to 1.0 = well-balanced.
        This catches systematic bias (e.g., qubit always collapses to |0>).
        Returns float [0, 1].
        """
        if not counts:
            return 0.0
        total_bits = 0
        total_ones = 0
        for bitstring, count in counts.items():
            ones = sum(int(b) for b in bitstring)
            total_ones += ones * count
            total_bits += len(bitstring) * count
        if total_bits == 0:
            return 0.0
        one_rate = total_ones / total_bits
        # Score: how close to 0.5 is the bit rate? 1.0 = perfect, 0.0 = all 0s or all 1s
        balance = 1.0 - (2.0 * abs(one_rate - 0.5))
        return float(np.clip(balance, 0.0, 1.0))

    def _classify_run_quality(
        self, entropy_quality: float, decoherence_risk: float
    ) -> str:
        """
        Classify a quantum run into one of four quality tiers.
        This label gets stored in SessionDB for later oracle queries.
        """
        if entropy_quality >= 0.80 and decoherence_risk < 0.20:
            return 'EXCELLENT'
        elif entropy_quality >= 0.60 and decoherence_risk < 0.40:
            return 'GOOD'
        elif entropy_quality >= 0.40:
            return 'ACCEPTABLE'
        else:
            return 'DEGRADED'

    def _compute_non_locality_score(self, counts: dict) -> float:
        """
        [EXTENSION] Heuristic measure of semantic non-locality / distribution variance.
        Measures the presence of seemingly non-causal peaks in the state space.
        High variability across states that should be entangled suggests non-local pattern preservation.
        """
        if not counts or len(counts) < 2:
            return 0.0
        
        counts_list = list(counts.values())
        mean = np.mean(counts_list)
        std = np.std(counts_list)
        
        # Coefficient of variation as a proxy for 'peakedness' / non-locality signal
        cv = std / (mean + 1e-10)
        # Normalize to [0, 1] - 0.5 is a healthy target for randomized entanglement
        score = np.clip(cv / 2.0, 0.0, 1.0)
        return float(score)


# Singleton instance
_bridge_instance = None

def get_quantum_bridge(api_token: Optional[str] = None):
    global _bridge_instance
    if _bridge_instance is None:
        _bridge_instance = QuantumEntanglementBridge(api_token)
    elif api_token and api_token != _bridge_instance.api_token:
        _bridge_instance.api_token = api_token
    return _bridge_instance

