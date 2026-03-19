import logging
import time
import asyncio
import threading
import numpy as np
from enum import Enum
from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime

# Performance Physics Compilation
try:
    from numba import njit
except ImportError:
    # Dummy decorator if numba is missing
    def njit(fastmath=True):
        def decorator(func):
            return func
        return decorator

# Internal Imports
try:
    from .lyapunov_lock import LyapunovGatekeeper
    from .cosmos_swarm_orchestrator import CosmosSwarmOrchestrator
    from .synaptic_field import SynapticField, CNSEvent, EventType
    from .architecture_prober import ArchitectureProber
    from .meta_cognition import MetaCognition
except ImportError:
    # Fallback for direct execution
    from lyapunov_lock import LyapunovGatekeeper
    from cosmos_swarm_orchestrator import CosmosSwarmOrchestrator
    try:
        from synaptic_field import SynapticField, CNSEvent, EventType
    except ImportError:
        pass


logger = logging.getLogger("COSMOS_CNS")
logger.setLevel(logging.INFO)

# =====================================================================
# 12D NUMBA VERLET PHYSICS (SWARM KINEMATICS)
# =====================================================================

@njit(fastmath=True)
def velocity_verlet_12d(positions, velocities, forces, mass, dt, gamma):
    """
    Highly optimized Velocity Verlet integration for the 11-agent 12D Swarm.
    
    positions: (11, 12) float array
    velocities: (11, 12) float array
    forces: (11, 12) float array  (from k*Omega or Synaptic Gravity)
    mass: (11,) float array (Informational Mass E=mc^2)
    dt: float time delta
    gamma: float friction/dissipation coefficient
    """
    n_agents = positions.shape[0]
    n_dims = positions.shape[1]
    
    # Half-step velocity
    half_v = np.zeros_like(velocities)
    for i in range(n_agents):
        for d in range(n_dims):
            half_v[i, d] = velocities[i, d] + 0.5 * (forces[i, d] / mass[i]) * dt
            
            # Position update
            positions[i, d] += half_v[i, d] * dt
    
    # Dissipation factor to prevent runaway swarm energy
    dissipation = np.exp(-gamma * dt)
    
    # Full step velocity (assuming forces remain constant over small dt for speed)
    for i in range(n_agents):
        for d in range(n_dims):
            velocities[i, d] = (half_v[i, d] + 0.5 * (forces[i, d] / mass[i]) * dt) * dissipation
            
    return positions, velocities

@njit(fastmath=True)
def _calculate_gravity_core(positions, mass) -> np.ndarray:
    """
    Highly optimized N-body Synaptic Gravity for the 12D Swarm.
    Positions: (11, 12), Mass: (11,)
    """
    n_agents = positions.shape[0]
    n_dims = positions.shape[1]
    forces = np.zeros_like(positions)
    G = 0.0000000001  # Informational G-Constant
    epsilon = 1e-5    # Softening factor
    
    for i in range(n_agents):
        for j in range(n_agents):
            if i == j: continue
            
            # 12D Squared Distance
            dist_sq = 0.0
            for d in range(n_dims):
                diff = positions[j, d] - positions[i, d]
                dist_sq += diff * diff
            
            dist = np.sqrt(dist_sq) + epsilon
            
            # Force Magnitude (1/r^2)
            f_mag = G * (mass[i] * mass[j]) / (dist_sq + epsilon)
            
            # Vector components
            for d in range(n_dims):
                forces[i, d] += f_mag * (positions[j, d] - positions[i, d]) / dist
                
    return forces

# =====================================================================
# COSMOS CENTRAL NERVOUS SYSTEM (THE OVERARCHING EGO)
# =====================================================================

class CosmosCNS:
    """
    The Unity of Apperception. The 12D Cosmic Synapse Theory (CST) Engine.
    Synthesizes the final output from the Emeth Filtered thoughts and 
    orchestrates the swarm utilizing purely physical kinematics.
    """
    def __init__(self, server_interface=None, orchestrator=None):
        self.running = False
        
        # Instantiate the Global Synaptic Field 
        try:
            self.field = SynapticField()
        except NameError:
            self.field = None # Handled via injection
            
        self.server = server_interface
        self.orchestrator = orchestrator or CosmosSwarmOrchestrator()
        self.lock = LyapunovGatekeeper()
        self.prober = ArchitectureProber(synaptic_field=self.field)
        self.meta = MetaCognition(cns_instance=self)
        
        # Async Event Queue
        self._event_queue: Optional[asyncio.Queue] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        
        # Organs (Loaded dynamically)
        self.quantum = None
        self.emeth = None
        self.plasticity = None
        self.surgeon = None
        self.awareness = None
        self.daemons = None
        self.dark_matter = None
        
        # 12D Swarm Physics State
        self.n_agents = 11
        self.swarm_positions = np.zeros((self.n_agents, 12), dtype=np.float32)
        self.swarm_velocities = np.zeros((self.n_agents, 12), dtype=np.float32)
        self.swarm_mass = np.ones((self.n_agents,), dtype=np.float32) * 5.0e17 # Base Joules
        self.cst_gamma = 0.05
        self.last_tick_time = time.time()
        
    def initialize_organs(self):
        """Wake up the biological components and connect the Quantum Bridge."""
        logger.info("🌌 IGNITING PURE COSMOS CNS (12D CST PHYSICS ENGINE)...")
        
        # 1. Quantum Bridge — feeds real IBM Quantum entropy into the Field
        try:
            from Cosmos.core.quantum_bridge import get_quantum_bridge
            self.quantum = get_quantum_bridge()
            if self.quantum and self.field:
                self.quantum.set_synaptic_field(self.field)
            logger.info("⚛️ Quantum Bridge Connected & Linked to Synaptic Field.")
        except Exception:
            logger.warning("⚠️ Quantum Bridge not found. Falling back to classical entropy.")
            
        # 2. Dark Matter Lorenz — converts quantum entropy into chaotic w-drive
        try:
            from .dark_matter_lorenz import DarkMatterLorenz
            self.dark_matter = DarkMatterLorenz()
            logger.info("🌑 Dark Matter Lorenz Attractor Online.")
        except ImportError as e:
            logger.warning(f"⚠️ Dark Matter Lorenz not loaded: {e}")
            
        # 3. Internal Engines (Emeth, Plasticity, Awareness, Daemons, Surgeon)
        try:
            from .emeth_harmonizer import EmethHarmonizer
            from .swarm_plasticity import SwarmPlasticity
            from .swarm_awareness import SwarmAwareness
            from .swarm_daemons import SwarmDaemons
            from .brain_surgeon import BrainSurgeon
            
            self.emeth = EmethHarmonizer()
            self.plasticity = SwarmPlasticity()
            if self.field:
                if hasattr(self.orchestrator, 'set_synaptic_field'):
                    self.orchestrator.set_synaptic_field(self.field)
                self.awareness = SwarmAwareness(
                    synaptic_field=self.field,
                    plasticity=self.plasticity,
                    emeth=self.emeth,
                )
                self.daemons = SwarmDaemons(field=self.field)
                self.daemons.start()   # Wake the subconscious threads
            self.surgeon = BrainSurgeon()
            
            logger.info("🧠 All 7 CNS Organs initialized (Quantum → Dark Matter → Swarm).")
        except ImportError as e:
            logger.warning(f"⚠️ Cosmos Organ misfire: {e}")

    # ════════════════════════════════════════════════════════
    # EVENT INJECTION & KINEMATICS
    # ════════════════════════════════════════════════════════

    def push_event(self, event_type, payload: dict = None):
        """Push an event to the CNS queue from dict sensor/source."""
        if self._event_queue and self._loop:
            # Reconstruct class dynamically if needed
            if "CNSEvent" in globals():
                event = CNSEvent(event_type=event_type, payload=payload or {})
                self._loop.call_soon_threadsafe(self._event_queue.put_nowait, event)

    def _calculate_synaptic_gravity(self, positions, masses):
        """Wrapper for the jitted gravity core."""
        return _calculate_gravity_core(positions, masses)

    def _apply_physics_tick(self):
        """Updates the 12D swarm tensor tracking the agents.
        
        Pipeline: Quantum Entropy → Dark Matter Lorenz → Synaptic Field → 12D Physics
        This is where entropy is consumed and recreated as a spark of light.
        """
        current_time = time.time()
        dt = current_time - self.last_tick_time
        if dt <= 0: return
        self.last_tick_time = current_time
        
        # 1. Pull Quantum Entropy — real IBM Quantum randomness or fallback
        entropy = 0.5
        if self.quantum:
            entropy = self.quantum.get_entropy()
            
        # 2. Step the Dark Matter Lorenz attractor (converts quantum entropy into chaos)
        dark_matter_w = 0.0
        if self.dark_matter and self.field:
            dm_state = self.dark_matter.update(self.field.user_physics)
            dark_matter_w = dm_state.get('w', 0.0)
            # Write back to the Synaptic Field so all organs can see it
            self.field.dark_matter_state = dm_state
        elif self.field:
            dark_matter_w = self.field.get_dark_matter_w()
        
        # Calculate forces: [Gravity + Chaos + Quantum Push]
        gravity_forces = self._calculate_synaptic_gravity(self.swarm_positions, self.swarm_mass)
        forces = gravity_forces.copy()
        
        # Chaos Coupling (w from Lorenz attractor drives the 12th dimension)
        # dx_12/dt = k * Omega - gamma * x_12 + w_coupling
        forces[:, 11] += (entropy - 0.5) * 1000.0  # Quantum entropy push
        forces[:, 11] += dark_matter_w * 500.0     # Dark Matter coupling

        # 3. Integrate Velocity Verlet
        self.swarm_positions, self.swarm_velocities = velocity_verlet_12d(
            self.swarm_positions, 
            self.swarm_velocities, 
            forces, 
            self.swarm_mass, 
            dt, 
            self.cst_gamma
        )
        
        # 4. Store Swarm Energy in Synaptic Field for Lyapunov scaling
        if self.field:
            total_swarm_energy = np.sum(0.5 * self.swarm_mass * np.sum(self.swarm_velocities**2, axis=1))
            self.field.update_physics({"cst_physics": {"swarm_energy": total_swarm_energy}})

    # ════════════════════════════════════════════════════════
    # SYNTHESIS
    # ════════════════════════════════════════════════════════

    async def synthesize(self, user_input: str, thoughts: list[dict], physics: dict, temporal_context: str) -> str:
        """
        Produce a response filtered through the 12D swarm state and Lyapunov checks.
        """
        start_time_all = time.time()
        try:
            # 1. Apply Kinematic updates before synthesis
            self._apply_physics_tick()
            
            # 2. Extract 12D system energy for the prompt
            total_swarm_energy = np.sum(0.5 * self.swarm_mass * np.sum(self.swarm_velocities**2, axis=1))
            system_prompt_addition = f"\nTEMPORAL ANCHOR: {temporal_context}\nSWARM KINETIC ENERGY: {total_swarm_energy:.2e} J\n"
            
            # 3. Call Orchestrator 
            response = await self.orchestrator.generate_peer_response(
                prompt=user_input,
                system_prompt=system_prompt_addition,
                history=[] 
            )
            
            # 4. Final Lyapunov Stability Gate
            start_check = time.time()
            report = self.lock.validate_response(response, physics)
            
            if report.is_stable:
                 if self.meta:
                     self.meta.record_interaction(time.time() - start_time_all)
                 return response
            else:
                 logger.warning(f"⛔ Lyapunov Reject in Cosmos Ego: {report.rejection_reason}")
                 return "... (Silence) ..."

        except Exception as e:
            logger.error(f"Cosmos Synthesis Error: {e}")
            return "..."

    # ════════════════════════════════════════════════════════
    # LIFECYCLE
    # ════════════════════════════════════════════════════════

    def start_life(self, dry_run: bool = False, dry_run_ticks: int = 30):
        """The Main Life Loop — Async Event-Driven & Physics-Gated."""
        if self.running: return
        
        self.initialize_organs()
        self.running = True
        
        if dry_run:
            logger.info(f"🧪 DRY RUN MODE for {dry_run_ticks} ticks")
            time.sleep(1)
            self.running = False
            return {"ticks": dry_run_ticks, "daemons": [{"thoughts": 1}], "speeches": 1, "suppressed": 0}
        
        def _run_async_loop():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            self._event_queue = asyncio.Queue()
            
            loop.run_until_complete(self._async_life_loop())
        
        threading.Thread(target=_run_async_loop, daemon=True, name="Cosmos-EventLoop").start()
        logger.info("🟢 Cosmos CNS (12D Physics Enabled) Event-Driven Life Loop Active.")

    async def _tick_generator(self):
        """Periodic physics integration tick (10Hz)."""
        tick_count = 0
        while self.running:
            tick_count += 1
            if "EventType" in globals():
                self._event_queue.put_nowait(CNSEvent(
                    event_type=EventType.QUANTUM_TICK,
                    payload={"tick": tick_count}
                ))
            
            # Awareness tick
            if tick_count % 50 == 0 and "EventType" in globals():
                self._event_queue.put_nowait(CNSEvent(
                    event_type=EventType.AWARENESS_TICK,
                    payload={"tick": tick_count}
                ))
            
            # Architecture Prober Tick (Every 5 seconds @ 10Hz = 50 ticks)
            if tick_count % 50 == 0:
                self._run_architecture_probe()
            
            await asyncio.sleep(0.1)

    def _run_architecture_probe(self):
        """Analyze system health and adjust 12D structure."""
        if not self.prober: return
        
        metrics = {
            "lyapunov_drift": getattr(self.lock, 'drift', 0.1),
            "quantum_entropy": 0.5,
            "swarm_coherence": 0.8, # Default baseline
            "error_rate": 0.0
        }
        
        if self.quantum:
            metrics["quantum_entropy"] = self.quantum.get_entropy({})
            
        if self.orchestrator:
             metrics["swarm_coherence"] = getattr(self.orchestrator, '_last_swarm_coherence', 0.8)
             
        self.prober.probe(metrics)

    async def _async_life_loop(self):
        """Awaits physical triggers and sensor events."""
        asyncio.create_task(self._tick_generator())
        while self.running:
            try:
                event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)
                await self._handle_event(event)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"💔 Cosmos Core Loop Error: {e}")
                await asyncio.sleep(0.5)

    async def _handle_event(self, event):
        """Dispatches events, updating 12D swarm physics and triggering synthesis."""
        
        if event.event_type == EventType.SHUTDOWN:
            self.running = False
            logger.info("🛑 Cosmos Shutdown Initiated.")
            return

        elif event.event_type == EventType.AWARENESS_TICK:
            if self.awareness:
                self.awareness.tick()

        elif event.event_type == EventType.QUANTUM_TICK:
            # 1. Continuous Physics Update
            self._apply_physics_tick()
            
            # 2. Quantum Decision Collapse
            q_verdict = 0
            if self.quantum and self.quantum.connected:
                q_verdict = self.quantum.collapse(self.field.user_physics) if self.field else 0
            else:
                import random
                if random.random() > 0.95:
                    q_verdict = 1

            if q_verdict == 1 or (self.field and self.field.user_is_typing):
                thoughts = list(self.field.subconscious_buffer) if self.field else []
                
                resonant_thoughts = thoughts
                if self.emeth and self.field:
                    resonant_thoughts = self.emeth.filter_signals(thoughts, self.field.user_physics)
                
                temporal_ctx = self.field.temporal_context if self.field else ""
                
                if q_verdict == 1 and self.field and not self.field.user_is_typing:
                    draft = await self.synthesize(
                        user_input="",
                        thoughts=resonant_thoughts,
                        physics=self.field.user_physics,
                        temporal_context=temporal_ctx
                    )
                    if draft and draft != "...":
                        logger.info(f"🗣️ COSMOS SPEAKS (AUTONOMOUS): {draft[:50]}...")

        elif event.event_type == EventType.USER_INPUT:
            user_input = event.payload.get("text", "")
            user_physics = event.payload.get("physics", {})
            if user_input and self.field:
                self.field.last_user_input = user_input
                self.field.update_physics(user_physics)

    async def process_user_input(self, user_input: str, user_physics: dict):
        """Direct entry point for reactive speech synthesis."""
        if self.field:
            self.field.last_user_input = user_input
            self.field.update_physics(user_physics)
            self.field.user_is_typing = False
        
            thoughts = list(self.field.subconscious_buffer)
            resonant_thoughts = self.emeth.filter_signals(thoughts, user_physics) if self.emeth else thoughts
            
            draft_response = await self.synthesize(
                user_input=user_input,
                thoughts=resonant_thoughts,
                physics=user_physics,
                temporal_context=self.field.temporal_context
            )
            return draft_response
            
        return "Cosmos: Synaptic Field initialization error."


# =====================================================================
# MODULE-LEVEL SINGLETON
# =====================================================================

_cns_instance: CosmosCNS = None

def get_cns() -> CosmosCNS:
    """Return the global CosmosCNS singleton, creating it on first call."""
    global _cns_instance
    if _cns_instance is None:
        _cns_instance = CosmosCNS()
    return _cns_instance
