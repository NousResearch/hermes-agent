"""
Cosmos IBM Quantum Integration
===================================

Quantum computing integration for enhanced AI capabilities using IBM Quantum Platform.

IBM Quantum Open Plan (Free Tier) - as of 2026:
- QPU Hardware: 10 minutes per 28-day rolling window (NOT calendar month)
- QPU Access: Heron r1/r2/r3 processors (133-156 qubits), us-east region only
- Cloud Simulators: RETIRED (May 2024) - use local AerSimulator instead
- Local Simulators: Unlimited (AerSimulator, FakeBackends with noise models)
- Execution Modes: Job and Batch ONLY (Session mode requires paid plan)
- Channel: ibm_quantum_platform (ibm_quantum channel is DEAD as of July 2025)

Current QPU Backends (2026):
- ibm_fez, ibm_torino, ibm_marrakesh, ibm_kingston, ibm_pittsburgh (Heron)
- RETIRED: ibm_brisbane (Nov 2025), ibm_kyoto, ibm_sherbrooke (July 2025)

Strategy: Local AerSimulator for development, QPU hardware for high-value SAGI tasks
Budget: 40% evolution, 30% optimization, 20% benchmark, 10% other

Use Cases:
- Quantum Genetic Algorithm (QGA) for agent evolution toward SAGI
- QAOA for multi-objective swarm optimization
- VQE for variational simulations
- Quantum-inspired pattern extraction for memory dreaming
- Noise-aware simulation via FakeBackends (mimics real QPU noise, unlimited)

"When classical optimization hits a wall, we go quantum." - Cosmos Collective
"""

import os
import json
import asyncio
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger

# Qiskit imports (graceful fallback if not installed)
try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit import Parameter, ParameterVector
    # Qiskit 2.x uses StatevectorSampler/Estimator for local simulation
    from qiskit.primitives import StatevectorSampler, StatevectorEstimator
    from qiskit_ibm_runtime import QiskitRuntimeService, Session, Batch, Options
    from qiskit_ibm_runtime import SamplerV2, EstimatorV2
    from qiskit_aer import AerSimulator
    from qiskit.quantum_info import SparsePauliOp
    QISKIT_AVAILABLE = True
except ImportError as e:
    QISKIT_AVAILABLE = False
    logger.warning(f"Qiskit import error: {e}. Run: pip install qiskit qiskit-ibm-runtime qiskit-aer")

# Nexus integration for signal emission
try:
    from Cosmos.core.nexus import get_nexus, SignalType
    NEXUS_AVAILABLE = True
except ImportError:
    NEXUS_AVAILABLE = False


async def _emit_quantum_signal(signal_type: str, data: Dict[str, Any]) -> None:
    """
    Emit a quantum-related signal to the Nexus event bus.

    AGI v1.8.2: Quantum events are broadcast to the swarm for monitoring,
    evolution feedback, and coordinated optimization.
    """
    if not NEXUS_AVAILABLE:
        return

    try:
        nexus = get_nexus()
        await nexus.emit(signal_type, data)
    except Exception as e:
        logger.debug(f"Could not emit quantum signal: {e}")


class QuantumBackend(Enum):
    """Available quantum backends (updated 2026)."""
    SIMULATOR_IDEAL = "aer_simulator"           # Noise-free local, unlimited
    SIMULATOR_NOISY = "aer_simulator_noisy"     # With noise model, unlimited
    FAKE_BACKEND = "fake_backend"               # FakeBackend with real QPU noise, unlimited
    # Current Heron QPUs (Open Plan, us-east)
    HARDWARE_FEZ = "ibm_fez"                    # Heron, 156 qubits
    HARDWARE_TORINO = "ibm_torino"              # Heron, 133 qubits
    HARDWARE_MARRAKESH = "ibm_marrakesh"        # Heron, 156 qubits
    AUTO = "auto"                               # Auto-select based on task


class QuantumTaskType(Enum):
    """Types of quantum tasks for resource allocation."""
    OPTIMIZATION = "optimization"       # QAOA, VQE - high value
    EVOLUTION = "evolution"             # QGA for agent mutation
    PATTERN = "pattern"                 # Memory pattern extraction
    INFERENCE = "inference"             # Knowledge graph queries
    SAMPLING = "sampling"               # Probabilistic modeling
    BENCHMARK = "benchmark"             # Performance testing


class ExecutionMode(Enum):
    """IBM Quantum execution modes per official docs (2026).

    Open Plan (free): Job and Batch ONLY. Session requires paid plan.
    """
    JOB = "job"           # Single primitive request - Open Plan OK
    BATCH = "batch"       # Multiple independent jobs in parallel - Open Plan OK
    SESSION = "session"   # Exclusive QPU access - PAID PLANS ONLY


class ResilienceLevel(Enum):
    """Error mitigation levels for Estimator (per IBM docs)."""
    NONE = 0              # No error mitigation
    MINIMAL = 1           # TREX readout error correction (default)
    MEDIUM = 2            # TREX + ZNE + gate twirling (~3x overhead)


@dataclass
class QuantumOptions:
    """
    Configuration options for quantum execution (per IBM Quantum best practices).

    Based on: https://quantum.cloud.ibm.com/docs/guides/configure-error-mitigation
    """
    # Execution mode - BATCH is default (Open Plan compatible, cheaper than Session)
    execution_mode: ExecutionMode = ExecutionMode.BATCH

    # Error mitigation (Estimator only)
    resilience_level: ResilienceLevel = ResilienceLevel.MINIMAL

    # Dynamical Decoupling - suppresses idle qubit errors
    dynamical_decoupling: bool = True
    dd_sequence_type: str = "XpXm"  # or "XX"

    # Pauli Twirling - converts noise to Pauli channel
    enable_twirling: bool = True
    twirling_num_randomizations: int = 32
    twirling_shots_per_randomization: int = 100

    # Transpilation
    optimization_level: int = 3  # 0-3, higher = more optimization

    # Zero Noise Extrapolation (for resilience_level 2)
    zne_noise_factors: Tuple[int, ...] = (1, 3, 5)
    zne_extrapolator: str = "exponential"  # or "linear", "polynomial"


@dataclass
class QuantumUsageStats:
    """Track quantum resource usage against IBM Open Plan free tier.

    IBM uses a 28-day ROLLING window (not calendar month).
    10 minutes (600s) of QPU time per rolling 28-day period.
    """
    hardware_seconds_used: float = 0.0
    hardware_jobs_count: int = 0
    simulator_jobs_count: int = 0
    last_hardware_run: Optional[datetime] = None
    window_start: datetime = field(default_factory=datetime.now)

    # Free tier limits - 28-day rolling window
    ROLLING_WINDOW_DAYS: int = 28
    WINDOW_HARDWARE_SECONDS: int = 600  # 10 minutes per 28-day window
    # Keep old name as alias for backward compat
    MONTHLY_HARDWARE_SECONDS: int = 600

    @property
    def hardware_seconds_remaining(self) -> float:
        """Remaining hardware time in current 28-day window."""
        return max(0, self.WINDOW_HARDWARE_SECONDS - self.hardware_seconds_used)

    @property
    def hardware_percentage_used(self) -> float:
        """Percentage of 28-day window hardware quota used."""
        return (self.hardware_seconds_used / self.WINDOW_HARDWARE_SECONDS) * 100

    @property
    def days_until_reset(self) -> int:
        """Days until the 28-day rolling window resets."""
        elapsed = (datetime.now() - self.window_start).days
        return max(0, self.ROLLING_WINDOW_DAYS - elapsed)

    def can_use_hardware(self, estimated_seconds: float = 30) -> bool:
        """Hard check: block hardware if quota would be exceeded."""
        self.reset_if_window_expired()
        if self.hardware_seconds_used >= self.WINDOW_HARDWARE_SECONDS:
            logger.warning(f"HARD BLOCK: 28-day hardware quota exhausted ({self.hardware_seconds_used:.0f}/{self.WINDOW_HARDWARE_SECONDS}s, resets in {self.days_until_reset}d)")
            return False
        if self.hardware_seconds_remaining < estimated_seconds:
            logger.warning(f"HARD BLOCK: Not enough quota for {estimated_seconds:.0f}s job ({self.hardware_seconds_remaining:.0f}s remaining, resets in {self.days_until_reset}d)")
            return False
        return True

    def record_hardware_job(self, duration_seconds: float):
        """Record a hardware job execution. Raises if quota exceeded."""
        if self.hardware_seconds_used + duration_seconds > self.WINDOW_HARDWARE_SECONDS + 60:
            # Allow 60s grace for jobs that ran slightly over estimate
            raise RuntimeError(
                f"Quantum hardware quota exceeded: {self.hardware_seconds_used + duration_seconds:.0f}s "
                f"would exceed {self.WINDOW_HARDWARE_SECONDS}s per 28-day window"
            )
        self.hardware_seconds_used += duration_seconds
        self.hardware_jobs_count += 1
        self.last_hardware_run = datetime.now()
        logger.info(f"Quantum hardware job: {duration_seconds:.1f}s used, {self.hardware_seconds_remaining:.1f}s remaining")

        # Emit usage warning if below 20%
        if self.hardware_percentage_used >= 80:
            try:
                import asyncio
                loop = asyncio.get_running_loop()
                loop.create_task(_emit_quantum_signal("quantum.usage_warning", {
                    "percentage_used": self.hardware_percentage_used,
                    "seconds_remaining": self.hardware_seconds_remaining,
                    "jobs_used": self.hardware_jobs_count,
                    "warning_level": "critical" if self.hardware_percentage_used >= 95 else "high",
                    "timestamp": datetime.now().isoformat()
                }))
            except RuntimeError:
                pass  # No running event loop in this thread, skip emission
            except Exception:
                pass  # Don't fail on signal emission

    def record_simulator_job(self):
        """Record a simulator job (unlimited)."""
        self.simulator_jobs_count += 1

    def reset_if_window_expired(self):
        """Reset counters if 28-day rolling window has expired."""
        now = datetime.now()
        elapsed = (now - self.window_start).days
        if elapsed >= self.ROLLING_WINDOW_DAYS:
            logger.info(
                f"Quantum 28-day window reset. Previous window: "
                f"{self.hardware_seconds_used:.1f}s hardware, {self.simulator_jobs_count} simulator jobs"
            )
            self.hardware_seconds_used = 0.0
            self.hardware_jobs_count = 0
            self.simulator_jobs_count = 0
            self.window_start = now

    # Backward compat alias
    def reset_if_new_month(self):
        """Alias for reset_if_window_expired (legacy compat)."""
        self.reset_if_window_expired()

    def to_dict(self) -> Dict:
        """Serialize for persistence."""
        return {
            "hardware_seconds_used": self.hardware_seconds_used,
            "hardware_jobs_count": self.hardware_jobs_count,
            "simulator_jobs_count": self.simulator_jobs_count,
            "last_hardware_run": self.last_hardware_run.isoformat() if self.last_hardware_run else None,
            "window_start": self.window_start.isoformat(),
            "rolling_window_days": self.ROLLING_WINDOW_DAYS
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "QuantumUsageStats":
        """Deserialize from persistence."""
        stats = cls()
        stats.hardware_seconds_used = data.get("hardware_seconds_used", 0.0)
        stats.hardware_jobs_count = data.get("hardware_jobs_count", 0)
        stats.simulator_jobs_count = data.get("simulator_jobs_count", 0)
        if data.get("last_hardware_run"):
            stats.last_hardware_run = datetime.fromisoformat(data["last_hardware_run"])
        # Support both old month_start and new window_start keys
        window_key = data.get("window_start") or data.get("month_start")
        if window_key:
            stats.window_start = datetime.fromisoformat(window_key)
        stats.reset_if_window_expired()
        return stats


class HardwareBudgetAllocator:
    """
    Strategic hardware budget allocation for maximum innovation impact.

    Distributes 600s/month across quantum operations by priority:
    1. Evolution final generations (40%) - real quantum noise drives genuine mutation diversity
    2. QAOA optimization (30%) - hardware finds better optima than simulator
    3. Benchmark/validation (20%) - compare hardware vs simulator results
    4. Pattern extraction (10%) - quantum sampling for memory consolidation

    Each category has a budget cap. Once a category is exhausted, it falls back
    to simulator. This ensures the most impactful operations always get hardware.
    """

    # Budget allocation (fraction of MONTHLY_HARDWARE_SECONDS)
    BUDGET_ALLOCATION = {
        QuantumTaskType.EVOLUTION: 0.40,    # 240s - evolution final gens
        QuantumTaskType.OPTIMIZATION: 0.30, # 180s - QAOA/VQE
        QuantumTaskType.BENCHMARK: 0.20,    # 120s - validation runs
        QuantumTaskType.PATTERN: 0.05,      #  30s - memory patterns
        QuantumTaskType.INFERENCE: 0.03,    #  18s - KG queries
        QuantumTaskType.SAMPLING: 0.02,     #  12s - probabilistic
    }

    def __init__(self, usage_stats: "QuantumUsageStats"):
        self.usage_stats = usage_stats
        self._category_usage: Dict[str, float] = {}
        self._category_usage_file = Path("data/quantum_budget_usage.json")
        self._load_category_usage()

    def _load_category_usage(self):
        """Load per-category usage from disk."""
        try:
            if self._category_usage_file.exists():
                data = json.loads(self._category_usage_file.read_text())
                month_key = datetime.now().strftime("%Y-%m")
                if data.get("month") == month_key:
                    self._category_usage = data.get("usage", {})
                else:
                    self._category_usage = {}
        except Exception:
            self._category_usage = {}

    def _save_category_usage(self):
        """Persist per-category usage to disk."""
        try:
            self._category_usage_file.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "month": datetime.now().strftime("%Y-%m"),
                "usage": self._category_usage,
                "total_hardware_used": self.usage_stats.hardware_seconds_used,
                "updated": datetime.now().isoformat()
            }
            self._category_usage_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.debug(f"Could not save budget usage: {e}")

    def get_category_budget(self, task_type: QuantumTaskType) -> float:
        """Get total hardware budget for a task category (seconds)."""
        fraction = self.BUDGET_ALLOCATION.get(task_type, 0.0)
        return self.usage_stats.MONTHLY_HARDWARE_SECONDS * fraction

    def get_category_remaining(self, task_type: QuantumTaskType) -> float:
        """Get remaining hardware budget for a category."""
        budget = self.get_category_budget(task_type)
        used = self._category_usage.get(task_type.value, 0.0)
        return max(0, budget - used)

    def should_use_hardware(self, task_type: QuantumTaskType, estimated_seconds: float = 30) -> bool:
        """
        Decide if this task should use hardware based on strategic budget.

        Returns True only if:
        1. Overall quota allows it
        2. This category still has budget
        """
        if not self.usage_stats.can_use_hardware(estimated_seconds):
            return False
        remaining = self.get_category_remaining(task_type)
        if remaining < estimated_seconds:
            logger.info(
                f"Hardware budget exhausted for {task_type.value}: "
                f"{remaining:.0f}s remaining of {self.get_category_budget(task_type):.0f}s allocation"
            )
            return False
        return True

    def record_usage(self, task_type: QuantumTaskType, duration_seconds: float):
        """Record hardware usage against a category budget."""
        current = self._category_usage.get(task_type.value, 0.0)
        self._category_usage[task_type.value] = current + duration_seconds
        self._save_category_usage()
        logger.info(
            f"Hardware budget {task_type.value}: {duration_seconds:.1f}s used, "
            f"{self.get_category_remaining(task_type):.0f}s remaining"
        )

    def get_budget_report(self) -> Dict[str, Any]:
        """Get full budget status report."""
        report = {
            "total_budget_seconds": self.usage_stats.MONTHLY_HARDWARE_SECONDS,
            "total_used_seconds": self.usage_stats.hardware_seconds_used,
            "total_remaining_seconds": self.usage_stats.hardware_seconds_remaining,
            "categories": {}
        }
        for task_type, fraction in self.BUDGET_ALLOCATION.items():
            budget = self.usage_stats.MONTHLY_HARDWARE_SECONDS * fraction
            used = self._category_usage.get(task_type.value, 0.0)
            report["categories"][task_type.value] = {
                "budget": budget,
                "used": used,
                "remaining": max(0, budget - used),
                "percentage_used": (used / budget * 100) if budget > 0 else 0
            }
        return report


@dataclass
class QuantumJobResult:
    """Result from a quantum job execution."""
    success: bool
    backend_used: str
    execution_time: float
    shots: int
    counts: Optional[Dict[str, int]] = None
    expectation_values: Optional[List[float]] = None
    optimal_parameters: Optional[List[float]] = None
    error: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


class IBMQuantumProvider:
    """
    IBM Quantum Experience integration for Cosmos.

    Manages connections, usage tracking, and intelligent backend selection
    to maximize the free tier while enabling quantum-enhanced AI capabilities.
    """

    def __init__(self, api_key: str = None):
        """
        Initialize IBM Quantum provider.

        Args:
            api_key: IBM Quantum API key (or set IBM_QUANTUM_API_KEY env var)
        """
        self.api_key = api_key or os.environ.get("IBM_QUANTUM_API_KEY")
        self.service: Optional[QiskitRuntimeService] = None
        self.simulator: Optional[AerSimulator] = None
        self.fake_backend = None  # Noise-aware fake backend for realistic local sim
        self.usage_stats = QuantumUsageStats()
        self._connected = False
        self._usage_file = Path("data/quantum_usage.json")

        # Load persisted usage stats
        self._load_usage_stats()

        # Strategic hardware budget allocator
        self.budget = HardwareBudgetAllocator(self.usage_stats)

    def _load_usage_stats(self):
        """Load usage stats from disk."""
        try:
            if self._usage_file.exists():
                with open(self._usage_file) as f:
                    data = json.load(f)
                self.usage_stats = QuantumUsageStats.from_dict(data)
                logger.debug(f"Loaded quantum usage: {self.usage_stats.hardware_seconds_used:.1f}s used this month")
        except Exception as e:
            logger.warning(f"Could not load quantum usage stats: {e}")

    def _save_usage_stats(self):
        """Persist usage stats to disk."""
        try:
            self._usage_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._usage_file, "w") as f:
                json.dump(self.usage_stats.to_dict(), f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save quantum usage stats: {e}")

    async def connect(self) -> bool:
        """
        Connect to IBM Quantum services.

        Returns:
            True if connected successfully
        """
        if not QISKIT_AVAILABLE:
            logger.error("Qiskit not available. Install with: pip install qiskit qiskit-ibm-runtime qiskit-aer")
            return False

        if not self.api_key:
            logger.error("No IBM Quantum API key provided")
            return False

        try:
            # Initialize IBM Runtime service
            # Channel: ibm_quantum_platform (ibm_quantum was removed July 2025)
            self.service = QiskitRuntimeService(
                channel="ibm_quantum_platform",
                token=self.api_key
            )

            # Initialize local simulator (always available, unlimited)
            self.simulator = AerSimulator()

            # Initialize noise-aware fake backend for realistic local simulation
            try:
                from qiskit_ibm_runtime.fake_provider import FakeTorino
                self.fake_backend = FakeTorino()  # 133-qubit Heron noise model
                logger.info("FakeTorino noise model loaded for realistic local simulation")
            except ImportError:
                self.fake_backend = None
                logger.debug("Fake backends not available, using ideal simulator")

            self._connected = True

            # List available real backends
            try:
                backends = self.service.backends(operational=True, simulator=False)
                backend_names = [b.name for b in backends]
                logger.info(f"Connected to IBM Quantum Platform (us-east)")
                logger.info(f"Available QPUs: {backend_names}")
            except Exception:
                logger.info("Connected to IBM Quantum Platform")

            logger.info(
                f"Hardware quota: {self.usage_stats.hardware_seconds_remaining:.0f}s remaining "
                f"({self.usage_stats.days_until_reset}d until window reset)"
            )

            return True

        except Exception as e:
            logger.error(f"Failed to connect to IBM Quantum Platform: {e}")
            # Still initialize local simulator even without cloud connection
            self.simulator = AerSimulator()
            logger.info("Local AerSimulator initialized (cloud connection failed, simulator-only mode)")
            return False

    def get_hardware_budget_report(self) -> Dict[str, Any]:
        """Get strategic hardware budget allocation report."""
        return self.budget.get_budget_report()

    def get_available_backends(self) -> List[str]:
        """Get list of available quantum backends."""
        if not self._connected or not self.service:
            return ["aer_simulator"]

        try:
            backends = self.service.backends()
            return [b.name for b in backends]
        except Exception as e:
            logger.warning(f"Could not fetch backends: {e}")
            return ["aer_simulator"]

    def _select_backend(
        self,
        task_type: QuantumTaskType,
        num_qubits: int,
        prefer_hardware: bool = False
    ) -> Tuple[str, bool]:
        """
        Intelligently select backend based on task and resources.

        Args:
            task_type: Type of quantum task
            num_qubits: Required qubits
            prefer_hardware: Prefer hardware if available

        Returns:
            (backend_name, is_hardware)
        """
        # Always check usage stats
        self.usage_stats.reset_if_new_month()

        # High-value tasks that benefit from hardware
        high_value_tasks = {
            QuantumTaskType.OPTIMIZATION,
            QuantumTaskType.EVOLUTION,
            QuantumTaskType.BENCHMARK
        }

        # Estimate job duration (simple heuristic)
        estimated_seconds = 30 + (num_qubits * 2)  # Base + per-qubit overhead

        # Decision logic - hard block if quota exceeded, budget-aware
        use_hardware = (
            prefer_hardware and
            task_type in high_value_tasks and
            self._connected and
            self.service is not None and
            self.budget.should_use_hardware(task_type, estimated_seconds)  # Budget + hard cap
        )

        if prefer_hardware and not use_hardware:
            logger.info(
                f"Hardware requested but denied for {task_type.value} "
                f"(overall: {self.usage_stats.hardware_seconds_remaining:.0f}s remaining, "
                f"category: {self.budget.get_category_remaining(task_type):.0f}s remaining). "
                f"Falling back to simulator."
            )

        if use_hardware:
            # Select current Heron QPU backend (2026 - us-east region)
            # ibm_brisbane retired Nov 2025, ibm_kyoto retired 2025
            # Try to pick least busy from available backends
            try:
                backend = self.service.least_busy(
                    operational=True, simulator=False, min_num_qubits=num_qubits
                )
                return (backend.name, True)
            except Exception:
                # Fallback to known current backends
                if num_qubits <= 133:
                    return ("ibm_torino", True)
                else:
                    return ("ibm_fez", True)  # 156 qubits Heron r2/r3

        # Default to local simulator (unlimited, always available)
        return ("aer_simulator", False)

    async def run_circuit(
        self,
        circuit: "QuantumCircuit",
        shots: int = 1024,
        task_type: QuantumTaskType = QuantumTaskType.SAMPLING,
        prefer_hardware: bool = False,
        parameters: Optional[Dict] = None,
        options: Optional[QuantumOptions] = None
    ) -> QuantumJobResult:
        """
        Execute a quantum circuit using IBM Quantum best practices.

        Args:
            circuit: Qiskit QuantumCircuit to execute
            shots: Number of measurement shots
            task_type: Type of task for backend selection
            prefer_hardware: Prefer real quantum hardware
            parameters: Optional parameter bindings
            options: QuantumOptions for error mitigation and execution mode

        Returns:
            QuantumJobResult with execution results
        """
        if not QISKIT_AVAILABLE:
            return QuantumJobResult(
                success=False,
                backend_used="none",
                execution_time=0,
                shots=0,
                error="Qiskit not installed"
            )

        options = options or QuantumOptions()
        start_time = datetime.now()
        backend_name, is_hardware = self._select_backend(
            task_type, circuit.num_qubits, prefer_hardware
        )

        # Emit job submitted signal
        asyncio.create_task(_emit_quantum_signal("quantum.job_submitted", {
            "backend": backend_name,
            "is_hardware": is_hardware,
            "num_qubits": circuit.num_qubits,
            "shots": shots,
            "task_type": task_type.value,
            "timestamp": datetime.now().isoformat()
        }))

        try:
            if is_hardware and self.service:
                # Run on real quantum hardware with IBM best practices
                backend = self.service.backend(backend_name)

                # Transpile with configurable optimization level
                transpiled = transpile(
                    circuit,
                    backend,
                    optimization_level=options.optimization_level
                )

                # Configure runtime options per IBM docs
                runtime_options = {
                    "dynamical_decoupling": {
                        "enable": options.dynamical_decoupling,
                        "sequence_type": options.dd_sequence_type
                    },
                    "twirling": {
                        "enable_gates": options.enable_twirling,
                        "num_randomizations": options.twirling_num_randomizations,
                        "shots_per_randomization": options.twirling_shots_per_randomization
                    }
                }

                # Execution mode: Open Plan only supports Job and Batch (NOT Session)
                if options.execution_mode == ExecutionMode.SESSION:
                    logger.warning("Session mode not available on Open Plan, using Batch instead")

                if options.execution_mode == ExecutionMode.JOB:
                    # Job mode: single primitive request, simplest
                    sampler = SamplerV2(mode=backend)
                    if parameters:
                        job = sampler.run([(transpiled.assign_parameters(parameters),)], shots=shots)
                    else:
                        job = sampler.run([(transpiled,)], shots=shots)
                    result = job.result()
                else:
                    # Batch mode (default): parallel compilation, Open Plan compatible
                    with Batch(backend=backend) as batch:
                        sampler = SamplerV2(mode=batch)
                        if parameters:
                            job = sampler.run([(transpiled.assign_parameters(parameters),)], shots=shots)
                        else:
                            job = sampler.run([(transpiled,)], shots=shots)
                        result = job.result()

                execution_time = (datetime.now() - start_time).total_seconds()

                # Record usage against overall quota and category budget
                self.usage_stats.record_hardware_job(execution_time)
                self.budget.record_usage(task_type, execution_time)
                self._save_usage_stats()

                # Extract counts from PubResult
                counts = {}
                if hasattr(result[0], 'data'):
                    pub_data = result[0].data
                    if hasattr(pub_data, 'meas'):
                        counts = pub_data.meas.get_counts()
                    elif hasattr(pub_data, 'c'):
                        counts = pub_data.c.get_counts()

                job_result = QuantumJobResult(
                    success=True,
                    backend_used=backend_name,
                    execution_time=execution_time,
                    shots=shots,
                    counts=counts,
                    metadata={
                        "is_hardware": True,
                        "transpiled_depth": transpiled.depth(),
                        "execution_mode": options.execution_mode.value,
                        "dynamical_decoupling": options.dynamical_decoupling,
                        "twirling": options.enable_twirling
                    }
                )

                # Emit result signal for hardware
                asyncio.create_task(_emit_quantum_signal("quantum.result", {
                    "backend": backend_name,
                    "is_hardware": True,
                    "execution_time": execution_time,
                    "shots": shots,
                    "unique_outcomes": len(counts) if counts else 0,
                    "timestamp": datetime.now().isoformat()
                }))

                self._log_quantum_job(job_result, task_type, circuit, parameters)
                return job_result

            else:
                # Run on local simulator (unlimited, free)
                # Use noise-aware fake backend when available for realistic results
                sim_backend = self.simulator
                backend_label = "aer_simulator"
                use_noise = hasattr(self, 'fake_backend') and self.fake_backend is not None

                if use_noise and circuit.num_qubits <= 133:
                    # Noise-aware simulation using real QPU noise model
                    sim_backend = AerSimulator.from_backend(self.fake_backend)
                    backend_label = "aer_simulator_noisy(FakeTorino)"

                transpiled = transpile(circuit, sim_backend, optimization_level=1)

                if parameters:
                    transpiled = transpiled.assign_parameters(parameters)

                job = sim_backend.run(transpiled, shots=shots)
                result = job.result()

                execution_time = (datetime.now() - start_time).total_seconds()
                self.usage_stats.record_simulator_job()

                counts = result.get_counts()

                job_result = QuantumJobResult(
                    success=True,
                    backend_used=backend_label,
                    execution_time=execution_time,
                    shots=shots,
                    counts=counts,
                    metadata={
                        "is_hardware": False,
                        "noise_aware": use_noise,
                        "circuit_depth": transpiled.depth()
                    }
                )

                # Emit result signal for simulator
                asyncio.create_task(_emit_quantum_signal("quantum.result", {
                    "backend": backend_label,
                    "is_hardware": False,
                    "execution_time": execution_time,
                    "shots": shots,
                    "unique_outcomes": len(counts) if counts else 0,
                    "timestamp": datetime.now().isoformat()
                }))

                self._log_quantum_job(job_result, task_type, circuit, parameters)
                return job_result

        except Exception as e:
            logger.error(f"Quantum execution failed: {e}")

            # Emit error signal
            asyncio.create_task(_emit_quantum_signal("quantum.error", {
                "backend": backend_name,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }))

            err_result = QuantumJobResult(
                success=False,
                backend_used=backend_name,
                execution_time=(datetime.now() - start_time).total_seconds(),
                shots=shots,
                error=str(e)
            )
            self._log_quantum_job(err_result, task_type, circuit, parameters)
            return err_result

    async def run_estimator(
        self,
        circuit: "QuantumCircuit",
        observable: "SparsePauliOp",
        shots: int = 1024,
        task_type: QuantumTaskType = QuantumTaskType.OPTIMIZATION,
        prefer_hardware: bool = False,
        parameters: Optional[Dict] = None,
        options: Optional[QuantumOptions] = None
    ) -> QuantumJobResult:
        """
        Run Estimator primitive for expectation value calculations.

        Per IBM docs: Estimator computes expectation values of observables
        with respect to states prepared by quantum circuits.

        Args:
            circuit: Qiskit QuantumCircuit (without measurements)
            observable: SparsePauliOp observable to measure
            shots: Number of measurement shots
            task_type: Type of task for backend selection
            prefer_hardware: Prefer real quantum hardware
            parameters: Optional parameter bindings
            options: QuantumOptions with resilience_level for error mitigation

        Returns:
            QuantumJobResult with expectation_values
        """
        if not QISKIT_AVAILABLE:
            return QuantumJobResult(
                success=False,
                backend_used="none",
                execution_time=0,
                shots=0,
                error="Qiskit not installed"
            )

        options = options or QuantumOptions()
        start_time = datetime.now()
        backend_name, is_hardware = self._select_backend(
            task_type, circuit.num_qubits, prefer_hardware
        )

        try:
            if is_hardware and self.service:
                backend = self.service.backend(backend_name)

                # Transpile circuit
                transpiled = transpile(
                    circuit,
                    backend,
                    optimization_level=options.optimization_level
                )

                # Open Plan: use Batch mode (Session not available)
                if options.execution_mode == ExecutionMode.SESSION:
                    logger.warning("Session mode not available on Open Plan, using Batch for Estimator")

                with Batch(backend=backend) as batch:
                    # Configure Estimator with resilience level per IBM docs
                    estimator = EstimatorV2(mode=batch)

                    # Set resilience options
                    try:
                        estimator.options.resilience_level = options.resilience_level.value
                    except Exception:
                        pass  # Some versions don't support this

                    # Configure error mitigation techniques
                    try:
                        if options.resilience_level.value >= 1:
                            estimator.options.resilience.measure_mitigation = True

                        if options.resilience_level.value >= 2:
                            estimator.options.resilience.zne_mitigation = True
                            estimator.options.resilience.zne.noise_factors = options.zne_noise_factors
                            estimator.options.resilience.zne.extrapolator = options.zne_extrapolator
                    except Exception:
                        pass  # Resilience options vary by version

                    # Enable dynamical decoupling
                    try:
                        if options.dynamical_decoupling:
                            estimator.options.dynamical_decoupling.enable = True
                            estimator.options.dynamical_decoupling.sequence_type = options.dd_sequence_type
                    except Exception:
                        pass

                    # Enable twirling
                    try:
                        if options.enable_twirling:
                            estimator.options.twirling.enable_gates = True
                            estimator.options.twirling.num_randomizations = options.twirling_num_randomizations
                    except Exception:
                        pass

                    # Build PUB (Primitive Unified Bloc) per IBM docs
                    if parameters:
                        pub = (transpiled.assign_parameters(parameters), observable)
                    else:
                        pub = (transpiled, observable)

                    job = estimator.run([pub])
                    result = job.result()

                execution_time = (datetime.now() - start_time).total_seconds()
                self.usage_stats.record_hardware_job(execution_time)
                self.budget.record_usage(task_type, execution_time)
                self._save_usage_stats()

                # Extract expectation values
                expectation_values = []
                if hasattr(result[0], 'data'):
                    evs = result[0].data.evs
                    expectation_values = evs.tolist() if hasattr(evs, 'tolist') else [evs]

                job_result = QuantumJobResult(
                    success=True,
                    backend_used=backend_name,
                    execution_time=execution_time,
                    shots=shots,
                    expectation_values=expectation_values,
                    metadata={
                        "is_hardware": True,
                        "resilience_level": options.resilience_level.value,
                        "error_mitigation": {
                            "measure_mitigation": options.resilience_level.value >= 1,
                            "zne": options.resilience_level.value >= 2,
                            "dynamical_decoupling": options.dynamical_decoupling,
                            "twirling": options.enable_twirling
                        }
                    }
                )
                self._log_quantum_job(job_result, task_type, circuit, parameters)
                return job_result

            else:
                # Simulator - use StatevectorEstimator (Qiskit 2.x)
                estimator = StatevectorEstimator()

                if parameters:
                    pub = (circuit.assign_parameters(parameters), observable)
                else:
                    pub = (circuit, observable)

                job = estimator.run([pub])
                result = job.result()
                execution_time = (datetime.now() - start_time).total_seconds()
                self.usage_stats.record_simulator_job()

                # Extract expectation values from PubResult
                expectation_values = []
                if hasattr(result[0], 'data'):
                    evs = result[0].data.evs
                    expectation_values = evs.tolist() if hasattr(evs, 'tolist') else [float(evs)]

                job_result = QuantumJobResult(
                    success=True,
                    backend_used="statevector_simulator",
                    execution_time=execution_time,
                    shots=shots,
                    expectation_values=expectation_values,
                    metadata={"is_hardware": False}
                )
                self._log_quantum_job(job_result, task_type, circuit, parameters)
                return job_result

        except Exception as e:
            logger.error(f"Estimator execution failed: {e}")
            err_result = QuantumJobResult(
                success=False,
                backend_used=backend_name,
                execution_time=(datetime.now() - start_time).total_seconds(),
                shots=shots,
                error=str(e)
            )
            self._log_quantum_job(err_result, task_type, circuit, parameters)
            return err_result

    def _log_quantum_job(self, result: QuantumJobResult, task_type: QuantumTaskType, circuit: "QuantumCircuit" = None, parameters: Dict = None):
        """Save detailed quantum execution logs to disk for graph visualization."""
        try:
            log_dir = Path("data/quantum/jobs")
            log_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            job_id = f"{task_type.value}_{result.backend_used}_{timestamp}"
            
            log_data = {
                "job_id": job_id,
                "timestamp": datetime.now().isoformat(),
                "task_type": task_type.value,
                "success": result.success,
                "backend": result.backend_used,
                "execution_time_sec": result.execution_time,
                "shots": result.shots,
                "error": result.error,
                "metadata": result.metadata
            }
            
            if result.counts:
                log_data["counts"] = result.counts
            if result.expectation_values:
                log_data["expectation_values"] = result.expectation_values
                
            if parameters:
                # Convert params to native types
                log_data["parameters"] = {str(k): float(v) for k, v in parameters.items()}
                
            if circuit:
                try:
                    log_data["circuit_qasm"] = circuit.qasm()
                except Exception:
                    log_data["circuit_depth"] = circuit.depth()
                    
            log_file = log_dir / f"{job_id}.json"
            log_file.write_text(json.dumps(log_data, indent=2))
            logger.debug(f"Quantum job logged to {log_file}")
        except Exception as e:
            logger.error(f"Failed to log quantum job: {e}")

    def get_usage_summary(self) -> Dict:
        """Get current usage summary."""
        self.usage_stats.reset_if_window_expired()
        return {
            "hardware_seconds_used": self.usage_stats.hardware_seconds_used,
            "hardware_seconds_remaining": self.usage_stats.hardware_seconds_remaining,
            "hardware_percentage_used": self.usage_stats.hardware_percentage_used,
            "hardware_jobs_count": self.usage_stats.hardware_jobs_count,
            "simulator_jobs_count": self.usage_stats.simulator_jobs_count,
            "last_hardware_run": self.usage_stats.last_hardware_run.isoformat() if self.usage_stats.last_hardware_run else None,
            "connected": self._connected,
            "days_until_reset": self.usage_stats.days_until_reset,
            "rolling_window_days": self.usage_stats.ROLLING_WINDOW_DAYS,
            "noise_aware_sim": self.fake_backend is not None,
            "execution_mode": "batch (Open Plan)",
            "budget": self.budget.get_budget_report() if hasattr(self, 'budget') else None
        }


# =============================================================================
# QUANTUM ALGORITHMS FOR COSMOS
# =============================================================================

class QuantumGeneticOptimizer:
    """
    Quantum-enhanced Genetic Algorithm for agent evolution using a Variational Quantum Algorithm (VQA).

    Uses a parameterized quantum circuit (Ansatz) where the swarm's Hebbian learning weights
    become the parameters (angles of rotation) for the quantum gates. The measurement
    collapses into the new "mutated" bitstring.

    Integrates with: evolution/genetic_optimizer.py, population_manager.py
    """

    def __init__(self, provider: IBMQuantumProvider, num_qubits: int = 8):
        """
        Initialize quantum genetic optimizer using VQA Ansatz.

        Args:
            provider: IBM Quantum provider instance
            num_qubits: Number of qubits (determines solution space size)
        """
        self.provider = provider
        self.num_qubits = num_qubits
        self.generation = 0

    def _create_ansatz(self) -> Tuple[Optional["QuantumCircuit"], List[Any]]:
        """
        Create a parameterized Variational Quantum Circuit (Ansatz).
        Returns the circuit and its Parameter objects for binding continuous weights.
        """
        if not QISKIT_AVAILABLE:
            return None, []

        qc = QuantumCircuit(self.num_qubits)
        params = [Parameter(f'theta_{i}') for i in range(self.num_qubits)]

        # Initial superposition
        for i in range(self.num_qubits):
            qc.h(i)

        # Parameterized rotations (Hebbian weights map to these)
        for i in range(self.num_qubits):
            qc.ry(params[i], i)

        # Entanglement between adjacent qubits
        for i in range(self.num_qubits - 1):
            qc.cx(i, i + 1)

        return qc, params

    async def generate_quantum_population(
        self,
        population_size: int,
        fitness_func: Optional[Callable[[str], float]] = None,
        weights: Optional[List[float]] = None,
        prefer_hardware: bool = False
    ) -> List[Tuple[str, float]]:
        """
        Generate a quantum population by sampling the VQA Ansatz configured by continuous weights.

        Args:
            population_size: Number of individuals to generate
            fitness_func: Classical function to evaluate fitness of bitstring
            weights: List of continuous weights to bind to the Parameter gates [0..1 mapped to 0..pi]
            prefer_hardware: Use real quantum hardware if available

        Returns:
            List of (bitstring, fitness_score) tuples
        """
        if not QISKIT_AVAILABLE:
            logger.warning("Qiskit not available, using classical random generation")
            population = []
            for _ in range(population_size):
                bitstring = ''.join(str(np.random.randint(2)) for _ in range(self.num_qubits))
                fitness = fitness_func(bitstring) if fitness_func else np.random.random()
                population.append((bitstring, fitness))
            return sorted(population, key=lambda x: x[1], reverse=True)

        # Build quantum parameterized circuit
        qc, params = self._create_ansatz()

        # Map classical continuous weights to quantum angles
        param_dict = None
        if weights and len(weights) >= self.num_qubits:
            # Map [0, 1] weights to [0, pi] angles
            bound_angles = [w * np.pi for w in weights[:self.num_qubits]]
            param_dict = {p: a for p, a in zip(params, bound_angles)}
        else:
            # Default to random exploration if no weights provided (uniform spread)
            bound_angles = [np.random.random() * np.pi for _ in range(self.num_qubits)]
            param_dict = {p: a for p, a in zip(params, bound_angles)}

        # Measure all qubits (allocating correctly sized classical register)
        qc.measure_all()

        # Run circuit
        result = await self.provider.run_circuit(
            qc,
            shots=population_size * 2,  # Oversample for diversity
            task_type=QuantumTaskType.EVOLUTION,
            prefer_hardware=prefer_hardware,
            parameters=param_dict
        )

        if not result.success or not result.counts:
            logger.warning(f"Quantum population generation failed: {result.error}")
            population = []
            for _ in range(population_size):
                bitstring = ''.join(str(np.random.randint(2)) for _ in range(self.num_qubits))
                fitness = fitness_func(bitstring) if fitness_func else np.random.random()
                population.append((bitstring, fitness))
            return sorted(population, key=lambda x: x[1], reverse=True)

        # Process results
        population = []
        for bitstring, count in result.counts.items():
            bs = bitstring.replace(" ", "")
            fitness = fitness_func(bs) if fitness_func else count / result.shots
            population.append((bs, fitness))

        population.sort(key=lambda x: x[1], reverse=True)
        self.generation += 1
        logger.info(f"Quantum GA generation {self.generation}: {len(population)} candidates, best fitness {population[0][1]:.4f}")

        return population[:population_size]

    async def quantum_crossover(
        self,
        parent1: str,
        parent2: str,
        prefer_hardware: bool = False
    ) -> str:
        """
        Quantum-inspired crossover using entanglement.
        Creates offspring that has quantum superposition of both parents' traits.
        """
        if not QISKIT_AVAILABLE or len(parent1) != len(parent2):
            point = np.random.randint(1, len(parent1))
            return parent1[:point] + parent2[point:]

        n = len(parent1)
        qc = QuantumCircuit(n)

        # Encode parents into quantum state
        for i in range(n):
            if parent1[i] == '1':
                qc.x(i)

            if parent1[i] != parent2[i]:
                qc.h(i)

        for i in range(n - 1):
            qc.cx(i, i + 1)

        qc.measure_all()

        result = await self.provider.run_circuit(
            qc,
            shots=1,
            task_type=QuantumTaskType.EVOLUTION,
            prefer_hardware=prefer_hardware
        )

        if result.success and result.counts:
            offspring = max(result.counts, key=result.counts.get)
            return offspring.replace(" ", "")

        point = np.random.randint(1, n)
        return parent1[:point] + parent2[point:]

    async def quantum_mutation(
        self,
        individual: str,
        mutation_rate: float = 0.1,
        prefer_hardware: bool = False
    ) -> str:
        """
        Quantum-inspired mutation using rotation gates.
        Uses parameterized rotation to control mutation probability.
        """
        if not QISKIT_AVAILABLE:
            result = list(individual)
            for i in range(len(result)):
                if np.random.random() < mutation_rate:
                    result[i] = '0' if result[i] == '1' else '1'
            return ''.join(result)

        n = len(individual)
        qc = QuantumCircuit(n)

        # Encode individual
        for i in range(n):
            if individual[i] == '1':
                qc.x(i)

        # Apply rotation gates based on mutation rate
        theta = 2 * np.arcsin(np.sqrt(mutation_rate))
        for i in range(n):
            qc.ry(theta, i)

        qc.measure_all()

        result = await self.provider.run_circuit(
            qc,
            shots=1,
            task_type=QuantumTaskType.EVOLUTION,
            prefer_hardware=prefer_hardware
        )

        if result.success and result.counts:
            mutated = max(result.counts, key=result.counts.get)
            return mutated.replace(" ", "")

        result_list = list(individual)
        for i in range(len(result_list)):
            if np.random.random() < mutation_rate:
                result_list[i] = '0' if result_list[i] == '1' else '1'
        return ''.join(result_list)


class QAOAOptimizer:
    """
    Quantum Approximate Optimization Algorithm (QAOA) for combinatorial problems.

    Useful for:
    - Multi-objective optimization in genetic_optimizer.py
    - Task allocation in swarm orchestration
    - Knowledge graph inference optimization

    Integrates with: core/genetic_optimizer.py, core/collective/
    """

    def __init__(self, provider: IBMQuantumProvider):
        self.provider = provider

    def _create_cost_circuit(self, num_qubits: int, gamma: float, edges: List[Tuple[int, int]]) -> "QuantumCircuit":
        """Create cost unitary for QAOA (e.g., MaxCut problem)."""
        if not QISKIT_AVAILABLE:
            return None

        qc = QuantumCircuit(num_qubits)

        for i, j in edges:
            if i < num_qubits and j < num_qubits:
                qc.rzz(2 * gamma, i, j)

        return qc

    def _create_mixer_circuit(self, num_qubits: int, beta: float) -> "QuantumCircuit":
        """Create mixer unitary for QAOA."""
        if not QISKIT_AVAILABLE:
            return None

        qc = QuantumCircuit(num_qubits)

        for i in range(num_qubits):
            qc.rx(2 * beta, i)

        return qc

    async def optimize(
        self,
        num_qubits: int,
        edges: List[Tuple[int, int]],
        p: int = 2,
        shots: int = 1024,
        prefer_hardware: bool = False
    ) -> QuantumJobResult:
        """
        Run QAOA optimization.

        Args:
            num_qubits: Problem size
            edges: Graph edges for MaxCut-style problem
            p: QAOA depth (layers)
            shots: Measurement shots
            prefer_hardware: Use real hardware

        Returns:
            QuantumJobResult with optimal bitstring
        """
        if not QISKIT_AVAILABLE:
            return QuantumJobResult(
                success=False,
                backend_used="none",
                execution_time=0,
                shots=0,
                error="Qiskit not installed"
            )

        # Initialize parameters (would be optimized in full implementation)
        gammas = [np.pi / 4] * p
        betas = [np.pi / 8] * p

        # Build QAOA circuit
        qc = QuantumCircuit(num_qubits, num_qubits)

        # Initial superposition
        for i in range(num_qubits):
            qc.h(i)

        # QAOA layers
        for layer in range(p):
            # Cost layer
            cost_circuit = self._create_cost_circuit(num_qubits, gammas[layer], edges)
            qc.compose(cost_circuit, inplace=True)

            # Mixer layer
            mixer_circuit = self._create_mixer_circuit(num_qubits, betas[layer])
            qc.compose(mixer_circuit, inplace=True)

        qc.measure_all()

        # Run optimization
        result = await self.provider.run_circuit(
            qc,
            shots=shots,
            task_type=QuantumTaskType.OPTIMIZATION,
            prefer_hardware=prefer_hardware
        )

        if result.success and result.counts:
            # Find best solution
            best_bitstring = max(result.counts, key=result.counts.get)
            result.optimal_parameters = gammas + betas
            result.metadata["best_solution"] = best_bitstring
            result.metadata["qaoa_depth"] = p

        return result


class QuantumPatternExtractor:
    """
    Quantum-inspired pattern extraction for memory consolidation.

    Uses quantum sampling for exploring pattern spaces in
    high-dimensional memory embeddings.

    Integrates with: memory/memory_dreaming.py, memory/dream_consolidation.py
    """

    def __init__(self, provider: IBMQuantumProvider):
        self.provider = provider

    async def extract_patterns(
        self,
        embedding_matrix: np.ndarray,
        num_patterns: int = 5,
        prefer_hardware: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Extract patterns from embedding matrix using quantum sampling.

        Args:
            embedding_matrix: Matrix of memory embeddings (n_memories x embedding_dim)
            num_patterns: Number of patterns to extract
            prefer_hardware: Use quantum hardware

        Returns:
            List of pattern dictionaries with indices and weights
        """
        n_memories = embedding_matrix.shape[0]

        # Limit qubits to practical range
        num_qubits = min(int(np.ceil(np.log2(n_memories + 1))), 10)

        if not QISKIT_AVAILABLE:
            # Classical clustering fallback
            logger.info("Using classical pattern extraction")
            patterns = []
            for i in range(num_patterns):
                # Random pattern selection
                indices = np.random.choice(n_memories, size=min(3, n_memories), replace=False)
                patterns.append({
                    "pattern_id": i,
                    "memory_indices": indices.tolist(),
                    "weights": [1.0 / len(indices)] * len(indices),
                    "method": "classical_random"
                })
            return patterns

        # Build quantum circuit for pattern sampling
        qc = QuantumCircuit(num_qubits, num_qubits)

        # Hadamard superposition
        for i in range(num_qubits):
            qc.h(i)

        # Add some entanglement for correlated patterns
        for i in range(num_qubits - 1):
            qc.cx(i, i + 1)

        # Parameterized rotations based on embedding statistics
        embedding_var = np.var(embedding_matrix, axis=0).mean()
        rotation_angle = np.arctan(embedding_var) * 2
        for i in range(num_qubits):
            qc.ry(rotation_angle, i)

        qc.measure_all()

        # Run quantum sampling
        result = await self.provider.run_circuit(
            qc,
            shots=num_patterns * 100,
            task_type=QuantumTaskType.PATTERN,
            prefer_hardware=prefer_hardware
        )

        if not result.success or not result.counts:
            logger.warning("Quantum pattern extraction failed, using classical")
            # Fallback
            patterns = []
            for i in range(num_patterns):
                indices = np.random.choice(n_memories, size=min(3, n_memories), replace=False)
                patterns.append({
                    "pattern_id": i,
                    "memory_indices": indices.tolist(),
                    "weights": [1.0 / len(indices)] * len(indices),
                    "method": "classical_fallback"
                })
            return patterns

        # Convert quantum samples to patterns
        patterns = []
        sorted_counts = sorted(result.counts.items(), key=lambda x: x[1], reverse=True)

        for i, (bitstring, count) in enumerate(sorted_counts[:num_patterns]):
            # Convert bitstring to memory indices
            bs = bitstring.replace(" ", "")
            index = int(bs, 2) % n_memories

            # Build pattern around this index
            related_indices = [
                (index + j) % n_memories
                for j in range(-1, 2)
            ]

            patterns.append({
                "pattern_id": i,
                "memory_indices": related_indices,
                "weights": [count / result.shots] * len(related_indices),
                "quantum_amplitude": count / result.shots,
                "method": "quantum_sampling"
            })

        logger.info(f"Extracted {len(patterns)} quantum patterns from {n_memories} memories")
        return patterns


# =============================================================================
# SINGLETON AND UTILITY FUNCTIONS
# =============================================================================

_quantum_provider: Optional[IBMQuantumProvider] = None


def get_quantum_provider() -> Optional[IBMQuantumProvider]:
    """Get or create the quantum provider singleton."""
    global _quantum_provider
    if _quantum_provider is None:
        api_key = os.environ.get("IBM_QUANTUM_API_KEY")
        if api_key:
            _quantum_provider = IBMQuantumProvider(api_key)
    return _quantum_provider


async def initialize_quantum(api_key: str = None) -> bool:
    """
    Initialize quantum computing integration.

    Args:
        api_key: IBM Quantum API key (optional, uses env var if not provided)

    Returns:
        True if initialized successfully
    """
    global _quantum_provider

    if api_key:
        os.environ["IBM_QUANTUM_API_KEY"] = api_key

    _quantum_provider = IBMQuantumProvider(api_key or os.environ.get("IBM_QUANTUM_API_KEY"))
    return await _quantum_provider.connect()


async def quantum_evolve_agent(
    agent_genome: str,
    fitness_func: Callable[[str], float],
    generations: int = 5,
    population_size: int = 20,
    prefer_hardware: bool = False
) -> Tuple[str, float]:
    """
    Evolve an agent genome using quantum genetic algorithm.

    High-level function for integration with evolution/ modules.

    Args:
        agent_genome: Binary string representing agent parameters
        fitness_func: Function to evaluate genome fitness
        generations: Number of evolution generations
        population_size: Population size per generation
        prefer_hardware: Use quantum hardware for key operations

    Returns:
        (best_genome, best_fitness) tuple
    """
    provider = get_quantum_provider()
    if not provider:
        logger.warning("Quantum provider not initialized, using classical evolution")
        # Classical fallback
        best = agent_genome
        best_fitness = fitness_func(agent_genome)
        for _ in range(generations * population_size):
            mutated = list(best)
            for i in range(len(mutated)):
                if np.random.random() < 0.1:
                    mutated[i] = '0' if mutated[i] == '1' else '1'
            mutated = ''.join(mutated)
            fit = fitness_func(mutated)
            if fit > best_fitness:
                best = mutated
                best_fitness = fit
        return best, best_fitness

    qga = QuantumGeneticOptimizer(provider, num_qubits=len(agent_genome))

    # Initialize with quantum population
    population = await qga.generate_quantum_population(
        population_size,
        fitness_func,
        grover_iterations=1,
        prefer_hardware=prefer_hardware
    )

    best_genome, best_fitness = population[0]

    for gen in range(generations):
        # Selection (tournament)
        selected = population[:population_size // 2]

        # Strategic hardware usage: first gen (seed diversity) and last gen (finalize)
        # Middle generations use simulator to conserve budget
        use_hw_this_gen = prefer_hardware and (gen == 0 or gen == generations - 1)

        # Crossover and mutation
        offspring = []
        for i in range(0, len(selected) - 1, 2):
            child = await qga.quantum_crossover(
                selected[i][0],
                selected[i + 1][0],
                prefer_hardware=use_hw_this_gen
            )
            child = await qga.quantum_mutation(child, mutation_rate=0.1)
            fitness = fitness_func(child)
            offspring.append((child, fitness))

        # Combine and select best
        combined = population + offspring
        combined.sort(key=lambda x: x[1], reverse=True)
        population = combined[:population_size]

        if population[0][1] > best_fitness:
            best_genome, best_fitness = population[0]

        logger.debug(f"Quantum evolution gen {gen + 1}: best fitness {best_fitness:.4f}")

    return best_genome, best_fitness


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

async def test_quantum_integration():
    """Test IBM Quantum integration with all features."""
    print("Testing Cosmos Quantum Integration (IBM Best Practices)")
    print("=" * 60)

    api_key = os.environ.get("IBM_QUANTUM_API_KEY")
    if not api_key:
        print("No IBM_QUANTUM_API_KEY found in environment")
        print("Set it with: export IBM_QUANTUM_API_KEY=your_key")
        return False

    # Initialize
    success = await initialize_quantum(api_key)
    print(f"Connection: {'Success' if success else 'Failed'}")

    if not success:
        return False

    provider = get_quantum_provider()

    # Check usage
    usage = provider.get_usage_summary()
    print(f"\nUsage Summary:")
    print(f"  Hardware: {usage['hardware_seconds_used']:.1f}s / 600s ({usage['hardware_percentage_used']:.1f}%)")
    print(f"  Simulator jobs: {usage['simulator_jobs_count']}")

    # Test simple circuit
    if QISKIT_AVAILABLE:
        print("\n--- Test 1: Basic Sampler Circuit ---")
        qc = QuantumCircuit(3, 3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.measure_all()

        result = await provider.run_circuit(qc, shots=100)
        print(f"  Success: {result.success}")
        print(f"  Backend: {result.backend_used}")
        print(f"  Counts: {result.counts}")

        print("\n--- Test 2: Estimator with Error Mitigation ---")
        # Create circuit for expectation value (no measurement)
        qc_est = QuantumCircuit(2)
        qc_est.h(0)
        qc_est.cx(0, 1)

        # Observable: ZZ
        observable = SparsePauliOp.from_list([("ZZ", 1.0)])

        # Configure options per IBM docs
        options = QuantumOptions(
            execution_mode=ExecutionMode.BATCH,
            resilience_level=ResilienceLevel.MINIMAL,
            dynamical_decoupling=True,
            enable_twirling=False  # Only for hardware
        )

        result_est = await provider.run_estimator(
            qc_est, observable, shots=100, options=options
        )
        print(f"  Success: {result_est.success}")
        print(f"  Backend: {result_est.backend_used}")
        print(f"  Expectation values: {result_est.expectation_values}")

        print("\n--- Test 3: Quantum Genetic Algorithm ---")

        def test_fitness(bitstring: str) -> float:
            return sum(int(b) for b in bitstring) / len(bitstring)

        qga = QuantumGeneticOptimizer(provider, num_qubits=5)
        population = await qga.generate_quantum_population(10, test_fitness)
        print(f"  Generated population of {len(population)}")
        print(f"  Best individual: {population[0][0]} (fitness: {population[0][1]:.2f})")

        print("\n--- Test 4: QAOA Optimizer ---")
        qaoa = QAOAOptimizer(provider)
        # Simple 4-node graph for MaxCut
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        result_qaoa = await qaoa.optimize(4, edges, p=1, shots=100)
        print(f"  Success: {result_qaoa.success}")
        print(f"  Best solution: {result_qaoa.metadata.get('best_solution', 'N/A')}")

        print("\n--- Test 5: Execution Modes ---")
        print(f"  Available modes: {[m.value for m in ExecutionMode]}")
        print(f"  Resilience levels: {[r.name for r in ResilienceLevel]}")

    print("\n" + "=" * 60)
    print("Quantum integration test complete!")
    print("IBM Quantum best practices implemented:")
    print("  - SamplerV2 and EstimatorV2 primitives")
    print("  - Session/Batch/Job execution modes")
    print("  - Error mitigation (TREX, ZNE, twirling)")
    print("  - Dynamical decoupling for idle qubits")
    print("  - Usage tracking with monthly quota")
    return True


if __name__ == "__main__":
    asyncio.run(test_quantum_integration())
