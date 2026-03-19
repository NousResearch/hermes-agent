"""
QUANTUM PROOF - Real IBM Quantum Hardware Execution
====================================================

Submits actual quantum circuits to IBM Quantum hardware.
Jobs appear in IBM Quantum portal under "Past Workloads".

Uses minimal qubit/shot counts to conserve quota while
demonstrating real quantum computation.
"""

import asyncio
import os
from datetime import datetime
from typing import Any, Dict, Optional
from dataclasses import dataclass, field

from loguru import logger


@dataclass
class QuantumJob:
    """A quantum job submitted to IBM hardware."""
    job_id: str
    backend: str
    circuit_name: str
    num_qubits: int
    shots: int
    status: str = "pending"
    created_at: datetime = field(default_factory=datetime.now)
    result: Optional[Dict[str, Any]] = None
    counts: Optional[Dict[str, int]] = None


class QuantumProof:
    """
    Real IBM Quantum hardware integration.

    Submits actual quantum circuits to demonstrate
    Cosmos's quantum capabilities.
    """

    def __init__(self):
        self.api_key = os.getenv("IBM_QUANTUM_API_KEY") or os.getenv("IBM_QUANTUM_TOKEN")
        self.jobs: Dict[str, QuantumJob] = {}

        # Preferred backends - current Heron QPUs (2026, us-east region)
        # RETIRED: ibm_brisbane (Nov 2025), ibm_kyoto, ibm_sherbrooke (July 2025)
        self.preferred_backends = [
            "ibm_fez",           # Heron, 156 qubits
            "ibm_torino",        # Heron, 133 qubits
            "ibm_marrakesh",     # Heron, 156 qubits
        ]

        self._service = None
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize IBM Quantum service."""
        if self._initialized:
            return True

        # Load env if not already loaded
        if not self.api_key:
            from dotenv import load_dotenv
            from pathlib import Path
            env_path = Path(__file__).parent.parent.parent.parent / ".env"
            load_dotenv(env_path)
            self.api_key = os.getenv("IBM_QUANTUM_API_KEY") or os.getenv("IBM_QUANTUM_TOKEN")

        if not self.api_key:
            logger.error("No IBM Quantum API key found")
            return False

        try:
            # Import qiskit runtime
            from qiskit_ibm_runtime import QiskitRuntimeService

            # Save and initialize service
            # Channel: ibm_quantum_platform (ibm_quantum channel removed July 2025)
            QiskitRuntimeService.save_account(
                channel="ibm_quantum_platform",
                token=self.api_key,
                overwrite=True
            )

            self._service = QiskitRuntimeService(channel="ibm_quantum_platform")
            self._initialized = True
            logger.info("IBM Quantum service initialized")
            return True

        except ImportError:
            logger.error("qiskit_ibm_runtime not installed. Run: pip install qiskit-ibm-runtime")
            return False
        except Exception as e:
            logger.error(f"IBM Quantum init error: {e}")
            return False

    def get_least_busy_backend(self) -> Optional[str]:
        """Get the least busy available backend."""
        if not self._service:
            return self.preferred_backends[0]

        try:
            from qiskit_ibm_runtime import QiskitRuntimeService

            # Get available backends
            backends = self._service.backends(
                simulator=False,
                operational=True,
                min_num_qubits=2
            )

            if not backends:
                return self.preferred_backends[0]

            # Find least busy
            least_busy = min(backends, key=lambda b: b.status().pending_jobs)
            return least_busy.name

        except Exception as e:
            logger.debug(f"Backend selection error: {e}")
            return self.preferred_backends[0]

    async def run_bell_state(
        self,
        shots: int = 100,
        backend: Optional[str] = None,
    ) -> QuantumJob:
        """
        Run a Bell state circuit on real hardware.

        Creates maximally entangled state |00⟩ + |11⟩
        Minimal resource usage: 2 qubits, configurable shots.

        This is the "Hello World" of quantum computing.
        """
        if not await self.initialize():
            raise RuntimeError("Failed to initialize IBM Quantum")

        try:
            from qiskit import QuantumCircuit
            from qiskit_ibm_runtime import SamplerV2 as Sampler
            from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

            # Create Bell state circuit
            qc = QuantumCircuit(2, 2)
            qc.h(0)           # Hadamard on qubit 0
            qc.cx(0, 1)       # CNOT: entangle qubits
            qc.measure([0, 1], [0, 1])  # Measure both

            # Get backend
            backend_name = backend or self.get_least_busy_backend()
            hw_backend = self._service.backend(backend_name)

            # Transpile for hardware
            pm = generate_preset_pass_manager(backend=hw_backend, optimization_level=1)
            transpiled = pm.run(qc)

            # Run on real hardware
            sampler = Sampler(hw_backend)
            job = sampler.run([transpiled], shots=shots)

            # Create job record
            quantum_job = QuantumJob(
                job_id=job.job_id(),
                backend=backend_name,
                circuit_name="bell_state",
                num_qubits=2,
                shots=shots,
                status="submitted",
            )

            self.jobs[quantum_job.job_id] = quantum_job
            logger.info(f"Quantum job submitted: {quantum_job.job_id} on {backend_name}")

            return quantum_job

        except Exception as e:
            logger.error(f"Bell state error: {e}")
            raise

    async def run_quantum_random(
        self,
        num_bits: int = 8,
        shots: int = 1,
        backend: Optional[str] = None,
    ) -> QuantumJob:
        """
        Generate true quantum random bits.

        Uses superposition to create genuine randomness
        from quantum measurement collapse.
        """
        if not await self.initialize():
            raise RuntimeError("Failed to initialize IBM Quantum")

        # Limit to reasonable size
        num_bits = min(num_bits, 16)

        try:
            from qiskit import QuantumCircuit
            from qiskit_ibm_runtime import SamplerV2 as Sampler
            from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

            # Create superposition circuit
            qc = QuantumCircuit(num_bits, num_bits)
            qc.h(range(num_bits))  # Hadamard on all qubits
            qc.measure(range(num_bits), range(num_bits))

            # Get backend
            backend_name = backend or self.get_least_busy_backend()
            hw_backend = self._service.backend(backend_name)

            # Transpile and run
            pm = generate_preset_pass_manager(backend=hw_backend, optimization_level=1)
            transpiled = pm.run(qc)

            sampler = Sampler(hw_backend)
            job = sampler.run([transpiled], shots=shots)

            quantum_job = QuantumJob(
                job_id=job.job_id(),
                backend=backend_name,
                circuit_name="quantum_random",
                num_qubits=num_bits,
                shots=shots,
                status="submitted",
            )

            self.jobs[quantum_job.job_id] = quantum_job
            logger.info(f"Quantum random job submitted: {quantum_job.job_id}")

            return quantum_job

        except Exception as e:
            logger.error(f"Quantum random error: {e}")
            raise

    async def run_ghz_state(
        self,
        num_qubits: int = 3,
        shots: int = 100,
        backend: Optional[str] = None,
    ) -> QuantumJob:
        """
        Run a GHZ (Greenberger-Horne-Zeilinger) state circuit.

        Creates multi-qubit entanglement: |000...⟩ + |111...⟩
        Demonstrates quantum advantage over classical correlation.
        """
        if not await self.initialize():
            raise RuntimeError("Failed to initialize IBM Quantum")

        # Limit qubits
        num_qubits = min(max(num_qubits, 3), 10)

        try:
            from qiskit import QuantumCircuit
            from qiskit_ibm_runtime import SamplerV2 as Sampler
            from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

            # Create GHZ circuit
            qc = QuantumCircuit(num_qubits, num_qubits)
            qc.h(0)  # Hadamard on first qubit
            for i in range(num_qubits - 1):
                qc.cx(i, i + 1)  # Chain of CNOTs
            qc.measure(range(num_qubits), range(num_qubits))

            # Get backend
            backend_name = backend or self.get_least_busy_backend()
            hw_backend = self._service.backend(backend_name)

            # Transpile and run
            pm = generate_preset_pass_manager(backend=hw_backend, optimization_level=1)
            transpiled = pm.run(qc)

            sampler = Sampler(hw_backend)
            job = sampler.run([transpiled], shots=shots)

            quantum_job = QuantumJob(
                job_id=job.job_id(),
                backend=backend_name,
                circuit_name="ghz_state",
                num_qubits=num_qubits,
                shots=shots,
                status="submitted",
            )

            self.jobs[quantum_job.job_id] = quantum_job
            logger.info(f"GHZ state job submitted: {quantum_job.job_id}")

            return quantum_job

        except Exception as e:
            logger.error(f"GHZ state error: {e}")
            raise

    async def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of a quantum job."""
        if not await self.initialize():
            return {"error": "Not initialized"}

        try:
            job = self._service.job(job_id)
            status = job.status().name

            # Handle backend - might be string or object
            backend = job.backend()
            backend_name = backend.name if hasattr(backend, 'name') else str(backend)

            result = {
                "job_id": job_id,
                "status": status,
                "backend": backend_name,
                "created": str(job.creation_date),
            }

            # If completed, get results
            if status == "DONE":
                job_result = job.result()
                # Get counts from the first pub result
                pub_result = job_result[0]
                counts = pub_result.data.meas.get_counts()
                result["counts"] = counts
                result["success"] = True

                # Update our record
                if job_id in self.jobs:
                    self.jobs[job_id].status = "completed"
                    self.jobs[job_id].counts = counts

            return result

        except Exception as e:
            logger.error(f"Job status error: {e}")
            return {"job_id": job_id, "error": str(e)}

    async def wait_for_result(
        self,
        job_id: str,
        timeout: float = 300.0,
        poll_interval: float = 5.0,
    ) -> Dict[str, Any]:
        """Wait for a job to complete and return results."""
        import time
        start = time.time()

        while time.time() - start < timeout:
            status = await self.get_job_status(job_id)

            if status.get("status") == "DONE":
                return status
            elif status.get("status") in ["ERROR", "CANCELLED"]:
                return status

            await asyncio.sleep(poll_interval)

        return {"job_id": job_id, "error": "Timeout waiting for result"}

    def get_jobs(self) -> list:
        """Get all submitted jobs."""
        return [
            {
                "job_id": j.job_id,
                "circuit": j.circuit_name,
                "backend": j.backend,
                "qubits": j.num_qubits,
                "shots": j.shots,
                "status": j.status,
                "created": j.created_at.isoformat(),
            }
            for j in self.jobs.values()
        ]


# Global instance
_quantum_proof: Optional[QuantumProof] = None


def get_quantum_proof() -> QuantumProof:
    """Get global QuantumProof instance."""
    global _quantum_proof
    if _quantum_proof is None:
        _quantum_proof = QuantumProof()
    return _quantum_proof


# Convenience functions

async def run_quantum_bell(shots: int = 100) -> Dict[str, Any]:
    """Quick Bell state execution."""
    qp = get_quantum_proof()
    job = await qp.run_bell_state(shots=shots)
    return {
        "job_id": job.job_id,
        "backend": job.backend,
        "circuit": "bell_state",
        "shots": shots,
        "portal_url": f"https://quantum.ibm.com/jobs/{job.job_id}",
    }


async def run_quantum_random_bits(num_bits: int = 8) -> Dict[str, Any]:
    """Quick quantum random generation."""
    qp = get_quantum_proof()
    job = await qp.run_quantum_random(num_bits=num_bits, shots=1)
    return {
        "job_id": job.job_id,
        "backend": job.backend,
        "circuit": "quantum_random",
        "bits": num_bits,
        "portal_url": f"https://quantum.ibm.com/jobs/{job.job_id}",
    }
