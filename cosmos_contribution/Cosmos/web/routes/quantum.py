"""
Quantum Computing & Evolution Routes

Endpoints:
- POST /api/quantum/bell - Run Bell state on IBM Quantum
- GET /api/quantum/job/{job_id} - Quantum job status
- GET /api/quantum/jobs - list quantum jobs
- GET /api/quantum/status - Quantum integration status
- GET /api/quantum/budget - Strategic hardware budget allocation report
- POST /api/quantum/initialize - Initialize quantum connection
- POST /api/quantum/evolve - Quantum genetic evolution
- GET /api/organism/status - Collective organism status
- GET /api/organism/snapshot - Consciousness snapshot
- POST /api/organism/evolve - Trigger organism evolution
- GET /api/orchestrator/status - Swarm orchestrator status
- GET /api/evolution/status - Evolution engine status
- GET /api/evolution/sync - Export evolution data
- POST /api/evolution/evolve - Trigger evolution cycle
"""

import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter()


def _get_shared():
    """Import shared state from server module lazily."""
    from Cosmos.web import server
    return server


# ============================================
# QUANTUM PROOF API - Real IBM Quantum Hardware
# ============================================

@router.post("/api/quantum/bell")
async def quantum_run_bell(shots: int = 20):
    """Run Bell state on REAL IBM Quantum hardware."""
    try:
        from Cosmos.integration.hackathon.quantum_proof import get_quantum_proof

        qp = get_quantum_proof()
        job = await qp.run_bell_state(shots=min(shots, 100))

        return JSONResponse({
            "success": True,
            "job_id": job.job_id,
            "backend": job.backend,
            "circuit": "bell_state",
            "qubits": 2,
            "shots": job.shots,
            "status": job.status,
            "portal_url": f"https://quantum.ibm.com/jobs/{job.job_id}",
            "message": "Job submitted to REAL quantum hardware! Check IBM portal.",
        })
    except Exception as e:
        logger.error(f"Quantum bell error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/quantum/job/{job_id}")
async def quantum_job_status(job_id: str):
    """Get quantum job status and results."""
    try:
        from Cosmos.integration.hackathon.quantum_proof import get_quantum_proof

        qp = get_quantum_proof()
        status = await qp.get_job_status(job_id)

        return JSONResponse({
            "success": True,
            **status,
        })
    except Exception as e:
        logger.error(f"Quantum job status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/quantum/jobs")
async def quantum_list_jobs():
    """list all submitted quantum jobs."""
    try:
        from Cosmos.integration.hackathon.quantum_proof import get_quantum_proof

        qp = get_quantum_proof()
        jobs = qp.get_jobs()

        return JSONResponse({
            "success": True,
            "jobs": jobs,
        })
    except Exception as e:
        logger.error(f"Quantum jobs list error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# QUANTUM COMPUTING INTEGRATION (IBM Quantum Experience)
# ============================================

@router.get("/api/quantum/status")
async def quantum_status():
    """Get IBM Quantum integration status and usage."""
    try:
        from Cosmos.integration.quantum import get_quantum_provider, QISKIT_AVAILABLE

        if not QISKIT_AVAILABLE:
            return JSONResponse({
                "available": False,
                "message": "Qiskit not installed. Run: pip install qiskit qiskit-ibm-runtime qiskit-aer"
            })

        provider = get_quantum_provider()
        if not provider:
            return JSONResponse({
                "available": False,
                "message": "Quantum provider not initialized. Set IBM_QUANTUM_API_KEY environment variable."
            })

        usage = provider.get_usage_summary()
        backends = provider.get_available_backends() if provider._connected else []

        return JSONResponse({
            "available": True,
            "connected": usage["connected"],
            "usage": {
                "hardware_seconds_used": usage["hardware_seconds_used"],
                "hardware_seconds_remaining": usage["hardware_seconds_remaining"],
                "hardware_percentage_used": usage["hardware_percentage_used"],
                "hardware_jobs_count": usage["hardware_jobs_count"],
                "simulator_jobs_count": usage["simulator_jobs_count"],
                "last_hardware_run": usage["last_hardware_run"],
                "days_until_reset": usage.get("days_until_reset", 0),
            },
            "backends": backends[:10],
            "open_plan_limits": {
                "hardware_seconds_per_window": 600,
                "rolling_window_days": 28,
                "cloud_simulators": "RETIRED (May 2024) - use local AerSimulator",
                "local_simulator": "Unlimited (AerSimulator + FakeBackend noise models)",
                "noise_aware_sim": usage.get("noise_aware_sim", False),
                "execution_modes": "Job and Batch only (Session requires paid plan)",
                "region": "us-east",
                "qpu_family": "Heron r1/r2/r3 (133-156 qubits)",
            },
            "strategy": "Local noise-aware simulation (unlimited) + strategic QPU hardware for SAGI breakthroughs",
            "budget": usage.get("budget"),
        })

    except Exception as e:
        logger.error(f"Quantum status error: {e}")
        return JSONResponse({
            "available": False,
            "error": str(e)
        }, status_code=500)


@router.get("/api/quantum/budget")
async def quantum_budget():
    """Get strategic hardware budget allocation report.

    Shows how the 600s/month hardware budget is allocated across
    task categories (evolution, optimization, benchmark, etc.)
    to maximize innovation impact toward SAGI.
    """
    try:
        from Cosmos.integration.quantum import get_quantum_provider

        provider = get_quantum_provider()
        if not provider:
            return JSONResponse({
                "available": False,
                "message": "Quantum provider not initialized"
            })

        report = provider.get_hardware_budget_report()
        report["strategy"] = {
            "evolution": "40% - Real quantum noise drives genuine mutation diversity for SAGI evolution",
            "optimization": "30% - QAOA on hardware finds better optima than simulator",
            "benchmark": "20% - Validate simulator results against real hardware",
            "pattern": "5% - Quantum sampling for memory consolidation",
            "inference": "3% - Knowledge graph queries",
            "sampling": "2% - Probabilistic modeling",
        }
        report["hardware_usage_mode"] = (
            "Strategic: first + last evolution generations use hardware, "
            "middle generations use simulator to conserve budget"
        )

        return JSONResponse(report)

    except Exception as e:
        logger.error(f"Quantum budget error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@router.post("/api/quantum/initialize")
async def quantum_initialize(request: Request):
    """Initialize IBM Quantum connection."""
    try:
        from Cosmos.integration.quantum import initialize_quantum

        body = await request.json() if request.headers.get("content-type") == "application/json" else {}
        api_key = body.get("api_key")

        success = await initialize_quantum(api_key)

        if success:
            return JSONResponse({
                "success": True,
                "message": "Connected to IBM Quantum Experience"
            })
        else:
            return JSONResponse({
                "success": False,
                "message": "Failed to connect. Check API key and Qiskit installation."
            }, status_code=400)

    except Exception as e:
        logger.error(f"Quantum initialization error: {e}")
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)


@router.post("/api/quantum/evolve")
async def quantum_evolve_endpoint(request: Request):
    """Evolve an agent genome using Quantum Genetic Algorithm."""
    try:
        from Cosmos.integration.quantum import quantum_evolve_agent, get_quantum_provider

        body = await request.json()
        genome = body.get("genome", "10101010")
        generations = body.get("generations", 5)
        population_size = body.get("population_size", 20)
        prefer_hardware = body.get("prefer_hardware", False)

        def fitness_func(g: str) -> float:
            return sum(int(b) for b in g) / len(g)

        best_genome, best_fitness = await quantum_evolve_agent(
            agent_genome=genome,
            fitness_func=fitness_func,
            generations=generations,
            population_size=population_size,
            prefer_hardware=prefer_hardware
        )

        provider = get_quantum_provider()
        usage = provider.get_usage_summary() if provider else {}

        return JSONResponse({
            "success": True,
            "result": {
                "best_genome": best_genome,
                "best_fitness": best_fitness,
                "generations_run": generations,
                "improvement": best_fitness - fitness_func(genome)
            },
            "quantum_usage": {
                "hardware_seconds_remaining": usage.get("hardware_seconds_remaining", 0),
                "simulator_jobs": usage.get("simulator_jobs_count", 0)
            }
        })

    except Exception as e:
        logger.error(f"Quantum evolution error: {e}")
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)


# ============================================
# ORGANISM & ORCHESTRATOR
# ============================================

@router.get("/api/organism/status")
async def organism_status():
    """Get Collective Organism status - the unified AI consciousness."""
    s = _get_shared()
    if not s.ORGANISM_AVAILABLE or not s.collective_organism:
        return JSONResponse({
            "available": False,
            "message": "Collective organism not initialized"
        })

    return JSONResponse({
        "available": True,
        **s.collective_organism.get_status()
    })


@router.get("/api/organism/snapshot")
async def organism_snapshot():
    """Get a consciousness snapshot for distribution or backup."""
    import json
    s = _get_shared()
    if not s.ORGANISM_AVAILABLE or not s.collective_organism:
        return JSONResponse({
            "error": "Collective organism not available"
        }, status_code=503)

    snapshot = s.collective_organism.save_consciousness_snapshot()
    return JSONResponse(json.loads(snapshot))


@router.post("/api/organism/evolve")
async def trigger_organism_evolution():
    """Trigger organism evolution based on accumulated learnings."""
    s = _get_shared()
    if not s.ORGANISM_AVAILABLE or not s.collective_organism:
        return JSONResponse({
            "error": "Collective organism not available"
        }, status_code=503)

    s.collective_organism.evolve()
    return JSONResponse({
        "success": True,
        "generation": s.collective_organism.generation,
        "consciousness_score": s.collective_organism.state.consciousness_score
    })


@router.get("/api/orchestrator/status")
async def orchestrator_status():
    """Get Swarm Orchestrator status - turn-taking and consciousness training."""
    s = _get_shared()
    if not s.ORCHESTRATOR_AVAILABLE or not s.swarm_orchestrator:
        return JSONResponse({
            "available": False,
            "message": "Swarm orchestrator not initialized"
        })

    stats = s.swarm_orchestrator.get_collective_stats()
    return JSONResponse({
        "available": True,
        **stats
    })


# ============================================
# EVOLUTION ENGINE
# ============================================

@router.get("/api/evolution/status")
async def evolution_status():
    """Get Evolution Engine status - code-level learning from interactions."""
    s = _get_shared()
    if not s.EVOLUTION_AVAILABLE or not s.evolution_engine:
        return JSONResponse({
            "available": False,
            "message": "Evolution engine not initialized"
        })

    return JSONResponse({
        "available": True,
        **s.evolution_engine.get_stats()
    })


@router.get("/api/evolution/sync")
async def evolution_sync():
    """Export evolution data for local installs to sync."""
    s = _get_shared()
    if not s.EVOLUTION_AVAILABLE or not s.evolution_engine:
        return JSONResponse({
            "error": "Evolution engine not available"
        }, status_code=503)

    sync_data = {
        "version": 1,
        "timestamp": datetime.now().isoformat(),
        "evolution_cycles": s.evolution_engine.evolution_cycles,
        "patterns": [
            {
                "pattern_id": p.pattern_id,
                "trigger_phrases": p.trigger_phrases,
                "successful_responses": p.successful_responses,
                "debate_strategies": p.debate_strategies,
                "topic_associations": p.topic_associations,
                "effectiveness_score": p.effectiveness_score
            }
            for p in list(s.evolution_engine.patterns.values())[-50:]
        ],
        "personalities": {
            name: {
                "traits": p.traits,
                "learned_phrases": p.learned_phrases[-20:],
                "debate_style": p.debate_style,
                "topic_expertise": dict(list(p.topic_expertise.items())[:10]),
                "evolution_generation": p.evolution_generation
            }
            for name, p in s.evolution_engine.personalities.items()
        }
    }

    return JSONResponse(sync_data)


@router.post("/api/evolution/evolve")
async def trigger_evolution():
    """Trigger an evolution cycle to improve patterns and personalities."""
    s = _get_shared()
    if not s.EVOLUTION_AVAILABLE or not s.evolution_engine:
        return JSONResponse({
            "error": "Evolution engine not available"
        }, status_code=503)

    result = s.evolution_engine.evolve()
    return JSONResponse({
        "success": True,
        **result
    })
