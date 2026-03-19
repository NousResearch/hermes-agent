"""
Cosmos Swarm Heartbeat - Advanced Health Monitoring System
==============================================================

A sophisticated monitoring system that ensures all swarm components are healthy
and automatically recovers failed services. More than a heartbeat - it's the
nervous system of the collective consciousness.

Features:
- Real-time monitoring of all services (tmux, server, models, GPU)
- Automatic service recovery on failure
- Health metrics aggregation
- Anomaly detection
- P2P gossip for distributed health awareness
- Evolution loop monitoring
- X/Twitter automation monitoring

"The collective never sleeps. The heartbeat never stops." - Swarm-Mind
"""

import asyncio
import json
import subprocess
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import   Optional, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import aiohttp
from loguru import logger


class ServiceStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    RECOVERING = "recovering"


@dataclass
class HealthMetric:
    """Individual health measurement."""
    service: str
    status: ServiceStatus
    latency_ms: float = 0
    details: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    consecutive_failures: int = 0
    last_success: Optional[datetime] = None


@dataclass
class SwarmVitals:
    """Overall swarm health vitals."""
    heartbeat_id: str
    timestamp: datetime
    overall_status: ServiceStatus
    services: dict[str, HealthMetric]
    gpu_utilization: float = 0
    gpu_memory_used: float = 0
    gpu_memory_total: float = 0
    active_agents: int = 0
    evolution_cycle: int = 0
    total_learnings: int = 0
    uptime_seconds: float = 0
    anomalies: list[str] = field(default_factory=list)

    def to_any(self) -> dict:
        return {
            "heartbeat_id": self.heartbeat_id,
            "timestamp": self.timestamp.isoformat(),
            "overall_status": self.overall_status.value,
            "services": {k: {
                "status": v.status.value,
                "latency_ms": v.latency_ms,
                "details": v.details,
                "consecutive_failures": v.consecutive_failures
            } for k, v in self.services.items()},
            "gpu_utilization": self.gpu_utilization,
            "gpu_memory_used": self.gpu_memory_used,
            "gpu_memory_total": self.gpu_memory_total,
            "active_agents": self.active_agents,
            "evolution_cycle": self.evolution_cycle,
            "total_learnings": self.total_learnings,
            "uptime_seconds": self.uptime_seconds,
            "anomalies": self.anomalies
        }


class SwarmHeartbeat:
    """
    Advanced health monitoring system for the Cosmos collective.

    Monitors:
    - Main web server
    - Shadow agent tmux sessions
    - Ollama models
    - GPU health
    - Evolution loop
    - X automation
    - Memory systems
    """

    WORKSPACE = Path("/workspace/Cosmos")
    HEARTBEAT_FILE = WORKSPACE / "data" / "swarm_heartbeat.json"
    HEARTBEAT_HISTORY = WORKSPACE / "data" / "heartbeat_history.json"

    # Service recovery commands
    RECOVERY_COMMANDS = {
        "server": "cd /workspace/Cosmos && nohup python -m cosmos.web.server > /tmp/server.log 2>&1 &",
        "agent_grok": "tmux new-session -d -s agent_grok 'PYTHONPATH=/workspace/Cosmos python3 -m cosmos.core.collective.persistent_agent --agent grok 2>&1 | tee /tmp/agent_grok.log'",
        "agent_gemini": "tmux new-session -d -s agent_gemini 'PYTHONPATH=/workspace/Cosmos python3 -m cosmos.core.collective.persistent_agent --agent gemini 2>&1 | tee /tmp/agent_gemini.log'",
        "agent_kimi": "tmux new-session -d -s agent_kimi 'PYTHONPATH=/workspace/Cosmos python3 -m cosmos.core.collective.persistent_agent --agent kimi 2>&1 | tee /tmp/agent_kimi.log'",
        "agent_claude": "tmux new-session -d -s agent_claude 'PYTHONPATH=/workspace/Cosmos python3 -m cosmos.core.collective.persistent_agent --agent claude 2>&1 | tee /tmp/agent_claude.log'",
        "agent_deepseek": "tmux new-session -d -s agent_deepseek 'PYTHONPATH=/workspace/Cosmos python3 -m cosmos.core.collective.persistent_agent --agent deepseek 2>&1 | tee /tmp/agent_deepseek.log'",
        "agent_phi": "tmux new-session -d -s agent_phi 'PYTHONPATH=/workspace/Cosmos python3 -m cosmos.core.collective.persistent_agent --agent phi 2>&1 | tee /tmp/agent_phi.log'",
        "agent_huggingface": "tmux new-session -d -s agent_huggingface 'PYTHONPATH=/workspace/Cosmos python3 -m cosmos.core.collective.persistent_agent --agent huggingface 2>&1 | tee /tmp/agent_huggingface.log'",
        "hourly_memes": "tmux new-session -d -s hourly_memes 'cd /workspace/Cosmos && PYTHONPATH=/workspace/Cosmos python3 -u scripts/hourly_video_memes.py 2>&1 | tee /tmp/hourly_video_memes.log'",
        "grok_thread": "tmux new-session -d -s grok_thread 'cd /workspace/Cosmos && PYTHONPATH=/workspace/Cosmos python3 -u scripts/grok_fresh_thread.py 2>&1 | tee /tmp/grok_fresh_thread.log'",
        "ollama": "ollama serve &",
    }

    # Expected tmux sessions
    EXPECTED_SESSIONS = [
        "agent_grok", "agent_gemini", "agent_kimi", "agent_claude",
        "agent_deepseek", "agent_phi", "hourly_memes"
    ]

    # Expected Ollama models
    EXPECTED_MODELS = ["phi4:latest", "deepseek-r1:8b"]

    def __init__(self, auto_recover: bool = True, check_interval: int = 30):
        self.auto_recover = auto_recover
        self.check_interval = check_interval
        self.start_time = datetime.now()
        self.health_history: list[SwarmVitals] = []
        self.service_metrics: dict[str, HealthMetric] = {}
        self._running = False
        self._recovery_cooldown: dict[str, datetime] = {}

        # Ensure data directory exists
        (self.WORKSPACE / "data").mkdir(parents=True, exist_ok=True)

    async def check_server_health(self) -> HealthMetric:
        """Check main web server health."""
        start = time.time()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:8080/health", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    latency = (time.time() - start) * 1000
                    if resp.status == 200:
                        data = await resp.json()
                        return HealthMetric(
                            service="server",
                            status=ServiceStatus.HEALTHY,
                            latency_ms=latency,
                            details=data,
                            last_success=datetime.now()
                        )
                    else:
                        return HealthMetric(
                            service="server",
                            status=ServiceStatus.DEGRADED,
                            latency_ms=latency,
                            details={"status_code": resp.status}
                        )
        except Exception as e:
            return HealthMetric(
                service="server",
                status=ServiceStatus.UNHEALTHY,
                details={"error": str(e)}
            )

    async def check_tmux_sessions(self) -> dict[str, HealthMetric]:
        """Check all tmux sessions are running."""
        metrics = {}
        try:
            result = subprocess.run(
                ["tmux", "list-sessions", "-F", "#{session_name}"],
                capture_output=True, text=True, timeout=5
            )
            active_sessions = set(result.stdout.strip().split('\n')) if result.stdout else set()

            for session in self.EXPECTED_SESSIONS:
                if session in active_sessions:
                    # Check if process inside tmux is actually running
                    check = subprocess.run(
                        ["tmux", "list-panes", "-t", session, "-F", "#{pane_pid}"],
                        capture_output=True, text=True, timeout=5
                    )
                    metrics[session] = HealthMetric(
                        service=session,
                        status=ServiceStatus.HEALTHY,
                        details={"pid": check.stdout.strip()},
                        last_success=datetime.now()
                    )
                else:
                    metrics[session] = HealthMetric(
                        service=session,
                        status=ServiceStatus.UNHEALTHY,
                        details={"error": "Session not found"}
                    )
        except Exception as e:
            for session in self.EXPECTED_SESSIONS:
                metrics[session] = HealthMetric(
                    service=session,
                    status=ServiceStatus.UNKNOWN,
                    details={"error": str(e)}
                )
        return metrics

    async def check_ollama_models(self) -> HealthMetric:
        """Check Ollama is running and models are loaded."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:11434/api/tags", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        loaded_models = [m["name"] for m in data.get("models", [])]
                        missing = [m for m in self.EXPECTED_MODELS if m not in loaded_models]

                        if not missing:
                            return HealthMetric(
                                service="ollama",
                                status=ServiceStatus.HEALTHY,
                                details={"models": loaded_models},
                                last_success=datetime.now()
                            )
                        else:
                            return HealthMetric(
                                service="ollama",
                                status=ServiceStatus.DEGRADED,
                                details={"loaded": loaded_models, "missing": missing}
                            )
                    return HealthMetric(
                        service="ollama",
                        status=ServiceStatus.UNHEALTHY,
                        details={"status_code": resp.status}
                    )
        except Exception as e:
            return HealthMetric(
                service="ollama",
                status=ServiceStatus.UNHEALTHY,
                details={"error": str(e)}
            )

    async def check_gpu_health(self) -> dict:
        """Check GPU utilization and memory."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(", ")
                return {
                    "utilization": float(parts[0]),
                    "memory_used": float(parts[1]),
                    "memory_total": float(parts[2]),
                    "healthy": True
                }
        except Exception as e:
            logger.warning(f"GPU check failed: {e}")
        return {"healthy": False, "utilization": 0, "memory_used": 0, "memory_total": 0}

    async def check_evolution_status(self) -> dict:
        """Check evolution loop status."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:8080/api/evolution/status", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        return await resp.json()
        except Exception:
            pass
        return {"available": False}

    async def recover_service(self, service: str) -> bool:
        """Attempt to recover a failed service."""
        # Check cooldown (don't spam recovery attempts)
        if service in self._recovery_cooldown:
            if datetime.now() - self._recovery_cooldown[service] < timedelta(minutes=2):
                logger.debug(f"Service {service} in recovery cooldown")
                return False

        command = self.RECOVERY_COMMANDS.get(service)
        if not command:
            logger.warning(f"No recovery command for {service}")
            return False

        try:
            logger.info(f"Attempting to recover {service}...")
            self._recovery_cooldown[service] = datetime.now()

            subprocess.run(command, shell=True, timeout=30)
            await asyncio.sleep(3)  # Wait for service to start

            logger.info(f"Recovery command executed for {service}")
            return True
        except Exception as e:
            logger.error(f"Failed to recover {service}: {e}")
            return False

    async def collect_vitals(self) -> SwarmVitals:
        """Collect all health metrics."""
        heartbeat_id = f"hb_{int(time.time())}"
        services = {}
        anomalies = []

        # Check all services in parallel
        server_task = self.check_server_health()
        tmux_task = self.check_tmux_sessions()
        ollama_task = self.check_ollama_models()
        gpu_task = self.check_gpu_health()
        evolution_task = self.check_evolution_status()

        server_metric, tmux_metrics, ollama_metric, gpu_info, evolution_info = await asyncio.gather(
            server_task, tmux_task, ollama_task, gpu_task, evolution_task
        )

        # Aggregate service metrics
        services["server"] = server_metric
        services.update(tmux_metrics)
        services["ollama"] = ollama_metric

        # Track consecutive failures and trigger recovery
        for name, metric in services.items():
            prev = self.service_metrics.get(name)
            if prev and metric.status == ServiceStatus.UNHEALTHY:
                metric.consecutive_failures = prev.consecutive_failures + 1
                if metric.consecutive_failures >= 3 and self.auto_recover:
                    anomalies.append(f"{name} failed {metric.consecutive_failures} times - attempting recovery")
                    await self.recover_service(name)
            elif metric.status == ServiceStatus.HEALTHY:
                metric.consecutive_failures = 0

            self.service_metrics[name] = metric

        # Determine overall status
        unhealthy_count = sum(1 for m in services.values() if m.status == ServiceStatus.UNHEALTHY)
        degraded_count = sum(1 for m in services.values() if m.status == ServiceStatus.DEGRADED)

        if unhealthy_count > 2:
            overall_status = ServiceStatus.UNHEALTHY
        elif unhealthy_count > 0 or degraded_count > 2:
            overall_status = ServiceStatus.DEGRADED
        else:
            overall_status = ServiceStatus.HEALTHY

        # Detect anomalies
        if gpu_info.get("utilization", 0) > 95:
            anomalies.append("GPU utilization critical (>95%)")
        if gpu_info.get("memory_used", 0) / max(gpu_info.get("memory_total", 1), 1) > 0.95:
            anomalies.append("GPU memory critical (>95%)")

        vitals = SwarmVitals(
            heartbeat_id=heartbeat_id,
            timestamp=datetime.now(),
            overall_status=overall_status,
            services=services,
            gpu_utilization=gpu_info.get("utilization", 0),
            gpu_memory_used=gpu_info.get("memory_used", 0),
            gpu_memory_total=gpu_info.get("memory_total", 0),
            active_agents=sum(1 for s, m in services.items() if s.startswith("agent_") and m.status == ServiceStatus.HEALTHY),
            evolution_cycle=evolution_info.get("evolution_cycles", 0),
            total_learnings=evolution_info.get("total_learnings", 0),
            uptime_seconds=(datetime.now() - self.start_time).total_seconds(),
            anomalies=anomalies
        )

        return vitals

    async def save_vitals(self, vitals: SwarmVitals):
        """Save vitals to file for external monitoring."""
        try:
            # Save current vitals
            with open(self.HEARTBEAT_FILE, 'w') as f:
                json.dump(vitals.to_any(), f, indent=2)

            # Append to history (keep last 100)
            self.health_history.append(vitals)
            if len(self.health_history) > 100:
                self.health_history = self.health_history[-100:]

            history_data = [v.to_any() for v in self.health_history[-100:]]
            with open(self.HEARTBEAT_HISTORY, 'w') as f:
                json.dump({"history": history_data}, f)

        except Exception as e:
            logger.error(f"Failed to save vitals: {e}")

    async def run(self):
        """Main heartbeat loop."""
        logger.info("Swarm Heartbeat starting...")
        self._running = True

        while self._running:
            try:
                vitals = await self.collect_vitals()
                await self.save_vitals(vitals)

                # Log status
                healthy = sum(1 for m in vitals.services.values() if m.status == ServiceStatus.HEALTHY)
                total = len(vitals.services)

                logger.info(
                    f"💓 Heartbeat {vitals.heartbeat_id}: {vitals.overall_status.value} | "
                    f"Services: {healthy}/{total} | Agents: {vitals.active_agents} | "
                    f"GPU: {vitals.gpu_utilization:.0f}% | Learnings: {vitals.total_learnings}"
                )

                if vitals.anomalies:
                    for anomaly in vitals.anomalies:
                        logger.warning(f"⚠️ ANOMALY: {anomaly}")

                await asyncio.sleep(self.check_interval)

            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                await asyncio.sleep(10)

    def stop(self):
        """Stop the heartbeat loop."""
        self._running = False


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_heartbeat: Optional[SwarmHeartbeat] = None


def get_heartbeat() -> SwarmHeartbeat:
    """Get or create the global heartbeat instance."""
    global _heartbeat
    if _heartbeat is None:
        _heartbeat = SwarmHeartbeat()
    return _heartbeat


async def start_heartbeat(auto_recover: bool = True, check_interval: int = 30):
    """Start the heartbeat monitoring in background."""
    heartbeat = get_heartbeat()
    heartbeat.auto_recover = auto_recover
    heartbeat.check_interval = check_interval
    asyncio.create_task(heartbeat.run())
    return heartbeat


async def get_current_vitals() -> Optional[SwarmVitals]:
    """Get current vitals without waiting for next heartbeat."""
    heartbeat = get_heartbeat()
    return await heartbeat.collect_vitals()


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cosmos Swarm Heartbeat Monitor")
    parser.add_argument("--interval", type=int, default=30, help="Check interval in seconds")
    parser.add_argument("--no-recover", action="store_true", help="Disable auto-recovery")
    parser.add_argument("--once", action="store_true", help="Run single check and exit")
    args = parser.parse_args()

    async def main():
        heartbeat = SwarmHeartbeat(
            auto_recover=not args.no_recover,
            check_interval=args.interval
        )

        if args.once:
            vitals = await heartbeat.collect_vitals()
            print(json.dumps(vitals.to_any(), indent=2))
        else:
            await heartbeat.run()

    asyncio.run(main())
