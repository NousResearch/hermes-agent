"""
Brain Surgeon — CNS Diagnostic & Recovery Organ
=================================================
The watchdog that monitors all CNS organs, detects failures,
performs hot-swaps, and runs self-healing routines.

Theory:
    - Every organ has a heartbeat. If it stops, the Surgeon intervenes.
    - Organs can be hot-swapped between COSMOS_12D (full physics)
      and FALLBACK_OLLAMA (lightweight/deterministic) mode.
    - The Surgeon tracks organ load times, failure rates, and
      recovery statistics for MetaCognition to consume.

Author: Cosmos CNS / Cory Shane Davis
Version: 2.0.0 (Full Diagnostic Engine)
"""

import time
import logging
import threading
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("BRAIN_SURGEON")


# ════════════════════════════════════════════════════════
# DATA CLASSES
# ════════════════════════════════════════════════════════

@dataclass
class OrganHealth:
    """Health record for a single CNS organ."""
    name: str
    loaded: bool = False
    healthy: bool = True
    load_time_ms: float = 0.0
    last_heartbeat: float = 0.0
    failure_count: int = 0
    last_error: Optional[str] = None

    @property
    def age_seconds(self) -> float:
        """Seconds since last heartbeat."""
        if self.last_heartbeat == 0.0:
            return float('inf')
        return time.time() - self.last_heartbeat


@dataclass
class SurgeryReport:
    """Report from a hot-swap or recovery procedure."""
    organ_name: str
    action: str               # "HOT_SWAP", "RESTART", "LOBOTOMY", "HEAL"
    success: bool
    old_mode: str
    new_mode: str
    duration_ms: float
    reason: str = ""


# ════════════════════════════════════════════════════════
# BRAIN SURGEON
# ════════════════════════════════════════════════════════

ORGAN_NAMES = [
    "quantum", "emeth", "plasticity", "awareness",
    "daemons", "surgeon", "dark_matter",
]

# Heartbeat timeout — if an organ hasn't ticked in this many seconds, it's dead
HEARTBEAT_TIMEOUT_S = 120.0


class BrainSurgeon:
    """
    The Diagnostic Organ.
    Manages temporal context, model health, organ hot-swapping,
    and auto-recovery for the entire CNS.
    """

    def __init__(self):
        self._lock = threading.RLock()

        # Active execution mode
        self.active_lobe = "FALLBACK_OLLAMA"
        self.knowledge_base_status = "STATIC_2023"
        self.lobotomy_active = False

        # Organ health registry
        self._organs: dict[str, OrganHealth] = {}
        for name in ORGAN_NAMES:
            self._organs[name] = OrganHealth(name=name)

        # Surgery history (last N operations)
        self._surgery_log: list[SurgeryReport] = []
        self._max_log_size = 50

        # Stats
        self._total_surgeries = 0
        self._total_recoveries = 0
        self._boot_time = time.time()

        logger.info("🔪 Brain Surgeon Online. Monitoring all CNS organs.")

    # ════════════════════════════════════════════════════════
    # ORGAN REGISTRATION & HEARTBEATS
    # ════════════════════════════════════════════════════════

    def register_organ(self, name: str, instance, load_time_ms: float = 0.0):
        """Register an organ as loaded and healthy."""
        with self._lock:
            health = self._organs.get(name, OrganHealth(name=name))
            health.loaded = instance is not None
            health.healthy = instance is not None
            health.load_time_ms = load_time_ms
            health.last_heartbeat = time.time()
            health.last_error = None
            self._organs[name] = health
            logger.debug(f"[SURGEON] Organ '{name}' registered. "
                         f"Loaded={health.loaded}, Time={load_time_ms:.1f}ms")

    def heartbeat(self, organ_name: str):
        """Record a heartbeat from an organ (call periodically during tick)."""
        with self._lock:
            if organ_name in self._organs:
                self._organs[organ_name].last_heartbeat = time.time()
                self._organs[organ_name].healthy = True

    def report_failure(self, organ_name: str, error: str):
        """Report an organ failure."""
        with self._lock:
            if organ_name in self._organs:
                organ = self._organs[organ_name]
                organ.healthy = False
                organ.failure_count += 1
                organ.last_error = error
                logger.error(f"🚨 [SURGEON] Organ '{organ_name}' FAILURE #{organ.failure_count}: {error}")

    # ════════════════════════════════════════════════════════
    # DIAGNOSTICS
    # ════════════════════════════════════════════════════════

    def diagnose(self) -> dict:
        """
        Full diagnostic scan of all CNS organs.
        Returns a comprehensive health report.
        """
        with self._lock:
            organ_reports = {}
            total_loaded = 0
            total_healthy = 0
            total_dead = 0

            for name, organ in self._organs.items():
                is_timed_out = organ.age_seconds > HEARTBEAT_TIMEOUT_S
                effective_health = organ.healthy and not is_timed_out

                organ_reports[name] = {
                    "loaded": organ.loaded,
                    "healthy": effective_health,
                    "load_time_ms": round(organ.load_time_ms, 1),
                    "age_seconds": round(organ.age_seconds, 1) if organ.last_heartbeat > 0 else None,
                    "failure_count": organ.failure_count,
                    "last_error": organ.last_error,
                }

                if organ.loaded:
                    total_loaded += 1
                if effective_health and organ.loaded:
                    total_healthy += 1
                if organ.loaded and not effective_health:
                    total_dead += 1

            total_organs = len(self._organs)
            health_pct = (total_healthy / max(1, total_loaded)) * 100 if total_loaded > 0 else 0.0

            return {
                "active_lobe": self.active_lobe,
                "knowledge_base": self.knowledge_base_status,
                "lobotomy_active": self.lobotomy_active,
                "uptime_seconds": round(time.time() - self._boot_time, 1),
                "organs": organ_reports,
                "summary": {
                    "total": total_organs,
                    "loaded": total_loaded,
                    "healthy": total_healthy,
                    "dead": total_dead,
                    "health_pct": round(health_pct, 1),
                },
                "surgery_stats": {
                    "total_surgeries": self._total_surgeries,
                    "total_recoveries": self._total_recoveries,
                    "recent": [
                        {
                            "organ": s.organ_name,
                            "action": s.action,
                            "success": s.success,
                            "reason": s.reason,
                        }
                        for s in self._surgery_log[-5:]
                    ],
                },
            }

    def get_dead_organs(self) -> list[str]:
        """Return names of loaded organs that have stopped responding."""
        with self._lock:
            dead = []
            for name, organ in self._organs.items():
                if organ.loaded and (not organ.healthy or organ.age_seconds > HEARTBEAT_TIMEOUT_S):
                    dead.append(name)
            return dead

    # ════════════════════════════════════════════════════════
    # HOT-SWAP / LOBOTOMY
    # ════════════════════════════════════════════════════════

    def lobotomy_switch(self, target_lobe: str) -> SurgeryReport:
        """
        Hot-swap the active model engine.

        Args:
            target_lobe: "COSMOS_12D" or "FALLBACK_OLLAMA"

        Returns:
            SurgeryReport with details of the operation.
        """
        start = time.time()
        old_mode = self.active_lobe

        logger.warning(f"🔪 BRAIN SURGEON: Initiating Lobotomy. "
                       f"{old_mode} → {target_lobe}")

        with self._lock:
            if target_lobe == "COSMOS_12D":
                self.active_lobe = "COSMOS_12D"
                self.knowledge_base_status = "LIVING_FIELD"
            else:
                self.active_lobe = "FALLBACK_OLLAMA"
                self.knowledge_base_status = "STATIC_2023"

            self.lobotomy_active = True
            duration_ms = (time.time() - start) * 1000

            report = SurgeryReport(
                organ_name="active_lobe",
                action="LOBOTOMY",
                success=True,
                old_mode=old_mode,
                new_mode=self.active_lobe,
                duration_ms=duration_ms,
                reason=f"Manual switch to {target_lobe}",
            )
            self._log_surgery(report)
            self.lobotomy_active = False

        logger.info(f"🧠 Lobotomy Complete. Active Lobe: {self.active_lobe}")
        return report

    # ════════════════════════════════════════════════════════
    # AUTO-RECOVERY
    # ════════════════════════════════════════════════════════

    def attempt_recovery(self, cns_instance) -> list[SurgeryReport]:
        """
        Scan for dead organs and attempt to restart them.

        Args:
            cns_instance: The CosmosCNS instance (needed to re-init organs).

        Returns:
            List of SurgeryReports for each recovery attempt.
        """
        dead = self.get_dead_organs()
        if not dead:
            return []

        reports = []
        logger.warning(f"🏥 [SURGEON] Auto-recovery triggered for: {dead}")

        for organ_name in dead:
            start = time.time()
            success = False
            error = ""

            try:
                if organ_name == "quantum":
                    from Cosmos.core.quantum_bridge import get_quantum_bridge
                    cns_instance.quantum = get_quantum_bridge()
                    if cns_instance.quantum and cns_instance.field:
                        cns_instance.quantum.set_synaptic_field(cns_instance.field)
                    success = True

                elif organ_name == "dark_matter":
                    from .dark_matter_lorenz import DarkMatterLorenz
                    cns_instance.dark_matter = DarkMatterLorenz()
                    success = True

                elif organ_name == "emeth":
                    from .emeth_harmonizer import EmethHarmonizer
                    cns_instance.emeth = EmethHarmonizer()
                    success = True

                elif organ_name == "plasticity":
                    from .swarm_plasticity import SwarmPlasticity
                    cns_instance.plasticity = SwarmPlasticity()
                    success = True

                elif organ_name == "awareness":
                    from .swarm_awareness import SwarmAwareness
                    if cns_instance.field:
                        cns_instance.awareness = SwarmAwareness(
                            synaptic_field=cns_instance.field,
                            plasticity=cns_instance.plasticity,
                            emeth=cns_instance.emeth,
                        )
                        success = True

                elif organ_name == "daemons":
                    from .swarm_daemons import SwarmDaemons
                    if cns_instance.field:
                        cns_instance.daemons = SwarmDaemons(field=cns_instance.field)
                        cns_instance.daemons.start()
                        success = True

                if success:
                    self.register_organ(organ_name, getattr(cns_instance, organ_name, None))
                    self._total_recoveries += 1

            except Exception as e:
                error = str(e)
                logger.error(f"🚨 [SURGEON] Recovery FAILED for '{organ_name}': {e}")

            duration_ms = (time.time() - start) * 1000
            report = SurgeryReport(
                organ_name=organ_name,
                action="RESTART",
                success=success,
                old_mode="DEAD",
                new_mode="ALIVE" if success else "DEAD",
                duration_ms=duration_ms,
                reason=error if not success else "Auto-recovered",
            )
            reports.append(report)
            self._log_surgery(report)

        return reports

    # ════════════════════════════════════════════════════════
    # INTERNAL
    # ════════════════════════════════════════════════════════

    def _log_surgery(self, report: SurgeryReport):
        """Append a surgery report to the log."""
        self._surgery_log.append(report)
        self._total_surgeries += 1
        if len(self._surgery_log) > self._max_log_size:
            self._surgery_log = self._surgery_log[-self._max_log_size:]

    def get_stats(self) -> dict:
        """Quick summary for telemetry."""
        diag = self.diagnose()
        return {
            "active_lobe": self.active_lobe,
            "health_pct": diag["summary"]["health_pct"],
            "loaded": diag["summary"]["loaded"],
            "dead": diag["summary"]["dead"],
            "total_surgeries": self._total_surgeries,
            "total_recoveries": self._total_recoveries,
            "uptime_s": round(time.time() - self._boot_time, 1),
        }
