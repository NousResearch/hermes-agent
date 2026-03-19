import time
import logging

from dataclasses import dataclass, field

logger = logging.getLogger("META_COGNITION")

@dataclass
class HealthReport:
    status: str
    uptime: float
    dimensions_active: int
    quantum_coherence: float
    last_thought_latency: float
    warnings: list[str] = field(default_factory=list)

class MetaCognition:
    """
    Cosmo's Meta-Cognition Module — Self-Monitoring & Diagnostics.
    
    Acts as a 'second brain' that monitors the primary CNS functions,
    tracks operational health, and provides introspection capabilities.
    """
    
    def __init__(self, cns_instance=None):
        self.cns = cns_instance
        self.start_time = time.time()
        self.total_thoughts = 0
        self.last_latencies = []
        
    def get_health_report(self) -> HealthReport:
        """Generate a comprehensive health report for the system."""
        if not self.cns:
            return HealthReport("ERROR", 0, 0, 0.0, 0.0, ["CNS Not Associated"])
            
        uptime = time.time() - self.start_time
        dimensions = 12 # Constant in CST
        
        q_coherence = 0.0
        if self.cns.quantum:
            q_coherence = self.cns.quantum.get_entropy({})
            
        warnings = []
        if q_coherence < 0.2:
            warnings.append("Low Quantum Entropy: System may be stagnating.")
        if hasattr(self.cns.lock, 'drift') and self.cns.lock.drift > 0.5:
            warnings.append(f"High Lyapunov Drift ({self.cns.lock.drift:.2f}): Stability at risk.")
            
        status = "OPTIMAL"
        if warnings:
            status = "STRESSED"
        if self.cns.field and self.cns.field.system_mode == "HEAL":
            status = "RECOVERING"
            
        lat = self.last_latencies[-1] if self.last_latencies else 0.0
        
        return HealthReport(
            status=status,
            uptime=uptime,
            dimensions_active=dimensions,
            quantum_coherence=q_coherence,
            last_thought_latency=lat,
            warnings=warnings
        )
        
    def record_interaction(self, latency: float):
        """Track cognitive performance."""
        self.total_thoughts += 1
        self.last_latencies.append(latency)
        if len(self.last_latencies) > 50:
            self.last_latencies.pop(0)
            
    def get_status_dict(self) -> dict:
        report = self.get_health_report()
        return {
            "status": report.status,
            "uptime_seconds": round(report.uptime, 2),
            "mode": self.cns.field.system_mode if self.cns.field else "UNKNOWN",
            "q_entropy": round(report.quantum_coherence, 4),
            "warnings": report.warnings,
            "latency_ms": round(report.last_thought_latency * 1000, 2),
            "thoughts_processed": self.total_thoughts
        }
