import logging
import time
from typing import Dict, Any, List

logger = logging.getLogger("META_COGNITION")

class MetaCognitionModule:
    """
    [UPGRADE 6] Meta-Cognition Module - Operational Self-Monitoring.
    Handles operational health: latency, loop detection, quality regression.
    Distinct from self_awareness.py (existential/ethical).
    """
    def __init__(self):
        self.start_time = time.time()
        self.loop_detector_threshold = 5  # repeated patterns
        self.message_history = []

    def log_operation(self, op_name: str, duration: float, metadata: Dict[str, Any]):
        """Log a system operation for bottleneck analysis."""
        if duration > 1.0:
            logger.info(f"[META] Bottleneck detected in {op_name}: {duration:.2f}s")
        
        self._detect_pathological_loops(op_name)

    def _detect_pathological_loops(self, op_name: str):
        """Identify pathological reasoning loops."""
        self.message_history.append(op_name)
        if len(self.message_history) > 20:
            self.message_history.pop(0)
            
        freq = self.message_history.count(op_name)
        if freq > self.loop_detector_threshold:
            logger.warning(f"[META] Pathological loop detected in {op_name}. Frequency: {freq}")

    def get_cognitive_health_metrics(self) -> Dict[str, Any]:
        """Expose operational metrics for a health endpoint."""
        return {
            'uptime': time.time() - self.start_time,
            'recent_operations': self.message_history[-5:],
            'stability_signal': 'STABLE'
        }

# Instance for background services
meta_cognition = MetaCognitionModule()
