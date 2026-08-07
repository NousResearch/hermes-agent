"""hermes mesh — provisioner package.

Layout (see design doc, "Code layout in `hermes-agent`"):
    provisioner.py   — SSH + render + scp + systemctl orchestration
    registry.py      — nodes.yaml + retained-topic registry + drift detection
    validation.py    — heartbeat wait, manifest validation
    templates/roles/ — role templates (bare, watchdog, compute for v0.1)
"""
from .provisioner import ControllerConfig, NodeSpec, ProbeResult
from .validation import DriftState, NodeStatus

__all__ = [
    "NodeSpec",
    "ControllerConfig",
    "ProbeResult",
    "NodeStatus",
    "DriftState",
]
