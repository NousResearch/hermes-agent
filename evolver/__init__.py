"""Harness Evolver — offline diagnose/edit/eval loop around the agent loop.

Phase 0 (this package's current scope): the failure archive, the trace
contract, the three credit gates (validity -> activation -> credit), and the
calibration harness that proves the gates separate known-good human patches
from known-bad ones. No proposer, no rollout yet.

The evolver never runs in the hot path and never ships in the wheel:
``pyproject.toml``'s ``packages.find`` include-list deliberately omits this
directory. It is offline tooling for improving the scaffold, not part of it.
"""

__version__ = "0.1.0"

# Version of the trace/archive JSONL contract. Bump when archive.py's
# record shape changes; readers must reject newer major versions.
SCHEMA_VERSION = 1
