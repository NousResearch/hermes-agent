"""Canonical stopped-runtime inventory for isolated Muncho canary mutations.

Any operation that can reboot the VM, rotate its host identity, or publish a
stopped release must prove this complete superset absent or disabled+inactive.
Keeping one narrow data-only module prevents the seven-unit release subset and
the twelve-unit storage subset from drifting again.
"""

from __future__ import annotations


CANARY_RUNTIME_UNITS = (
    "hermes-cloud-gateway.service",
    "muncho-canary-discord-edge.service",
    "muncho-canonical-writer-phase-b-readiness.service",
    "muncho-canonical-writer.service",
    "muncho-canonical-writer-export.service",
    "muncho-canonical-writer-export.timer",
    "muncho-isolated-worker.socket",
    "muncho-isolated-worker.service",
    "muncho-capability-browser.service",
    "muncho-discord-connector.service",
    "muncho-discord-egress.service",
    "muncho-mac-ops-edge.service",
)


__all__ = ["CANARY_RUNTIME_UNITS"]
