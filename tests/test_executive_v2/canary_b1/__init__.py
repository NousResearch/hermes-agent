"""HERMES B-1 Knowledge Discovery Canary (hermetic, READONLY at runtime).

This package is a self-contained canary implementation of the EvidencePack
v1 + Knowledge Discovery flow described in
~/.hermes/reports/HERMES_B1_KNOWLEDGE_DISCOVERY_CANARY_DESIGN_READONLY/.

The canary is **completely hermetic** by design:

* No real GBrain invocation (FakeGBrainSource is in-memory).
* No real Obsidian vault walk (FakeObsidianSource is in-memory).
* No real subprocess (caller never invokes subprocess).
* No outbound network (caller never opens sockets).
* No state.db writes (uses in-memory FakeDB from conftest).
* No audit log writes (audit_capture is in-memory).
* No skills/profiles/configs touched at runtime.

The canary's purpose is to validate the EvidencePack v1 contract end-to-end
with deterministic, frozen-time inputs, using fake providers that are
indistinguishable from the real providers in their behavior.
"""

from __future__ import annotations

__all__ = [
    "__version__",
]

__version__ = "canary_b1_v1.0.0"
