"""Hermes Enterprise: multi-tenant control plane for deploying and operating
Hermes agents.

This package is the control-plane counterpart to the hermes-agent runtime
(the "Harness" in platform terms). It is deliberately standalone: nothing in
the core agent imports it, it adds no model tools, and it never runs inside
an agent conversation. It is operated via the ``hermes enterprise`` CLI and
(in later work) the controller service.

Layout:
    resources.py  - platform resource model (Namespace, Agent, AgentRevision, ...)
    iam.py        - IAM entities (Principal, Role, AccessBinding, ...)
    store.py      - SQLite-backed resource store with optimistic concurrency
    audit.py      - append-only audit log
    contracts.py  - Driver / Adapter ABCs (compute, sandbox, secrets, IAM)
    errors.py     - typed error hierarchy (all failures are fail-closed)
"""

__all__ = [
    "resources",
    "store",
    "audit",
    "contracts",
    "errors",
]

ENTERPRISE_SCHEMA_VERSION = 1
