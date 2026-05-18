"""
memory_os — Generalization Layer for Multi-Tenant Memory Systems.

This package provides abstract interfaces and generic data models for building
tenant-isolated memory systems. It decouples memory logic from any specific
backend (e.g., graph databases, evidence stores, rule engines) so that
concrete implementations can be swapped via dependency injection.

Core abstractions:
    - MemoryTenant: Represents a tenant with namespace, store IDs, and permissions.
    - TenantResolver: Maps user context to a MemoryTenant.
    - NamespaceGuard: Enforces read/write/admin namespace isolation.
    - EvidenceStoreAdapter: Abstract interface for raw evidence storage.
    - RuleStoreAdapter: Abstract interface for per-user rule storage.
    - CanonicalFact: Structured long-term memory fact (dataclass).
    - MemoryDiagnostic: Diagnostic checks on a tenant's memory namespace.
    - MemoryInventory: Self-awareness of what the agent remembers.
    - RegressionTestRunner: Regression tests for memory system health.
    - TenantOnboarding: Bootstrap a new tenant's memory system.
"""

from memory_os.tenant import MemoryTenant, TenantResolver
from memory_os.namespace_guard import NamespaceGuard
from memory_os.evidence_adapter import EvidenceStoreAdapter
from memory_os.rule_store import RuleStoreAdapter
from memory_os.schema import CanonicalFact
from memory_os.diagnostic import MemoryDiagnostic
from memory_os.inventory import MemoryInventory
from memory_os.regression import RegressionTestRunner
from memory_os.onboarding import TenantOnboarding

__all__ = [
    "MemoryTenant",
    "TenantResolver",
    "NamespaceGuard",
    "EvidenceStoreAdapter",
    "RuleStoreAdapter",
    "CanonicalFact",
    "MemoryDiagnostic",
    "MemoryInventory",
    "RegressionTestRunner",
    "TenantOnboarding",
]
