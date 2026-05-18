# Namespace Isolation for Multi-Tenant Memory Systems

## Overview

This document describes the namespace isolation architecture for multi-tenant long-term memory systems. It ensures that each tenant's memories, evidence, rules, and diagnostics are completely isolated from other tenants.

## Architecture

The isolation is defense-in-depth, using three independent layers:

```
┌─────────────────────────────────────────────┐
│ Layer 1: Memory Graph Namespace             │
│ - Each tenant has a unique namespace        │
│ - All CRUD operations filter by namespace   │
│ - Admin sees all, users see only their own  │
├─────────────────────────────────────────────┤
│ Layer 2: Evidence Store Isolation           │
│ - Each tenant has a dedicated evidence store│
│ - Recall/retain operations scoped to store  │
│ - No cross-store leakage                    │
├─────────────────────────────────────────────┤
│ Layer 3: Rule Store Isolation               │
│ - Each tenant has per-user rule injection   │
│ - System prompt rules are tenant-specific   │
│ - No cross-tenant rule contamination        │
└─────────────────────────────────────────────┘
```

## Namespace Guard

The `NamespaceGuard` class enforces isolation at the API level:

```python
from memory_os.namespace_guard import NamespaceGuard
from memory_os.tenant import MemoryTenant

guard = NamespaceGuard()

alice = MemoryTenant(namespace="user:alice", ...)
bob = MemoryTenant(namespace="user:bob", ...)

# Alice can read her own namespace
assert guard.can_read(alice, "user:alice") == True

# Alice cannot read Bob's namespace
assert guard.can_read(alice, "user:bob") == False

# Admin can read all namespaces
admin = MemoryTenant(namespace="", permissions={"is_admin": True}, ...)
assert guard.can_read(admin, "user:bob") == True
```

## Tenant Resolution

The `TenantResolver` maps user context to a `MemoryTenant`:

```python
from memory_os.tenant import TenantResolver, MemoryTenant

class MyTenantResolver(TenantResolver):
    def resolve(self, user_context):
        return MemoryTenant(
            namespace=f"user:{user_context['user_id']}",
            evidence_store_id=f"evidence_{user_context['user_id']}",
            rule_store_id=f"rules_{user_context['user_id']}",
            ...
        )
```

## Test Suite

The test suite uses neutral Alice/Bob/Core fixtures:

- **ALICE_ONLY_TOKEN**: Unique identifier visible only in Alice's namespace
- **BOB_ONLY_TOKEN**: Unique identifier visible only in Bob's namespace
- **COMMON_RULE_TOKEN**: Shared identifier visible to all tenants

### Covered Isolation Paths

| Path | Test |
|------|------|
| Read | Alice cannot read Bob's private data |
| Search | Alice search results exclude Bob's data |
| Write | Alice cannot write to Bob's namespace |
| Delete | Alice cannot delete Bob's data |
| Glossary | Alice glossary scan excludes Bob's entities |
| Diagnostic | Alice diagnostic shows only her namespace stats |
| Recent | Alice recent updates exclude Bob's changes |
| Fact History | Alice cannot see Bob's version chain |
| Core Read | Both Alice and Bob can read shared core rules |
| Admin Auth | Empty namespace does not imply admin access |

## Running Tests

```bash
pytest tests/test_namespace_isolation.py -v
```

## Compatibility

- Backward compatible with existing single-tenant deployments
- Namespace is optional — default namespace "" works for single-user
- No changes required to existing memory content
- Evidence and rule stores are adapter-based — any backend can be used
