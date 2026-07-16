# BETA-002 — Declarative specialist registry

Beta loads specialists from the packaged `agent/beta/specialists.yaml` catalog. The registry validates every entry, rejects duplicate IDs, and exposes only enabled specialists to routing. Adding a specialist changes the catalog, not the router.

## Manifest example

```yaml
specialists:
  - id: dba
    name: Database Administrator
    description: Diagnoses database performance and locking.
    capabilities: [database, postgresql, query-performance]
    keywords: [postgresql, query, lock]
    allowed_toolsets: [terminal, file, web]
    blocked_tools: [delegate_task, clarify, memory, send_message]
    model: null
    provider: null
    max_risk: medium
    memory_access: read_write
    memory_scope: specialist:dba
    max_concurrency: 2
    enabled: true
```

`max_risk` is the highest operation risk the specialist may receive. `memory_scope` is technical and specialist-owned; it never grants access to Beta's strategic memory.

## Validation

```bash
python -m pytest -q tests/agent/beta/test_specialists.py
```
