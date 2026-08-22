# MEM0 Observability Layer Pattern

## When This Pattern Applies

Use this pattern when a mission requires adding **local observability** to a trusted audit pipeline — deterministic metrics, structured event schemas, and log entries — without external telemetry or network calls.

This is designed for the production readiness gate's `observability` dimension, which requires `has_metrics=true`, `has_logs=true`, `has_audit_trail=true`.

## Design Constraints

| Constraint | Rationale |
|------------|-----------|
| No external telemetry | No network calls, no file writes (return dicts only), no secrets |
| No secret leakage | No provider keys, tokens, or credentials in any output |
| Deterministic | Same inputs produce same event IDs (stable hash of content only) |
| Local counters only | Module-level ints, no Redis/StatsD/Prometheus |
| Scalar-only log summaries | Nested dicts/lists stripped to avoid secret leakage |
| Timestamp in event body | `datetime.datetime.now(datetime.UTC).isoformat()` — NOT `utcnow()` (deprecated) |
| Event ID from content hash | SHA256 truncated to 16 hex chars, from canonical JSON of payload only (no timestamp in hash input) |

## Module Structure

```
v2/memory/abstraction/observability.py
├── Module-level counters (4)
│   ├── _controlled_write_count
│   ├── _verification_count
│   ├── _snapshot_count
│   └── _rollback_count
├── _recent_event_log (FIFO, bounded to 100 entries)
├── _stable_hash(payload) — deterministic sha256 trunc-16
├── _now_iso() — UTC ISO-8601 timestamp
├── get_metrics() -> dict
│     Returns: {controlled_write_count, verification_count, snapshot_count, rollback_count}
├── build_controlled_write_event(candidate, target_store, before_hash, after_hash, snapshot_state) -> dict
│     Increments: controlled_write_count + snapshot_count
├── build_rollback_drill_event(target_path, before_hash, after_hash, simulation_result) -> dict
│     Increments: verification_count
├── build_rollback_execution_event(target_path, restored_from_hash, verified_hash) -> dict
│     Increments: rollback_count
├── build_log_entry(event, source_module, log_level='info') -> dict
│     Returns: {level, source, timestamp, event_id, event_type, payload_summary}
│     payload_summary: scalar values only (dicts/lists stripped)
├── build_observability_context() -> dict
│     Returns: {metrics: {...}, recent_event_log: [...]}
```

## Event Schema

Every event has the same top-level shape:

```python
{
    "event_type": "controlled_write" | "rollback_drill" | "rollback_execution",
    "event_id": "<sha256-trunc-16-of-canonical-payload>",
    "timestamp": "2026-07-02T01:30:00.123456+00:00",
    "payload": { ... },           # event-specific content
    "deterministic": True,
}
```

The `event_id` is computed from `_stable_hash({"event_type": ..., "payload": ...})` — timestamp is NOT part of the hash input so identical payloads produce identical IDs.

## Log Entry Schema

```python
{
    "level": "info",
    "source": "v2.memory.abstraction.controlled_write_pilot",
    "timestamp": "2026-07-02T01:30:00.123456+00:00",
    "event_id": "<sha256-trunc-16>",
    "event_type": "controlled_write",
    "payload_summary": {
        "target_store": "...",
        "snapshot_state": "...",
        # No nested dicts or lists
    }
}
```

## Test Coverage (10 tests)

| Test | What it verifies |
|------|-----------------|
| `test_metrics_start_at_zero` | All 4 counters are 0 on import |
| `test_*_event_has_correct_schema` | Each event type has all required fields |
| `test_log_entry_structure` | Log envelope has all fields, payload_summary is scalar-only |
| `test_no_secret_leakage` | No key/token/secret/password at any nesting level |
| `test_deterministic_output` | Same inputs → same event_id |
| `test_observability_context_includes_metrics` | Context has `metrics` + `recent_event_log` |
| `test_event_recording_accumulates` | Multiple events accumulate in log + increment counters |
| `test_no_network_telemetry` | No http/url/endpoint at any nesting level |

## Pitfalls

- **Counter mapping is event-specific:** `build_controlled_write_event` increments BOTH `controlled_write_count` AND `snapshot_count`. `rollback_drill` increments `verification_count`. Only `rollback_execution` increments `rollback_count`.
- **Timestamp in event body, NOT in event_id hash:** Event IDs are deterministic from content only. Timestamp is present in the body for human reading but excluded from the hash.
- **Scalar-only log summaries:** `payload_summary` strips all dict/list values to prevent accidental credential leakage. If the payload contains nested structures, they are not included in the log summary.
- **Module-level state is NOT thread-safe:** Counters and event log are module-level Python variables. This is acceptable for the single-threaded Hermes agent context but would need locking for concurrent access.
- **`datetime.utcnow()` is deprecated:** Use `datetime.now(datetime.UTC)` instead. The test file will emit warnings if this is wrong.

## Production Readiness Gate Integration

The observability layer feeds directly into `evaluate_production_readiness()`:

```python
result = evaluate_production_readiness({
    ...
    'observability': {
        'has_metrics': True,       # get_metrics() returns non-zero counts
        'has_logs': True,          # build_observability_context() returns event_log
        'has_audit_trail': True,   # existing JSON artifacts
    },
    ...
})
assert result['decision'] == 'go'
```

When the observability dimension failed (previous state), only `has_audit_trail` was `true`. After this layer, all three are `true`.
