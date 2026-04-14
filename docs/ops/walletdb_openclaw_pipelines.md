# WalletDB OpenClaw deterministic pipeline helpers (JSON-only)

## JSON-only rendering convention
All pipeline responses MUST be a single JSON object (no prose, no markdown). This keeps downstream parsing deterministic.

Required top-level keys:
- `status`: "ok" | "error" | "noop"
- `summary`: short human summary (string)
- `outputs`: object with named results
- `actions`: array of planned actions (may be empty)
- `errors`: array of error objects (may be empty)

Optional keys:
- `metrics`: numeric counters
- `debug`: string or object (only when explicitly requested)

### Example (success)
```json
{
  "status": "ok",
  "summary": "Dispatched 3 alerts; 0 failures",
  "outputs": {
    "dispatched": 3,
    "alerts": [
      {"alert_id": 123, "channel": "telegram", "target": "5936691533"}
    ]
  },
  "actions": [
    {"type": "notify", "channel": "telegram", "target": "5936691533"}
  ],
  "errors": []
}
```

### Example (error)
```json
{
  "status": "error",
  "summary": "Webhook payload missing idempotency_key",
  "outputs": {},
  "actions": [],
  "errors": [
    {"code": "missing_field", "field": "idempotency_key"}
  ]
}
```

## Deterministic pipeline guidelines
1. **Validate input schema first**; return JSON error immediately on mismatch.
2. **Idempotency before side effects**; check `idempotency_key` and `event_id`.
3. **Explicit outputs**; no implicit state changes unless returned in `actions`.
4. **Stable ordering**; keep arrays deterministically sorted (e.g., by id).
5. **No free-form text**; JSON only unless user explicitly requests prose.
