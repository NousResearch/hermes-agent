# WalletDB webhook-first triggers schema

## Trigger envelope (JSON)
```json
{
  "event_id": "evt_2026_02_10_0001",
  "event_type": "walletdb.alert.created",
  "idempotency_key": "walletdb:alert:0001",
  "occurred_at": "2026-02-10T02:55:00Z",
  "source": "walletdb",
  "payload": {
    "alert_id": 123,
    "chain": "solana",
    "entity_id": "CL_abc",
    "alert_type": "large_transfer",
    "amount_usd": 8250.12
  },
  "trace": {
    "request_id": "req_01",
    "span_id": "span_01"
  },
  "reply_to": {
    "channel": "telegram",
    "target": "5936691533"
  }
}
```

## Required fields
- `event_id`: unique event id (string)
- `event_type`: stable event name (string)
- `idempotency_key`: unique key to dedupe (string)
- `occurred_at`: ISO8601 UTC timestamp
- `source`: emitter id (e.g., "walletdb")
- `payload`: event body (object)

## Optional fields
- `trace`: trace metadata
- `reply_to`: default reply route

## Expected response (JSON)
```json
{
  "status": "ok",
  "accepted": true,
  "idempotency_key": "walletdb:alert:0001",
  "message": "queued"
}
```

## Idempotency rules
- If `idempotency_key` already processed: return `{status:"ok", accepted:false, message:"duplicate"}`.
- Store `event_id` and `idempotency_key` with TTL >= 24h.
- Side effects must only happen on the **first** accept.
