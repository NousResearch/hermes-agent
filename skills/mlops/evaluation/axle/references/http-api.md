# AXLE HTTP API Reference

## Base URL

```text
https://axle.axiommath.ai
```

## Authentication

```text
Authorization: Bearer your_api_key
```

Without key: 10 concurrent active requests. With key: 20.
Get keys at https://axle.axiommath.ai/app/console

## Endpoints

### Utility (GET, no /api prefix)

```bash
# List environments
curl https://axle.axiommath.ai/v1/environments

# Health check
curl https://axle.axiommath.ai/v1/status
```

### Tools (POST /api/v1/<tool_name>)

All tools use POST with JSON body:

```bash
curl -X POST https://axle.axiommath.ai/api/v1/check \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $AXLE_API_KEY" \
  -d '{
    "content": "import Mathlib\ntheorem foo : 1+1=2 := by norm_num",
    "environment": "lean-4.28.0"
  }'
```

Available endpoints:
- `POST /api/v1/verify_proof`
- `POST /api/v1/check`
- `POST /api/v1/extract_theorems`
- `POST /api/v1/rename`
- `POST /api/v1/theorem2lemma`
- `POST /api/v1/theorem2sorry`
- `POST /api/v1/merge`
- `POST /api/v1/simplify_theorems`
- `POST /api/v1/repair_proofs`
- `POST /api/v1/have2lemma`
- `POST /api/v1/have2sorry`
- `POST /api/v1/sorry2lemma`
- `POST /api/v1/disprove`
- `POST /api/v1/normalize`

Note the URL pattern inconsistency: tools use `/api/v1/`, environments uses `/v1/`.

## Common request parameters

All tools accept:

```json
{
  "content": "string (required)",
  "environment": "string (required, e.g., lean-4.28.0)",
  "timeout_seconds": 120,
  "ignore_imports": false
}
```

`timeout_seconds`: default 120, max 300 for non-admin.

## Common response format

```json
{
  "okay": true,
  "content": "import Mathlib\n\ntheorem foo ...",
  "lean_messages": {"errors": [], "warnings": [], "infos": []},
  "tool_messages": {"errors": [], "warnings": [], "infos": []},
  "failed_declarations": [],
  "timings": {"parse_ms": 35, "total_ms": 41},
  "info": {
    "request_id": "uuid-v4",
    "environment": "lean-4.28.0",
    "total_request_time_ms": 47,
    "queue_time_ms": 4,
    "execution_time_ms": 42,
    "cached_response": false
  }
}
```

Not all fields present on all endpoints:
- `okay` — only on check, verify_proof, repair_proofs
- `failed_declarations` — only on check, verify_proof
- `documents` — only on extract_theorems
- `results`, `disproved_theorems` — only on disprove
- `lemma_names` — only on have2lemma, sorry2lemma
- `simplification_stats` — only on simplify_theorems
- `repair_stats` — only on repair_proofs
- `normalize_stats` — only on normalize

## Error responses

Fatal errors return one of three top-level error types (each is a separate response format):

```json
{"user_error": "Invalid argument: ..."}
```
```json
{"internal_error": "Server bug: ..."}
```
```json
{"error": "Runtime failure: timeout/OOM/crash"}
```

HTTP status codes:
- 200 — success
- 400 — invalid argument (user_error)
- 403 — forbidden (auth issue)
- 404 — not found
- 409 — conflict
- 429 — rate limited (retryable)
- 500 — internal error (bug)
- 503 — service unavailable (retryable)

## verify_proof example

```bash
curl -X POST https://axle.axiommath.ai/api/v1/verify_proof \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $AXLE_API_KEY" \
  -d '{
    "formal_statement": "theorem foo : 1 + 1 = 2 := by sorry",
    "content": "theorem foo : 1 + 1 = 2 := by norm_num",
    "environment": "lean-4.28.0",
    "ignore_imports": true,
    "use_def_eq": true
  }'
```

## merge example

```bash
curl -X POST https://axle.axiommath.ai/api/v1/merge \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $AXLE_API_KEY" \
  -d '{
    "documents": [
      "import Mathlib\ntheorem a : 1 = 1 := rfl",
      "import Mathlib\ntheorem b : 2 = 2 := rfl"
    ],
    "environment": "lean-4.28.0",
    "ignore_imports": true
  }'
```

## Environment response

```json
[
  {
    "name": "lean-4.28.0",
    "lean_toolchain": "leanprover/lean4:v4.28.0",
    "repo_url": null,
    "revision": null,
    "subdir": null,
    "imports": "import Mathlib",
    "description": "Lean 4.28.0 with Mathlib"
  }
]
```
