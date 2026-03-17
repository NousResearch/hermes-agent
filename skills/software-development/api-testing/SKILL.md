---
name: api-testing
description: Systematic API debugging and testing — REST and GraphQL. Use when diagnosing failing requests, unexpected responses, auth issues, or building API integration tests.
version: 1.1.0
author: eren-karakus0
license: MIT
metadata:
  hermes:
    tags: [api, rest, graphql, http, debugging, testing, curl, integration]
    related_skills: [systematic-debugging, test-driven-development]
---

# API Testing & Debugging

## Overview

API failures are deceptive. A 200 OK can hide broken data. A 500 can mask a simple auth typo.

**Core principle:** Isolate the layer, then fix. Never guess which part of the request chain is broken.

## When to Use

**Use this skill when:**
- API returns unexpected status code or body
- Auth flow fails (401/403 after token refresh, OAuth, API key)
- Request works in Postman/browser but fails in code
- Debugging webhook or callback integrations
- Building or reviewing API integration tests
- Investigating rate limiting or pagination issues

**Don't use when:**
- Issue is in UI rendering (not an API problem)
- Database query optimization (use profiling instead)
- Network infrastructure (DNS/firewall — escalate to ops)

## 5-Minute Quickstart

### REST

```bash
# Basic GET — see full request/response exchange
curl -v https://api.example.com/users/1

# POST with JSON body
curl -X POST https://api.example.com/users \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"name": "test", "email": "test@example.com"}'

# Inspect response headers only
curl -sI https://api.example.com/health

# Pretty-print JSON response
curl -s https://api.example.com/users | python -m json.tool
```

### GraphQL

```bash
# Query
curl -X POST https://api.example.com/graphql \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"query": "{ user(id: 1) { name email } }"}'

# Mutation
curl -X POST https://api.example.com/graphql \
  -H "Content-Type: application/json" \
  -d '{"query": "mutation { createUser(input: { name: \"test\" }) { id } }"}'

# Introspection (discover schema)
curl -X POST https://api.example.com/graphql \
  -H "Content-Type: application/json" \
  -d '{"query": "{ __schema { types { name } } }"}'
```

### GraphQL: Beware 200 OK with Errors

GraphQL often returns HTTP 200 even when errors exist. Some servers may return 400/500, but many don't. **Always check the `errors` field regardless of status code:**

```python
resp = requests.post(url, json={"query": query}, headers=headers)
data = resp.json()
if data.get("errors"):
    for err in data["errors"]:
        print(f"GraphQL error: {err['message']} (path: {err.get('path')})")
    # Treat as failure even though status is 200
```

**Rule:** For GraphQL, always inspect the response body for `errors` — status code alone is unreliable.

### Python (requests)

```python
import requests

resp = requests.get(
    "https://api.example.com/users/1",
    headers={"Authorization": f"Bearer {token}"},
    timeout=10,
)
print(resp.status_code)
print(resp.headers)
print(resp.json())
```

## Systematic Debug Flow

Work through each layer in order. Do NOT skip layers — a TLS error masquerading as a timeout wastes hours.

```
Step 1: Connectivity      → Can we reach the host at all?
Step 1.5: Timeouts        → Is the connection slow or hanging?
Step 2: TLS/SSL           → Is the certificate valid and trusted?
Step 3: Authentication    → Are credentials correct and unexpired?
Step 4: Request format    → Is the payload what the server expects?
Step 5: Response parse    → Is the response what our code expects?
Step 6: Semantics         → Does the data mean what we think it does?
```

### Step 1: Connectivity

```bash
# DNS resolution
nslookup api.example.com

# TCP connectivity
curl -v --connect-timeout 5 https://api.example.com/health
```

**Common failures:** DNS not resolving, firewall blocking port, VPN required.

### Step 1.5: Timeouts

Distinguish between connection timeout (can't reach host) and read timeout (host accepted but is slow):

```bash
# Separate connect vs total timeout
curl --connect-timeout 5 --max-time 30 https://api.example.com/slow-endpoint

# Show timing breakdown
curl -w "dns: %{time_namelookup}s\nconnect: %{time_connect}s\ntls: %{time_appconnect}s\nfirst_byte: %{time_starttransfer}s\ntotal: %{time_total}s\n" \
  -o /dev/null -s https://api.example.com/endpoint
```

```python
# Python: always use tuple timeout (connect, read)
resp = requests.get(url, timeout=(3.05, 30))  # 3s connect, 30s read

# Catch timeouts separately
from requests.exceptions import ConnectTimeout, ReadTimeout
try:
    resp = requests.get(url, timeout=(3.05, 30))
except ConnectTimeout:
    print("Cannot reach host — check DNS, firewall, VPN")
except ReadTimeout:
    print("Host accepted connection but response is too slow")
```

**Diagnosis guide:**
- `time_connect` high → network/firewall issue, not the API
- `time_starttransfer` high but `time_connect` low → server is slow processing
- Request hangs indefinitely → missing timeout param (Python `requests` has no default timeout!)

### Step 2: TLS/SSL

```bash
# Show certificate chain and expiry
curl -vI https://api.example.com 2>&1 | grep -E "SSL|subject|expire|issuer"

# Skip TLS verification (debug only, never in production)
curl -k https://api.example.com/health
```

**Common failures:** Expired cert, self-signed cert, wrong hostname on cert, missing CA bundle.

### Step 3: Authentication

```bash
# Test token validity
curl -s -o /dev/null -w "%{http_code}" \
  -H "Authorization: Bearer $TOKEN" \
  https://api.example.com/me

# Check token expiry (JWT) — handles base64url padding
python -c "import json,base64,sys; p=sys.argv[1].split('.')[1]; p+='='*(-len(p)%4); print(json.dumps(json.loads(base64.urlsafe_b64decode(p)),indent=2))" "$TOKEN"
```

**Debug checklist:**
- Is the token expired? Check `exp` claim in JWT.
- Is it the right token type? (Bearer vs Basic vs API key)
- Is the header name correct? (`Authorization` vs `X-Api-Key` vs custom)
- Is the token for the right environment? (staging key on production)

### Step 4: Request Format

```bash
# Log exact request being sent
curl -v -X POST https://api.example.com/endpoint \
  -H "Content-Type: application/json" \
  -d '{"key": "value"}' 2>&1
```

**Debug checklist:**
- Content-Type matches body format? (`application/json` vs `multipart/form-data`)
- Required fields present?
- Correct HTTP method? (POST vs PUT vs PATCH)
- Query params properly encoded? (`%20` not spaces)
- Request body valid JSON/XML? (use `python -m json.tool` to validate)

**Content-Type mismatch — the silent 415/400 cause:**

```python
# WRONG: data= sends form-encoded, but header says JSON
requests.post(url, data='{"key": "value"}', headers={"Content-Type": "application/json"})

# RIGHT: json= auto-sets Content-Type and serializes
requests.post(url, json={"key": "value"})

# WRONG: Accept header doesn't match what you parse
requests.get(url, headers={"Accept": "text/xml"})  # then calling .json()

# For file uploads — don't set Content-Type manually, let requests handle multipart
requests.post(url, files={"file": open("doc.pdf", "rb")})
```

Common mismatches:
- `json=` vs `data=` in Python requests (different Content-Type, different encoding)
- Server expects `application/x-www-form-urlencoded` but you send JSON
- `Accept` header says XML but you parse as JSON
- Manual `Content-Type: multipart/form-data` without boundary (let the library set it)

### Step 5: Response Parsing

```python
resp = requests.post(url, json=payload, timeout=10)

# Don't just check status — inspect everything
print(f"Status: {resp.status_code}")
print(f"Headers: {dict(resp.headers)}")
print(f"Body: {resp.text[:500]}")

# Check content type before parsing
if "application/json" in resp.headers.get("Content-Type", ""):
    data = resp.json()
else:
    print(f"Unexpected content type: {resp.headers.get('Content-Type')}")
    print(f"Raw body: {resp.text[:500]}")
```

**Common failures:** HTML error page returned instead of JSON, empty body, wrong encoding.

### Step 6: Semantic Validation

The response parsed successfully — but is the data correct?

- Does `"status": "active"` actually mean what your code assumes?
- Is the ID in the response the one you requested?
- Are timestamps in the expected timezone?
- Is pagination returning all results or just the first page?

## HTTP Status Playbook

### 401 Unauthorized

```
Diagnosis: Credentials missing or invalid.
```

1. Is the `Authorization` header present? (`curl -v` to verify)
2. Is the token/key correct and unexpired?
3. Is the auth scheme correct? (`Bearer` vs `Basic` vs `Token`)
4. API key: header vs query param? (some APIs use `?api_key=`)

### 403 Forbidden

```
Diagnosis: Authenticated but not authorized.
```

1. Does this user/token have the required permissions/scopes?
2. Is the resource owned by a different account?
3. Is there an IP allowlist blocking you?
4. Is CORS blocking a browser request? (check `Access-Control-Allow-Origin`)

### 404 Not Found

```
Diagnosis: Resource doesn't exist or URL is wrong.
```

1. Is the URL path correct? (trailing slash, typo, wrong version prefix)
2. Does the resource ID exist?
3. Is the API version correct? (`/v1/` vs `/v2/`)
4. Is the base URL correct? (staging vs production)

### 409 Conflict

```
Diagnosis: State conflict — usually duplicate or concurrent modification.
```

1. Does the resource already exist? (duplicate create)
2. Is there a stale `ETag` or `If-Match` header?
3. Is another process modifying the same resource?

### 422 Unprocessable Entity

```
Diagnosis: Valid JSON but invalid data.
```

1. Read the error body — it usually lists exactly which fields are wrong.
2. Check field types (string vs int, date format).
3. Check required vs optional fields.
4. Check enum values — is your value in the allowed set?

### 429 Too Many Requests

```
Diagnosis: Rate limited.
```

1. Check `Retry-After` header for wait time.
2. Check `X-RateLimit-*` headers for limit/remaining/reset.
3. Implement exponential backoff:

```python
import time

def request_with_backoff(method, url, **kwargs):
    max_retries = 5
    for attempt in range(max_retries):
        resp = requests.request(method, url, **kwargs)
        if resp.status_code != 429:
            return resp
        wait = int(resp.headers.get("Retry-After", 2 ** attempt))
        time.sleep(wait)
    return resp
```

### 5xx Server Errors

```
Diagnosis: Server-side failure — usually not your fault.
```

1. **500 Internal Server Error** — Server bug. Log your request for the provider.
2. **502 Bad Gateway** — Upstream server down. Retry with backoff.
3. **503 Service Unavailable** — Server overloaded or in maintenance. Check status page.
4. **504 Gateway Timeout** — Upstream timeout. Reduce payload size or increase timeout.

**For all 5xx:** retry with exponential backoff, alert if persistent.

## Pagination & Idempotency

**Pagination:** Always verify you're getting all results. Check for `next_cursor`, `next_page`, or `total_count` in responses. Two common patterns:
- **Offset-based:** `?limit=100&offset=200` — simple but can skip items if data changes.
- **Cursor-based:** `?cursor=abc123` — preferred for large or live datasets.

**Idempotency:** For non-idempotent operations (POST), send an `Idempotency-Key` header (UUID) to prevent duplicate execution on retries. Critical for payments, orders, and any create operation.

## Contract Validation

Validate response structure before using the data. Catch schema drift early:

```python
def validate_user_response(data: dict) -> list[str]:
    """Validate required fields exist and have expected types."""
    errors = []
    required = {"id": int, "email": str, "created_at": str}
    for field, expected_type in required.items():
        if field not in data:
            errors.append(f"Missing required field: {field}")
        elif not isinstance(data[field], expected_type):
            errors.append(f"Field '{field}': expected {expected_type.__name__}, got {type(data[field]).__name__}")
    return errors

resp = requests.get(f"{base}/users/1", headers=headers)
issues = validate_user_response(resp.json())
if issues:
    print(f"Contract violations: {issues}")
```

**When to validate:**
- After any API upgrade or version change
- When integrating a new third-party API
- In CI smoke tests to catch breaking changes early

## Correlation IDs

Most APIs return a request ID for tracing. **Always capture it** — it's the fastest way to get provider support:

```python
resp = requests.post(url, json=payload, headers=headers)
request_id = (
    resp.headers.get("X-Request-Id")
    or resp.headers.get("X-Trace-Id")
    or resp.headers.get("CF-Ray")  # Cloudflare
)

if resp.status_code >= 400:
    print(f"Request failed: {resp.status_code}")
    print(f"Correlation ID: {request_id}")
    print(f"Timestamp: {resp.headers.get('Date')}")
    # Include these when filing a bug with the API provider
```

**Bug report template for API providers:**

```
Endpoint: POST /api/v1/orders
Request ID: req_abc123xyz
Timestamp: 2026-03-17T14:30:00Z
Status: 500
Expected: 201 with order object
Actual: 500 {"error": "internal server error"}
Repro: curl -X POST ... (with <REDACTED> auth)
```

## Regression Test Template

Minimal pytest smoke test for API endpoints — run in CI to catch breaking changes:

```python
import os
import pytest
import requests

BASE_URL = os.environ.get("API_BASE_URL", "https://api.example.com")
TOKEN = os.environ.get("API_TOKEN", "")
HEADERS = {"Authorization": f"Bearer {TOKEN}"}


class TestAPISmoke:
    """Lightweight smoke tests — verify endpoints are reachable and return expected shapes."""

    def test_health(self):
        resp = requests.get(f"{BASE_URL}/health", timeout=5)
        assert resp.status_code == 200

    def test_list_users_returns_array(self):
        resp = requests.get(f"{BASE_URL}/users", headers=HEADERS, timeout=10)
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data.get("data", data), list)

    def test_get_user_returns_required_fields(self):
        resp = requests.get(f"{BASE_URL}/users/1", headers=HEADERS, timeout=10)
        assert resp.status_code in (200, 404)
        if resp.status_code == 200:
            user = resp.json()
            assert "id" in user
            assert "email" in user

    def test_invalid_auth_returns_401(self):
        resp = requests.get(
            f"{BASE_URL}/users",
            headers={"Authorization": "Bearer invalid-token"},
            timeout=10,
        )
        assert resp.status_code == 401
```

**Usage:** `API_BASE_URL=https://staging.example.com API_TOKEN=sk-... pytest test_api_smoke.py -v`

## Security

### Token Handling

- **Never log full tokens.** Redact in output: `Bearer <REDACTED>`
- **Never hardcode tokens** in scripts or skill output. Use env vars.
- **Rotate immediately** if a token appears in logs, error messages, or version control.

### Safe Logging

```python
def redact_auth(headers: dict) -> dict:
    """Redact sensitive headers for safe logging."""
    sensitive = {"authorization", "x-api-key", "cookie", "set-cookie"}
    return {
        k: ("<REDACTED>" if k.lower() in sensitive else v)
        for k, v in headers.items()
    }

# Log safely
print(f"Request headers: {redact_auth(dict(resp.request.headers))}")
```

### Sensitive Data Leak Checklist

After every API integration, scan for these leaks:

- [ ] **Credentials in URLs** — API keys in query strings end up in server logs, browser history, and referrer headers. Use headers instead.
- [ ] **PII in error responses** — Does a 404 on `/users/123` reveal that the user exists vs doesn't? (user enumeration)
- [ ] **Stack traces in production** — 500 responses should not include file paths, line numbers, or framework internals.
- [ ] **Internal hostnames/IPs** — Error messages or headers leaking `10.x.x.x`, `internal-api.corp.local`, or database hostnames.
- [ ] **Tokens in error bodies** — Some APIs echo back the auth token in error details. Verify they don't.
- [ ] **Verbose headers** — `Server: Apache/2.4.41 (Ubuntu)` or `X-Powered-By: Express` reveal stack info. Not your problem to fix, but note for security reviews.

### Request Logging for Reproduction

When reporting API issues, provide a reproducible curl command with redacted secrets:

```bash
curl -X POST https://api.example.com/endpoint \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <REDACTED>" \
  -d '{"field": "value"}'
# Response: 422 {"error": "field 'email' is required"}
```

## Output Format

When reporting API debug findings, use this structure:

```
## Finding

**Endpoint:** POST /api/v1/users
**Status:** 422 Unprocessable Entity

## Repro

curl -X POST https://api.example.com/api/v1/users \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <REDACTED>" \
  -d '{"name": "test"}'

## Root Cause

Missing required field `email` in request body.
Server validation rejects the request before processing.

## Fix

Add `email` field to the request payload:
  -d '{"name": "test", "email": "test@example.com"}'
```

## Hermes Agent Integration

### With terminal

Use `terminal` to run curl commands and inspect responses directly:

```python
terminal("curl -s -w '\\nHTTP_CODE:%{http_code}' https://api.example.com/health")
```

### With execute_code

For complex flows (auth + paginate + validate):

```python
execute_code("""
import requests

token = "<from env or config>"
base = "https://api.example.com"

# Step 1: Auth check
me = requests.get(f"{base}/me", headers={"Authorization": f"Bearer {token}"})
print(f"Auth: {me.status_code}")

# Step 2: Fetch data
resp = requests.get(f"{base}/users", headers={"Authorization": f"Bearer {token}"})
print(f"Users: {resp.status_code}, count={len(resp.json().get('data', []))}")
""")
```

### With delegate_task

For comprehensive API test suites:

```python
delegate_task(
    goal="Test all CRUD endpoints for /api/v1/users",
    context="""
    Follow api-testing skill.
    Base URL: https://api.example.com
    Auth: Bearer token from EXAMPLE_API_KEY env var

    Test each endpoint:
    1. POST /users — create
    2. GET /users/:id — read
    3. PATCH /users/:id — update
    4. DELETE /users/:id — delete

    For each: verify status code, response schema, error cases (400, 404, 422).
    Log repro curl for any failure.
    Redact all tokens in output.
    """,
    toolsets=['terminal', 'file']
)
```

### With systematic-debugging

API returning errors? Follow the debug flow:
1. Use this skill to isolate the failing layer (connectivity → TLS → auth → request → response)
2. Use systematic-debugging to trace root cause in your code
3. Use test-driven-development to write regression tests for the fix
