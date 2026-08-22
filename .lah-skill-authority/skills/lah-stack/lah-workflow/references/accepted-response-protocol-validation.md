# Accepted-Response Protocol Validation

HTTP `2xx` alone is **not sufficient** to classify a transport response as `ACCEPTED`. Protocol-level evidence must be verified before accepting the response. This is a general transport-boundary security pattern applicable to any HTTP-based adapter.

## Principle

Transport success and protocol validity are separate conditions:

```
2xx + valid protocol evidence → ACCEPTED
2xx + malformed/invalid evidence → PROTOCOL_ERROR + unknownOutcome (if request may have been sent)
```

## Minimum Protocol Evidence Requirements

For a `2xx` response to be classified as `ACCEPTED`, ALL must be true:

| Requirement | Check |
|-------------|-------|
| Response body exists | Not null, not undefined |
| Body is valid JSON | Parsable |
| Body is a JSON object | Not a primitive, not an array |
| At least one valid remote identifier | `approval_request_id` or `entry_id` |
| No contradictory fields | No field that contradicts the HTTP 2xx status |

## Remote Field Validation

Each remote identifier must be validated explicitly:

| Property | Behavior |
|----------|----------|
| **PROPERTY_ABSENT** | Field does not exist in the response object → allowed for optional fields |
| **PROPERTY_PRESENT_VALID** | Field exists, is a non-empty string ≤ 256 chars, contains only safe chars → use the value |
| **PROPERTY_PRESENT_INVALID** | Field exists but fails validation → protocol error with reason code |

### Validation rules per remote field:

- **Type**: Must be a string (numbers, arrays, objects, booleans → invalid)
- **Empty**: Must not be zero-length
- **Whitespace**: Must not be whitespace-only after trim
- **Length**: Must not exceed 256 characters (or documented protocol limit)
- **Character policy**: Only safe identifier characters (`[a-zA-Z0-9_\-./:@]+`)
- **Absent v. empty distinction**: Use `Object.prototype.hasOwnProperty.call(body, field)` to detect absent vs present-but-empty

## Contradictory Response Detection

A `2xx` response is contradictory (and therefore protocol-invalid) when:

```
response.status === "REJECTED"
response.accepted === false
response.rejected === true
```

These should be detected and classified as `PROTOCOL_ERROR`, not `ACCEPTED`.

## Classification Flow

```text
Raw transport result
  → Is there a _normalizationError?
    → PROTOCOL_ERROR + unknownOutcome=true
  → Is it timed out?
    → UNKNOWN_OUTCOME (if request may have been sent)
    → TRANSPORT_ERROR (if not sent)
  → Is it connection reset?
    → Same as timeout
  → Has HTTP status?
    → 2xx → validateAcceptedResponse()
      → Valid → ACCEPTED
      → Invalid → PROTOCOL_ERROR
        → unknownOutcome = requestMayHaveBeenSent
        → reconciliationRequired = requestMayHaveBeenSent
    → 4xx (rejected) → REJECTED
    → 5xx → UNKNOWN_OUTCOME or TRANSPORT_ERROR
    → Other → PROTOCOL_ERROR
  → No HTTP status?
    → Not sent → TRANSPORT_ERROR
    → Possibly sent → UNKNOWN_OUTCOME
```

## Reason Codes

Machine-readable codes for protocol errors:

```
TRANSPORT_RESPONSE_BODY_MALFORMED    — Body is not valid JSON
TRANSPORT_RESPONSE_BODY_TYPE_INVALID — Body is not a JSON object
ACCEPTED_RESPONSE_MISSING_BODY       — Body is null/undefined
REQUIRED_REMOTE_ID_MISSING           — No valid remote identifier found
REMOTE_APPROVAL_REQUEST_ID_INVALID   — approval_request_id present but invalid
REMOTE_ENTRY_ID_INVALID              — entry_id present but invalid
REMOTE_RECEIPT_ID_INVALID            — receipt_id present but invalid
REMOTE_STATUS_INVALID                — status present but invalid
REMOTE_PROTOCOL_VERSION_INVALID      — protocol_version present but invalid
TRANSPORT_RESPONSE_CONTRADICTORY     — Body contradicts HTTP status
```

## Implementation Pattern (JavaScript/Node.js)

```javascript
const MAX_REMOTE_ID_LENGTH = 256;
const REMOTE_ID_PATTERN = /^[a-zA-Z0-9_\-./:@]+$/;

function validateRemoteField(parsedBody, fieldName) {
  const hasOwn = Object.prototype.hasOwnProperty.call(parsedBody, fieldName);
  if (!hasOwn) return { present: false };

  const value = parsedBody[fieldName];
  if (typeof value !== 'string') {
    return { present: true, valid: false, error: 'wrong type' };
  }
  if (value.length === 0) {
    return { present: true, valid: false, error: 'empty' };
  }
  if (value.trim().length === 0) {
    return { present: true, valid: false, error: 'whitespace-only' };
  }
  if (value.length > MAX_REMOTE_ID_LENGTH) {
    return { present: true, valid: false, error: 'too long' };
  }
  if (!REMOTE_ID_PATTERN.test(value)) {
    return { present: true, valid: false, error: 'invalid chars' };
  }
  return { present: true, valid: true, value };
}

function validateAcceptedResponse(parsedBody, rawBody) {
  if (!rawBody) return { valid: false, code: 'ACCEPTED_RESPONSE_MISSING_BODY' };
  if (!parsedBody || typeof parsedBody !== 'object' || Array.isArray(parsedBody)) {
    return { valid: false, code: 'TRANSPORT_RESPONSE_BODY_TYPE_INVALID' };
  }
  // ... validate IDs, contradiction check ...
}
```

## Common Pitfalls

| Pitfall | Why it's wrong | Fix |
|---------|----------------|-----|
| `const id = body.id \|\| null` | Collapses absent, empty, `0`, `false` into same result | Use `hasOwnProperty` + explicit type check |
| `if (body) { parse }` without null check | `body` could be `null` → still falsy, but no error evidence | Check `typeof rawBody === 'string'` before parsing |
| Accepting 2xx from HTTP status alone | Ignores malformed bodies, contradictions, missing IDs | Run `validateAcceptedResponse()` before classifying |
| `catch { /* ignore */ }` on JSON parse | Silently treats unparseable body as valid | Set a protocol error flag in the catch block |
