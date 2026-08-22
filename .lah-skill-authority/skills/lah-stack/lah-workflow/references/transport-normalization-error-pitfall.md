# Transport `_normalizationError` — E2 Classification Precedence Pitfall

## Symptom

A transport wrapper catches an HTTP rejection from `submitApprovalRaw` (e.g. 400), extracts `httpStatus=400`, but also sets `_normalizationError: err.message`. E2's `normalizeTransportResult` classifies it as `PROTOCOL_ERROR` instead of `REJECTED`, losing the valid HTTP status classification.

## Root cause

E2's `normalizeTransportResult` checks `_normalizationError` FIRST, before any HTTP status classification:

```javascript
// In normalizeTransportResult:
if (rawResult._normalizationError) {
    classification = ATTEMPT_CLASSIFICATION.SUBMISSION_ATTEMPT_PROTOCOL_ERROR;
    unknownOutcome = true;
    reconciliationRequired = true;
}
// ... later checks for httpStatus, timeout, etc.
```

This means ANY error message in `_normalizationError` overrides HTTP status classification, even when the transport correctly extracted a meaningful status code.

## Fix

Only set `_normalizationError` when:

1. No HTTP status could be extracted from the error, AND
2. The error is not a known transport failure (timeout, connection reset)

```javascript
// Correct pattern:
const hasStructuredError = httpStatus !== null;
const normalizationError = (!hasStructuredError && !isTimeout && !isConnectionError)
  ? (err.message || 'Transport threw an error')
  : null;

return {
  transportStatus: 'error',
  httpStatus,
  // ... other fields
  _normalizationError: normalizationError, // null when HTTP status is known
};
```

## Rules

| Condition | `httpStatus` | `_normalizationError` | E2 classification |
|-----------|-------------|----------------------|-------------------|
| Timeout before send | null | not set | TRANSPORT_ERROR |
| Timeout after send | null | not set | UNKNOWN_OUTCOME |
| HTTP 400 (explicit rejection) | 400 | null | REJECTED |
| HTTP 500 (server error) | 500 | null | UNKNOWN_OUTCOME / TRANSPORT_ERROR |
| Unknown error, no status | null | err.message | PROTOCOL_ERROR |
| Connection reset | null | not set | UNKNOWN_OUTCOME / TRANSPORT_ERROR |

## Test pattern

```javascript
// Transport throws an HTTP rejection
const httpErr = new Error('LAHB approval submit failed: 400 {"error":"bad request"}');
const client = buildFakeRawSubmitClientThatThrows(httpErr);
const result = await performGovernedX402RealLahbSubmission({ ... });
assert.equal(result.classification, 'REAL_SUBMISSION_ATTEMPT_REJECTED');
```

The mock `submitApprovalRaw` throws with `"failed: 400"` in the message. The production transport extracts `httpStatus=400` and sets `_normalizationError=null`. E2 classifies by HTTP status as REJECTED.
