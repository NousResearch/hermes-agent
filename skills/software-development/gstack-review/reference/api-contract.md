# API Contract Specialist Review Checklist

Scope: When SCOPE_API=true
Output: JSON objects, one finding per line. Schema:
{"severity":"CRITICAL|INFORMATIONAL","confidence":N,"path":"file","line":N,"category":"api-contract","summary":"...","fix":"...","fingerprint":"path:line:api-contract","specialist":"api-contract"}
Optional: line, fix, fingerprint, evidence, test_stub.
If no findings: output `NO FINDINGS` and nothing else.

---

## Categories

### Breaking Changes
- Removed fields from response bodies (clients may depend on them)
- Changed field types (string → number, object → array)
- New required parameters added to existing endpoints
- Changed HTTP methods (GET → POST) or status codes (200 → 201)
- Renamed endpoints without maintaining the old path as a redirect/alias
- Changed authentication requirements (public → authenticated)

### Versioning Strategy
- Breaking changes made without a version bump (v1 → v2)
- Multiple versioning strategies mixed in the same API (URL vs header vs query param)
- Deprecated endpoints without a sunset timeline or migration guide
- Version-specific logic scattered across controllers instead of centralized

### Error Response Consistency
- New endpoints returning different error formats than existing ones
- Error responses missing standard fields (error code, message, details)
- HTTP status codes that don't match the error type (200 for errors, 500 for validation)
- Error messages that leak internal implementation details (stack traces, SQL)

### Rate Limiting & Pagination
- New endpoints missing rate limiting when similar endpoints have it
- Pagination changes (offset → cursor) without backwards compatibility
- Changed page sizes or default limits without documentation
- Missing total count or next-page indicators in paginated responses

### Documentation Drift
- OpenAPI/Swagger spec not updated to match new endpoints or changed params
- README or API docs describing old behavior after changes
- Example requests/responses that no longer work
- Missing documentation for new endpoints or changed parameters

### Backwards Compatibility
- Clients on older versions: will they break?
- Mobile apps that can't force-update: does the API still work for them?
- Webhook payloads changed without notifying subscribers
- SDK or client library changes needed to use new features
