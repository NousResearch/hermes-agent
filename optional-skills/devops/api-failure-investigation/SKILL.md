---
name: api-failure-investigation
description: Systematic approach to diagnose API connectivity issues through endpoint matrix testing, avoiding infinite retry loops and token waste.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [api, troubleshooting, debugging, connectivity]
    related_skills: [systematic-debugging, request-code-review]
---

# API Failure Investigation

## When to Use
When API calls repeatedly fail and you need to determine:
- Whether endpoints are actually available
- Root cause (wrong URL, auth issues, service down, version mismatch)
- Whether to retry, adjust parameters, or seek alternative approaches

**Use this to avoid:**
- Infinite retry loops
- Wasting tokens on repeated failed calls
- User frustration from prolonged troubleshooting
- Guessing without evidence

## Principles
1. **Test antes de commit** - Verify endpoint availability with minimal requests first
2. **Pattern before persistence** - If 3+ endpoints fail the same way, service is likely unavailable
3 **Document every attempt** - Keep clear record of what was tested and results
4. **Token-conscious investigation** - Short timeouts, small payloads, abort early on systematic failure

## Steps

### Phase 1: Environment & Configuration Check
```python
# Verify configuration exists and load values
- Check config files (e.g., ~/.config/ima/client_id, api_key)
- Validate non-empty credentials
- Print masked values to confirm loading
- Note base URL and any paths
```

### Phase 2: Create Test Matrix
List all candidate endpoints based on documentation and common patterns:
```
- /
- /api/v1/resource
- /v1/resource
- /openapi/resource
- /resource (legacy)
```

Include variations:
- Different HTTP methods (GET to test availability, then POST for actual operations)
- With/without auth headers
- Different content types

### Phase 3: Minimal GET Tests (Discovery)
For each endpoint with GET (or HEAD):
- Request with authentication headers
- Short timeout (10s)
- Record: status code, response body (first 200 chars), latency
- **Stop after first successful response** - base path found

**Analysis:**
- 200 → Endpoint exists, proceed to Phase 4
- 404 → Path not found (systematic issue)
- 401/403 → Auth problem (verify headers, permissions)
- 400 → Bad request (check payload format)
- 500+ → Server error (transient or bug)
- No response/timeout → Network or service down

**Decision rule:** If ≥3 endpoints return 404 with same pattern, API service is likely unavailable. Document and report. Do NOT continue testing POST.

### Phase 4: Targeted POST Tests (If GET succeeded)
Only if base path confirmed working:
- Use smallest possible payload (1-2 lines)
- Test expected content-type (multipart/form-data, application/json)
- Include required fields only
- Keep payload <1KB
- Test one endpoint at a time

### Phase 5: Pattern Analysis & Recommendation
Based on evidence:
- **All 404:** API not deployed or wrong base URL → Contact support or check documentation
- **Mixed 200/404:** Version mismatch → Try documented versions
- **401/403:** Authentication failure → Verify headers, token expiry, permissions
- **400:** Payload format wrong → Read API spec, adjust structure
- **500:** Server error → Transient, retry later or report
- **Timeout:** Service down or network issue → Check service status

## Pitfalls
- ❌ Don't keep retrying same failing endpoint hoping for different result
- ❌ Don't test large payloads before endpoint confirmed
- ❌ Don't assume "temporary glitch" without evidence
- ✅ Do mask credentials in logs
- ✅ Do provide user with clear verdict and next steps

## Integration with systematic-debugging
- Use systematic-debugging Phase 1 to gather error logs and evidence
- This skill provides specific API testing methodology
- Combine: understand error → test hypothesis with endpoint matrix

## Example Output Format
```
=== API Endpoint Test Results ===
Base URL: https://api.example.com
Auth: headers present (masked)

GET /
  Status: 200
  Body: HTML landing page

GET /api/v1/upload
  Status: 404
  Body: {"error":"Not Found"}

GET /v1/upload
  Status: 404
  Body: {"error":"Not Found"}

Conclusion: All API paths return 404. Root path works but API service appears unavailable. Recommend: contact API provider to confirm endpoint URLs and service status.
```

## Success Criteria
- All reasonable endpoints tested (≤10)
- Results documented with status codes
- Clear root cause hypothesis
- User receives actionable recommendation (retry, contact support, adjust config, alternative method)
- Token usage minimized (≤5-10 API calls total)
