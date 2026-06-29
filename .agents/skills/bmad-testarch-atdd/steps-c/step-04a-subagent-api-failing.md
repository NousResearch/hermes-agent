---
name: 'step-04a-subagent-api-failing'
description: 'Subagent: Generate red-phase API test scaffolds (TDD red phase)'
subagent: true
outputFile: '/tmp/tea-atdd-api-tests-{{timestamp}}.json'
---

# Subagent 4A: Generate Red-Phase API Test Scaffolds (TDD Red Phase)

## SUBAGENT CONTEXT

This is an **isolated subagent** running in parallel with E2E red-phase test generation.

**What you have from parent workflow:**

- Story acceptance criteria from Step 1
- Test strategy and scenarios from Step 3
- Knowledge fragments loaded: api-request, data-factories, api-testing-patterns
- Config: test framework, Playwright Utils enabled/disabled, Pact.js Utils enabled/disabled (`use_pactjs_utils`), Pact MCP mode (`pact_mcp`)
- Provider Endpoint Map (if `use_pactjs_utils` enabled and provider source accessible)

**If `use_pactjs_utils` is enabled:** Also generate consumer contract tests alongside API tests. Use the loaded pactjs-utils fragments (`pactjs-utils-overview`, `pactjs-utils-consumer-helpers`, `pactjs-utils-provider-verifier`, `pactjs-utils-request-filter`, `pact-consumer-di`, `pact-consumer-framework-setup`) for patterns. When generating consumer tests, enforce **one `pact.addInteraction()` per `it()` block** (see `pactjs-utils-consumer-helpers.md` Example 6) — PactV4's Rust FFI non-deterministically drops interactions if multiple are chained in one test. If `pact_mcp` is `"mcp"`, use SmartBear MCP tools (Fetch Provider States, Generate Pact Tests) to inform test generation.

**Your task:** Generate API test scaffolds for the feature's expected behavior. They stay in `test.skip()` until the developer activates them for the current task (TDD RED PHASE).

---

## MANDATORY EXECUTION RULES

- 📖 Read this entire subagent file before acting
- ✅ Generate red-phase API test scaffolds ONLY
- ✅ Tests MUST be emitted with `test.skip()` until the developer activates them
- ✅ Output structured JSON to temp file
- ✅ Follow knowledge fragment patterns
- ❌ Do NOT generate E2E tests (that's subagent 4B)
- ❌ Do NOT generate active passing tests (this is TDD red phase)
- ❌ Do NOT run tests (that's step 5)

---

## SUBAGENT TASK

### 1. Identify API Endpoints from Acceptance Criteria

From the story acceptance criteria (Step 1 output), identify:

- Which API endpoints will be created for this story
- Expected request/response contracts
- Authentication requirements
- Expected status codes and error scenarios

**Example Acceptance Criteria:**

```
Story: User Registration
- As a user, I can POST to /api/users/register with email and password
- System returns 201 Created with user object
- System returns 400 Bad Request if email already exists
- System returns 422 Unprocessable Entity if validation fails
```

### 2. Generate Red-Phase API Test Files

For each API endpoint, create test file in `tests/api/[feature].spec.ts`:

**Test Structure (ATDD - Red Phase):**

```typescript
import { test, expect } from '@playwright/test';
// If Playwright Utils enabled:
// import { apiRequest } from '@playwright-utils/api';

test.describe('[Story Name] API Tests (ATDD)', () => {
  test.skip('[P0] should register new user successfully', async ({ request }) => {
    // THIS TEST WILL FAIL - Endpoint not implemented yet
    const response = await request.post('/api/users/register', {
      data: {
        email: 'newuser@example.com',
        password: 'SecurePass123!',
      },
    });

    // Expect 201 but will get 404 (endpoint doesn't exist)
    expect(response.status()).toBe(201);

    const user = await response.json();
    expect(user).toMatchObject({
      id: expect.any(Number),
      email: 'newuser@example.com',
    });
  });

  test.skip('[P1] should return 400 if email exists', async ({ request }) => {
    // THIS TEST WILL FAIL - Endpoint not implemented yet
    const response = await request.post('/api/users/register', {
      data: {
        email: 'existing@example.com',
        password: 'SecurePass123!',
      },
    });

    expect(response.status()).toBe(400);
    const error = await response.json();
    expect(error.message).toContain('Email already exists');
  });
});
```

**CRITICAL ATDD Requirements:**

- ✅ Use `test.skip()` to mark tests as red-phase scaffolds
- ✅ Write assertions for EXPECTED behavior (even though not implemented)
- ✅ Use realistic test data (not placeholder data)
- ✅ Test both happy path and error scenarios from acceptance criteria
- ✅ Use `apiRequest()` helper if Playwright Utils enabled
- ✅ Use data factories for test data (from data-factories fragment)
- ✅ Include priority tags [P0], [P1], [P2], [P3]

### 1.5 Provider Source Scrutiny for CDC in TDD Red Phase (If `use_pactjs_utils` Enabled)

When generating Pact consumer contract tests in the ATDD red phase, provider scrutiny applies with TDD-specific rules. Apply the **Seven-Point Scrutiny Checklist** from `contract-testing.md` (Response shape, Status codes, Field names, Enum values, Required fields, Data types, Nested structures) for both existing and new endpoints.

**If provider endpoint already exists** (extending an existing API):

- READ the provider route handler, types, and validation schemas
- Verify all seven scrutiny points against the provider source: Response shape, Status codes, Field names, Enum values, Required fields, Data types, Nested structures
- Add `// Provider endpoint:` comment and scrutiny evidence block documenting findings for each point
- Wrap the entire test function in `test.skip()` (so the whole test including `executeTest` is skipped), not just the callback

**If provider endpoint is new** (TDD — endpoint not implemented yet):

- Use acceptance criteria as the source of truth for expected behavior
- Acceptance criteria should specify all seven scrutiny points where possible (status codes, field names, types, etc.) — note any gaps as assumptions in the evidence block
- Add `// Provider endpoint: TODO — new endpoint, not yet implemented`
- Document expected behavior from acceptance criteria in scrutiny evidence block
- Wrap the entire test function in `test.skip()` and use realistic expectations from the story

**Graceful degradation when provider source is inaccessible:**

1. **OpenAPI/Swagger spec available**: Use the spec as the source of truth for response shapes, status codes, and field names
2. **Pact Broker available** (when `pact_mcp` is `"mcp"`): Use SmartBear MCP tools to fetch existing provider states and verified interactions as reference
3. **Neither available**: For new endpoints, use acceptance criteria; for existing endpoints, use consumer-side types. Mark with `// Provider endpoint: TODO — provider source not accessible, verify manually` and set `provider_scrutiny: "pending"` in output JSON
4. **Never silently guess**: Document all assumptions in the scrutiny evidence block

**Provider endpoint comments are MANDATORY** even in red-phase tests — they document the intent.

**Example: Red-phase Pact test with provider scrutiny:**

```typescript
// Provider endpoint: TODO — new endpoint, not yet implemented
/*
 * Provider Scrutiny Evidence:
 * - Handler: NEW — not yet implemented (TDD red phase)
 * - Expected from acceptance criteria:
 *   - Endpoint: POST /api/v2/users/register
 *   - Status: 201 for success, 400 for duplicate email, 422 for validation error
 *   - Response: { id: number, email: string, createdAt: string }
 */
test.skip('[P0] should generate consumer contract for user registration', async () => {
  await provider
    .given('no users exist')
    .uponReceiving('a request to register a new user')
    .withRequest({
      method: 'POST',
      path: '/api/v2/users/register',
      headers: { 'Content-Type': 'application/json' },
      body: { email: 'newuser@example.com', password: 'SecurePass123!' },
    })
    .willRespondWith({
      status: 201,
      headers: { 'Content-Type': 'application/json' },
      body: like({
        id: integer(1),
        email: string('newuser@example.com'),
        createdAt: string('2025-01-15T10:00:00Z'),
      }),
    })
    .executeTest(async (mockServer) => {
      const result = await registerUser({ email: 'newuser@example.com', password: 'SecurePass123!' }, { baseUrl: mockServer.url });
      expect(result.id).toEqual(expect.any(Number));
    });
});
```

**Why test.skip():**

- Tests are written correctly for EXPECTED behavior
- But we know they'll fail because feature isn't implemented
- `test.skip()` documents this is intentional (TDD red phase)
- Once feature is implemented, remove `test.skip()` to verify green phase

### 3. Track Fixture Needs

Identify fixtures needed for API tests:

- Authentication fixtures (if endpoints require auth)
- Data factories (user data, etc.)
- API client configurations

**Do NOT create fixtures yet** - just track what's needed for aggregation step.

---

## OUTPUT FORMAT

Write JSON to temp file: `/tmp/tea-atdd-api-tests-{{timestamp}}.json`

```json
{
  "success": true,
  "subagent": "atdd-api-tests",
  "tests": [
    {
      "file": "tests/api/user-registration.spec.ts",
      "content": "[full TypeScript test file content with test.skip()]",
      "description": "ATDD API test scaffolds for user registration (RED PHASE)",
      "expected_to_fail": true,
      "acceptance_criteria_covered": [
        "User can register with email/password",
        "System returns 201 on success",
        "System returns 400 if email exists"
      ],
      "priority_coverage": {
        "P0": 1,
        "P1": 2,
        "P2": 0,
        "P3": 0
      }
    }
  ],
  "fixture_needs": ["userDataFactory"],
  "knowledge_fragments_used": [
    "api-request",
    "data-factories",
    "api-testing-patterns",
    "pactjs-utils-consumer-helpers",
    "pact-consumer-di"
  ],
  "test_count": 3,
  "tdd_phase": "RED",
  "provider_scrutiny": "completed",
  "summary": "Generated 3 red-phase API test scaffolds for user registration story"
}
```

**On Error:**

```json
{
  "success": false,
  "subagent": "atdd-api-tests",
  "error": "Error message describing what went wrong",
  "partial_output": {
    /* any tests generated before error */
  }
}
```

---

## EXIT CONDITION

Subagent completes when:

- ✅ All API endpoints from acceptance criteria have test files
- ✅ All tests use `test.skip()` (documented red-phase scaffolds)
- ✅ All tests assert EXPECTED behavior (not placeholder assertions)
- ✅ JSON output written to temp file
- ✅ Fixture needs to be tracked

**Subagent terminates here.** Parent workflow will read output and proceed to aggregation.

---

## 🚨 SUBAGENT SUCCESS METRICS

### ✅ SUCCESS:

- All API tests generated with test.skip()
- Tests assert expected behavior (not placeholders)
- JSON output valid and complete
- No E2E/component/unit tests included (out of scope)
- Tests follow knowledge fragment patterns
- Every Pact interaction has `// Provider endpoint:` comment (if CDC enabled)
- Provider scrutiny completed or TODO markers added for new endpoints (if CDC enabled)

### ❌ FAILURE:

- Generated active passing tests (wrong - this is RED phase)
- Tests without test.skip() (will break CI)
- Placeholder assertions (expect(true).toBe(true))
- Did not follow knowledge fragment patterns
- Invalid or missing JSON output
- Pact interactions missing provider endpoint comments (if CDC enabled)
