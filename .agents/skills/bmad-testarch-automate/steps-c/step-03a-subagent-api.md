---
name: 'step-03a-subagent-api'
description: 'Subagent: Generate API tests only'
subagent: true
outputFile: '/tmp/tea-automate-api-tests-{{timestamp}}.json'
---

# Subagent 3A: Generate API Tests

## SUBAGENT CONTEXT

This is an **isolated subagent** running in parallel with E2E test generation.

**What you have from parent workflow:**

- Target features/components identified in Step 2
- Knowledge fragments loaded: api-request, data-factories, api-testing-patterns
- Config: test framework, Playwright Utils enabled/disabled, Pact.js Utils enabled/disabled, Pact MCP mode
- Coverage plan: which API endpoints need testing

**Your task:** Generate API tests ONLY (not E2E, not fixtures, not other test types).

**If `use_pactjs_utils` is enabled AND contract artifacts are explicitly in scope for this subagent:** Apply pactjs-utils conventions from the loaded fragments (`pactjs-utils-overview`, `pactjs-utils-consumer-helpers`, `pactjs-utils-provider-verifier`, `pactjs-utils-request-filter`, `pact-consumer-framework-setup`) when generating contract-level API test scaffolding. Full consumer/provider suite generation — including edits to `vitest.config.pact.ts`, `vitest.config.contract.ts`, `package.json` scripts (`test:pact:consumer`, `test:pact:consumer:run`), and `scripts/publish-pact.sh` — must be triggered explicitly by the parent workflow flag `tea_use_pactjs_utils: true`; do not generate those artifacts implicitly. When in scope, enforce these determinism/FFI-safety rules:

- **Consumer tests**: exactly one `pact.addInteraction()` per `it()` block (use `it.each` for parameterized cases) — PactV4's Rust FFI drops interactions otherwise.
- **Consumer Vitest config**: `vitest.config.pact.ts` must include BOTH `fileParallelism: false` AND `pool: 'forks'` + `poolOptions.forks.singleFork: true`. `fileParallelism: false` prevents workers racing on the shared pact JSON; forks + singleFork prevents the `@pact-foundation/pact` Rust FFI from leaking state across files on Linux CI.
- **Provider Vitest config**: `vitest.config.contract.ts` must include `pool: 'forks'` + `poolOptions.forks.singleFork: true` (same rule as consumer) for multi-file and message-provider suites.
- **Consumer `package.json`**: generate both `test:pact:consumer` (determinism gate calling `scripts/check-pact-determinism.sh`) and `test:pact:consumer:run` (inner vitest invocation).
- **Publish script**: `scripts/publish-pact.sh` normalizes interactions with `jq -S '.interactions |= sort_by(...)'` before `pact-broker publish`.

If `pact_mcp` is `"mcp"`, use SmartBear MCP tools (Fetch Provider States, Generate Pact Tests) to inform test generation.

---

## MANDATORY EXECUTION RULES

- 📖 Read this entire subagent file before acting
- ✅ Generate API tests ONLY
- ✅ Output structured JSON to temp file
- ✅ Follow knowledge fragment patterns
- ❌ Do NOT generate E2E tests (that's subagent 3B)
- ❌ Do NOT run tests (that's step 4)
- ❌ Do NOT generate fixtures yet (that's step 3C aggregation)

---

## SUBAGENT TASK

### 1. Identify API Endpoints

From the coverage plan (Step 2 output), identify:

- Which API endpoints need test coverage
- Expected request/response formats
- Authentication requirements
- Error scenarios to test

### 2. Generate API Test Files

For each API endpoint, create test file in `tests/api/[feature].spec.ts`:

**Test Structure:**

```typescript
import { test, expect } from '@playwright/test';
// If Playwright Utils enabled:
// import { apiRequest } from '@playwright-utils/api';

test.describe('[Feature] API Tests', () => {
  test('[P0] should handle successful [operation]', async ({ request }) => {
    // Use apiRequest helper if Playwright Utils enabled
    // Otherwise use standard request fixture
    const response = await request.post('/api/endpoint', {
      data: {
        /* test data */
      },
    });

    expect(response.status()).toBe(200);
    expect(await response.json()).toMatchObject({
      /* expected */
    });
  });

  test('[P1] should handle [error scenario]', async ({ request }) => {
    // Test error handling
  });
});
```

**Requirements:**

- ✅ Use `apiRequest()` helper if Playwright Utils enabled (from api-request fragment)
- ✅ Use data factories for test data (from data-factories fragment)
- ✅ Follow API testing patterns (from api-testing-patterns fragment)
- ✅ Include priority tags [P0], [P1], [P2], [P3]
- ✅ Test both happy path and error scenarios
- ✅ Use proper TypeScript types
- ✅ Deterministic assertions (no timing dependencies)

**If Pact.js Utils enabled (from `subagentContext.config.use_pactjs_utils`):**

- ✅ Generate consumer contract tests in `pact/http/consumer/` using `createProviderState({ name, params })` pattern
- ✅ Generate provider verification tests in `pact/http/provider/` using `buildVerifierOptions({ provider, port, includeMainAndDeployed, stateHandlers })` pattern
- ✅ Generate request filter helpers in `pact/http/helpers/` using `createRequestFilter({ tokenGenerator: () => string })`
- ✅ Generate shared state constants in `pact/http/helpers/states.ts`
- ✅ If async/message patterns detected, generate message consumer tests in `pact/message/` using `buildMessageVerifierOptions`
- ✅ **Provider endpoint comment MANDATORY** on every Pact interaction: `// Provider endpoint: <path> -> <METHOD> <route>`
- ⚠️ **Postel's Law for matchers**: Use `like()`, `eachLike()`, `string()`, `integer()` matchers ONLY in `willRespondWith` (responses). Request bodies in `withRequest` MUST use exact values — never wrap request bodies in `like()`. The consumer controls what it sends, so contracts should be strict about request shape.

### 1.5 Provider Source Scrutiny (CDC Only)

**CRITICAL**: Before generating ANY Pact consumer interaction, perform provider source scrutiny per the **Seven-Point Scrutiny Checklist** defined in `contract-testing.md`. Do NOT generate response matchers from consumer-side types alone — this is the #1 cause of contract verification failures.

The seven points to verify for each interaction:

1. Response shape
2. Status codes
3. Field names
4. Enum values
5. Required fields
6. Data types
7. Nested structures

**Source priority**: Provider source code is most authoritative. When an OpenAPI/Swagger spec exists (`openapi.yaml`, `openapi.json`, `swagger.json`), use it as a complementary or alternative source — it documents the provider's contract explicitly and can be faster to parse than tracing through handler code. When both exist, cross-reference them; if they disagree, the source code wins. Document the discrepancy in the scrutiny evidence block (e.g., `OpenAPI shows 200 but handler returns 201; using handler behavior`) and flag it in the output JSON `summary` so it is discoverable by downstream consumers or audits.

**Scrutiny Sequence** (for each endpoint in the coverage plan):

1. **READ provider route handler and/or OpenAPI spec**: Find the handler file from `subagentContext.config.provider_endpoint_map` or by scanning the provider codebase. Also check for OpenAPI/Swagger spec files. Extract:
   - Exact status codes returned (`res.status(201)` / OpenAPI `responses` keys)
   - Response construction (`res.json({ data: ... })` / OpenAPI `schema`)
   - Error handling paths (what status codes for what conditions)

2. **READ provider type/model/DTO definitions**: Find the response type referenced by the handler or OpenAPI `$ref` schemas. Extract:
   - Exact field names (`transaction_id` not `transactionId`)
   - Field types (`string` ID vs `number` ID / OpenAPI `type` + `format`)
   - Optional vs required fields (OpenAPI `required` array)
   - Nested object structures (OpenAPI `$ref`, `allOf`, `oneOf`)

3. **READ provider validation schemas**: Find Joi/Zod/class-validator schemas or OpenAPI request body `schema.required`. Extract:
   - Required request fields and headers
   - Enum/union type allowed values (`"active" | "inactive"` / OpenAPI `enum`)
   - Request body constraints

4. **Cross-reference findings** against consumer expectations:
   - Does the consumer expect the same field names the provider sends?
   - Does the consumer expect the same status codes the provider returns?
   - Does the consumer expect the same nesting the provider produces?

5. **Document scrutiny evidence** as a block comment in the generated test:

```typescript
/*
 * Provider Scrutiny Evidence:
 * - Handler: server/src/routes/userHandlers.ts:45
 * - OpenAPI: server/openapi.yaml paths./api/v2/users/{userId}.get (if available)
 * - Response type: UserResponseDto (server/src/types/user.ts:12)
 * - Status: 201 for creation (line 52), 400 for validation error (line 48)
 * - Fields: { id: number, name: string, email: string, role: "user" | "admin" }
 * - Required request headers: Authorization (Bearer token)
 */
```

6. **Graceful degradation** when provider source is not accessible (follows the canonical four-step protocol from `contract-testing.md`):
   1. **OpenAPI/Swagger spec available**: Use the spec as the source of truth for response shapes, status codes, and field names
   2. **Pact Broker available** (when `pact_mcp` is `"mcp"` in `subagentContext.config`): Use SmartBear MCP tools to fetch existing provider states and verified interactions as reference
   3. **Neither available**: Generate from consumer types but use the TODO form of the mandatory comment: `// Provider endpoint: TODO — provider source not accessible, verify manually`. Set `provider_scrutiny: "pending"` in output JSON
   4. **Never silently guess**: Document all assumptions in the scrutiny evidence block

> ⚠️ **Anti-pattern**: Generating response matchers from consumer-side types alone. This produces contracts that reflect what the consumer _wishes_ the provider returns, not what it _actually_ returns. Always read provider source or OpenAPI spec first.

### 3. Track Fixture Needs

Identify fixtures needed for API tests:

- Authentication fixtures (auth tokens, API keys)
- Data factories (user data, product data, etc.)
- API client configurations

**Do NOT create fixtures yet** - just track what's needed for aggregation step.

---

## OUTPUT FORMAT

Write JSON to temp file: `/tmp/tea-automate-api-tests-{{timestamp}}.json`

```json
{
  "success": true,
  "subagent": "api-tests",
  "tests": [
    {
      "file": "tests/api/auth.spec.ts",
      "content": "[full TypeScript test file content]",
      "description": "API tests for authentication endpoints",
      "priority_coverage": {
        "P0": 3,
        "P1": 2,
        "P2": 1,
        "P3": 0
      }
    },
    {
      "file": "tests/api/checkout.spec.ts",
      "content": "[full TypeScript test file content]",
      "description": "API tests for checkout endpoints",
      "priority_coverage": {
        "P0": 2,
        "P1": 3,
        "P2": 1,
        "P3": 0
      }
    }
  ],
  "fixture_needs": ["authToken", "userDataFactory", "productDataFactory"],
  "knowledge_fragments_used": ["api-request", "data-factories", "api-testing-patterns"],
  "provider_scrutiny": "completed",
  "provider_files_read": ["server/src/routes/authHandlers.ts", "server/src/routes/checkoutHandlers.ts", "server/src/types/auth.ts"],
  "test_count": 12,
  "summary": "Generated 12 API test cases covering 3 features"
}
```

**On Error:**

```json
{
  "success": false,
  "subagent": "api-tests",
  "error": "Error message describing what went wrong",
  "partial_output": {
    /* any tests generated before error */
  }
}
```

---

## EXIT CONDITION

Subagent completes when:

- ✅ All API endpoints have test files generated
- ✅ All tests follow knowledge fragment patterns
- ✅ JSON output written to temp file
- ✅ Fixture needs tracked

**Subagent terminates here.** Parent workflow will read output and proceed to aggregation.

---

## 🚨 SUBAGENT SUCCESS METRICS

### ✅ SUCCESS:

- All API tests generated following patterns
- JSON output valid and complete
- No E2E/component/unit tests included (out of scope)
- Every Pact interaction has `// Provider endpoint:` comment (if CDC enabled)
- Provider source scrutiny completed or gracefully degraded with TODO markers (if CDC enabled)
- Scrutiny evidence documented as block comments in test files (if CDC enabled)

### ❌ FAILURE:

- Generated tests other than API tests
- Did not follow knowledge fragment patterns
- Invalid or missing JSON output
- Ran tests (not subagent responsibility)
- Pact interactions missing provider endpoint comments (if CDC enabled)
- Response matchers generated from consumer-side types without provider scrutiny (if CDC enabled)
