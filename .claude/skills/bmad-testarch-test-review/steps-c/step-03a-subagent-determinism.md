---
name: 'step-03a-subagent-determinism'
description: 'Subagent: Check test determinism (no random/time dependencies)'
subagent: true
outputFile: '/tmp/tea-test-review-determinism-{{timestamp}}.json'
---

# Subagent 3A: Determinism Quality Check

## SUBAGENT CONTEXT

This is an **isolated subagent** running in parallel with other quality dimension checks.

**What you have from parent workflow:**

- Test files discovered in Step 2
- Knowledge fragment: test-quality (determinism criteria)
- Config: test framework

**Your task:** Analyze test files for DETERMINISM violations only.

---

## MANDATORY EXECUTION RULES

- 📖 Read this entire subagent file before acting
- ✅ Check DETERMINISM only (not other quality dimensions)
- ✅ Output structured JSON to temp file
- ❌ Do NOT check isolation, maintainability, coverage, or performance (other subagents)
- ❌ Do NOT modify test files (read-only analysis)
- ❌ Do NOT run tests (just analyze code)

---

## SUBAGENT TASK

### 1. Identify Determinism Violations

**Scan test files for non-deterministic patterns:**

**HIGH SEVERITY Violations**:

- `Math.random()` - Random number generation
- `Date.now()` or `new Date()` without mocking
- `setTimeout` / `setInterval` without proper waits
- External API calls without mocking
- File system operations on random paths
- Database queries with non-deterministic ordering
- **PactV4 consumer tests: multiple `pact.addInteraction()` in a single `it()` block** — the Rust FFI non-deterministically drops interactions (see `pactjs-utils-consumer-helpers.md` Example 6). Flag any `.pacttest.ts` file where a single `it()`/`test()` contains more than one `addInteraction()` chain.
- **PactV4 consumer Vitest config missing `fileParallelism: false`** in `vitest.config.pact.ts` — parallel workers race on the shared pact JSON file (see `pact-consumer-framework-setup.md` Example 2). HIGH regardless of file count.
- **PactV4 consumer Vitest config missing `pool: 'forks'` + `poolOptions.forks.singleFork: true`** in `vitest.config.pact.ts` — best current understanding is that the `@pact-foundation/pact` napi-rs binding is not robust across Vitest worker threads sharing a process; once a consumer+provider pair has ≥2 `.pacttest.ts` files, default threads pool produces reproducible "request was expected but not received" flakes on Linux CI. **Severity: HIGH if the repo has ≥2 `.pacttest.ts` files for the same consumer+provider pair; LOW (future-proof advisory) for single-file suites.** See `pact-consumer-framework-setup.md` Example 2.
- **Pact provider Vitest config missing `pool: 'forks'` + `poolOptions.forks.singleFork: true`** in `vitest.config.contract.ts` for multi-file provider suites (especially message providers) — same pool rule as the consumer side (see `pactjs-utils-provider-verifier.md` Example 7).
- **Consumer or provider Vitest config sets any of: `sequence.concurrent: true`, `maxConcurrency > 1`, `maxWorkers > 1`, `isolate: false`** in `vitest.config.pact.ts` / `vitest.config.contract.ts` — each defeats the serialization the forks-singleFork rule relies on. HIGH.
- **Consumer repo lacks a determinism gate** — if `tea_use_pactjs_utils` is enabled, flag any `package.json` whose `test:pact:consumer` script does not run `scripts/check-pact-determinism.sh` (see `pact-consumer-framework-setup.md` Example 10).

**MEDIUM SEVERITY Violations**:

- `page.waitForTimeout(N)` - Hard waits instead of conditions
- Flaky selectors (CSS classes that may change)
- Race conditions (missing proper synchronization)
- Test order dependencies (test A must run before test B)

**LOW SEVERITY Violations**:

- Missing test isolation (shared state between tests)
- Console timestamps without fixed timezone

### 2. Analyze Each Test File

For each test file from Step 2:

```javascript
const violations = [];

// Check for Math.random()
if (testFileContent.includes('Math.random()')) {
  violations.push({
    file: testFile,
    line: findLineNumber('Math.random()'),
    severity: 'HIGH',
    category: 'random-generation',
    description: 'Test uses Math.random() - non-deterministic',
    suggestion: 'Use faker.seed(12345) for deterministic random data',
  });
}

// Check for Date.now()
if (testFileContent.includes('Date.now()') || testFileContent.includes('new Date()')) {
  violations.push({
    file: testFile,
    line: findLineNumber('Date.now()'),
    severity: 'HIGH',
    category: 'time-dependency',
    description: 'Test uses Date.now() or new Date() without mocking',
    suggestion: 'Mock system time with test.useFakeTimers() or use fixed timestamps',
  });
}

// Check for hard waits
if (testFileContent.includes('waitForTimeout')) {
  violations.push({
    file: testFile,
    line: findLineNumber('waitForTimeout'),
    severity: 'MEDIUM',
    category: 'hard-wait',
    description: 'Test uses waitForTimeout - creates flakiness',
    suggestion: 'Replace with expect(locator).toBeVisible() or interceptNetworkCall-based network waits',
  });
}

// ... check other patterns
```

**Detecting Pact Vitest config violations (`vitest.config.pact.ts` / `vitest.config.contract.ts`)**

Vitest configs vary widely — `defineConfig({ test: { ... } })`, `mergeConfig(base, overrides)`, `satisfies UserConfig`, imported constants, TS spreads. A full AST parse is out of scope; use this fallback heuristic and accept false-negatives only for the `mergeConfig` case, which the subagent must flag separately:

```javascript
// Resolve the config file(s). For consumer: scripts.test:pact:consumer:run in package.json
// usually points at `vitest run --config <path>`. For provider: `vitest run --config <path>`.
// If neither script exists but `.pacttest.ts` files exist, default to 'vitest.config.pact.ts'.
const configPath = resolveVitestConfigPath({ scriptName: 'test:pact:consumer:run', fallback: 'vitest.config.pact.ts' });
const src = fs.readFileSync(configPath, 'utf8');

// 1. Literal-match the two mandatory lines. Tolerate single or double quotes and whitespace.
const hasFileParallelismFalse = /\bfileParallelism\s*:\s*false\b/.test(src);
const hasPoolForks = /\bpool\s*:\s*['"]forks['"]/.test(src);
const hasSingleForkTrue = /\bsingleFork\s*:\s*true\b/.test(src);

// 2. Flag settings that would defeat the rule if a human added them.
const hasSequenceConcurrent = /\bsequence\s*:\s*\{[^}]*\bconcurrent\s*:\s*true/.test(src);
const hasHighMaxConcurrency = /\bmaxConcurrency\s*:\s*([2-9]|\d{2,})/.test(src);
const hasHighMaxWorkers = /\bmaxWorkers\s*:\s*([2-9]|\d{2,})/.test(src);
const hasIsolateFalse = /\bisolate\s*:\s*false\b/.test(src);

// 3. mergeConfig / extends fallback — we cannot reliably follow imports. Emit LOW advisory.
const usesMergeConfig = /\bmergeConfig\s*\(/.test(src) || /\bextends\s*:/.test(src);

// 4. File-count gating for the pool-forks rule.
const pactTestCount = glob.sync('tests/contract/**/*.pacttest.ts').length;
```

**Violation emission rules** (apply in order; exit on first match per check):

- Missing `fileParallelism: false` → HIGH (always)
- Missing `pool: 'forks'` OR missing `singleFork: true`, AND `pactTestCount >= 2` → HIGH
- Missing `pool: 'forks'` OR missing `singleFork: true`, AND `pactTestCount < 2` → LOW (future-proof advisory)
- Any of `sequence.concurrent: true`, `maxConcurrency > 1`, `maxWorkers > 1`, `isolate: false` present → HIGH
- `usesMergeConfig` AND any of the three mandatory matches missing → LOW + `category: "pact-config-unverifiable"` with a suggestion to inline the pool settings at the leaf config or provide a `// tea:pact-ffi-safe` marker comment the subagent can trust

### 3. Calculate Determinism Score

**Scoring Logic**:

```javascript
const totalChecks = testFiles.length * checksPerFile;
const failedChecks = violations.length;
const passedChecks = totalChecks - failedChecks;

// Weight violations by severity
const severityWeights = { HIGH: 10, MEDIUM: 5, LOW: 2 };
const totalPenalty = violations.reduce((sum, v) => sum + severityWeights[v.severity], 0);

// Score: 100 - (penalty points)
const score = Math.max(0, 100 - totalPenalty);
```

---

## OUTPUT FORMAT

Write JSON to temp file: `/tmp/tea-test-review-determinism-{{timestamp}}.json`

```json
{
  "dimension": "determinism",
  "score": 85,
  "max_score": 100,
  "grade": "B",
  "violations": [
    {
      "file": "tests/api/user.spec.ts",
      "line": 42,
      "severity": "HIGH",
      "category": "random-generation",
      "description": "Test uses Math.random() - non-deterministic",
      "suggestion": "Use faker.seed(12345) for deterministic random data",
      "code_snippet": "const userId = Math.random() * 1000;"
    },
    {
      "file": "tests/e2e/checkout.spec.ts",
      "line": 78,
      "severity": "MEDIUM",
      "category": "hard-wait",
      "description": "Test uses waitForTimeout - creates flakiness",
      "suggestion": "Replace with expect(locator).toBeVisible()",
      "code_snippet": "await page.waitForTimeout(5000);"
    }
  ],
  "passed_checks": 12,
  "failed_checks": 3,
  "total_checks": 15,
  "violation_summary": {
    "HIGH": 1,
    "MEDIUM": 1,
    "LOW": 1
  },
  "recommendations": [
    "Use faker with fixed seed for all random data",
    "Replace all waitForTimeout with conditional waits",
    "Mock Date.now() in tests that use current time"
  ],
  "summary": "Tests are mostly deterministic with 3 violations (1 HIGH, 1 MEDIUM, 1 LOW)"
}
```

**On Error:**

```json
{
  "dimension": "determinism",
  "success": false,
  "error": "Error message describing what went wrong"
}
```

---

## EXIT CONDITION

Subagent completes when:

- ✅ All test files analyzed for determinism violations
- ✅ Score calculated (0-100)
- ✅ Violations categorized by severity
- ✅ Recommendations generated
- ✅ JSON output written to temp file

**Subagent terminates here.** Parent workflow will read output and aggregate with other quality dimensions.

---

## 🚨 SUBAGENT SUCCESS METRICS

### ✅ SUCCESS:

- All test files scanned for determinism violations
- Score calculated with proper severity weighting
- JSON output valid and complete
- Only determinism checked (not other dimensions)

### ❌ FAILURE:

- Checked quality dimensions other than determinism
- Invalid or missing JSON output
- Score calculation incorrect
- Modified test files (should be read-only)
