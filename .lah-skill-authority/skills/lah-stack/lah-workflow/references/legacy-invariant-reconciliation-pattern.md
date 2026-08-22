# Legacy Static Invariant Reconciliation Pattern

Use when a MIXED mission's bounded repair phase involves replacing a stale
blanket test assertion (e.g. a regex substring match that was once valid
but now false-positives on legitimate new code) with bounded architectural
authority tests. Zero production code changes allowed.

## When to apply

- A legacy test uses a **blanket assertion** (regex substring, grepping
  for `/keyword` anywhere in a file) that was designed for the old codebase
- New legitimate code patterns now match the blanket assertion (e.g. regex
  literal patterns in `EXECUTE_APPROVED_PATTERNS` caught by a `/execute/`
  check)
- The underlying safety concern (no direct executor authority) is still valid
  but the assertion is too broad

## How it works

### Step 1 — Identify the stale assertion

Find the exact line or regex that false-positive flags the new code.
Record what it matches that it shouldn't. Example:

```
if (/\/execute\b/.test(trimmed)) → matched regex literals in EXECUTE_APPROVED_PATTERNS
```

### Step 2 — Document why it became invalid

List the legitimate new symbols that the blanket assertion catches.
Show the architectural justification for each:

| Symbol | Type | Justification |
|--------|------|---------------|
| `EXECUTE_APPROVED_PATTERNS` | Regex patterns | Deterministic intent detection for explicit second-consent |
| `buildExecuteApprovedResponse()` | Handler function | Delegates to canonical approved-execution entrypoint |

### Step 3 — Design the replacement assertions

Replace the single blanket assertion with multiple **bounded** tests that each
check a specific invariant. Each test must:

- Still catch the **original danger** (e.g. no direct executor import)
- **Explicitly allow** the new legitimate patterns
- Use **comment stripping** to avoid false positives from JSDoc/regex
- Never use blanket substring checks that create the same problem again

Example replacement structure:

```
G1 — Router has no direct executor authority
  • No import of executor.js
  • No call to executeAction()
  • No call to executeGoverned()
  • No call to callTool()
  • Affirmatively checks that executeApprovedGovernedAction IS imported

G2 — Router has no provider mutation authority
  • No ExoClick imports
  • No campaign-control imports
  • No pauseCampaigns/playCampaigns calls
  • No provider-auth/provider-credential imports
  • No axios/child_process references

G3 — Canonical delegation is bounded and unique
  • executeApprovedGovernedAction comes from canonical entrypoint only
  • buildExecuteApprovedResponse passes only action_id + approval_id
  • No spawn, worker, queue, scheduler, setInterval, polling

G4 — Packet builder has no execution authority (preserved unchanged)
```

### Step 4 — Verify both directions

The new tests must prove:

1. **Safety preserved**: Direct `executeAction`, executor imports, provider
   imports, and alternative execution mechanisms are still forbidden
2. **Capability allowed**: Canonical delegation (`executeApprovedGovernedAction`,
   `buildExecuteApprovedResponse`) is explicitly verified as present

Sanity-check: if someone later introduces a true bypass (importing executor.js
directly in the router), the new tests must still catch it.

### Step 5 — Full gate suite

| Gate | What | Required |
|------|------|----------|
| A | Reconciled legacy suite | 100% pass |
| B | Focused tests for the capability | 100% pass (unchanged) |
| C | Combined regression suite | 100% pass |
| D | Repeatability (5 runs) | 5/5 pass |
| E | Static diff proof | Zero production files changed |

### Step 6 — Static diff proof

Confirm with `git diff HEAD --stat`:

- Only the intended test file(s) changed
- Zero files under `src/` changed
- No `.only`, `.skip`, `.todo`
- No test silently deleted

## Key techniques

### Comment stripping before static analysis

```javascript
const noComments = source
  .replace(/\/\*[\s\S]*?\*\//g, '')  // multi-line comments
  .replace(/\/\/.*$/gm, '');          // single-line comments
```

This prevents false positives from comments, JSDoc, and documentation strings
that happen to contain the target keyword.

### Import-specific checks

```javascript
// Check for direct executor import
assert.ok(!/from\s+['"].*executor\.?[^.]*['"]/.test(noComments),
  'Router must not import executor.js');
```

### Function-body bounded checks

```javascript
// Verify function passes only expected arguments to the delegation target
const fnMatch = source.match(/async function buildExecuteApprovedResponse[\s\S]*?\n\}/);
if (fnMatch) {
  const body = fnMatch[0];
  assert.ok(body.includes('action_id'), 'Must reference action_id');
  assert.ok(body.includes('approval_id'), 'Must reference approval_id');
}
```

## Key pitfalls

| Trap | Symptom | Prevention |
|------|---------|------------|
| **Replacing with an equally blanket assertion** | New test still uses `/execute/` substring check | Each assertion must check ONE specific invariant (import name, function call, symbol presence) |
| **Silent test deletion** | Old assertion removed, no replacement catches the original danger | Verify: if someone imports executor.js tomorrow, does the new test still fail? |
| **False confidence from `.only`** | Only selected tests run, hiding failures elsewhere | Run combined suite, check for `.only`/`.skip`/`.todo` |
| **Affirming canonical delegation without bounding it** | Test checks that `executeApprovedGovernedAction` is imported but doesn't verify it comes from the canonical entrypoint | Check the import source path, not just the symbol name |
