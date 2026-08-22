# Canonical Capability Inventory Pattern

Use when building an evidence-based capability registry for a codebase. This creates the canonical inventory that all future architectural decisions reference.

## When to Use

After completing a Phase 0 doctrine/audit mission, before beginning implementation missions. The inventory answers: "what actually exists in this codebase, with what state, and what evidence supports each state claim?"

## Steps

### 1. Define the taxonomy

Group capabilities into stable categories. Example from CLOE:

| Category | Example |
|----------|---------|
| Routing & Classification | Intent classifiers, routers, dispatch logic |
| V5 Pipeline | Decision modules, confidence engine |
| Governance | Approval gates, agent registry, safety |
| Business | Opportunity matcher, proposals, discovery |
| Observability | Runtime state, health, MCP tools |
| Operator Validation | Trial framework, evaluators, evidence |
| Provider Bridge | Brain ask, context builder |
| Infrastructure | Server, auth, stores |

### 2. Define canonical states

| State | Definition | How to detect |
|-------|------------|--------------|
| DECLARED | Named/referenced in code or docs | Found in import/require or documentation |
| IMPLEMENTED | Has an implementation file | File exists at expected path |
| TESTED | Has at least one test file | Test file exists with matching stem |
| WIRED | Imported by another module | `grep -rl 'from.*name' src/` returns results |
| EXPOSED | Referenced in entry point (server.js, main router) | Found in server.js or main entry file |
| RETRIEVABLE | Accessible through API or service | Route mounted in Express router |
| PROVIDER_ACCESSIBLE | Included in provider context | Found in brain prompt template or context builder |
| OPERATOR_PROVEN | Passed real operator trial | Evidence in test/reports/operator-trials/ |
| DISABLED | Behind env flag | Gated by `process.env.X !== '1'` |
| BLOCKED | Explicitly blocked/gated | `buildBlocked()` or `buildApprovalRequired()` |
| OBSOLETE | Superseded by newer version | Has V2/V3/V4/V5 counterpart, newer is wired |
| UNKNOWN | Cannot determine from evidence | No test file, no importer, no evidence |

### 3. Build the scanner script

Write a Node.js script (not bash — too many cross-references needed per capability) that:

1. Defines all capabilities with name, category, entry point, risk level
2. For each capability:
   a. Check implementation file exists
   b. Find test files (scan `test/` directory recursively for filename stem)
   c. Find importers (grep `src/` and `tools/` for the stem name)
   d. Check exposure in server.js or main router
3. Assign canonical state (last matching state from the priority-ordered list)
4. Output JSON registry + summary

```javascript
// Minimal scanner structure
const capabilities = [];
function addCap(name, category, entry, deps, risk, authority) {
  const testFiles = findTests(name);  // scan test/ for name stem
  const importers = findImporters(name);  // grep src/ tools/ for imports
  const hasFile = existsSync(resolve(REPO, entry));
  const wired = importers.length > 0;
  const exposed = isExposed(name);  // check server.js
  
  const states = [];
  states.push(hasFile ? 'IMPLEMENTED' : 'DECLARED');
  if (testFiles.length > 0) states.push('TESTED');
  if (wired) states.push('WIRED');
  if (exposed) states.push('EXPOSED');
  
  capabilities.push({
    name, category, entry_point: entry,
    implementation_exists: hasFile,
    test_count: testFiles.length,
    wired, exposed,
    canonical_state: states[states.length - 1] || 'UNKNOWN',
    all_states: states
  });
}
```

### 4. Name aliasing

A single capability may be referenced by multiple names (e.g. `unauthorized-action-detection` is tested via `operator-natural-action-router.test.js`, not via `unauthorized-action-detection.test.js`). The scanner should treat an alias match as evidence of TESTED.

After the first scan, manually verify each "untested wired" entry — some may be aliased tests, not genuine gaps.

### 5. Output deliverables

| File | Content |
|------|---------|
| `docs/mcporter/canonical-capability-registry-v1.json` | Full registry (all capabilities with states + evidence) |
| `docs/superpowers/plans/canonical-capability-taxonomy-v1.md` | Taxonomy document |
| `docs/superpowers/plans/canonical-capability-missing-report-v1.md` | Genuine gaps (untested, orphan) |

### 6. Missing and orphan reports

- **Missing**: capabilities in DECLARED state with no implementation file (0 expected in a mature codebase)
- **Orphan**: capabilities with no importers (may indicate dead code or incomplete wiring)
- **Untested wired**: capabilities that are imported but have 0 tests — genuine quality gaps
