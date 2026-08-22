# Preflight Integration Tests — LAH Workflow + Repo Router

These tests verify that the `lah-repo-router` integration into `lah-workflow` is
correctly documented and mandatory. Tests are deterministic — they check
documented invariants, not runtime behavior.

## Test 1: Routing preflight exists before Gate 1

**Check:** The SKILL.md contains a "Repository Routing Preflight" section
before the "Gate 1 — CodeGraph" section.

**Method:** grep for the section marker.

```bash
grep -c "## Repository Routing Preflight" ~/.hermes/skills/lah-stack/lah-workflow/SKILL.md
# Expected: 1
```

**Verdict:** PREFLIGHT_SECTION_EXISTS

---

## Test 2: Routing is MANDATORY

**Check:** The preflight section states it is mandatory.

**Method:** Check for MANDATORY keyword.

```bash
grep -c "MANDATORY" ~/.hermes/skills/lah-stack/lah-workflow/SKILL.md
# Expected: >= 2 (in preflight + skills branch table)
```

**Verdict:** MANDATORY_STATED

---

## Test 3: Decision handling documented for all outcomes

**Check:** RESOLVED, AMBIGUOUS, and UNRESOLVED outcomes are documented.

**Method:** Check for each outcome keyword under the routing preflight section.

```bash
grep -c "RESOLVED" ~/.hermes/skills/lah-stack/lah-workflow/SKILL.md
# Expected: >= 1

grep -c "AMBIGUOUS" ~/.hermes/skills/lah-stack/lah-workflow/SKILL.md
# Expected: >= 1

grep -c "UNRESOLVED" ~/.hermes/skills/lah-stack/lah-workflow/SKILL.md
# Expected: >= 1
```

**Verdict:** DECISION_HANDLING_DOCUMENTED

---

## Test 4: Routing context propagation documented

**Check:** The routing receipt fields are propagated to later phases.

**Method:** Check for the routing context propagation table.

```bash
grep -c "Where each field is used" ~/.hermes/skills/lah-stack/lah-workflow/SKILL.md
# Expected: 1
```

**Verdict:** CONTEXT_PROPAGATION_DOCUMENTED

---

## Test 5: CodeGraph ordering is correct

**Check:** The workflow documents that CodeGraph inside the router comes before
general CodeGraph analysis.

**Method:** Check for the CodeGraph ordering section.

```bash
grep -c "targeted CodeGraph escalation" ~/.hermes/skills/lah-stack/lah-workflow/SKILL.md
# Expected: >= 1
```

**Verdict:** CODEGRAPH_ORDERING_CORRECT

---

## Test 6: Repository context enforcement documented

**Check:** Workspace verification, git root verification, and forbidden root
checks are documented.

```bash
grep -c "verify the workspace" ~/.hermes/skills/lah-stack/lah-workflow/SKILL.md
# Expected: 1

grep -c "git rev-parse" ~/.hermes/skills/lah-stack/lah-workflow/SKILL.md
# Expected: >= 2 (once in preflight, once in enforcement)

grep -c "write_forbidden" ~/.hermes/skills/lah-stack/lah-workflow/SKILL.md
# Expected: >= 1
```

**Verdict:** CONTEXT_ENFORCEMENT_DOCUMENTED

---

## Test 7: Memory writes use memory_repo

**Check:** The memory write section references ROUTING_MEMORY, not the
implementation repo.

```bash
grep -c "ROUTING_MEMORY" ~/.hermes/skills/lah-stack/lah-workflow/SKILL.md
# Expected: >= 2
```

**Verdict:** MEMORY_REPO_SEPARATE

---

## Test 8: Router not duplicated in workflow

**Check:** The workflow does NOT duplicate the router mapping logic. It must
reference the `lah-repo-router` skill, not redefine routing rules.

```bash
grep -c "lah-repo-router" ~/.hermes/skills/lah-stack/lah-workflow/SKILL.md
# Expected: >= 1 (reference to external router)

grep -c "prefixPriority\|canonicalMissionRepoMap\|aliasIndex" ~/.hermes/skills/lah-stack/lah-workflow/SKILL.md
# Expected: 0 (router internals not duplicated in workflow)
```

**Verdict:** ROUTER_NOT_DUPLICATED

---

## Test 9: Fail-closed for ambiguous/unresolved

**Check:** The workflow blocks on AMBIGUOUS and UNRESOLVED decisions.

```bash
grep -c "STOP.*Do not proceed\|STOP.*Fail-closed\|Stop fail-closed" ~/.hermes/skills/lah-stack/lah-workflow/SKILL.md
# Expected: >= 2
```

**Verdict:** FAIL_CLOSED_DOCUMENTED

---

## Test 10: Router invocation exactly once

**Check:** The workflow documents that the router is invoked exactly once per
execution and the receipt is saved.

```bash
grep -c "exactly once" ~/.hermes/skills/lah-stack/lah-workflow/SKILL.md
# Expected: >= 1 (in pitfall table)
```

**Verdict:** SINGLE_INVOCATION_DOCUMENTED

---

## Test 11: Aggregated verdict

| Test | Expected | Actual |
|------|----------|--------|
| 1. Preflight section exists | 1 | |
| 2. MANDATORY stated | >=2 | |
| 3. All outcomes documented | >=1 each | |
| 4. Context propagation | 1 | |
| 5. CodeGraph ordering | >=1 | |
| 6. Context enforcement | documented | |
| 7. Memory repo separate | >=2 | |
| 8. Router not duplicated | >=1 router, 0 internals | |
| 9. Fail-closed | >=2 | |
| 10. Single invocation | >=1 | |
