# Commit-Integrity Audit Pattern

Audit whether a Git commit contains only its intended scope, detect absorbed pre-existing user work, and repair safely when contamination is found.

## When to Use

- A mission brief questions whether a commit's scope is clean
- A continuity JSON / operator packet claims pre-existing state was preserved but the commit file count suggests over-scope
- You need to verify that reverting a mission commit doesn't delete user work
- A commit contained 71 files but only 10 belong to the mission

## 8-Phase Protocol

### Phase 0 — Establish Authoritative Baseline

Collect without mutating anything:

```
git rev-parse --show-toplevel
git branch --show-current
git rev-parse HEAD
git status --short --branch
git remote -v
git branch -vv
git worktree list --porcelain
git log --oneline --decorate --graph -20
git branch --contains <TARGET_COMMIT>
git tag --contains <TARGET_COMMIT>
```

Key questions:
- **Is the commit local-only or published?** Check `git branch --contains`, upstream tracking, and downstream references.
- **Are baseline refs valid?** Verify the claimed initial SHA exists with `git rev-parse --verify`. If it doesn't, find the actual parent.
- **Are there commits after the target?** Any downstream work that references it.

### Phase 1 — Classify Every Changed Path

For each file in `git diff --name-status <PARENT>..<TARGET>`:

1. Get the complete diff metadata: `git diff --stat`, `git diff --numstat`, `git show --format=fuller`
2. Classify each path as exactly one of:
   - **MISSION_CREATED** — new file that implements mission scope
   - **MISSION_MODIFIED** — existing file changed to implement mission scope (dependency updates, integration wiring)
   - **PREEXISTING_USER_MODIFICATION** — file was already dirty before the mission; mission committed it
   - **PREEXISTING_USER_DELETION** — file was already deleted from working tree; mission committed the deletion
   - **PREEXISTING_STAGED_CHANGE** — file was staged before mission; mission committed it
   - **HISTORICAL_EVIDENCE** — receipts, continuity files
   - **GENERATED_ARTIFACT** — auto-generated files, build output
   - **AMBIGUOUS_ORIGIN** — cannot determine provenance
   - **UNRELATED_CHANGE** — clearly outside mission scope
   - **REQUIRED_DEPENDENCY_CHANGE** — needed because a deleted module's imports must be removed

**How to determine origin:**
- Compare blob hashes at parent, commit, and current working tree (`git show PARENT:path | sha1sum` etc.)
- Check whether the current working tree has a DIFFERENT modification to the same file (proves the commit change was pre-existing, now superseded by new dirty state)
- Check whether the modified file's changes are directly correlated with deleted modules (e.g., import removal + module deletion = coordinated, not accidental)

### Phase 2 — Build the Mission File Allowlist

The mission's OWN files are usually obvious: NEW files in a specific directory (e.g., `tools/repo-hygiene-authority/*`).

For MODIFIED files: determine whether each change is:
- **Dependency cleanup** (necessary because a module was deleted — correlated change)
- **Mission scope addition** (a new feature wired into existing code)
- **Pre-existing user work** (change existed before the mission and was absorbed)

The allowlist = MISSION_CREATED + MISSION_MODIFIED (mission scope only, excluding absorbed pre-existing work).

### Phase 3 — Determine User-Change Absorption

Answer explicitly:

1. Were pre-existing user changes included in the commit?
2. How many paths?
3. Were their contents preserved exactly?
4. Did their Git status change from uncommitted to committed?
5. Would reverting the commit revert or delete user work?
6. Can the intended mission implementation be reconstructed independently?
7. Are any KEEP-candidate changes now incorrectly owned by the mission commit?

**Blob comparison technique** for suspected user paths:
```bash
sha1_at_parent=$(git show <PARENT>:path | sha1sum)
sha1_at_commit=$(git show <TARGET>:path | sha1sum)
sha1_current=$(sha1sum path)
# If parent ≠ commit but current ≠ commit, the file in the commit is different from
# both — the committed change was the pre-existing dirty state that's now been
# replaced by NEW dirty state.
```

### Phase 4 — Select the Safe Repair Strategy

| Strategy | When | Procedure |
|----------|------|-----------|
| **A — No repair** | User changes were NOT absorbed; report merely described the diff incorrectly | Fix only reports, receipts, and continuity metadata |
| **B — Clean commit, isolated branch** | Target commit is local-only, unreferenced | 1. Create branch from parent. 2. Apply only allowlist files. 3. Clean commit. 4. Preserve original branch for reference |
| **C — Corrective commits** | Target commit is published / consumed downstream | Preserve hygiene implementation, restore user changes to uncommitted state, document limitations |
| **D — Blocked preservation package** | Ownership unclear, safe repair impossible | Full patch set + blob hashes + proposed plan. No history mutation |

**Critical: Never use `git reset --hard`, `git clean`, blanket stash, or history rewriting without proving safety first.**

### Phase 5 — Certification Truthfulness Audit

For each claimed capability in the operator packet, report, and continuity JSON:

| Status | Meaning |
|--------|---------|
| PROVEN | Runtime test demonstrates it working |
| PARTIALLY_PROVEN | Works in limited scenarios or with manual setup |
| SCHEMA_ONLY | Schema/model defines it but no runtime code |
| STUBBED | Function exists but returns hardcoded/empty values |
| NOT_IMPLEMENTED | Not present at any level |
| NOT_TESTED | Could work but no test covers it |

**Rule: Schema fields and interface definitions are NOT runtime capability.**
A field like `rollback_available: true` in a receipt schema does not prove rollback works — it proves a field exists.

### Phase 6 — Rollback Proof Audit

When a commit or module claims rollback capability, verify:

1. Is `rollback_available` computed from evidence or hard-coded?
2. Is there an executable rollback command (not null/empty)?
3. Is original content preserved (diff file, backup, or reverse patch)?
4. Has rollback ever been executed in any test?
5. Run a bounded fixture test:
```
Create fixture with known content A
→ Apply simulated mutation → state B
→ Verify backup was made
→ Execute rollback
→ Verify byte-for-byte return to A
```

### Phase 7 — Verdict Identifier and Metadata Repair

Validate the verdict identifier used in continuity JSON, operator packet, and commit message against the mission's allowed list. Common discrepancy patterns:
- **Abbreviated identifier**: `HYGIENE_AUTHORITY_V1` instead of `HYGIENE_AUTHORITY_PATTERN_EXTRACTION_AND_IMPLEMENTATION_V1`
- **Wrong casing or separator style**

**Pitfall: Don't preserve an inflated certification for compatibility. If the mission didn't prove what it claimed, downgrade the verdict.**

### Phase 8 — Reverification

After any repair or metadata fix, run:
- Focused unit tests (must match or exceed baseline pass rate)
- Revert-proof test: reverting the clean commit removes ONLY mission-owned files
- User-work ownership test: reverting does NOT remove files owned by pre-existing user work
- FastSafe gate or equivalent
- Continuity JSON schema validation
- Cross-file consistency: operator packet, continuity JSON, and commit message agree on verdict

## Common Pitfalls

| Trap | Fix |
|------|-----|
| **Baseline SHA doesn't exist** | Find actual parent with `git log --oneline --parents -1 <TARGET>` |
| **Continuity JSON verdict doesn't match contract** | Validate against the mission's allowed list before writing |
| **Schema fields reported as capabilities** | Distinguish SCHEMA_ONLY, STUBBED, NOT_IMPLEMENTED in audit |
| **rollback_available hardcoded truthy** | Check that actual content/diff is preserved, not just the field |
| **Cross-repo report confusion** | Always check which repository a report's `repository` field targets |
| **Clean commit absorbs user work** | Verify: revert removes only the intended allowlist files |
| **Inflated certification preserved for compatibility** | Fix the verdict to match actual evidence, not the summary tag |
