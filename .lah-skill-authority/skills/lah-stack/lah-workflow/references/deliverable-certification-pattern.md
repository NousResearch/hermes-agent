# Deliverable Certification Pattern

## When to use

After a prior mission claims to have produced a deliverable with a specific verdict, and you need to independently verify its repository truth, claim accuracy, and safety boundaries. This is a verification-only pattern — no files are created or modified.

## Pattern overview

Structured as a series of gated checks. Each gate produces evidence that either supports or refutes the prior mission's claims.

### Gate 0 — Repository routing and path truth

Resolve the reported deliverable path using the canonical `lah-repo-router`. Do not trust the path from the prior report — re-run routing. Then:

```bash
realpath <reported_path>
readlink -f <reported_path>
stat --format="%F %i" <reported_path>
```

Check for symlinks, multiple clones (compare inodes), and whether the path resolves to the canonical checkout or a workspace clone.

### Gate 1 — Deliverable identity

```bash
wc -l <path>
stat --format="%s" <path>
sha256sum <path>
git ls-files --error-unmatch <relative_path_in_repo> 2>&1
```

Prove: exists, size, fingerprint, git tracked status. If untracked, no commit was created — flag as non-durable.

### Gate 2 — Git scope truth

```bash
git branch --show-current
git rev-parse --short HEAD
git remote -v
git status --short
git log --oneline --diff-filter=A -- "<relative_path>"
git log --all --oneline --diff-filter=A -- "<relative_path>"
```

Prove: which branch and SHA, whether the document was committed, whether it appears in any branch, whether scope contamination occurred.

### Gate 3 — Claim-to-evidence matrix

For every material claim from the prior mission, build a structured matrix:

| Claim | Document location | Supporting evidence | Status |
|-------|-------------------|-------------------|--------|

Use direct `grep -c` and `grep -n` counts to verify presence. Do not rely on section headings alone — verify substantive content exists.

Allowed statuses: `VERIFIED | PARTIALLY_VERIFIED | UNSUPPORTED | CONTRADICTED | AMBIGUOUS`

### Gate 4 — Artifact completeness

Define the artifact list from the prior mission spec. For each artifact, run a `grep` for its section marker. Classify:

`COMPLETE | PRESENT_BUT_INCOMPLETE | REFERENCED_ONLY | MISSING | INCONSISTENT`

### Gate 5 — Component verification

For claims about existing reusable components:
```bash
test -f <path> && wc -l <path>
```

For claims about missing components:
```bash
test -f <path> && echo "EXISTS" || echo "MISSING"
```

### Gate 6 — Safety boundary verification

Check all forbidden actions explicitly:

```bash
# Wallet/secret files
find <repo> -name "*wallet*" -o -name "*seed*" -o -name "*priv*key*" | grep -v node_modules

# SDK installation
grep -c "<sdk_name>" package.json

# Git changes
git diff --name-only
git commit history for the mission timeframe
```

Prove each as `NO` or `YES` — never guess. Use `UNPROVEN` when direct evidence is unavailable.

### Gate 7 — Source verification

Inspect the document's source register. Classify each source:

`PROTOCOL_SPECIFICATION | OFFICIAL_VENDOR_DOCUMENTATION | PRIMARY_RESEARCH | ACADEMIC_PREPRINT | STANDARDS`

Source URLs may be cited but not independently re-fetched during a read-only certification — distinguish `DOCUMENT_CITATION_PRESENT` from `SOURCE_FETCH_VERIFIED`.

### Gate 8 — Metric reconciliation

For numerical claims (transaction volume, ecosystem size, age), cross-reference with the cited source in the document. Flag rounding and context.

## Required output format

```text
1. VERDICT
2. EXECUTIVE SUMMARY
3. REPOSITORY ROUTING TABLE
4. PATH AND REALPATH TABLE
5. DELIVERABLE IDENTITY RECORD
6. GIT MISSION SCOPE RECORD
7. CLAIM-TO-EVIDENCE MATRIX
8. ARTIFACT COMPLETENESS MATRIX
9. COMPONENT VERIFICATION
10. SAFETY BOUNDARY VERIFICATION
11. LIMITATIONS
12. EXPLICITLY NOT DONE
13. FINAL GIT STATUS
14. NEXT RECOMMENDED MOVE
```

## Pitfalls

- **Trusting the prior report's path**: Always re-run `lah-repo-router`. The reported path might be in a workspace clone, not the canonical checkout.
- **Trusting prior git status claims**: The earlier mission may claim `Branch: main` when the actual branch differs. Verify with `git branch --show-current`.
- **Assuming tracked == committed**: A file may be untracked (`??` in `git status --short`) even though it exists on disk. This means no commit was created.
- **Metric definition mismatch**: The document may cite "75M+ transactions" but the source dashboard may count HTTP requests, settled payments, or blockchain transactions — these are different metrics. Flag without resolving unless you can independently verify.
- **Percentage claims without denominator**: "~70% reuse" without a reproducible component count is a qualitative estimate, not a measured figure. Flag as such.
- **Certification spec claims ≠ document claims**: The certification mission spec may list claims the document is *expected* to make (e.g. "29 threats catalogued") that the document never actually states. Always verify each claim by searching the actual document text (`grep -c` against the document). Do not assume the certification spec's claim list reflects the document's actual content. If a claimed metric is absent from the document, it is an error in the certification spec's prior-knowledge framing, not a defect in the document itself. Report the discrepancy without requiring the document to match an external expectation.
