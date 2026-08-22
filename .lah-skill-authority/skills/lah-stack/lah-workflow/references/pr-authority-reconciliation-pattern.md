# Open-PR Authority Reconciliation Pattern (LAH)

Mission class: audit ALL open PRs in the LAH org, classify authority, close only
provably obsolete PRs without merge, and leave a reusable deterministic
classifier. Established during LAH_OPEN_PR_AUTHORITY_RECONCILIATION_AND_SUPERSEDED_CLEANUP_V1.

## Sequencing (LOTs)

1. **Inventory** — capture default-branch SHAs + full PR snapshot (state, head_sha,
   base, mergeable, labels, changed_files) persisted to JSON BEFORE any mutation.
   `gh api repos/{org}/{repo}/pulls` (REST) gives base.sha; `gh pr view` does not.
2. **Reuse audit** — locate existing GitHub Watcher / PR Factory / Engineering
   Memory / supersession resolver in the repo that OWNS them. For LAH that is
   `openclaw-runtime/lah-openclaw-mvp` (src/services/github-watcher-readonly.js,
   src/dev-agent/pr-factory/, engineering-memory/). Do NOT rebuild.
3. **Implement** pure classifier + mutation gate + read-only audit CLI + tests
   (RED→GREEN). Merge via a normal governed PR.
4. **Read-only audit** of all open PRs using an evidence file (repo#pr → typed evidence).
5. **Deep unique-delta check** per proposed closure: blob comparison vs default branch.
6. **Freeze mutation set** with exact head SHAs.
7. **Per PR**: re-fetch state → assertMutationSafe → factual comment → close
   WITHOUT merge → verify state=closed && merged=false → receipt.
8. **Engineering memory card + continuity JSON** (use the official tools).
9. **Final inventory + certification** (gates A–I).

## Classifier design (deterministic, fail-closed)

Classifications: `KEEP_ACTIVE`, `CLOSE_SUPERSEDED`, `CLOSE_DUPLICATE`,
`CLOSE_HISTORICAL_FIXTURE`, `REVIEW_REQUIRED`, `EXTERNAL_OUT_OF_SCOPE`.
NEVER emit a MERGE action.

- Weak signals (age, inactivity, behind main, conflicts, similar title, draft,
  failed CI, no comments) NEVER authorize closure alone.
- `unique_delta` present+required → REVIEW_REQUIRED veto (no-loss rule).
- Race guards first: pr_not_open / head_sha_changed / pr_merged → mutation blocked.
- Evidence types: explicit_successor (merged), merged_implementation +
  main_containment, historical_fixture + completed_by, duplicate_of
  (merged|surviving), external_upstream, weak_signal.

## Mutation gate ordering (pitfall)

A concurrently merged PR reports `state=closed` AND `merged=true`. Check
`merged` BEFORE `state`, else you return PR_NOT_OPEN instead of
PR_MERGED_CONCURRENTLY and the operator misreads the block reason.

## gh CLI JSON field quirks (cost real round-trips)

- `gh pr view --json merged` → "Unknown JSON field: merged". Use
  `mergedAt`/`mergeCommit`; merged = `mergedAt != null || mergeCommit != null`.
- `gh pr view --json repository` → unknown field.
- `gh pr list --json draft` → unknown field. The field is `isDraft`.
- `gh pr view --json files --jq '.files[].filename'` → null. Use
  `gh pr diff N --name-only` for filenames.
- `gh pr view --json baseRefOid` → unknown. Base SHA comes from REST
  `repos/{org}/{repo}/pulls` (base.sha).

## Blob-identity supersession proof

Two PRs are the same change when `git rev-parse <ref>:<path>` yields the same
blob for the same path (e.g. cloe.mjs 70411d2..af641df in both #81/#82;
observation_inbox.py blob 34cf1d8 on base). Use `git cat-file -e origin/main:<path>`
for presence, `git diff --stat origin/main...<head>` for merge-base deltas vs
`git diff origin/main <head>` for direct deltas.

## Pre-existing test-failure proof (control worktree)

Fresh worktrees need `npm ci` first (missing deps → ERR_MODULE_NOT_FOUND, e.g. zod).
To prove suite failures are pre-existing: run the full suite in the feature
worktree AND a control worktree of origin/main; extract failing TEST FILES only
(`grep '^not ok' | grep 'test/' | sort -u`); `comm -13 <(origin) <(mine)` empty
→ zero new regressions. Compare file NAMES, not assertion counts — one file can
carry hundreds of failures (e.g. openclaw-aionui.test.js + readonly-operator-cli-client.test.js).

## Full-suite pollution (lah-openclaw-mvp)

The suite regenerates tracked `test/reports/*.json` (restore before commit) and
creates untracked `data/` artifacts: cloe-governed-action-packets.json[.bak],
creative-assets/, decision-records/, memory-events/, test-closure-dt-jobs.json.
These are NOT gitignored (only `data/export/`) and NOT tracked. Never stage them;
commit only the mission's files, one by one.

## Engineering memory card generation (pitfall)

`finalize-mission-card.mjs` → `extract-core.mjs` reads at REPORT ROOT:
- `tests` (array | string | object) → validation.tests
- `runtime_evidence` (non-empty object) → runtime_verified=true
- `deployment.verified === true` → deployment_verified

Placing tests under a `validation` section is ignored → "SUCCESS card without
any validation" → validate-mission-card FAILS → INDEX BUILD BLOCKED (blocks the
whole engineering-memory index). Fix: root-level `tests: [...]` +
`runtime_evidence: {...}` in the mission report.

## Closure comment template

```
Closing without merge.

This PR is no longer the current implementation authority.

Classification: <CLOSE_SUPERSEDED | CLOSE_DUPLICATE | CLOSE_HISTORICAL_FIXTURE>

Evidence:
- <successor or canonical implementation>
- <merged PR / commit / continuity authority>
- <current-main containment or historical-fixture reason>

The historical discussion and branch are intentionally preserved.
No code from this PR is being merged as part of this closure.

Canonical current authority:
- <PR / commit / subsystem>

Final disposition:
SUPERSEDED_OR_HISTORICAL_PR_CLOSED_WITHOUT_MERGE
```

## Dead required CI check (merge guard)

When ci-governance (or any required check) fails on main itself — e.g. billing:
"spending limit needs to be increased" / "job not acquired by Runner" — document
the pre-existing failure in the PR, then merge with
`gh pr merge N --admin --merge --match-head-commit <full-sha>` after verifying
head SHA. Project governance = LOCAL_CI_VERIFIED local deterministic validation.

## Evidence persistence

Keep evidence OUTSIDE repos: `/home/deploy/evidence/<mission>/` — routing receipt,
initial snapshot, classification, audit report, evidence-for-close, mutation set,
closure receipts, mission report. Continuity JSON + engineering memory card go
into the repo (docs/mcporter/ + engineering-memory/cards/).
