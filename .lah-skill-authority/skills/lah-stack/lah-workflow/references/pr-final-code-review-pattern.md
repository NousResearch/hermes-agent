# PR Final Code Review Pattern (READ_ONLY_AUDIT of an existing PR)

Established during CLOE_RETRIEVAL_TOOLS_PRESERVATION_PR701_FINAL_CODE_REVIEW_V1.
Use for review-only missions: verdict only, no code changes, no push/merge/deploy.

## Phase 0 — PR authority (never skip)
- `gh pr view <N> --repo <owner>/<repo> --json state,isDraft,mergeable,mergeStateStatus,baseRefName,headRefName,headRefOid,additions,deletions,changedFiles,url,createdAt,updatedAt`
- Verify `headRefOid` equals the mission's EXPECTED HEAD — if moved, inspect new commits before continuing.
- `gh pr view <N> --json commits -q '.commits[] | .oid[0:8] + " " + .messageHeadline'`
- Reviews/checks: `gh pr view <N> --json reviews`, `gh pr checks <N>`, and the check rollup.
- **Dead CI check proof (do not trust "unavailable"):** `gh api repos/<owner>/<repo>/commits/<BASE_SHA>/check-runs --jq '.check_runs[] | {name,conclusion}'` — if `ci-governance` (or any required check) reports `failure` on the BASE SHA itself, it is pre-existing on main, not introduced by the PR. Record this as API evidence.

## Phase 1 — Diff scope
- `git diff --name-status <BASE>..<HEAD>` and `git diff --stat`.
- Assert: exactly the expected files; no package.json/package-lock changes; no route/parser/runtime/campaign/config files.
- Assert no uncommitted drift in the review worktree: `git diff HEAD -- <file> | wc -l` must be 0.

## Phase 2 — Production patch review
Read the FULL control flow around each hunk (not just the diff): the effective-state initialization (e.g. `effectiveTools`, `effectiveToolChoice`, `forcedSynthesis` declared once), the adjacent governance branches, and the budget/duplicate branches that follow. Answer the 14 questions (tools preservation, tool_choice preservation, no injection, empty-array semantics, grounding retention, forcedSynthesis semantics, no duplicate requests, greeting exposure, budget/duplicate/fallback/text-path untouched, tool_calls preservation). Identify the exact FIRST behavioral difference base→PR.

## Phase 3 — Tool choice contract matrix (run it, don't reason about it)
Write a throwaway probe that calls the REAL entry point (e.g. `buildNativeChatCompletions`) with a mock fetch that captures the provider payload, iterating:

- tools: absent / undefined / null / `[]` / `[valid]`
- tool_choice: absent / auto / none / required / `{type:'function',...}`

Report per combination: payload tools, payload tool_choice, decision, grounding injected, provider called. Special attention:
- `tools=[valid] + tool_choice=none` — explicit client choice must NOT be overridden.
- `tools absent + tool_choice=auto` in the patched branch — must not send a stale/invalid contract (ANSWER_SUFFICIENT forces `none`; safe).
- `tools null` — rejected upstream by `validateToolDefinitions` (decision n/a), behavior unchanged.

Delete the probe after capture; it is review tooling, not a deliverable.

## Phase 4 — Test quality classification
Check the test file imports the REAL production module (`buildNativeChatCompletions`), not a reimplementation. If the test copies the patched logic (e.g. its own `hasClientTools`) instead of asserting on the captured real payload, that is LOGIC_DUPLICATION = blocking. Classify groups: UNIT_REAL_CODE (real entry, mocked boundary), INTEGRATION_REAL_CODE (real entry + real knowledge stack), MOCK_HEAVY, LOGIC_DUPLICATION, INSUFFICIENT.

## Phase 7 — Pre-existing failure verification (dual-run, byte-comparable)
Do NOT accept "pre-existing" from file ownership or import relationships. Reproduce on BOTH SHAs:

```bash
# base worktree (clean inspection — allowed by review missions)
cd /home/deploy/lah-stack-repos/<repo>
git worktree add --detach /home/deploy/lah-stack-worktrees/review-<tag>-base <BASE_SHA>
cp -r <head-wt>/<subdir>/node_modules <base-wt>/<subdir>/node_modules
cd <base-wt>/<subdir> && node --test --test-concurrency=1 test/<file>.test.js | tee /tmp/review-base.log
cd <head-wt>/<subdir> && node --test --test-concurrency=1 test/<file>.test.js | tee /tmp/review-head.log
```

Same Node version, same command, same concurrency, same fixtures. Compare: failing test names, line numbers, error text (`+ undefined - true`), timing. Classify exactly one: IDENTICAL_PRE_EXISTING_FAILURE / PR_WORSENS / PR_FIXES / ENVIRONMENTALLY_UNSTABLE / INSUFFICIENT_EVIDENCE. Clean up with `git worktree remove --force` from the canonical checkout.

## Phase 9 — Static scans: use search_files, NOT terminal grep
Terminal grep for literal destructive patterns (`rm -rf`, `git push --force`) trips the terminal hardline blocklist — the pattern text itself is the trigger, even inside a regex; obfuscated variants can still be denied by the operator. Use the `search_files` tool (pure read-only primitive, no blocklist) for: secret scan, conflict markers, EXOCLICK activation, campaign mutation, raw payload/session logging. Reserve terminal for `node --check`, `git diff --check`, test runs.

## Phase 10 — Docs review gotcha
Operator packet / continuity JSON may carry INTERMEDIATE branch SHAs (written before later doc commits). The PR head is authoritative: `gh pr view <N> --json headRefOid`. Intermediate-SHA references are minor doc inaccuracies (non-blocking) — flag them, don't fail the review for them.

## Verdict taxonomy
- READY_TO_MERGE — minimal patch, real-code tests, contract preserved, no injection, no DSML weakening, pre-existing failures independently reproduced, no new failures, docs truthful.
- READY_TO_MERGE_WITH_DOCUMENTED_PRE_EXISTING_FAILURES — same but CI/dead checks or doc-SHA inaccuracies documented.
- REQUEST_CHANGES / REVIEW_BLOCKED — any unexpected source file, test-logic duplication, contract override, or DSML weakening.
