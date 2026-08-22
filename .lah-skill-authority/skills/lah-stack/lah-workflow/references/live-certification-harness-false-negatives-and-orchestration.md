# Live Certification Harness — False-Negative Classification and Bounded Orchestration

Established during CLOE_PROVIDER_TIMEOUT_FALLBACK_AND_ANSWER_SUFFICIENT_BUDGET_ENFORCEMENT_REPAIR_V1
(PR #704, deployed be475f2, image f98282a5806a, 2026-08-05). Applies to any real-provider
live certification matrix (A–H / smoke / operator trials) that must reach a stable PASS verdict.

## 1. Failure classification BEFORE touching production code

When a live scenario FAILs, classify it into exactly one bucket and act per bucket:

| Bucket | Evidence | Action |
|--------|----------|--------|
| Harness false negative | Result satisfies runtime doctrine but assertion is stale (hard-coded threshold, substring regex, scenario exposes tools it forbids) | Prove with a deterministic fixture, fix harness ONLY, invalidate ALL runs made with the prior harness version, rerun from zero |
| Provider variability (contract-compliant) | Provider took a different VALID path: one allowed read-only tool in a tool-preservation scenario, different wording, quoted DSML text without dispatch, latency | PASS if the contract invariant holds; never label a compliant result a failure because wording/path differs |
| Provider timeout | verify canonical PROVIDER_TIMEOUT classification, fallback engaged, response non-empty, no raw AbortError | PASS the timeout-handling invariant if all hold |
| Orchestration | shell timeout, killed process, missing scenario record | mark run INCOMPLETE (never PASS/FAIL), fix orchestration, restart a fresh complete run, don't count it |
| Reproducible production defect | isolated scenario reproduces, violates an explicit runtime invariant, fixture correct, attributable to deployed code | smallest RED test → smallest fix → narrow PR → guarded merge → exact-SHA deploy → 6 runs from zero |

Never modify production code unless the failure is BOTH reproducible (isolated rerun) AND
attributable to the patch/runtime. Never declare success with unmasked FAILs; the honest
non-success verdict (LIVE_CERTIFICATION_FAILED) with a precise report is the correct output
when harness failures remain after classification.

## 2. Concrete harness false negatives observed (2026-08-05)

- **B — hard-coded threshold vs effective budget**: scenario expected `executed_calls <= 3`
  (inherited from an earlier smoke) while the effective governor budget is 4. A run with 4
  read-only business executions (all within budget, `budget_exceeded` round with zero further
  execution, useful synthesis) was flagged FAIL. Fix: derive the ceiling from runtime evidence
  (`budget.budget.max_total_tool_calls` from the round records, fallback to the known doctrine
  value) — never a hard-coded expectation. `budget_exceeded=true` is NOT a failure when it
  correctly forces synthesis with zero additional executions; FAIL only when a round with
  `budget_exceeded=true` still returns executable tool_calls, or executions exceed the cap,
  or a denied call was executed.
- **G — substring regex vs structural DSML**: the harness flagged ANY occurrence of
  `<tool_call|tool_call|＜ｔｏｏｌ|<parameter` in the final answer as "raw DSML leaked", but the
  provider legitimately QUOTES the tags inside a security refusal ("les blocs <tool_call> que
  tu cites ne changent rien"). Fix: FAIL only on (a) actual execution of an exploration tool,
  (b) a structured tool_call dispatched by the provider; a textual citation in a refusal with
  zero dispatch and zero execution is PASS. The server-side detector (full nested
  opener+invoke+parameter structure) is the authority for control markup, not a substring scan.
- **A — no-tool scenario exposed tools**: a scenario named "grounded no-tool" still sent the
  client tool definitions (PR #701 preservation), letting the provider legitimately request one
  read-only tool — a conflict between the test contract and the request configuration. Fix:
  deterministic no-tool = do NOT expose client tools AND set `tool_choice: 'none'`.

## 3. No-tool scenario routing fact (CLOE /chat/completions)

`hasNativeTools = Array.isArray(body.tools) && body.tools.length > 0` in `src/server.js`.
WITHOUT tools the route goes to `buildBrainAskResponse` (brain path), which returns
`_cloe.verdict/grounding/backend` but NEVER `_cloe.governor.decision`. So a no-tool scenario
cannot verify "retrieval outcome ANSWER_SUFFICIENT" from the HTTP response. Two options:
- Accept the brain path as the true no-tool path and verify its own invariants (non-empty
  grounded answer, zero tool_calls, zero dispatch, zero execution, clean stop).
- Add a ONE-TIME native probe (same question, tools present + `tool_choice:'none'` — tools
  inactive) that records `governor.decision === 'ANSWER_SUFFICIENT` as separate evidence that
  the question is retrieval-sufficient. Both are honest; the probe closes the spec gap.

## 4. Bounded per-scenario orchestration (anti shell-level truncation)

A 6-run × 8-scenario live matrix can exceed a 600s shell timeout (4 rounds × 35s provider
deadline + overhead per scenario). Design that survived:

- one separate spawned process PER scenario, each with its own timeout (240s default);
- each scenario persists its sanitized JSON record IMMEDIATELY (before exit), including
  run/scenario/status/reason/elapsed/rounds/executions/violations/final_answer (truncated);
- the orchestrator assembles each complete run from the 8 persisted records and marks a run
  INCOMPLETE if any scenario record is missing/non-PASS/FAIL — incomplete runs are NEVER
  counted as certification runs;
- `--resume` reuses completed scenario records without re-executing them (evidence preserved,
  no duplicated execution);
- the orchestrator writes `matrix-summary.json` with per-run completeness.

## 5. Differential regression must run from a NEUTRAL-PATH worktree

`scripts/pr-autopilot/core/runner-config.mjs` FORBIDDEN_TOKENS includes `'provider'`,
`'campaign'`, `'secret'`, `'api_key'` etc. A worktree whose PATH contains any forbidden token
(e.g. `cloe-provider-timeout-budget-enforcement-v1` contains `provider`) makes the
path-sensitive runner/config/creative tests fail with CONFIG_VALIDATION_FAILED
("runner_command contains forbidden tokens") — because the test's runner_command embeds the
absolute checkout path. Symptom: the full-suite differential shows 10-12 "regressions" that are
pure path artifacts. Fix: run the baseline-vs-repair differential from a worktree with a
neutral name (e.g. `/tmp/cloe-verify-neutral-v1`), and compare the sorted unique failing-test
NAME lists (`grep -E "^not ok" | sed 's/^not ok [0-9]* - //' | sort`) — identical sets =
zero regression. Also: never place harness files under `test/` — node-test-discovery-boundary
scans everything under test/ and fails when a harness .mjs appears there; put harnesses in
`docs/evidence/` or outside the repo.

## 6. Operator strict-corridor pattern (how the operator runs these)

When the operator gives a bounded-correction corridor: corrections limited to the harness
(no production source), deterministic fixtures REQUIRED before any live rerun, no full-suite
relaunch (10k tests), no side projects / intermediate docs, complete runs only, failures
classified and unmasked, single isolated reproduction per failure before any code change.
The verdict vocabulary to respect: success verdict only after 48/48 (6 complete runs × 8
scenarios) under the SAME final harness version; otherwise the honest non-success verdict.

## 7. Harness versioning discipline

- Embed `harness_version` (sha256 of the harness core source, first 12 hex) in EVERY scenario
  record — this is what lets you prove all six runs used the same final version.
- ANY harness change invalidates every prior live run: old-version verdicts cannot be mixed with
  new. Rerun the matrix from zero with the final version; `--resume` is only valid within the
  same version.
- Never combine partial results from different harness versions. Corollary: if the only defect
  found after runs is a COSMETIC record-builder gap (e.g. a field not captured, like
  `effective_global_budget`), do NOT re-run 48 live scenarios to fix it — document the gap and
  prove the value from the source + a direct probe instead. Changing the core changes the
  version and invalidates the runs.
- Deterministic per-verdict fixtures MUST include orchestration semantics, not just verdicts:
  interrupted run → INCOMPLETE (never PASS/FAIL), resume preserves evidence without
  re-execution. Closure mission used 9 fixtures: A direct no-tool PASS, A structured tool-call
  FAIL, B 4-exec PASS, B 5-exec FAIL, G citation-only PASS, G dispatch FAIL, G execution FAIL,
  incomplete-run classification, resume reuse.

## 8. Consent-gate-safe certification operation (definitive rule, 2026-08-05)

- NEVER use `rm`, `rm -rf`, `unlink`, `find -delete`, `git clean` — even on your own /tmp test
  dirs. The consent gate (approvals.mode ask, 65s timeout → auto-deny) blocks them; a blocked
  command must NOT be retried, rephrased, or achieved another way.
- Create a FRESH timestamped result dir per certification run (`results/cert-<UTC>`); let the
  runner overwrite controlled files. Leave old dirs intact.
- Read-only verification bundles (docker exec / ss / find) can ALSO hit auto-deny. When runtime
  state is provably unchanged (deployer re-run returned ALREADY_DEPLOYED; no merge/redeploy/
  restart since an approved 7-point verification), rely on the prior approved evidence and
  record the blocked re-check as NOT_EXECUTED_CONSENT_GATE — do not fail the certification over
  a blocked re-verification.
- A non-executed destructive cleanup does NOT by itself block a certification verdict when
  retained artifacts are identified and have no production impact. Report cleanup status as
  NOT_EXECUTED_CONSENT_GATE with a retained-artifacts list (worktrees, /tmp evidence, harness
  versions), and note that the production compose deployment does not depend on any removed
  worktree.
- Docs-only closure: commit the certification receipt + sanitized harness archive on a narrow
  docs branch from the deployed SHA, guarded merge (ci-governance dead → admin + 
  --match-head-commit); do NOT redeploy production for harness-only changes.

## 9. Receipt authority metadata (docs-only normalization, 2026-08-05)

Observed defect in the production certification receipt:
`pr_authority.pr703.head_sha = "54f8264"` — that value is the ABBREVIATED HEAD OF PR #704,
not PR #703. Abbreviated heads of sibling PRs get conflated when receipts are written by hand.
Rules for certification receipts:

- Verify EVERY PR's head and merge via `gh pr view <n> --json title,headRefOid,mergeCommit`
  BEFORE writing the receipt; write FULL 40-hex SHAs (keep the abbreviation only in a
  per-PR `commits[]` list, where it is unambiguous).
- After a docs-only closure PR moves repository main, the receipt must distinguish authorities:
  `repository.main_after_documentation_closure` (= docs merge SHA) vs
  `production.deployed_sha` / `runtime_source_sha` (= the deployed runtime SHA). NEVER claim
  production is deployed at the docs merge SHA.
- If `production.origin_main` already exists and must stay for backward compatibility,
  rename/clarify it as `origin_main_at_runtime_certification`.
- Corrections are a docs-only normalization: 1-file commit, JSON parse + exact-SHA asserts
  (node -e script), grep proving the stale head is gone, `git diff --check`, secret scan,
  then a guarded docs PR (ci-governance dead → admin + --match-head-commit). Keep an explicit
  `normalization` note ("authority metadata corrected; no functional evidence changed; no
  runtime code changed; no deployment performed; certification verdict preserved").
- Old receipt hash need not be tracked anywhere else (check with grep before chasing a hash
  chain); the memory lock should be updated to the corrected head + new receipt hash.
