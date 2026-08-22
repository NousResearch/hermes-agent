# Controlled Canonicalization: merge-in-dependency-order + post-deploy validation

Pattern proven 2026-08-16 on `LAH_MATH_STACK_CONTROLLED_CANONICALIZATION_AND_POST_DEPLOY_VALIDATION_V1`
(M1 merge #219 → gate, M2 merge #220 → gate, M3 #809 blocked). Use for any mission whose
objective is ONLY "merge certified code in dependency order, let auto-deploy run, verify,
stop on any unexpected change" — explicitly NOT an authority-promotion mission.

## Mission shape

- Merge PRs strictly in dependency order; each merge is followed by a post-deploy gate.
- Next merge proceeds ONLY if the previous gate PASSED.
- Any blocker → `BLOCKED_SAFE` with evidence; never improvise a fix in production.
- Keep all authority flags false; execute zero promotions (H11-style) unless separately authorized.

## Pre-merge reconciliation (run before ANY merge)

1. `git fetch origin --prune` both repos; record `origin/main` SHAs and confirm they are
   UNCHANGED since certification. Any change = material state change → stop.
2. Re-read every PR with `gh pr view <PR> --json state,mergeable,mergeStateStatus,headRefOid`.
   Compare `headRefOid` to the certified SHA — must match exactly.
3. Verify dependency topology with `git merge-base --is-ancestor <base_head> <pr_head>`.
4. Verify additive-only: grep production entry points (`src/server.js` requires, all `require(`
   of the new module dir) — new layer must NOT be imported by any production path. Also check
   repo workflows/hooks to know the deploy mechanism (no workflow = external/Hostinger auto-deploy).
5. Re-run the minimum certification suites on the exact PR heads (symlink node_modules into the
   bare worktree, run, then `rm node_modules`). Prove equivalence with certified artifacts.
6. Check branch protections (`gh api repos/<owner>/<repo>/branches/main/protection`): no
   protection → normal merge fine; required checks → a failing required check blocks the
   normal path and is NOT bypassable without explicit operator authorization.

## Verify the effective diff after an upstream merge (stacked-PR hygiene)

When PR #B was opened against main while containing commits shared with PR #A (its base),
after #A merges GitHub recomputes #B against the new main. Re-fetch and prove the effective
scope BEFORE merging #B:

```bash
git fetch origin main
git log --oneline origin/main..<branch-B>          # must be ONLY the new commits
git rev-list --count origin/main..<branch-B>        # count matches expected delta
git diff --name-only origin/main..<branch-B> | grep -E "<shared-phase-files>"   # must be EMPTY
```

If GitHub still presents duplicate/conflicting changes from the shared phase → stop, do not merge.

## Deployed-commit verification via prod /version (external auto-deploy)

For repos with NO GitHub workflow/hook (deploy is external — Hostinger-style main→live on merge),
verify the deployment actually landed by hitting the prod endpoint that exposes commit+time:

```bash
curl -s https://<prod-host>/version    # expect: commit == merge commit SHA, build_time == merge instant
```

Then gate checks (read-only probes only — never re-run the deployer):
- `/health` HTTP 200, twice a few seconds apart → no restart loop (build_time stable across probes)
- `db_connected` / provider guard flags still true
- No new routes exposed by the new layer: probe `/autocut /portfolio /bandit /discovery /causal`
  etc → 404; existing decision endpoints unchanged (401 auth is "exists", 404 is "never existed")
- No new math/policy processes: `ps aux | grep -iE "thompson|bandit|autocut|portfolio|..."`
- Authority invariants from code (grep the layer for `SHADOW_ONLY`, `v1_authoritative`,
  `NOT_IDENTIFIED`, `DATASET_INSUFFICIENT`, `GOVERNANCE_DECISION_REQUIRED`)

Record per-merge verdict `M<N>_POST_DEPLOY_PASS` / `FAIL` + a JSON receipt in the evidence dir.

## Environment-blocked required check → BLOCKED_SAFE, not admin-merge

When a required check fails in 2-3s, before treating it as a code failure diagnose:

```bash
gh pr checks <PR>                                   # shows the failing required check
gh run view <RUN_ID> --repo <owner>/<repo>          # ANNOTATIONS section explains
gh run view <RUN_ID> --log-failed                   # EMPTY output = job never started
```

Known environment blocker: GitHub Actions `billing_exhaustion` — annotation "The job was not
started because recent account payments have failed or your spending limit needs to be
increased". This is an ACCOUNT issue, NOT a code issue: the run never starts, `--log-failed`
is empty, and the same failure appears on unrelated branches/PRs simultaneously.

Rules:
- Do NOT use a prepared admin-merge packet for such a PR unless a NEW explicit operator
  authorization is given (mission text may pre-authorize merges but NOT the bypass exception).
- Leave the PR OPEN; report "ready once billing resolved — same commit, no code change expected".
- Do NOT label it an urgent incident unless an actual production incident exists.
- Check the OTHER open PRs on the same repo: same 2s billing failure = environment-wide proof.

## Rollback capability between steps

- Each merge is atomic and reversible via revert of the merge commit (or revert in REVERSE
  dependency order when #B's base is #A's merge — revert #B first, then #A).
- Keep the rollback statement explicit in the receipt: "revert #220 before #219".
- Branch protection absence means no protection bypass is involved — normal merge path only.
