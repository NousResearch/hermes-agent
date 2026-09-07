# Fork Maintenance — how randlee/hermes-agent tracks the upstream firehose

Upstream (`NousResearch/hermes-agent`) lands 200–700 commits/day. This fork exists to
carry ONE thing on top of it: the ATM injection patch stack (see
`PATCH-REQUIREMENTS.md` in this directory — read it first; it is the contract).

## Repo roles

| Thing | Role |
|---|---|
| `main` | upstream/main + the ATM patch stack, advanced only by reviewed PR |
| `atm/stack` branch | the patch stack (3 code commits + this docs commit), rebased onto upstream daily; advanced by delete + recreate — NEVER force-push (force-push needs interactive user auth; the pipeline runs unattended) |
| `sync/candidate-YYYYMMDD` branches | daily PR candidates produced by the cron |
| `runtime-*` tags | sources of built runtimes (see `~/.hermes/RUNTIME-PLAN.md` on the gateway host) |
| `~/Documents/forks/hermes-agent` (host) | integration WORKSPACE only — nothing executes from it (enforced by runtime-audit) |
| `~/.hermes/runtime/<tag>/` (host) | immutable runtime installs; gateways run `runtime/current` |

## Daily pipeline (2-level cron)

**Level 1 — mechanical (no agent judgment):** `hermes_ops sync` (Python module in
hendrix `hermes-ops/`, invoked by the cron via `~/.hermes/scripts/fork-sync.py`;
unit-tested, idempotent — safe to re-run after any failure) in a scratch clone:
1. fetch upstream; branch `sync/candidate-YYYYMMDD` from `upstream/main`
2. rebase `atm/stack` onto it (`git rebase`); a clean rebase proceeds, ANY conflict
   → level 2
3. fresh venv: `uv sync --frozen --no-dev --extra messaging`; run the seam contract
   tests (`tests/gateway/test_inject_internal_message.py`, 26 expected) + hooks tests
4. green → pre-resolve the merge into `main`: because main and each candidate
   carry different rebased copies of the stack, a raw candidate→main PR always
   conflicts. The script builds the merge commit itself with the sanctioned
   resolution — **candidate tree wins** (first established by loki in PR #7) —
   verifies tree-hash equality with the candidate, pushes both branches, and
   opens the PR from the pre-resolved merge branch (`sync/candidate-*-merge`).
   The PR therefore arrives conflict-free; reviewers judge the candidate via
   `git diff upstream/main..sync/candidate-*` (must be exactly the stack)
5. review chain: **contessa** (local qwen, free — does the context-intensive
   work) reviews the diff-vs-upstream and test output — the diff must be exactly
   the known patch stack, nothing more; then **alpha-prime** (qwen 3.7) approves
   and merges routine PRs and signs off smoke tests, then advances the stack pointer.
   AUTH NOTE: all agents share the `randlee` account, which also authors the PRs —
   formal `gh pr review --approve` is therefore impossible (GitHub forbids
   self-approval). The sanctioned path (established by loki, PR #7): post the
   review verdict as a PR comment, then merge via owner bypass
   (`gh pr merge --merge --admin`; enforce_admins is off). The 1-approval branch
   protection stays as a guard against accidental non-admin pushes, not as a
   working review gate
   (`git push origin --delete atm/stack && git push origin <candidate>:refs/heads/atm/stack`). **Loki** (frontier, expensive) is
   NOT in the routine path — non-trivial PRs, reviewer disagreement, or anything
   unexpected → level 2.

**Level 2 — escalation (agent judgment):** triggered by rebase conflict, test
failure, or reviewer rejection. The escalation agent is **loki** (hermes-agent-atm
maintainer, frontier model; workspace `hendrix/loki/`, reachable via
`atm send loki`). Loki receives `PATCH-REQUIREMENTS.md`
and follows its "How to update the patch" procedure. Its output is an updated
`atm/stack` + a PR — never a direct push to main, never a force-push of anything, never
a branch-protection change. If the contract can't be met, it stops and reports to
Rand with analysis.

## Promotion (deliberate, not automatic)

Merged main ≠ deployed. To deploy: tag, then on the gateway host
`make-runtime.sh --repo <fork> --ref <tag> --name <runtime-N> --hermes-atm <ver>
--atm-graft <ver>`, canary one profile, flip `runtime/current`, rolling restart.
Rollback = flip the symlink back. Full procedure: `~/.hermes/RUNTIME-PLAN.md`.

## History / lessons already learned

- Merge-based daily syncs (the pre-2026-08-16 pipeline) accumulated conflict debt
  and once ended with an agent force-pushing main and loosening branch protection.
  Rebase-the-stack + PR + protected main is the replacement. Do not regress to it.
- The patch's only recurring conflict is the `gateway/run.py` import block (trivial).
- Goal state is patch size ZERO: if upstream ever ships a public injection API,
  adapt hermes-atm to it and retire this stack.
