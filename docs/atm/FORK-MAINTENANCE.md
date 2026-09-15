# Fork Maintenance — how randlee/hermes-agent tracks the upstream firehose

Upstream (`NousResearch/hermes-agent`) lands 200–700 commits/day. This fork exists to
carry ONE thing on top of it: the ATM injection patch stack (see
`PATCH-REQUIREMENTS.md` in this directory — read it first; it is the contract).

## Repo roles

| Thing | Role |
|---|---|
| `main` | upstream/main + the ATM patch stack, advanced only by reviewed PR |
| `atm/stack` branch | the patch stack (4 linear commits: seam reseat, HGF-001..004 fixes, HGF-005 whitespace, heartbeat-stub test fix, + this docs commit), rebased onto upstream per release; advanced by delete + recreate to the LINEAR candidate tip — NEVER force-push (force-push needs interactive user auth; the pipeline runs unattended) and NEVER advance to a merge commit (see lessons) |
| `sync/candidate-YYYYMMDD` branches | per-release PR candidates produced by the cron |
| `release/vX-atm` branch + `vX-atm` annotated tag | **the published, installable artifact**: `atm/stack` cherry-picked onto the upstream release TAG `vX` (not main head). Host runtimes (`make_runtime --ref vX-atm`) and the colima testbed install from this tag. One per upstream release. See "Release publishing" below |
| `runtime-*` tags | sources of built runtimes (see `~/.hermes/RUNTIME-PLAN.md` on the gateway host) |
| `~/Documents/forks/hermes-agent` (host) | integration WORKSPACE only — nothing executes from it (enforced by runtime-audit) |
| `~/.hermes/runtime/<tag>/` (host) | immutable runtime installs; gateways run `runtime/current` |

## Per-release pipeline (2-level cron)

The cron runs daily but is **release-gated**: it acts ONLY when upstream has cut a
release tag the fork hasn't synced (`git fetch --tags` is mandatory — a one-shot
refspec fetch does not import tags; this exact bug silently froze the gate for 11
days, grecon-995). No new upstream tag → status `current`, nothing rebased, no PR,
no review chain, zero tokens. So "daily" describes the schedule, not the work: real
work happens once per upstream release.

**Level 1 — mechanical (no agent judgment):** `hermes_ops sync` (Python module in
hendrix `hermes-ops/`, invoked by the cron via `~/.hermes/scripts/fork-sync.py`;
unit-tested, idempotent — safe to re-run after any failure) in a scratch clone:
1. fetch upstream (with `--tags`); branch `sync/candidate-YYYYMMDD` from `upstream/main`
2. rebase `atm/stack` onto it (`git rebase`); a clean rebase proceeds, ANY conflict
   → level 2
3. fresh venv: `uv sync --frozen --no-dev --extra messaging`; run the seam contract
   tests (`tests/gateway/test_inject_internal_message.py`, 31 expected) + hooks tests
   (`tests/gateway/test_hooks.py`) = 38 total
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

## Release publishing (the installable artifact) — Rand ruling 2026-09-15

> "we build randlee/hermes-agent and patch every release. That should be what is
> installed."

The publishable unit is the **upstream release tag + ATM stack**, never a floating
branch head. The level-1 sync keeps `main`/`atm/stack` current; **publishing** turns a
synced release into the named artifact hosts and the testbed install from. Division of
labor: **grecon triggers** (release-gated runner → green candidate), **loki publishes**
(this procedure), **Rand approves** any `runtime/current` flip.

For each new upstream release `vX` with no `vX-atm` yet:

1. **Inputs.** `atm/stack` must be LINEAR — `git log -1 --format='%P' origin/atm/stack`
   shows ONE parent. If merge-tipped, rebuild the linear stack first (see lessons); never
   cherry-pick a merge tip.
2. **Branch from the TAG, not main head.** `BASE=$(git rev-parse 'vX^{commit}')` (deref
   the annotated tag); `git worktree add -b release/vX-atm <path> $BASE`.
3. **Cherry-pick the linear stack:** `git cherry-pick $(git merge-base origin/atm/stack
   upstream/main)..origin/atm/stack` — replays ONLY the stack commits onto the tag.
4. **Gate (all three):** `git diff --stat $BASE..HEAD` == exactly the 5-file stack
   (`gateway/run.py`, `gateway/run_startup.py`, `tests/gateway/test_inject_internal_message.py`,
   `docs/atm/PATCH-REQUIREMENTS.md`, `docs/atm/FORK-MAINTENANCE.md`); `git diff --check`
   clean; frozen-venv 38/38 (`uv sync --frozen --no-dev --extra messaging` then
   `uv run --frozen --no-dev --extra messaging --with pytest --with pytest-asyncio python -m pytest
   tests/gateway/test_inject_internal_message.py tests/gateway/test_hooks.py -q`).
5. **Publish (tags only, never force-push):** `git push origin release/vX-atm`;
   `git tag -a vX-atm <tip> -m "<base tag>, stack <tip>, 38/38 tests, <diffstat>"` (the
   annotated-tag message is self-certifying evidence); `git push origin vX-atm`; verify
   both with `git ls-remote origin refs/heads/release/vX-atm refs/tags/vX-atm`.
6. **Report** branch sha + tag-object sha to fenix@atm-dev (and grecon for bead closure).

`make_runtime --ref vX-atm` and the colima testbed (`HERMES_REF=vX-atm`; the testbed
resolver refuses branch heads and unpatched tags) install from this tag. Reference
instance: `v2026.9.14-atm` = b78679eb3b (tag object e399ff9d15), base v2026.9.14
(345cd2b057, hermes-agent 0.21.3), 4 stack commits, zero conflicts, 38/38.

The full operator checklist lives in loki's `fork-release-publishing` skill (hendrix
`loki/skills/`); this section is the canonical in-repo summary that travels with the
stack.

## Promotion (deliberate, not automatic)

Merged main ≠ published ≠ deployed — three distinct steps. A synced `main` becomes an
installable artifact only when published as `vX-atm` (above); a published tag becomes
live only when promoted. To deploy a published release: on the gateway host
`make_runtime --repo <fork> --ref vX-atm --name runtime-N --hermes-atm <ver>
--atm-graft <ver>` (all args required — the tool refuses to default), canary one profile,
**Rand approves**, flip `runtime/current`, rolling restart. Rollback = flip the symlink
back. Full procedure: `~/.hermes/RUNTIME-PLAN.md`.

## History / lessons already learned

- Merge-based daily syncs (the pre-2026-08-16 pipeline) accumulated conflict debt
  and once ended with an agent force-pushing main and loosening branch protection.
  Rebase-the-stack + PR + protected main is the replacement. Do not regress to it.
- **The stack pointer must stay LINEAR.** Advancing `atm/stack` to merged main (a merge
  commit) instead of the linear candidate makes the next runner rebase die with a
  misleading `conflict_files:1` / "patch stack no longer applies" — the runner cannot
  replay a merge tip (observed 2026-09-09, grecon-1hh: the advancement after the P1 fix
  PRs pushed merged main). Diagnose with `git log -1 --format='%P' <stack-tip>` (two
  parents = merge-tipped); fix by rebuilding the linear stack onto upstream and advancing
  to the LINEAR candidate tip. Check the tip's parent count after EVERY advancement.
- **The release gate needs `--tags`.** A one-shot refspec fetch (`fetch upstream main`)
  does NOT import tags, so the gate compared against a frozen tag set and reported
  `current` for 11 days while upstream shipped two releases (grecon-995, fixed in hendrix
  c965d1c). Always `git fetch --tags`.
- The patch's only recurring conflict is the `gateway/run.py` import block (trivial);
  upstream churn occasionally extends the seam's test stubs (e.g. heartbeat-restore added
  to `_start_post_connect_services` — a test-only fixture extension, production untouched).
- Goal state is patch size ZERO: if upstream ever ships a public injection API,
  adapt hermes-atm to it and retire this stack.
