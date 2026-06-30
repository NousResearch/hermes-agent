# PRD — Recurring Upstream-Sync Cron (v0.2 follow-up)

**Status:** DRAFT (roadmap v0.2 — ships AFTER the v0.1 parity merge lands clean) · **Version:** v0.1
**Owner:** Apollo · **Parent:** `2026-06-29-upstream-parity-merge-PRD.md` (D-8, Roadmap v0.2)
**Trigger:** v0.1 merge deployed + stable on the fleet.

---

## 1. Summary & Goal
Stop drift from ever re-walling. A scheduled job keeps `fork/main`'s distance from
`NousResearch/main` **small and visible**, so a sync is a 5-hunk chore, never an 83-hunk project.
The cron does **not** auto-merge into the fleet — it does the cheap, safe part autonomously
(measure, attempt a throwaway test-merge, report) and **escalates to Apollo** when a real merge is
warranted. The merge-and-deploy itself stays the gated, Apollo-driven, Ace-signed-off flow from
v0.1.

**Why:** the v0.1 retro is unambiguous — we let it run 1,734 behind because nothing *watched* the
gap. A weekly heartbeat that surfaces drift + conflict cost turns a quarterly crisis into a
routine.

## 2. Non-Goals
- **NOT auto-merge to `fork/main`** and **NOT auto-deploy.** The cron proposes; Apollo disposes
  (drives the v0.1 pipeline); Ace signs off on fleet deploy. Same irreversibility gate as v0.1.
- **NOT a second conflict-resolver.** It runs a *throwaway* test-merge to count conflicts, then
  `--abort`s. It never resolves anything.
- **NOT noisy.** Healthy/small-drift weeks are silent or a one-line #logs tick. It only goes LOUD
  when drift crosses a threshold or a test-merge reveals a conflict spike.

## 3. Constitution / Invariants
- **INV-1 — Read-only against live refs.** The job only `git fetch`es + runs a test-merge in a
  **throwaway worktree it removes**; it never writes `fork/main`, never pushes, never deploys.
  *Proof:* the script has no `git push`/`merge --no-ff fork`/`hermes update` call; a grep gate in
  CI/test asserts absent.
- **INV-2 — Escalation, not action.** On a trip, it sends Apollo an alert with the numbers; it
  does not start a merge. *Proof:* the only side-effect on trip is a `notify`/`fleet_alert` send.
- **INV-3 — Self-cleaning.** Every run removes its scratch worktree even on failure (trap/finally).
  *Proof:* `git worktree list` is unchanged before/after a run.
- **INV-4 — Cheap.** No agent loop on the happy path — a `no_agent` shell script on a cheap
  cadence (per `scheduler` + `cron-alert-discipline`). Escalation hands off to Apollo only when
  there's real work.

## 4. Resolved Decisions
- **D-1 — `no_agent` shell script, weekly.** Pure bash + git; no model spend on the measure step.
  Weekly (Mon 09:00 PT) balances freshness vs noise. Cadence tunable.
- **D-2 — Two-threshold trip.** Escalate if **behind > 150 commits** OR a **test-merge conflict
  count > 15 files**. Either signals "merge now, while it's cheap." (Numbers seeded from v0.1
  pain; tune after 4–6 weeks of observed baselines.)
- **D-3 — Quiet by default.** Below threshold → one line to #logs (or silent). Above → LOUD to
  #alerts via the house `fleet_alert` envelope, addressed to Apollo.
- **D-4 — Reuses v0.1 machinery.** When Apollo picks up the escalation, it runs the **same** v0.1
  PRD pipeline (spec is reusable as a template; the conflict-class triage + two-stage spine apply).

## 5. Architecture
launchd/cron (`no_agent`) → `scripts/upstream-drift-watch.sh`:
1. `git fetch origin fork` (read-only).
2. Compute `ahead`/`behind` counts + merge-base age.
3. Throwaway worktree at upstream HEAD; `git merge --no-commit fork/main`; count conflict files;
   `git merge --abort`; `git worktree remove --force`.
4. Compare against thresholds (D-2). Below → #logs one-liner / silent. Above → `fleet_alert warn
   upstream-drift "<N> behind, <M> conflict files — time to sync"` to #alerts.
5. Append a row to a small drift-history log (`docs/sync/drift-history.tsv`) so we can see the
   trend and tune thresholds.

## 6. Implementation Phases
- **Phase 1 — the watch script.**
  - *Unit/script check:* run it manually against current refs; prints `ahead/behind/merge-base-age/conflict-files` and exits 0.
  - *E2E:* on a synthetic "way behind" state (point at an old base), it trips and sends a test alert to a scratch channel.
  - *Negative/adversarial:* kill the script mid-test-merge → trap still removes the scratch worktree (`git worktree list` clean). Grep proves no `push`/`hermes update`/`merge fork` call exists (INV-1).
  - *Verify with:* `bash scripts/upstream-drift-watch.sh --dry-run` → correct numbers, scratch worktree gone, no live writes.
- **Phase 2 — schedule it.**
  - *Unit/script check:* registered via `scheduler` (`no_agent`), weekly Mon 09:00 PT; `launchctl`/cron lists it.
  - *E2E:* first scheduled run lands a #logs tick (healthy) or an #alerts escalation (if already drifted).
  - *Verify with:* the job fires on schedule and routes per D-3.

## 7. Security / Ops
Read-only; no credentials beyond git fetch. Alert routing per house envelope (skill `notify`).
Self-cleaning worktree (INV-3). Cheap model-free happy path (INV-4).

## 8. Risks & Mitigations
| Risk | Mitigation |
|---|---|
| Scratch worktree leak on crash | trap/finally `git worktree remove --force` (INV-3) + Phase-1 kill test |
| Threshold noise (cries wolf or stays silent too long) | drift-history log → tune D-2 after 4–6 weeks of real baselines |
| Cron silently dies (no heartbeat) | `scheduler` watchdog pattern; a missing weekly tick is itself detectable |
| Test-merge cost on a huge drift | bounded — it's one merge + abort in a throwaway tree, no resolution |

## 9. Open Questions
1. **Weekly vs biweekly?** *Rec: weekly to start; relax to biweekly if 4–6 weeks show consistently small drift.*
2. **Auto-open the v0.1-style spec on a trip, or just alert?** *Rec: just alert first; let Apollo decide scope. Auto-spec-drafting is a later nicety once the cadence is proven.*

## 10. Acceptance Criteria
- [ ] `scripts/upstream-drift-watch.sh` prints accurate ahead/behind/conflict counts and self-cleans. Evidence: manual dry-run + `git worktree list` unchanged.
- [ ] Grep proves no live-write call path (INV-1). Evidence: `grep -nE 'push|hermes update|merge.*fork/main' scripts/upstream-drift-watch.sh` returns only the read-only test-merge.
- [ ] Trips to #alerts above threshold, quiet below. Evidence: synthetic over/under-threshold runs route correctly.
- [ ] Scheduled `no_agent` weekly; fires on cadence. Evidence: scheduler registration + first live tick.
- [ ] Drift-history row appended each run. Evidence: `docs/sync/drift-history.tsv` grows by one row per run.
