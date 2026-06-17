# Independent Senior Review (Opus)

## Verdict: APPROVE WITH CHANGES

## Critical Blockers (severity-ordered, cite section/evidence)

None reach BLOCK. Every prior-review blocker is genuinely closed in the spec text, and the revision did not paper over them — it operationalized them (lsof guard §4.2, post-quiesce authoritative count §4.3, KeepAlive-driven bootout vs kill §4.2, schema-forcing ping §4.5, disk assert §4.1, advisory lock §4.1, label pin §4.0). The remaining issues are Required Changes, not blockers — but two of them are load-bearing enough that shipping the runbook *as literally written* will misfire on a correct outcome.

## Required Changes

1. **The schema-forcing ping is asserted to work but its failure mode is unhandled, and it pollutes the very thing AC-2 measures (§4.5, AC-2).** The fix for prior-blocker #1 was "poke the gateway with one recall turn to force schema init." Two unguarded gaps: (a) **What if the ping turn does NOT touch the LCM store?** A `-Q` recall on an empty store may short-circuit (nothing to recall) and never trigger the lazy `CREATE TABLE` — then `SELECT COUNT(*) FROM messages` still throws `no such table` and the gate fails on a correct outcome, exactly the failure prior-#1 tried to kill. You must define "clean" as **(table absent OR count∈{0,1}) AND ...**, i.e. tolerate the table *not existing yet*, rather than relying on the ping to guarantee it exists. The ping is a best-effort schema poke, not a guarantee. (b) The ping writes a real message row, so AC-2's "0 messages (or the single documented ping row)" is now a **range**, not a count — and §4.6 rollback / §4.3 backup-count logic must not later treat 1 as anomalous. Minor, but specify whether the ping row is left in or deleted before the campaign, because the campaign's first trial now starts against a non-empty store.

2. **Restart-path branch selection is under-specified and can leave Aegis down (§4.5 vs §4.2).** §4.2 branches on KeepAlive: `bootout` if true, `kill TERM` if false. §4.5 must restart with the *matching inverse*: `bootstrap` after a `bootout`, `kickstart -k` after a `kill`. The spec lists both restart verbs with an "if booted out" parenthetical but does not bind the choice to the **recorded §4.0 KeepAlive value** as a single source of truth — and critically, after a `bootout` you need the **plist path** for `bootstrap gui/$UID <plist>`, which §4.0 never captures. Add to §4.0: resolve and pin the plist path alongside `$LABEL` (`launchctl print gui/$UID/$LABEL` shows it, or derive from the label). Without it, the `bootout` branch has no way to come back up and you've taken down the break-glass agent with a manual-recovery-only path.

3. **No post-restart wait-for-ready before the schema-forcing ping (§4.5).** "Wait until a fresh `$DB` file exists AND the gateway pid is fresh/running" then immediately ping. A just-bootstrapped gateway may not yet be accepting `hermes -p aegis chat` turns (socket not bound / engine not warm). A ping fired into a not-yet-ready gateway fails, and the runbook reads that as AC-3 failure on a healthy outcome. Add a bounded readiness wait (retry the ping ≤N times / ≤60s) before treating a ping failure as a real failure.

4. **Rollback's own restart path inherits the same plist gap (§4.6).** §4.6 says "restart per §4.2" — but §4.2 is the *stop* procedure; restart is §4.5. And the bootout-rollback needs the pinned plist (change #2). Tighten the cross-reference and confirm the rollback one-liner echoed into the report actually contains the resolved plist path, not a `<plist>` placeholder.

5. **Lock-file staleness on hard-abort (§4.1).** The advisory lock is removed "in a trap on exit," but a `kill -9`, panic, or host reboot mid-op leaves `.lcm-reset.lock` orphaned and the campaign launcher permanently blocked. Add: lock contains the operator PID + `$TS`; the campaign launcher treats a lock whose PID is dead as stale and overrides (with a logged warning). Otherwise a crashed reset silently wedges Phase-2.

## Lens Notes (one line each)

- **Architecture:** Unchanged and sound — snapshot+move+lazy-recreate, scope creep correctly rejected (§3); revision added the right guards without changing the shape.
- **Security/identity-isolation:** Now *enforced*, not just asserted — lsof empty-fd gate (§4.2/AC-6) + literal path + single-label abort (§4.0) close the mis-resolved-label-touches-live-db hole.
- **DevOps/SRE:** Launch contract is the residual weak point — stop path is solid, *restart* path lacks the pinned plist (change #2) and a readiness wait (change #3); rollback inherits both (change #4).
- **Implementation/maintainability:** One-shot runbook; brittle bits are the ping's best-effort nature (change #1) and the plist gap — pin both as evidence in §4.0.
- **QA:** AC-1/AC-4/AC-6 are real, two-path, empirically-printed gates; AC-2 is now a *range* not a count (change #1a) and AC-3 is a genuine write+read probe — good, but needs the readiness wait or it false-fails (change #3).
- **Config-drift:** `$LABEL` pin is the right instinct (§4.0) — extend it to the plist path so the bootout-restart isn't a re-guess.

## Residual Risks / Open Questions

- **Does `hermes -p aegis chat -Q -q "ping"` actually exercise the LCM write path?** If recall-only (`-Q`) skips persistence on an empty store, the ping proves the gateway answers but NOT that the store initialized — AC-3 then certifies less than it claims. Confirm the ping writes a row; if not, use a turn that does.
- **Campaign starts against a non-empty store if the ping row is retained.** One stale row is far better than 1,421, but it's still a planted-needle-adjacent artifact; decide explicitly: delete the ping row post-verification, or document it as benign in the AC-2 baseline.
- **KeepAlive=true + bootout leaves Aegis administratively disabled until bootstrap.** If the run aborts between bootout and bootstrap, the break-glass agent is *down*, not just bounced. The LOUD-alert-on-abort (§4.2) must explicitly cover "Aegis left booted-out" as a distinct, paged state — not just "timeout."
- **Plist path drift:** if the job was loaded from a path that no longer exists (moved/renamed since boot), `bootstrap` fails. Verify the pinned plist path exists on disk during §4.0 pre-flight, not at restart time.
- **Numbering from prior review:** the duplicate "#3" is resolved here; no carried-over numbering defects.