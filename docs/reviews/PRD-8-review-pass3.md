# Independent Senior Review (Opus)

## Verdict: APPROVE WITH CHANGES

## Critical Blockers (severity-ordered, cite section/evidence)

None reach BLOCK. This is a v2 that absorbed the entire prior review into the spec body rather than deflecting it — every prior Required Change is now written into the runbook with a matching AC. I verified each one:

- Prior #1 (ping best-effort + AC-2 becomes a range) → **closed**: §4.5 "Prove clean — TOLERANT of lazy schema" + guarded `sqlite_master` lookup + AC-2 "table absent OR count∈{0,1}". The unhandled `no such table` throw is genuinely gone.
- Prior #2 (plist pin + KeepAlive-matched restart) → **closed**: §4.0 pins `$PLIST` and asserts it exists on disk pre-flight; §4.5 binds the restart verb to the recorded `$KEEPALIVE`.
- Prior #3 (readiness wait before ping) → **closed**: §4.5 bounded ≤60s readiness wait, ping-failure only real after window expires.
- Prior #4 (rollback cross-ref + resolved plist) → **closed**: §4.6 now points at §4.5 for restart, echoes fully-resolved command set.
- Prior #5 (stale lock on hard-abort) → **closed**: §4.1 `<pid> <ts>` + dead-PID stale-override rule.

The residual items below are Required Changes because the runbook is a one-shot privileged op and these are the seams where a *correct* outcome can still misreport or a step can silently no-op. None justify BLOCK.

## Required Changes

1. **The ping read-only/write disposition is still left as a run-time "confirm which" instead of being resolved before go (§4.5 ping-row disposition, AC-3, Open Q).** The spec now branches three ways — delete the row (noted unsafe), document-as-benign, or "confirm it's read-only at run time." This is the one unresolved fork from the prior review, and it's load-bearing for AC-3: if `-Q` is read-only, the readiness ping proves the gateway answers but does **not** prove the store initialized or that persistence works — so AC-3's "real write+read probe" is unmet by the ping alone. The spec half-acknowledges this ("if the ping is confirmed read-only, add one trivial write-then-recall"). Make that unconditional: **always** do one explicit write-then-recall as the AC-3 probe, independent of whether `-Q` writes, and have *that* known row be the one AC-2 tolerates (count∈{0,1}). That collapses the three-way fork into one deterministic state and removes the only "decide at run time" in a privileged runbook.

2. **`wal_checkpoint(TRUNCATE)` return code is not asserted (§4.3).** The spec asserts `-wal`/`-shm` are absent/zero-length *after* the checkpoint, which is the right post-condition — but `PRAGMA wal_checkpoint(TRUNCATE)` returns a `(busy, log, checkpointed)` triple where a leading `1` (busy) means it could **not** complete even though it didn't error. On a clean lsof-verified quiesce this should be `0|...`, but capture and assert the busy flag == 0 explicitly; otherwise a partial checkpoint that still leaves a short WAL is caught only indirectly by the zero-length assertion (which a racing reader could transiently satisfy). Cheap belt-and-suspenders on the load-bearing snapshot step.

3. **Lock acquisition is TOCTOU-racy as written (§4.1).** "Assert no live campaign (pgrep empty) AND drop an advisory lock" is check-then-act — two operators (or operator + campaign launcher) can both pass the pgrep, both write the lock, last-writer-wins. For a single-human-operated break-glass op this is low-probability, but the fix is free: create the lock with `O_EXCL` semantics (`set -o noclobber; > lockfile` or `mkdir` lock) so the second writer fails atomically, then apply the dead-PID stale-override on the failure path. As written, the stale-override and the acquire are not sequenced against a concurrent acquirer.

4. **Readiness/clean-check ordering can mask a non-empty fresh store (§4.5).** The readiness wait pings until the gateway accepts a turn, *then* the clean-proof runs. If the write-then-recall AC-3 probe (change #1) runs as part of readiness, the clean check must run **after** and must expect exactly the probe's row count — sequence it explicitly so the AC-2 count and the AC-3 probe-row are the same known artifact, not two independent writes that sum to 2 and trip the ≤1 tolerance. State the order: reset → restart → readiness-ping (no-write or counted) → single AC-3 write+recall → clean-proof expecting that exact count.

## Lens Notes (one line each)

- **Architecture:** Unchanged, sound, scope-disciplined (§3, §7) — store hygiene only, future per-run `--lcm-db` correctly deferred not smuggled in.
- **Security/identity-isolation:** Strong — literal aegis path, single-label abort (§4.0), lsof empty-fd split-brain gate (§4.2/AC-6), no globs; isolation is enforced not asserted.
- **DevOps/SRE:** Stop/restart/rollback are now symmetric and plist-pinned; remaining gaps are the unasserted checkpoint busy-flag (change #2) and the readiness↔clean-check ordering (change #4).
- **Implementation/maintainability:** One-shot runbook reads as executable; the single "decide at run time" fork (change #1) is the one maintainability smell — resolve it to a fixed sequence.
- **QA:** AC-1/4/5/6/8 are real two-path empirical gates; AC-2 is a correctly-bounded range; AC-3 is genuine *only if* the unconditional write+recall (change #1) replaces the conditional one.
- **Config-drift:** `$LABEL`+`$PLIST`+`$KEEPALIVE` pinned as echoed evidence and existence-asserted pre-flight — this is the right pattern; nothing re-guessed at restart.

## Residual Risks / Open Questions

- **Does the campaign launcher actually honor `.lcm-reset.lock`?** The spec says it "checks" the sentinel and treats dead-PID as stale — but that's a contract on a *different* component (PRD-7's launcher) asserted here, not proven. Confirm the launcher code reads this lock, or the lock is decorative.
- **Single benign row vs. zero:** even with change #1, the campaign's first trial starts against one known AC-3 probe row. Confirm PRD-7's `lcm_grep`/sentinel logic can't match the probe text — pick probe content that is provably outside any sentinel namespace (`LCM-LIVE-RECOVERY-*`, `LCM-ARMB-*`, `recover-*`).
- **`launchctl print` output format drift across macOS versions:** the `grep -i 'path ='` / `keepalive` parsing in §4.0 is screen-scraping a human-formatted command; a macOS point-release wording change silently breaks the pin. Low-probability, but the pre-flight existence-assert on `$PLIST` is what saves you — keep it.
- **296 MB copy under the ≥1 GB free assert:** fine today, but the assert is a fixed 1 GB while the store grows run-over-run; phrase it as `≥ 2× current $DB size` so it scales rather than silently becoming tight.
- **Backup retention vs. accidental campaign-fail:** §4.7 deletes `.polluted`/`.backup` only after PASS + Ace accepts — good, but state who fires the delete (it's a deliberate human-gated step, not a trap), so a failed campaign never auto-reaps the only recovery path.