# Independent Senior Review (Opus)

## Verdict: APPROVE WITH CHANGES

## Critical Blockers (severity-ordered, cite section/evidence)

None rise to BLOCK. The plan is reversible (§4.6), profile-scoped (§4.1), and gated on a verified backup (§4.3/AC-1). But several "required changes" are load-bearing — ship without them and the empirical-proof and reversibility claims are partly fiction.

## Required Changes

1. **Schema assumption is unverified — AC-2 can throw instead of proving "clean" (§4.5, AC-2).** The proof query hardcodes table names `messages` and `summary_nodes`. A *fresh* store is created lazily by the engine — tables may not exist until the first write, in which case `SELECT COUNT(*) FROM messages` returns **`Error: no such table`**, not `0`. Your own §4.5 says "wait until a fresh lcm.db exists" — existence of the file ≠ schema initialized. Required: define "clean" as **(file exists AND (table absent OR count==0))**, or force schema creation by poking the gateway with one no-op recall before asserting. Otherwise the acceptance gate fails on a *correct* outcome.

2. **No verification that the moved-aside files are the ones the gateway actually had open (§4.2/§4.4).** You TERM the gateway then move `lcm.db{,-wal,-shm}`. If the launchd label resolution is wrong (you flag "resolve, don't assume" but never assert the *result*), `launchctl kill` no-ops, the process stays up holding the fd, and you `mv` a live, open database. On macOS the `mv` succeeds (inode survives) but the running gateway keeps writing to the now-renamed inode and **recreates nothing** — you get a silent split-brain. Required: after the bounded wait, assert **no process holds the db** (`lsof ~/.hermes/profiles/aegis/lcm.db` empty) before any move. This is the single most likely real-world failure and it's currently unguarded.

3. **`launchctl kickstart` on an already-running (or KeepAlive-respawned) service is ambiguous (§4.2/§4.5).** If the Aegis job has `KeepAlive=true`, `launchctl kill TERM` does **not** stop it — launchd respawns it within seconds, racing your move. Required: use `launchctl bootout` (or `disable`) to actually hold it down during the move window, then `bootstrap`/`enable`+`kickstart`. State the KeepAlive policy explicitly; the whole quiesce step is unsound if you don't know it.

3. **AC-1 count-match has a TOCTOU gap (§4.1/§4.3).** Pre-flight records counts (§4.1) *before* quiesce (§4.2). If anything writes between the count and the checkpoint, "count matches pre-flight" fails on a correct backup. Record the authoritative count **after** quiesce, from the checkpointed file, and compare backup-to-source — not backup-to-a-stale-preflight-number.

4. **Disk-space pre-flight missing (§4.3).** Snapshot (296 MB) + `.polluted` move + fresh store needs headroom; a verified backup of a 296 MB db plus retained polluted trio is ~600 MB resident. Add a free-space assert before §4.3 or the "abort on failure" promise hides a half-copied backup.

5. **WAL-checkpoint-then-still-running ordering (§4.3).** §4.3 runs `wal_checkpoint(TRUNCATE)` on `lcm.db`, but §4.2 already required the gateway stopped. Confirm the checkpoint runs against a *closed* db (good) — but then the `.db-wal`/`.db-shm` in §4.4 should be **absent/empty** after a TRUNCATE checkpoint. If they're non-empty post-checkpoint, that's evidence the gateway wasn't actually down (see change #2). Make their post-checkpoint state an explicit assertion, not an afterthought.

## Lens Notes (one line each)

- **Architecture:** Sound — snapshot+move+lazy-recreate is the right minimal shape; correctly rejects per-run `--lcm-db` as scope creep (§3).
- **Security/identity-isolation:** Good intent (literal aegis path, no globs, AC-5) but isolation is *asserted* by convention, not *enforced* — add an `lsof`/fd check so a mis-resolved label can't touch a live db.
- **DevOps/SRE:** Rollback is clean (§4.6); launch contract is the weak point — KeepAlive policy and actual-stop verification are undefined (changes #2/#3).
- **Implementation/maintainability:** One-shot operational runbook, fine; the hardcoded label and table names are the brittle bits (changes #1/#3).
- **QA:** AC-2/AC-3 are real empirical gates (printed counts, fresh pid) — but AC-2 is fragile against lazy schema (change #1) and "reports healthy" (AC-3) is undefined — specify *what* healthy check.
- **Config-drift:** "resolve the real label, don't assume" appears 3× but the resolved value is never pinned/echoed into the report — capture it as evidence so the next run isn't a re-guess.

## Residual Risks / Open Questions

- **What does "gateway reports running/healthy" concretely mean (§4.5/AC-3)?** Pid alive ≠ LCM store writable. Without one real write+read probe, AC-3 certifies process liveness, not store function — the exact "demo works ≠ production-safe" trap.
- **Does the fresh store need *any* seed state** (schema migrations, version row) that the engine writes only on a path your restart doesn't trigger? If the engine expects a `schema_version` row, an "empty" store may be subtly different from a *correctly initialized* one.
- **Concurrency with PRD-7:** §4.1 checks no campaign running, but nothing prevents one from *starting* mid-operation. Take an advisory lock (or set a sentinel file the campaign launcher checks) for the quiesce window.
- **Backup retention/cleanup:** unspecified. After a successful Phase-2 run, who deletes the 296 MB `.polluted`/`.backup` pair, and on what criteria? Left undefined, this recurs every dev cycle and silently fills the disk.
- **Numbering nit:** two "#3" items in Required Changes above (kickstart-ambiguity and TOCTOU) — both stand; renumber on revision.