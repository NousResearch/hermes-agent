# PRD-8 — Aegis LCM Store Snapshot + Fresh-Reset for a Clean Phase-2 Benchmark

**Status:** DRAFT (pre-review)
**Owner:** Apollo
**Blast radius:** Aegis (break-glass agent) gateway + its LCM store. NOT Apollo, NOT any other profile.
**Privilege:** Bouncing the Aegis gateway is a privileged action (SOUL §7). Requires explicit go.

## 1. Problem

The Phase-2 LCM benchmark (PRD-7, Arm A raw-store + Arm B DAG node-served) drives the **live
Aegis** profile and writes every planted sentinel into Aegis's real LCM store
`~/.hermes/profiles/aegis/lcm.db`. After ~dozens of dev/validation runs that store now holds:

- **17,485 messages**, **296 MB**, **44 summary nodes**
- **1,421 sentinel-bearing rows** from PRIOR runs (`LCM-LIVE-RECOVERY-*`, `LCM-ARMB-*`, `recover-*`)

A confirmed consequence (smoke `arm-a-smoke-n3-tight`, trial `exact-1729-000`): the model called
`lcm_grep`, but the polluted store returned **other runs' sentinels** and it answered with the wrong
one. An N=180 run against this store measures "find the right needle among 1,421 stale needles," not
"does LCM recover the planted fact." The benchmark result would be **contaminated and pessimistic** —
unacceptable for a gate that decides the privileged Apollo cutover.

## 2. Goal

Give the Phase-2 campaign a **clean LCM store** so each trial recovers only its own planted fact,
while **losing nothing** (full reversibility) and **not disturbing any non-Aegis profile**.

Non-goal: changing the harness, the gate math, or the engine. This is store hygiene only.

## 3. Approach (Option 1 — snapshot + fresh store)

1. **Quiesce** the Aegis gateway (it holds `lcm.db` open with a 7.3 MB WAL).
2. **Checkpoint + snapshot** the current store to a timestamped backup (verifiable copy, incl. WAL).
3. **Move** the polluted db aside (do not delete) so Aegis recreates a fresh, empty store on start.
4. **Restart** Aegis; confirm a fresh `lcm.db` exists and is empty (0 messages / 0 nodes).
5. Campaign then runs against the clean store.

Rejected alternatives (documented): (2) per-run `--lcm-db` throwaway — more code, deferred; (3) run
dirty + post-filter — leaves `lcm_grep` noise, gives a contaminated number. Both rejected for THIS
gate; (2) is a reasonable future enhancement.

## 4. Detailed steps & commands

All paths under `~/.hermes/profiles/aegis/`. `$TS` = `date +%Y%m%d_%H%M%S`.
`DB=~/.hermes/profiles/aegis/lcm.db`.

### 4.0 Resolve + PIN the launchd label AND plist (config-drift defense)
- Resolve the real Aegis gateway label once: `launchctl list | grep -i 'hermes.*aegis'` →
  capture the exact label into `$LABEL`. **Echo `$LABEL` into the report** — it is evidence, never
  re-guessed on the next run. Abort if zero or >1 match (ambiguous → human).
- Resolve and PIN the **plist path** too: `launchctl print gui/$UID/$LABEL | grep -i 'path ='` (or
  the known `~/Library/LaunchAgents/<label>.plist`). **Assert the plist file exists on disk now**
  (pre-flight, not at restart time) — a `bootout` with a missing/moved plist strands Aegis with no
  `bootstrap` path. Echo `$PLIST` into the report. Abort if absent.
- Record the job's **KeepAlive policy** (`launchctl print gui/$UID/$LABEL | grep -i keepalive`) into
  `$KEEPALIVE`. This single recorded value drives BOTH the stop branch (§4.2) and the matching
  restart branch (§4.5) — one source of truth, no re-derivation.

### 4.1 Pre-flight (abort if any fails)
- Assert target is **aegis** only: operate on the literal `$DB`; never a glob, never another profile.
- Assert **no live campaign** running (`pgrep -f 'lcm_live_recovery|lcm_arm_b'` empty) AND drop an
  advisory lock `~/.hermes/profiles/aegis/.lcm-reset.lock` containing **`<operator-pid> <$TS>`**
- Assert **no live campaign** running (`pgrep -f 'lcm_live_recovery|lcm_arm_b'` empty) AND acquire an
  advisory lock **atomically** — `mkdir ~/.hermes/profiles/aegis/.lcm-reset.lock` (O_EXCL semantics:
  the second concurrent acquirer fails atomically, no check-then-act race), then write
  `<operator-pid> <$TS>` into it. Remove it in a trap on exit. **Stale-lock rule:** on a failed
  acquire, if the recorded PID is **dead**, override (rmdir + reacquire) with a logged warning — so a
  `kill -9`/reboot mid-op cannot permanently wedge Phase-2, but a *live* concurrent op is refused.
- Assert **free disk space ≥ 2× current `$DB` size** (scales as the store grows, vs a fixed 1 GB that
  silently becomes tight). Abort early if short — never start a copy that can half-finish.
- Record pre-flight counts/bytes for the report (informational only — the AUTHORITATIVE count is
  taken post-quiesce in §4.3 to avoid a TOCTOU mismatch).

### 4.2 Quiesce Aegis — and PROVE it stopped
- Branch on the recorded `$KEEPALIVE`: if true → `launchctl bootout gui/$UID/$LABEL` (actually holds
  it down). Else → `launchctl kill TERM gui/$UID/$LABEL`.
- Bounded wait (≤60s) until BOTH: the gateway pid is gone, AND
  **`lsof "$DB"` returns empty** (no process holds the db fd). This is the load-bearing guard —
  on macOS a `mv` of a still-open db silently splits brain (running gateway keeps writing the
  renamed inode, recreates nothing). **Do not proceed to any move until `lsof` is clean.** Timeout →
  abort + LOUD alert, touch nothing.
- **Abort-state alerting:** if we booted-out (KeepAlive path) and then abort before §4.5 bootstrap,
  the LOUD alert MUST say **"Aegis left BOOTED-OUT / DOWN — run rollback §4.6 bootstrap"** as a
  distinct paged state, not a generic timeout. The break-glass agent being administratively down is
  itself page-worthy.

### 4.3 Snapshot (reversible safety net) — authoritative count taken HERE
- With the gateway confirmed down: `sqlite3 "$DB" "PRAGMA wal_checkpoint(TRUNCATE);"` and **assert
  the returned `(busy, log, checkpointed)` triple has `busy == 0`** — a leading `1` means the
  checkpoint could not complete (a reader still attached) even though it didn't error; abort on busy
  rather than snapshot a partially-checkpointed db.
- **Assert** `$DB-wal` and `$DB-shm` are now absent/zero-length. Non-empty ⇒ something still has the
  db open ⇒ §4.2 failed ⇒ abort (cross-check on the lsof guard + the busy flag above).
- Take the **authoritative** message count from the checkpointed `$DB` now (post-quiesce).
- Copy `$DB` → `$DB.backup-$TS`; write `$DB.backup-$TS.sha256`.
- **Verify the backup against the source (not a stale pre-flight number):**
  `sqlite3 "$DB.backup-$TS" "PRAGMA integrity_check;"` == `ok` AND its message count == the
  authoritative §4.3 count. Abort the whole op if the backup doesn't verify — never move the original
  on an unverified backup.

### 4.4 Reset
- Move (NOT delete) the live trio aside: `$DB`, `$DB-wal`, `$DB-shm` → `*.polluted-$TS`
  (whichever exist). Move = recovery is `mv` back.

### 4.5 Restart + confirm clean — deterministic ordered sequence
Execute in **this exact order** (collapses the prior run-time forks into one known state):
1. **Restart** with the matching inverse of §4.2, driven by the SAME `$KEEPALIVE`: booted-out →
   `launchctl bootstrap gui/$UID "$PLIST"`; killed → `launchctl kickstart -k gui/$UID/$LABEL`.
2. **Readiness wait (bounded ≤60s, no-write):** retry `hermes -p aegis chat -Q -q "ping"` until the
   gateway accepts a turn. `-Q` recall is treated as read-only (proves liveness only). A just-started
   gateway may not have bound its socket / warmed the engine; only treat failure as real AFTER the
   window expires (else false-fails AC-3 on a healthy outcome).
3. **AC-3 functional probe — ONE explicit write+recall (unconditional):** plant a single
   probe fact whose text is **provably outside any sentinel namespace** (NOT `LCM-LIVE-RECOVERY-*`,
   `LCM-ARMB-*`, or `recover-*` — e.g. `RESET-PROBE-$TS`), then recall it. This is the real write+read
   that satisfies AC-3 — not the read-only readiness ping. It writes **exactly one** known row.
4. **Prove clean — TOLERANT of lazy schema, expecting EXACTLY the probe row:** guarded queries
   (`SELECT name FROM sqlite_master WHERE type='table' AND name=?` first, so a missing table reads as
   "clean (uninitialized)", never an unhandled `no such table` throw):
   - `messages`: **table absent OR `COUNT(*)` == 1** (exactly the §4.5.3 probe row), AND
   - `summary_nodes`: **table absent OR `COUNT(*)` == 0**, AND
   - `schema_version` consistent when present.
   Because the readiness ping is no-write and the AC-3 probe is the single write, the count is a
   deterministic `1`, never an ambiguous `{0,1}` sum-of-two-writes. Stage all printed output in the
   report so Ace sees the clean slate himself.
5. **Probe-row disposition:** the one `RESET-PROBE-$TS` row is documented as a benign baseline
   artifact (outside every sentinel namespace, so PRD-7's `lcm_grep`/sentinel matching cannot collide
   with it). The campaign's own sentinels dominate; `1` known row is acceptable and never anomalous in
   backup/rollback logic.

### 4.6 Rollback (documented, copy-paste; uses the pinned plist)
- Stop Aegis (bootout/kill per §4.2, branch on `$KEEPALIVE`) → confirm `lsof "$DB"` empty →
  `mv $DB.polluted-$TS $DB` (+ wal/shm) → **restart per §4.5** (bootstrap `"$PLIST"` if booted-out,
  else kickstart). Restores the exact prior store. **Echo the fully-resolved rollback command set
  (real `$LABEL` + `$PLIST`, no placeholders) into the report** so it's copy-paste ready. The
  verified `*.backup-$TS` is the second-line restore.

### 4.7 Retention / cleanup (don't silently fill the disk)
- After the Phase-2 campaign PASSES and Ace accepts the result, the `*.polluted-$TS` and
  `*.backup-$TS` pair (~600 MB) may be deleted. Until then, keep both. State this in the final report
  with the explicit deletion command so cleanup is a deliberate, logged step — not an orphaned 600 MB.

## 5. Acceptance criteria (all must hold; empirical, not asserted)
1. A `$DB.backup-$TS` exists, passes `integrity_check=ok`, and its message count == the
   **authoritative post-quiesce** count (proven, printed). (Not compared to a stale pre-flight number.)
2. Post-reset, the live `$DB` is **clean** = (`messages` table absent OR count == 1 (exactly the one
   `RESET-PROBE-$TS` row), `summary_nodes` table absent OR count == 0), via **guarded** queries that
   never throw `no such table` — proven by printed output. A lazily-uninitialized table reads as
   clean, not a fail; the count is deterministic (no-write readiness ping + single AC-3 write probe).
3. The Aegis gateway is **running and functionally healthy** — proven by a real accepted turn after a
   bounded readiness wait, not merely a live pid.
4. The polluted store is preserved at `*.polluted-$TS` AND a verified `*.backup-$TS` exists (two
   independent recovery paths).
5. **No file outside `~/.hermes/profiles/aegis/` is touched.** No other profile's gateway restarted.
   The pinned `$LABEL` resolved to exactly one aegis job; `$LABEL` and `$PLIST` are echoed in the
   report and `$PLIST` was proven to exist on disk in pre-flight.
6. Before any move, **`lsof "$DB"` was proven empty** (no process held the db) — the split-brain guard.
7. Rollback is a documented, copy-paste set with **resolved** `$LABEL`/`$PLIST` (no placeholders)
   echoed into the report; the advisory lock carries `<pid> <ts>`, is released on exit, and is
   stale-overridable (dead PID) so a crashed reset cannot wedge Phase-2.
8. If the KeepAlive/bootout path was taken, any abort between bootout and bootstrap pages a distinct
   **"Aegis left DOWN"** alert (not a generic timeout), and rollback restores it via the pinned plist.

## 6. Risks & mitigations
- **Wrong-profile blast:** hardcode the literal aegis path; assert profile==aegis; no globs. (AC-5)
- **Unverified backup → data loss:** integrity_check + count-match gate before the move. (AC-1)
- **Gateway won't come back:** bounded waits + LOUD alert; `.polluted-$TS` move is reversible. (§4.6)
- **WAL not checkpointed → torn snapshot:** explicit `wal_checkpoint(TRUNCATE)` before copy. (§4.3)
- **Privileged gateway bounce:** Aegis is break-glass/idle; still, requires Ace's go before run.

## 7. Out of scope
Harness changes, gate math, engine behavior, any non-aegis profile, deleting (vs moving) the old store.
