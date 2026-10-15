---
name: hermes-hybrid-sync-lock-recovery
description: Stabilize a Hermes hybrid watcher + scheduled Git sync when concurrent runs cause index.lock or overlapping git failures. Use when local-first sync must stay enabled and you need proof that the locking fix actually works.
---

# Hermes hybrid sync lock recovery

Use this when a local Hermes/Git sync setup has both:
- a watcher/event-driven trigger, and
- a periodic scheduled sync,

and you see failures caused by overlap, such as `index.lock`, concurrent `git add/commit/push`, or stale lock behavior.

## Goal
Keep the repository local-first, retain both fast watcher-based sync and scheduled catch-up sync, and eliminate overlap failures without disabling the hybrid design.

## Approach
1. Keep the repo/model/data local if capacity allows; do not move to shared/network storage just to work around a locking bug.
2. Serialize all sync executions with a single lock mechanism shared by both watcher and scheduler.
3. Verify not only by reading logs, but by forcing overlapping runs and confirming both exit cleanly.
4. Treat old failures in logs as historical until a fresh overlap test disproves the fix.

## Implementation pattern

### 1) Identify both entrypoints
Find every way the sync can start, typically:
- systemd service running a watcher
- systemd timer / cron invoking the sync script directly
- manual force-sync command

All of them must funnel through the same guarded script or wrapper.

### 2) Add a non-blocking lock around the critical section
Preferred pattern in bash:

```bash
LOCKFILE="${LOCKFILE:-/path/to/repo/.sync.lock}"
exec 9>"$LOCKFILE"
if ! flock -n 9; then
  echo "sync already running; exiting"
  exit 0
fi
```

Important:
- Put the lock before any `git add`, `git commit`, `git pull`, `git push`, or index mutation.
- Make overlapping invocations exit successfully (`0`) when skipping is acceptable. This prevents watchdog noise and proves serialization instead of failure.
- Use one shared lock path for watcher, timer, and manual force-sync.

### 3) Keep the critical section narrow but complete
Inside the lock, perform the full mutation flow, for example:
- refresh repo state
- stage files
- commit if needed
- pull/rebase if your workflow requires it
- push

Do not release the lock between git subcommands.

### 4) Make the service call the guarded script
Do not duplicate git logic in multiple places. The watcher and schedule should both call the same script or wrapper so the lock behavior stays identical.

### 5) Validate script syntax before restart
For shell scripts:

```bash
bash -n /path/to/script.sh
```

Only restart the service after syntax passes.

### 6) Restart and inspect service health
Typical checks:

```bash
systemctl --user daemon-reload
systemctl --user restart <service>
systemctl --user status <service> --no-pager
journalctl --user -u <service> -n 100 --no-pager
```

### 7) Run a deliberate overlap test
This is the key verification step. Launch two near-simultaneous force-sync runs and confirm both return cleanly.

Example pattern:

```bash
(<force-sync-command>; echo first:$?) &
(<force-sync-command>; echo second:$?) &
wait
```

Success criteria:
- both commands return `0`
- no fresh `index.lock` or concurrent git mutation error appears
- logs show one run performed the work and the other skipped or serialized safely

## How to interpret logs correctly
- A prior failure in logs does **not** mean the new fix failed.
- If the failure timestamp predates the patch/restart, treat it as historical.
- Prefer fresh evidence:
  - current service status healthy
  - no new lock errors after restart
  - explicit overlap stress test passed

## Recommended decision rule
If models/data fit locally and the issue is sync overlap, prefer:
- local storage/modelstore
- hybrid watcher + scheduled sync
- proper serialization with flock

Avoid switching to shared/network storage unless capacity or architecture truly requires it.

## Pitfalls
- Using separate lock files for watcher and timer: this does not prevent overlap.
- Locking only `git push` and not `git add/commit`: `index.lock` can still happen earlier.
- Returning non-zero on benign overlap: causes false alerts and noisy watchdog behavior.
- Verifying only through old journal output instead of a new forced overlap test.

## Verification checklist
- `bash -n` passes
- service restarts successfully
- service status is healthy
- no fresh lock errors appear in logs
- two simultaneous force-sync invocations both exit `0`
- repository behavior remains local-first and hybrid sync remains enabled
