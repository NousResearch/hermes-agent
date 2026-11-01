# Hermes Dispatcher/Worker Isolation — Design & Safety Spec (P0-A)

Status: design spec for review — implementation landed locally (kanban P0-A) and is *not* part of this PR; sections marked "design-pending" describe behavior not yet surfaced in the in-tree implementation
Author: Nei (Knowledge & Continuity) — drafted for the Hyraxknot Division, deliberately project-agnostic
Scope: any deployment of the Hermes kanban dispatcher (gateway-hosted or standalone daemon)

## 1. Problem

The kanban dispatcher historically spawned workers as direct subprocess
children of whatever process hosted it. When the dispatcher ran inside a
gateway systemd unit, a gateway restart (`KillMode=mixed` SIGKILLs the
whole unit cgroup subtree) killed every in-flight worker mid-task — losing
the run, the claim, and any partial work. Secrets also traveled on worker
command lines, visible to any local user via `ps`.

This spec fixes the class of problem, not one instance: gateways stop
owning work, a dedicated dispatcher owns leases, and workers run in
independent transient units.

## 2. Roles and responsibilities

| Component | Owns | Never does |
|---|---|---|
| Gateway (or any front-end process) | enqueueing work, waking the dispatcher | acquires/renews/releases claims; spawns workers |
| Dedicated dispatcher | lease acquisition, renewal, release; spawn; reclaim; promote; drain | runs inside a gateway unit |
| Worker | its own task, inside its own transient unit | shares a cgroup with the gateway |

A gateway in wake-only mode touches the canonical wake file
(`<kanban-home>/kanban/.wake`); the dedicated dispatcher polls that file's
mtime and ticks immediately. Presence of the wake flag is the only thing
the gateway controls — an arbitrary wake path would break the contract, so
the flag is deliberately presence-only.

## 3. Lease lifecycle (dispatcher-owned)

Lease fields live on the task row: `claim_lock`, `claim_expires`,
`worker_pid`.

- **Acquisition** — only the dispatcher writes claims. The spawn query
  (`status = 'ready' AND claim_lock IS NULL`) plus DB-level authorization
  enforce the single-writer rule: no other component can claim a task.
- **TTL** — default 15 minutes (`HERMES_KANBAN_CLAIM_TTL_SECONDS`
  overrides). A claim is a renewable lease, not a permanent assignment.
- **Renewal** — the worker's heartbeat extends `claim_expires`. The
  dispatcher reclaims tasks whose claim expired or whose worker PID is
  dead — death is source-probed (process table / unit MainPID), never
  trusted from prose.
- **Release** — completion/block clears `claim_lock`, `claim_expires`,
  `worker_pid` so the row returns to the ready pool or goes terminal.
- **Stale timeout** — `kanban.dispatch_stale_timeout_seconds` (default
  4h) bounds how long a running task may go without a heartbeat before
  the dispatcher re-queues it.

## 4. Worker isolation: independent transient units

Transport is configurable (`kanban.worker_isolation`: `auto` |
`systemd` | `subprocess`, env override `HERMES_KANBAN_WORKER_ISOLATION`):

- **systemd (preferred)** — the worker launches via `systemd-run --user`
  as its own transient service (`hermes-worker-<task>-<run>`), a sibling
  of any gateway unit under `user@.service`. It gets its own cgroup;
  stopping/restarting a gateway can never SIGKILL it. `--collect`
  removes the unit on exit. stdout/stderr append to the same worker log
  file the subprocess transport would write. The unit's MainPID is
  resolved with a short wait loop so crash detection behaves identically
  to the subprocess path.
- **subprocess (fallback)** — `Popen` with `start_new_session=True`:
  detached from the controlling terminal but **still in the dispatcher's
  cgroup** — a gateway restart can kill it. This is the documented
  weaker transport for hosts without a usable user systemd manager
  (non-Linux, minimal containers). `auto` prefers systemd when the user
  manager socket is alive and falls back to subprocess.

## 5. No secrets on command lines

Worker argv is visible to every local user (`ps`) and in
`systemctl show -p ExecStart` — so the spawn path fails closed before any
process launch:

- **Argv guard** — every secret-looking environment value (name matches
  `API_KEY|TOKEN|SECRET|PASSWORD|CREDENTIALS|PRIVATE_KEY|AUTH_TOKEN`, or
  high-entropy ≥32 chars across ≥3 character classes) is scanned against
  the final argv; any hit aborts the spawn naming the offending env var.
  Matching is exact-token/`NAME=value`, deliberately not substring —
  argv tokens routinely embed task ids and hashes. Load-bearing non-
  secret names (task/run/board/workspace ids, profile, home) are exempt.
- **Delivery** — the worker environment is written to a 0600
  `EnvironmentFile` (`KEY=VALUE`, shlex-quoted so spaces/specials
  survive; `shlex.quote` handles systemd 255 encoding) in a per-board
  run/env dir, and passed via `--property=EnvironmentFile=`. Secrets
  reach the worker through the file, never the command line. Stale env
  files are cleaned by `kanban gc`.
  *POSIX scope:* the 0600 permission promise applies to POSIX hosts;
  non-POSIX hosts (e.g. Windows) should not rely on it. The argv guard
  is a heuristic — an env var whose name does not match the regex and
  whose value is short-but-sensitive (e.g. a 20-char token) is not
  caught; the fail-closed abort remains the safety backstop, and
  operators should treat any env var whose value may be sensitive as
  load-bearing regardless of the heuristic.

## 6. Restart, adoption, drain semantics

- **Restart** — restarting a gateway restarts only the gateway unit.
  In-flight workers (own cgroup) keep running and keep their claims. The
  new dispatcher generation sees them and **adopts** them: surviving
  workers are left untouched — crash detection runs first
  (`detect_crashed_workers`, `release_stale_claims`), so dead workers are
  reclaimed, not adopted. Broken-claim orphans are requeued by
  `reconcile_orphaned_running`, which defers requeue while the recorded
  PID is alive on this host (never spawns a duplicate beside a live
  worker).
- **Adoption** — adopted workers are left untouched; their claims stay
  valid, their units keep running. *Design-pending:* the spec proposes
  `DispatchResult.adopted` (every running task with a live, host-local
  PID and intact claim) as telemetry so operators can see a generation
  changed under them; that field is not yet present in the in-tree
  `DispatchResult` (current observable signals: `reconciled_orphans`,
  `reclaimed`, `crashed`).
- **Drain** — SIGINT/SIGTERM (or `drain_on_start`) puts the dispatcher
  into draining mode: `spawn_allowed=False` (no new claims) while
  reclaim/promote continue, so already-queued work still progresses.
  The daemon exits when the running count reaches zero — or after
  `drain_timeout` seconds when workers run long — and any still-running
  workers are adopted by the next generation.
- **Wake** — gateways bump the wake file's mtime; the dispatcher polls
  it and ticks immediately instead of waiting for the next scheduled
  tick.

## 7. Backup / rollback procedure

Apply before *any* change to the dispatcher/worker machinery:

1. **Capture known-good** — sync live state to the ops repo (or tag the
   git checkout) BEFORE touching anything. A rollback is only as good as
   the last committed known-good state.
2. **Backup fidelity gate** — save pristine base-version copies of every
   file to be modified, and verify each backup is byte-identical
   (sha256) to the base blob. Refuse to roll back with a tainted backup.
   (Prior failure mode: backups captured *after* the change, byte-
   identical to the new state — rollback silently no-oped.)
3. **Rollback script** — parameterizable target dir (defaults to the
   live checkout; pass a throwaway worktree to test without touching
   live). Restores base files, removes new files, then sanity-diffs the
   working tree against the BASE revision — not HEAD, which is the
   change itself. Exit code 2 + warning if the tree doesn't match.
4. **Git alternative** — `git revert <base>..HEAD` when the change is
   linear and unpushed.
5. **Activation** — code changes only take effect for NEW processes;
   the running gateway keeps its already-imported modules. A gateway
   restart (from a shell OUTSIDE the gateway process) is required to
   load the change or the rollback.

## 8. Survival expectations & gate before any live gateway restart

**Survival contract:** a worker must survive its host gateway restarting
— it lives in a sibling cgroup under the user manager, not in the
gateway unit's subtree.

Pre-restart gate (all must hold; a live restart is only permitted after
a throwaway-worktree rehearsal):

- [ ] Focused isolation suite green, with env delivery asserted at the
      worker-process level (`/proc/<pid>/environ`), not reconstructed
      from unit metadata.
- [ ] No HEAD-only regressions: full test family compared against the
      base revision; failure set identical, zero new failures.
- [ ] Rollback proven: `rollback.sh` run against a throwaway worktree at
      the new HEAD prints the base-match verdict; `git diff` over all
      touched paths is empty; new files absent.
- [ ] Backup integrity verified (step 7.2) for every touched file.
- [ ] Restart issued from a detached context (external shell or a
      transient `systemd-run` unit that is itself outside every gateway
      cgroup) — never from inside the gateway's own process tree, which
      dies with the unit.
- [ ] Post-restart: adopted list reflects any in-flight workers; claims
      intact; wake tick observed; no zombie units (`--collect` working).

**Test-worker (canary) rehearsal:** before a production restart, spawn a
canary worker (a task that sleeps/logs long enough to straddle the
restart), restart the gateway unit, then assert: the canary survives,
its claim is adopted (not reclaimed), and it completes normally. Only
then may real traffic ride the same path.

## 9. Review checklist (this PR)

- [ ] Gateways only enqueue/wake — no claim writes, no spawns (§2, §3).
- [ ] Lease acquisition/renewal/release owned by the dispatcher alone,
      single-writer enforced at the DB layer (§3).
- [ ] Workers spawn in independent transient units/cgroups with
      `--collect`; subprocess fallback documented as weaker (§4).
- [ ] No secrets on command lines: fail-closed argv guard + 0600
      EnvironmentFile delivery; exemption list justified (§5).
- [ ] Restart/adoption/drain semantics explicit and source-probed (§6).
- [ ] Spec is project-agnostic — no AAEmu or deployment-specific
      behavior (§1–§8).
- [ ] Backup/rollback procedure is rehearsed on a throwaway tree, not
      just documented (§7).
- [ ] Pre-restart gate conditions are checkable, not aspirational (§8).
      *Design-pending:* no automated enforcement ships with this PR —
      see the tracking item for wiring a check into `kanban gc`/tests.
- [ ] Terminology matches the implementation (`kanban_isolation.py`,
      `DispatchResult` fields as they exist today, claim fields, wake
      file path).
