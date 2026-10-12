import { hasCronRunVerdict, isCronRunReadOnly, recordCronRunVerdict } from '@/store/read-only-transcript'
import type { SessionInfo } from '@/types/hermes'

type CronRunLiveness = Pick<SessionInfo, 'ended_at' | 'last_active' | 'scheduler_owned'> &
  Partial<Pick<SessionInfo, 'is_active'>>

type CronRunRow = CronRunLiveness & Pick<SessionInfo, 'id'>

// Mirrors the backend's activity window (`hermes_cli/web_routers/cron.py`,
// `sessions.py::_ACTIVE_WINDOW_S`). Used ONLY against an older backend that
// predates `scheduler_owned`.
const ACTIVITY_WINDOW_MS = 300_000

// `cron/scheduler.py::run_job` names every agent run `cron_{job_id}_{YYYYmmdd_HHMMSS}`.
const CRON_RUN_SESSION_ID = /^cron_.+_\d{8}_\d{6}$/

/**
 * Cron run sessions are autonomous scheduled executions, never interactive
 * chat targets (#88443). Two states may be written to as a normal desktop
 * chat: the run is still LIVE, or it was properly CLOSED (`ended_at` stamped
 * by `_finalize_cron_session`).
 *
 * LIVE means the scheduler OWNS the run (`scheduler_owned`: its in-flight
 * execution is held by a live process). It is NOT `is_active`, which is a
 * 300s activity window: a live run inside a long `terminal` / `execute_code`
 * / `delegate_task` call writes no heartbeat and reads inactive exactly like a
 * zombie. A run with `ended_at` NULL that the scheduler does not own never got
 * its `end_session` — a ZOMBIE — and is view-only.
 *
 * An older backend without `scheduler_owned` falls back to the activity window
 * (the backend's `is_active`, or the same formula over `last_active` for rows
 * that carry no flag). That is the best such a backend can say, and the verdict
 * is re-evaluated on every refresh and send, so it never latches.
 */
export function isResumableCronRun(run: CronRunLiveness, nowMs = Date.now()): boolean {
  if (run.ended_at != null) {
    return true
  }

  if (typeof run.scheduler_owned === 'boolean') {
    return run.scheduler_owned
  }

  if (typeof run.is_active === 'boolean') {
    return run.is_active
  }

  return typeof run.last_active === 'number' && nowMs - run.last_active * 1000 < ACTIVITY_WINDOW_MS
}

export function isCronRunSessionId(storedSessionId: null | string | undefined): boolean {
  return Boolean(storedSessionId && CRON_RUN_SESSION_ID.test(storedSessionId.trim()))
}

function recordRunVerdict(run: CronRunRow): void {
  recordCronRunVerdict(run.id, !isResumableCronRun(run))
}

/**
 * The single door every Cron surface (the sidebar run peek and the Cron page
 * history) uses to open a run's session, so the policy above is applied in one
 * place and cannot drift between callers. The verdict is recorded before the
 * route flips, so the first send already sees it.
 */
export function openCronRun<Run extends CronRunRow>(run: Run, open: (sessionId: string, session: Run) => void): void {
  recordRunVerdict(run)

  // The ROW rides along so the open can pin its owning (connection, profile)
  // (#82527).
  open(run.id, run)
}

/**
 * Fold a fresh page of run rows (a Cron surface poll) into the verdicts of runs
 * the desktop already evaluated, so a zombie-looking run that later ticks or
 * closes becomes writable again — and a live one that dies becomes view-only —
 * without waiting for a send. Rows never opened are left alone.
 */
export function reconcileCronRunVerdicts(runs: readonly CronRunRow[]): void {
  for (const run of runs) {
    if (hasCronRunVerdict(run.id)) {
      recordRunVerdict(run)
    }
  }
}

/**
 * Re-evaluate a cron run's write gate against its AUTHORITATIVE row right
 * before a send, then report whether the send must be refused.
 *
 * Covers what the Cron surfaces cannot: a verdict that went stale while no
 * surface was polling, and a restored route/tab after an app restart, where no
 * surface ever evaluated the run (the in-memory verdicts start empty). Such a
 * session is recognised by its cron run id. Any other session returns
 * immediately without a request.
 *
 * An unreadable row keeps a known verdict; an unknown run fails CLOSED — a
 * send into a possibly-dead cron session is exactly the misroute to prevent.
 */
export async function refreshCronRunWriteGate(
  storedSessionId: null | string | undefined,
  fetchRow: (storedSessionId: string) => Promise<CronRunLiveness & Partial<Pick<SessionInfo, 'source'>>>
): Promise<boolean> {
  const id = storedSessionId?.trim()

  if (!id || !(isCronRunSessionId(id) || hasCronRunVerdict(id))) {
    return false
  }

  try {
    const row = await fetchRow(id)

    if (row.source && row.source !== 'cron') {
      recordCronRunVerdict(id, false)
    } else {
      recordRunVerdict({ ...row, id })
    }
  } catch {
    if (!hasCronRunVerdict(id)) {
      recordCronRunVerdict(id, true)
    }
  }

  return isCronRunReadOnly(id)
}
