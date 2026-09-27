import { markCronRunReadOnly } from '@/store/read-only-transcript'
import type { SessionInfo } from '@/types/hermes'

/**
 * Cron run sessions are autonomous scheduled executions, never interactive
 * chat targets (#88443). Two states may be opened as a normal, WRITABLE
 * desktop chat: the run is still LIVE (`is_active`, the scheduler's agent is
 * executing it), or the run was properly CLOSED (`ended_at` stamped by
 * `_finalize_cron_session`).
 *
 * A run with `ended_at` NULL that is not live never got its `end_session` —
 * the watchdog killed the process, the run crashed, or the connection
 * dropped. It is a ZOMBIE: the row still looks open while nothing owns it, and
 * resuming it as a desktop chat routes the user's messages into a dead
 * `source='cron'` session, where the cron agent executes unrelated desktop
 * work. Such a run opens READ-ONLY instead.
 *
 * `is_active` is computed by the runs endpoint (`hermes_cli/web_routers/
 * cron.py`) as `ended_at IS NULL` plus a recent-activity window, so an older
 * backend that omits the flag fails SAFE here: the run reads as a zombie and
 * opens read-only rather than writable.
 */
export function isResumableCronRun(run: Pick<SessionInfo, 'ended_at' | 'is_active'>): boolean {
  return run.is_active === true || run.ended_at != null
}

/**
 * The single door every Cron surface (the sidebar run peek and the Cron page
 * history) uses to open a run's session, so the policy above is applied in one
 * place and cannot drift between callers.
 *
 * A never-closed run is latched read-only before the route flips: the
 * transcript still paints, but `submit` refuses to route a send into the dead
 * session (see `store/read-only-transcript`).
 */
export function openCronRun<Run extends Pick<SessionInfo, 'ended_at' | 'id' | 'is_active'>>(
  run: Run,
  open: (sessionId: string, session: Run) => void
): void {
  if (!isResumableCronRun(run)) {
    markCronRunReadOnly(run.id)
  }

  // The ROW rides along so the open can pin its owning (connection, profile)
  // (#82527).
  open(run.id, run)
}
