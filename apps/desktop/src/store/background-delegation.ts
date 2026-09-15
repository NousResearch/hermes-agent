import { computed } from 'nanostores'

import { $activeSessionId, $busy } from './session'
import { $subagentsBySession, type SubagentProgress } from './subagents'

export interface BackgroundResume {
  /** Latest live activity from the primary child, for on-demand detail without
   *  replacing the notice's explicit background-resume label. */
  activity: string | null
  /** Running/queued background children for the active session. */
  count: number
}

const RUNNING = (s: SubagentProgress) => s.status === 'running' || s.status === 'queued'

/**
 * "Parked" background-delegation signal for the active session.
 *
 * A top-level `delegate_task` always runs in the background: the parent turn
 * ends (`$busy` -> false) while the subagent keeps running, and its result
 * re-enters the conversation as a fresh turn when it finishes. During that
 * window the parent is idle but its work is not finished. Surface an explicit
 * "will resume" status, with child activity available on demand, rather than
 * presenting the child's thinking as if the parent were still running.
 *
 * Null while `$busy`: an active turn already owns the main loader, and subagents
 * spawned inside a running turn (synchronous orchestrator children) are part of
 * that turn, not parked background work the user is waiting on.
 */
export const $backgroundResume = computed(
  [$subagentsBySession, $activeSessionId, $busy],
  (bySession, sid, busy): BackgroundResume | null => {
    if (busy || !sid) {
      return null
    }

    const running = (bySession[sid] ?? []).filter(RUNNING)

    if (running.length === 0) {
      return null
    }

    const activity = (running[0]!.stream.at(-1)?.text ?? '').trim() || null

    return { activity, count: running.length }
  }
)
