import { type ThreadMessage, useThreadRuntime } from '@assistant-ui/react'
import { useStore } from '@nanostores/react'
import { type FC, useEffect } from 'react'

import { $now, sessionScheduledRetry, setScheduledRetry } from '@/store/scheduled-retry'
import { $activeSessionId } from '@/store/session'
import { $sessionStates } from '@/store/session-states'

// ── Fires scheduled retries inside the runtime ─────────────────────────────
// The scheduled-retry store holds "re-run this failed turn at HH:mm"; this
// leaf performs the run. It must mount INSIDE an AssistantRuntimeProvider:
// the reload is the same thread.startRun({parentId}) the Retry button's
// ActionBarPrimitive.Reload issues, so the retried turn re-enters the
// conversation exactly like a manual retry.

/** Session-busy retry re-arm step: short enough that the user's schedule
 *  survives a long turn, long enough not to spin on a stuck session. */
const RETRY_BUSY_REARM_MS = 60_000

const ScheduledRetryRunner: FC = () => {
  const thread = useThreadRuntime()
  const now = useStore($now)
  const activeSessionId = useStore($activeSessionId)
  const retry = useStore(sessionScheduledRetry(activeSessionId))

  useEffect(() => {
    if (!retry || now < retry.at) {
      return
    }

    // The turn the schedule pointed at may be gone (history reload, branch
    // switch) — a fired schedule with no target simply retires.
    const messages = thread.getState().messages as (ThreadMessage & { parentId?: null | string })[]
    const failed = messages.find(message => message.id === retry.messageId)

    if (!failed) {
      setScheduledRetry(retry.sessionId, null)

      return
    }

    if ($sessionStates.get()[retry.sessionId]?.busy) {
      // The session is mid-turn: retrying into it would truncate live work.
      // Re-arm a minute later instead of dropping the user's schedule.
      setScheduledRetry(retry.sessionId, { ...retry, at: Date.now() + RETRY_BUSY_REARM_MS })

      return
    }

    thread.startRun({ parentId: failed.parentId ?? null, sourceId: failed.id })
    setScheduledRetry(retry.sessionId, null)
  }, [now, retry, thread])

  return null
}

/**
 * Timer half of "Retry in X hours": wakes at each pending schedule, fires the
 * reload via the runtime, and clears/defers the record. Rendered once per
 * chat boundary (primary chat + session tiles); each instance only acts on
 * the schedule of its OWN session, so parallel tiles never cross-fire.
 */
export const ScheduledRetryScheduler: FC = () => {
  const now = useStore($now)
  const activeSessionId = useStore($activeSessionId)
  const retry = useStore(sessionScheduledRetry(activeSessionId))

  // Arm the clock for the earliest pending moment — a single timeout per
  // boundary, re-armed when the schedule or the session changes.
  useEffect(() => {
    if (!retry) {
      return
    }

    const delay = Math.max(0, retry.at - now)
    const id = window.setTimeout(() => $now.set(Date.now()), delay)

    return () => window.clearTimeout(id)
  }, [now, retry])

  return <ScheduledRetryRunner />
}
