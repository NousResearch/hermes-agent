/*
 * Retry on the update-failure screen.
 *
 * A live foreign updater PID must not re-enter UpdateMarkerGuard::acquire.
 * Name that PID and either offer to stop it, or say it is still running and
 * wait. A missing owner is the only path that may start another update.
 * Non-positive PIDs are not stoppable (pid 0 signals the process group).
 *
 * The installer command already drops a marker that names this process.
 * Callers pass only a foreign live owner, or null.
 */

export interface LiveMarkerOwner {
  pid: number
  ageSecs: number
}

export type UpdateFailureRetryAction =
  | { kind: 'acquire' }
  | {
      kind: 'stop_or_wait'
      pid: number
      ageSecs: number
      waitMessage: string
      stopLabel: string
    }

function formatMarkerAge(ageSecs: number): string {
  const secs = Math.max(0, Math.floor(ageSecs))
  const mins = Math.floor(secs / 60)

  if (mins > 0) {return `${mins}m ${secs % 60}s`}

  return `${secs}s`
}

export function updateFailureRetryAction(
  owner: LiveMarkerOwner | null
): UpdateFailureRetryAction {
  if (!owner || !Number.isInteger(owner.pid) || owner.pid <= 0) {
    return { kind: 'acquire' }
  }

  const age = formatMarkerAge(owner.ageSecs)

  return {
    kind: 'stop_or_wait',
    pid: owner.pid,
    ageSecs: owner.ageSecs,
    waitMessage:
      `Another Hermes update is still running (PID ${owner.pid}, started ${age} ago). ` +
      'Wait for it to finish, or stop that updater and try again.',
    stopLabel: `Stop updater ${owner.pid}`
  }
}
