/**
 * Retry on the update-failure screen must not re-enter the marker lock
 * while a live updater still owns it (issue #75422).
 *
 * The reporter withdrew the Gatekeeper and self-PID theories. What remains
 * is Retry citing a live foreign PID and calling acquire again. The failure
 * screen has to name that PID and either offer to stop it or say it is still
 * running and wait.
 */
import { describe, expect, it } from 'vitest'

import { updateFailureRetryAction } from '../apps/bootstrap-installer/src/lib/update-failure-retry'

describe('updateFailureRetryAction', () => {
  it('does not re-enter acquire while a live updater PID holds the marker', () => {
    const action = updateFailureRetryAction({ pid: 50214, ageSecs: 6, canStop: true })

    expect(action.kind).toBe('stop_or_wait')
    if (action.kind !== 'stop_or_wait') {
      return
    }
    expect(action.pid).toBe(50214)
    expect(action.ageSecs).toBe(6)
    expect(action.waitMessage).toContain('50214')
    expect(action.waitMessage.toLowerCase()).toContain('still running')
    expect(action.waitMessage.toLowerCase()).toContain('wait')
    expect(action.stopLabel?.toLowerCase()).toContain('stop')
    expect(action.stopLabel).toContain('50214')
  })

  it('allows acquire only when no live foreign updater owns the marker', () => {
    expect(updateFailureRetryAction(null)).toEqual({ kind: 'acquire' })
  })

  it('does not offer to stop a non-positive PID', () => {
    expect(updateFailureRetryAction({ pid: 0, ageSecs: 4 })).toEqual({ kind: 'acquire' })
    expect(updateFailureRetryAction({ pid: -1, ageSecs: 4 })).toEqual({ kind: 'acquire' })
  })

  it('names the elapsed age so the wait explanation is specific', () => {
    const action = updateFailureRetryAction({ pid: 50214, ageSecs: 6, canStop: true })

    expect(action.kind).toBe('stop_or_wait')
    if (action.kind !== 'stop_or_wait') {
      return
    }
    expect(action.waitMessage).toContain('6s')
  })

  it('does not offer to stop a live PID whose generation was not proved', () => {
    const action = updateFailureRetryAction({ pid: 50214, ageSecs: 6 })

    expect(action.kind).toBe('stop_or_wait')
    if (action.kind !== 'stop_or_wait') {
      return
    }
    expect(action.stopLabel).toBeUndefined()
    expect(action.waitMessage).toContain('50214')
    expect(action.waitMessage.toLowerCase()).toContain('wait')
    expect(action.waitMessage.toLowerCase()).not.toContain('stop that updater')
  })

  it('rolls marker age into minutes without dropping the PID', () => {
    const action = updateFailureRetryAction({ pid: 88, ageSecs: 125, canStop: true })

    expect(action.kind).toBe('stop_or_wait')
    if (action.kind !== 'stop_or_wait') {
      return
    }
    expect(action.waitMessage).toContain('PID 88')
    expect(action.waitMessage).toContain('2m 5s')
  })
})
