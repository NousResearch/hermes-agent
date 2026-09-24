import assert from 'node:assert/strict'

import { describe, test } from 'vitest'

import { formatBootTransitionLog, trySendBootProgress } from './boot-transition-log'

// #96743 item 3: every main→renderer boot state transition should land in
// desktop.log so a stuck post-ready boot is diagnosable after the fact
// instead of the reported 9-minute silent gap.

describe('formatBootTransitionLog', () => {
  test('logs a transition line with from→to phases', () => {
    const line = formatBootTransitionLog(
      { phase: 'backend.remote', error: null, running: true },
      { phase: 'backend.ready', error: null, running: true }
    )

    assert.match(line, /^\[boot\] transition backend\.remote -> backend\.ready/)
  })

  test('includes the error text when the new state carries an error', () => {
    const line = formatBootTransitionLog(
      { phase: 'backend.remote', error: null, running: true },
      { phase: 'backend.error', error: 'update-in-progress', running: false }
    )

    assert.match(line, /backend\.remote -> backend\.error/)
    assert.match(line, /error="update-in-progress"/)
  })

  test('marks stalled delivery when the main window is gone', () => {
    const line = formatBootTransitionLog(
      { phase: 'backend.remote', error: null, running: true },
      { phase: 'backend.ready', error: null, running: true },
      { delivered: false }
    )

    assert.match(line, /not-delivered/)
  })

  test('returns null when the phase did not change (pure progress ticks stay quiet)', () => {
    const line = formatBootTransitionLog(
      { phase: 'backend.remote', error: null, running: true },
      { phase: 'backend.remote', error: null, running: true }
    )

    assert.equal(line, null)
  })

  test('still logs when only the error flag flips within the same phase', () => {
    const line = formatBootTransitionLog(
      { phase: 'backend.remote', error: null, running: true },
      { phase: 'backend.remote', error: 'boom', running: false }
    )

    assert.match(line, /backend\.remote/)
    assert.match(line, /error="boom"/)
  })
})

describe('trySendBootProgress', () => {
  test('returns true when the send succeeds', () => {
    let sent: [string, unknown] | null = null

    const ok = trySendBootProgress(
      {
        send(channel: string, payload: unknown) {
          sent = [channel, payload]
        }
      },
      'hermes:boot-progress',
      { phase: 'backend.ready' }
    )

    assert.equal(ok, true)
    assert.deepEqual(sent, ['hermes:boot-progress', { phase: 'backend.ready' }])
  })

  test('swallows a thrown send (window destroyed mid-call) and returns false', () => {
    // The window can be destroyed between the caller's isDestroyed() checks
    // and the actual send; the throw must be contained so the transition log
    // still records the transition as not-delivered (#96743 review nit).
    const ok = trySendBootProgress(
      {
        send() {
          throw new Error('Object has been destroyed')
        }
      },
      'hermes:boot-progress',
      { phase: 'backend.ready' }
    )

    assert.equal(ok, false)
  })
})
