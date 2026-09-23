import assert from 'node:assert/strict'

import { test } from 'vitest'

import { createDesktopPrimaryBackendRuntime } from './desktop-primary-backend-runtime'

function state() {
  return {
    isPrimaryInstance: true,
    isQuittingForHandoff: false,
    primaryStartsInFlight: 0,
    primaryRecoverySuppressed: false,
    bootstrapFailure: null as Error | null,
    backendStartFailure: null as Error | null,
    remoteReauthFailure: null as Error | null,
    bootstrapRepairAttempt: 0
  }
}

test('a ready primary exit claims one recovery and its retry sees the settled start', async () => {
  const current = state()
  const events: string[] = []
  let rejectStart!: (reason: Error) => void

  const pendingStart = new Promise<never>((_resolve, reject) => {
    rejectStart = reject
  })

  const runtime = createDesktopPrimaryBackendRuntime({
    state: current,
    backendConnectionState: { getProcess: () => null, getPromise: () => null },
    backendShutdown: { hasStarted: () => false },
    primaryExitRecovery: {
      claim: snapshot => {
        events.push(`claim:${snapshot.hasPendingStart}`)

        return true
      },
      retryAfterFailedStart: snapshot => {
        events.push(`retry:${snapshot.hasPendingStart}`)

        return false
      },
      isCrashLooping: () => false
    },
    localBackendLifecycle: { start: () => pendingStart },
    rememberLog: (message: string) => events.push(message),
    sendBackendExit: () => events.push('exit-event'),
    firstLine: (message: string) => message
  } as any)

  assert.equal(runtime.scheduleUnexpectedPrimaryRecovery({ ready: false }), false)
  assert.deepEqual(events, [])

  assert.equal(runtime.scheduleUnexpectedPrimaryRecovery({ ready: true }), true)
  assert.equal(current.primaryStartsInFlight, 1)
  assert.deepEqual(events.slice(0, 3), [
    'claim:false',
    '[supervisor] backend exit left no primary owner and no start in flight; respawning',
    'exit-event'
  ])

  rejectStart(new Error('transient start failure'))
  await new Promise<void>(resolve => setImmediate(resolve))

  assert.equal(current.primaryStartsInFlight, 0)
  assert.ok(events.includes('retry:false'))
})

test('primary instance authority is live and a latched failure prevents a new connection', async () => {
  const current = state()
  const failure = new Error('repair required')
  const events: string[] = []
  current.isPrimaryInstance = false
  current.bootstrapFailure = failure

  const runtime = createDesktopPrimaryBackendRuntime({
    state: current,
    rememberLog: () => events.push('log'),
    reapOrphanedBackendsOnce: async () => events.push('reap'),
    localBackendLifecycle: {
      start: (run: () => Promise<unknown>) => run(),
      assertCanStart: () => events.push('admitted')
    },
    backendConnectionState: {
      getPromise: () => {
        throw new Error('connection access after latch')
      }
    }
  } as any)

  await assert.rejects(runtime.startHermes(), /already running in another window/)
  assert.deepEqual(events, ['log'])
  events.length = 0
  current.isPrimaryInstance = true

  await assert.rejects(runtime.startHermes(), error => error === failure)
  assert.deepEqual(events, ['reap', 'admitted'])
  assert.equal(current.primaryStartsInFlight, 0)
})
