import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  createPowerLifecycleGuard,
  registerPowerLifecycleListeners,
  stopPowerLifecycleGuardSafely
} from './power-lifecycle'

test('power lifecycle guard holds an app-suspension blocker only once', () => {
  const calls: Array<[string, string | number]> = []
  const active = new Set<number>()

  const blocker = {
    start(type: 'prevent-app-suspension') {
      calls.push(['start', type])
      active.add(42)

      return 42
    },
    isStarted(id: number) {
      return active.has(id)
    },
    stop(id: number) {
      calls.push(['stop', id])
      active.delete(id)
    }
  }

  const logs: string[] = []
  const guard = createPowerLifecycleGuard(blocker, line => logs.push(line))

  assert.equal(guard.start(), 42)
  assert.equal(guard.start(), 42, 'start must be idempotent')
  assert.equal(guard.active, true)
  assert.deepEqual(calls, [['start', 'prevent-app-suspension']])

  assert.equal(guard.stop(), true)
  assert.equal(guard.stop(), false, 'stop must be idempotent')
  assert.equal(guard.active, false)
  assert.deepEqual(calls, [
    ['start', 'prevent-app-suspension'],
    ['stop', 42]
  ])
  assert.deepEqual(logs, ['[power] app-suspension guard started id=42', '[power] app-suspension guard stopped id=42'])
})

test('power lifecycle guard retains its blocker id when the native status check fails', () => {
  let isStartedCalls = 0
  const stoppedIds: number[] = []

  const guard = createPowerLifecycleGuard({
    start() {
      return 42
    },
    isStarted(id) {
      assert.equal(id, 42)
      isStartedCalls += 1

      if (isStartedCalls === 1) {
        throw new Error('status failed')
      }

      return true
    },
    stop(id) {
      stoppedIds.push(id)
    }
  })

  assert.equal(guard.start(), 42)
  assert.throws(() => guard.stop(), /status failed/)
  assert.equal(guard.stop(), true, 'a failed native status check must be retryable')
  assert.equal(guard.stop(), false, 'a successful retry must clear the blocker id')
  assert.deepEqual(stoppedIds, [42])
})

test('power lifecycle guard retains its blocker id when the native stop fails', () => {
  let stopCalls = 0

  const guard = createPowerLifecycleGuard({
    start() {
      return 42
    },
    isStarted(id) {
      assert.equal(id, 42)

      return true
    },
    stop(id) {
      assert.equal(id, 42)
      stopCalls += 1

      if (stopCalls === 1) {
        throw new Error('stop failed')
      }
    }
  })

  assert.equal(guard.start(), 42)
  assert.throws(() => guard.stop(), /stop failed/)
  assert.equal(guard.stop(), true, 'a failed native stop must be retryable')
  assert.equal(guard.stop(), false, 'a successful retry must clear the blocker id')
  assert.equal(stopCalls, 2)
})

test('safe power lifecycle cleanup catches and logs guard failures', () => {
  const logs: string[] = []

  const guard = {
    get active() {
      return true
    },
    start() {
      return 42
    },
    stop() {
      throw new Error('native cleanup failed')
    }
  }

  assert.equal(
    stopPowerLifecycleGuardSafely(guard, line => logs.push(line)),
    false
  )
  assert.deepEqual(logs, ['[power] app-suspension guard cleanup failed: Error: native cleanup failed'])
})

test('power lifecycle listeners guard lock-screen work and release on unlock', () => {
  const listeners = new Map<string, () => void>()

  const powerMonitor = {
    on(event: string, listener: () => void) {
      listeners.set(event, listener)
    }
  }

  const calls: string[] = []

  const guard = {
    start() {
      calls.push('start')

      return 1
    },
    stop() {
      calls.push('stop')

      return true
    },
    get active() {
      return calls.at(-1) === 'start'
    }
  }

  registerPowerLifecycleListeners(powerMonitor, guard, () => calls.push('resume'))

  listeners.get('lock-screen')?.()
  assert.deepEqual(calls, ['start'])

  listeners.get('unlock-screen')?.()
  assert.deepEqual(calls, ['start', 'stop', 'resume'])

  listeners.get('resume')?.()
  assert.deepEqual(calls, ['start', 'stop', 'resume', 'resume'])
})

test('unlock still signals the renderer when blocker cleanup throws', () => {
  const listeners = new Map<string, () => void>()
  const logs: string[] = []
  let resumeSignals = 0

  registerPowerLifecycleListeners(
    {
      on(event, listener) {
        listeners.set(event, listener)
      }
    },
    {
      get active() {
        return true
      },
      start() {
        return 42
      },
      stop() {
        throw new Error('stop failed')
      }
    },
    () => {
      resumeSignals += 1
    },
    line => logs.push(line)
  )

  assert.doesNotThrow(() => listeners.get('unlock-screen')?.())
  assert.equal(resumeSignals, 1)
  assert.deepEqual(logs, ['[power] app-suspension guard cleanup failed: Error: stop failed'])
})
