import assert from 'node:assert/strict'

import { test } from 'vitest'

import { createStreamThrottle, type ThrottleWindowLike } from './stream-throttle'

function makeTimers() {
  const pending = new Map<number, () => void>()
  let nextId = 1

  return {
    clearTimeout: (handle: unknown) => {
      pending.delete(handle as number)
    },
    fire() {
      const jobs = [...pending.values()]
      pending.clear()

      for (const job of jobs) {
        job()
      }
    },
    get pendingCount() {
      return pending.size
    },
    setTimeout: (fn: () => void, _ms: number) => {
      const id = nextId++
      pending.set(id, fn)

      return id
    }
  }
}

function makeWindow() {
  const calls: boolean[] = []
  const listeners = new Map<string, () => void>()
  let destroyed = false

  const win = {
    calls,
    close() {
      destroyed = true
      listeners.get('closed')?.()
    },
    isDestroyed: () => destroyed,
    on(event: string, fn: () => void) {
      listeners.set(event, fn)
    },
    webContents: {
      isDestroyed: () => destroyed,
      setBackgroundThrottling(allowed: boolean) {
        calls.push(allowed)
      }
    }
  }

  return win
}

test('registering a window applies the current throttle state immediately', () => {
  const timers = makeTimers()
  const throttle = createStreamThrottle(timers)
  const idle = makeWindow()
  throttle.register(idle)

  // Idle default: throttling allowed.
  assert.deepEqual(idle.calls, [true])

  throttle.update(true)
  const late = makeWindow()
  throttle.register(late)

  // A window created mid-stream starts unthrottled.
  assert.deepEqual(late.calls, [false])
})

test('a turn in flight unthrottles every chat window; settling re-throttles after the trailing delay', () => {
  const timers = makeTimers()
  const throttle = createStreamThrottle(timers)
  const win = makeWindow()
  throttle.register(win)

  throttle.update(true)
  assert.deepEqual(win.calls, [true, false])
  assert.equal(throttle.isUnthrottled(), true)

  // Turn ends: not re-throttled synchronously — the tail flush needs full
  // cadence — only after the trailing timer fires.
  throttle.update(false)
  assert.deepEqual(win.calls, [true, false])
  assert.equal(throttle.isUnthrottled(), true)

  timers.fire()
  assert.deepEqual(win.calls, [true, false, true])
  assert.equal(throttle.isUnthrottled(), false)
})

test('a new turn during the trailing window cancels the pending re-throttle', () => {
  const timers = makeTimers()
  const throttle = createStreamThrottle(timers)
  const win = makeWindow()
  throttle.register(win)

  throttle.update(true)
  throttle.update(false)
  assert.equal(timers.pendingCount, 1)

  // Busy again before the delay elapses: stay unthrottled, timer cancelled.
  throttle.update(true)
  assert.equal(timers.pendingCount, 0)
  assert.equal(throttle.isUnthrottled(), true)

  // The cancelled timer firing late must be a no-op.
  timers.fire()
  assert.equal(throttle.isUnthrottled(), true)
})

test('repeated busy reports do not re-apply or stack timers', () => {
  const timers = makeTimers()
  const throttle = createStreamThrottle(timers)
  const win = makeWindow()
  throttle.register(win)

  throttle.update(true)
  throttle.update(true)
  throttle.update(true)
  assert.deepEqual(win.calls, [true, false])

  throttle.update(false)
  throttle.update(false)
  assert.equal(timers.pendingCount, 1)
})

test('closed and destroyed windows drop out without throwing', () => {
  const timers = makeTimers()
  const throttle = createStreamThrottle(timers)
  const closedWin = makeWindow()
  throttle.register(closedWin)
  closedWin.close()

  const gone: ThrottleWindowLike & { on?: never } = {
    isDestroyed: () => true,
    webContents: null
  }

  throttle.register(gone)

  throttle.update(true)
  // Only the registration-time call landed; nothing after close.
  assert.deepEqual(closedWin.calls, [true])
})

test('a boot hold keeps an idle renderer runnable until release', () => {
  const timers = makeTimers()
  const throttle = createStreamThrottle(timers)
  const win = makeWindow()
  throttle.register(win)

  const release = throttle.hold()
  assert.deepEqual(win.calls, [true, false])
  assert.equal(throttle.isUnthrottled(), true)

  release()
  release()
  assert.equal(timers.pendingCount, 1)

  timers.fire()
  assert.deepEqual(win.calls, [true, false, true])
})

test('releasing a boot hold does not throttle a turn that started meanwhile', () => {
  const timers = makeTimers()
  const throttle = createStreamThrottle(timers)
  const win = makeWindow()
  throttle.register(win)

  const release = throttle.hold()
  throttle.update(true)
  release()

  assert.equal(timers.pendingCount, 0)
  assert.equal(throttle.isUnthrottled(), true)
  assert.deepEqual(win.calls, [true, false])
})
