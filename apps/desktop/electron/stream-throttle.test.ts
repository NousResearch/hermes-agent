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

  throttle.update(idle, true)
  const late = makeWindow()
  throttle.register(late)

  // Another renderer's stream does not wake a newly created sibling window.
  assert.deepEqual(late.calls, [true])
  assert.equal(throttle.isUnthrottled(idle), true)
  assert.equal(throttle.isUnthrottled(late), false)
})

test('a turn in flight unthrottles its own window; settling re-throttles after the trailing delay', () => {
  const timers = makeTimers()
  const throttle = createStreamThrottle(timers)
  const win = makeWindow()
  throttle.register(win)

  throttle.update(win, true)
  assert.deepEqual(win.calls, [true, false])
  assert.equal(throttle.isUnthrottled(), true)

  // Turn ends: not re-throttled synchronously — the tail flush needs full
  // cadence — only after the trailing timer fires.
  throttle.update(win, false)
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

  throttle.update(win, true)
  throttle.update(win, false)
  assert.equal(timers.pendingCount, 1)

  // Busy again before the delay elapses: stay unthrottled, timer cancelled.
  throttle.update(win, true)
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

  throttle.update(win, true)
  throttle.update(win, true)
  throttle.update(win, true)
  assert.deepEqual(win.calls, [true, false])

  throttle.update(win, false)
  throttle.update(win, false)
  assert.equal(timers.pendingCount, 1)
})

test('concurrent windows throttle independently', () => {
  const timers = makeTimers()
  const throttle = createStreamThrottle(timers)
  const first = makeWindow()
  const second = makeWindow()
  throttle.register(first)
  throttle.register(second)

  throttle.update(first, true)
  assert.deepEqual(first.calls, [true, false])
  assert.deepEqual(second.calls, [true])

  throttle.update(second, true)
  throttle.update(first, false)
  assert.equal(timers.pendingCount, 1)
  assert.equal(throttle.isUnthrottled(first), true)
  assert.equal(throttle.isUnthrottled(second), true)

  timers.fire()
  assert.deepEqual(first.calls, [true, false, true])
  assert.deepEqual(second.calls, [true, false])
  assert.equal(throttle.isUnthrottled(first), false)
  assert.equal(throttle.isUnthrottled(second), true)
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

  throttle.update(closedWin, true)
  // Only the registration-time call landed; nothing after close.
  assert.deepEqual(closedWin.calls, [true])
})
