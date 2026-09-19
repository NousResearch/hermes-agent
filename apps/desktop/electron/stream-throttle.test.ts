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
    listeners,
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

// ---------------------------------------------------------------------------
// Fullscreen keep-painting (#94865 — Hyprland/Wayland white screen)
// ---------------------------------------------------------------------------

function makeFullscreenableWindow() {
  const win = makeWindow()
  let fullscreen = false
  const ext = win as ReturnType<typeof makeWindow> & {
    isFullScreen: () => boolean
    goFullscreen(on: boolean): void
  }
  ext.isFullScreen = () => fullscreen
  ext.goFullscreen = (on: boolean) => {
    fullscreen = on
    win.listeners.get(on ? 'enter-full-screen' : 'leave-full-screen')?.()
  }

  return ext
}

test('entering fullscreen unthrottles even when idle; leaving re-arms throttling', () => {
  const timers = makeTimers()
  const throttle = createStreamThrottle(timers)
  const win = makeFullscreenableWindow()
  throttle.register(win)

  assert.deepEqual(win.calls, [true]) // idle default

  // Fullscreen with zero turns in flight: must keep painting.
  win.goFullscreen(true)
  assert.deepEqual(win.calls, [true, false])
  assert.equal(throttle.isUnthrottled(), true)

  // A settle report during fullscreen must NOT re-throttle it.
  throttle.update(true)
  throttle.update(false)
  assert.equal(timers.pendingCount, 0)
  assert.deepEqual(win.calls, [true, false])

  // Leaving fullscreen re-arms the normal settle path.
  win.goFullscreen(false)
  throttle.update(false)
  assert.equal(timers.pendingCount, 1)
  timers.fire()
  assert.deepEqual(win.calls, [true, false, true])
  assert.equal(throttle.isUnthrottled(), false)
})

test('fullscreen keeps the fleet live on settle; everything re-throttles after leave', () => {
  const timers = makeTimers()
  const throttle = createStreamThrottle(timers)
  const fsWin = makeFullscreenableWindow()
  const normalWin = makeWindow()

  throttle.register(fsWin)
  throttle.register(normalWin)

  // Fullscreen lifts the whole fleet (global dial, same as streaming).
  fsWin.goFullscreen(true)
  assert.deepEqual(fsWin.calls.slice(-1), [false])
  assert.deepEqual(normalWin.calls.slice(-1), [false])

  // A settle report during fullscreen must NOT arm re-throttling.
  throttle.update(false)
  assert.equal(timers.pendingCount, 0)
  assert.deepEqual(normalWin.calls, [true, false])

  // Leaving fullscreen while idle arms the trailing re-throttle for everyone.
  fsWin.goFullscreen(false)
  assert.equal(timers.pendingCount, 1)
  timers.fire()
  assert.deepEqual(fsWin.calls, [true, false, true])
  assert.deepEqual(normalWin.calls, [true, false, true])
  assert.equal(throttle.isUnthrottled(), false)
})

test('closing the only fullscreen window re-arms throttling for the remaining fleet', () => {
  const timers = makeTimers()
  const throttle = createStreamThrottle(timers)
  const fsWin = makeFullscreenableWindow()
  const normalWin = makeWindow()

  throttle.register(fsWin)
  throttle.register(normalWin)
  fsWin.goFullscreen(true)
  fsWin.close()

  assert.equal(timers.pendingCount, 1)
  assert.equal(throttle.isUnthrottled(), true)

  timers.fire()
  assert.equal(throttle.isUnthrottled(), false)
  assert.deepEqual(normalWin.calls, [true, false, true])
})

test('leaving one of two fullscreen windows does not re-arm throttling', () => {
  const timers = makeTimers()
  const throttle = createStreamThrottle(timers)
  const first = makeFullscreenableWindow()
  const second = makeFullscreenableWindow()
  const normalWin = makeWindow()

  throttle.register(first)
  throttle.register(second)
  throttle.register(normalWin)
  first.goFullscreen(true)
  second.goFullscreen(true)
  first.goFullscreen(false)

  assert.equal(timers.pendingCount, 0)
  timers.fire()
  assert.equal(throttle.isUnthrottled(), true)
  assert.deepEqual(second.calls.slice(-1), [false])
  assert.deepEqual(normalWin.calls.slice(-1), [false])

  second.goFullscreen(false)
  assert.equal(timers.pendingCount, 1)
})

test('leaving fullscreen while a turn is still in flight keeps both windows live', () => {
  const timers = makeTimers()
  const throttle = createStreamThrottle(timers)
  const win = makeFullscreenableWindow()
  throttle.register(win)

  throttle.update(true) // streaming: everything unthrottled
  assert.deepEqual(win.calls, [true, false])

  win.goFullscreen(false) // user exits fullscreen mid-stream — nothing changes
  assert.deepEqual(win.calls, [true, false])
  assert.equal(throttle.isUnthrottled(), true)

  throttle.update(false)
  assert.equal(timers.pendingCount, 1, 'normal trailing re-throttle resumes')
})
