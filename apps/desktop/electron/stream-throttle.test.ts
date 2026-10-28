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

test('fullscreen events are inert when the Wayland workaround is disabled', () => {
  const timers = makeTimers()
  const throttle = createStreamThrottle(timers)
  const win = makeFullscreenableWindow()
  throttle.register(win)

  win.goFullscreen(true)
  assert.deepEqual(win.calls, [true])
  assert.equal(throttle.isUnthrottled(), false)
  assert.equal(timers.pendingCount, 0)
})

test('Wayland fullscreen unthrottles while idle and leaving re-arms the trailing throttle', () => {
  const timers = makeTimers()
  const throttle = createStreamThrottle(timers, 5_000, { keepFullscreenPainting: true })
  const win = makeFullscreenableWindow()
  throttle.register(win)

  win.goFullscreen(true)
  assert.deepEqual(win.calls, [true, false])
  assert.equal(throttle.isUnthrottled(), true)

  // A settle report during fullscreen must not re-throttle the visible surface.
  throttle.update(false)
  assert.equal(timers.pendingCount, 0)

  win.goFullscreen(false)
  assert.equal(timers.pendingCount, 1)
  timers.fire()
  assert.deepEqual(win.calls, [true, false, true])
  assert.equal(throttle.isUnthrottled(), false)
})

test('closing the only fullscreen Wayland window re-arms throttling for the remaining fleet', () => {
  const timers = makeTimers()
  const throttle = createStreamThrottle(timers, 5_000, { keepFullscreenPainting: true })
  const fullscreen = makeFullscreenableWindow()
  const normal = makeWindow()
  throttle.register(fullscreen)
  throttle.register(normal)

  fullscreen.goFullscreen(true)
  fullscreen.close()

  assert.equal(timers.pendingCount, 1)
  assert.equal(throttle.isUnthrottled(), true)

  timers.fire()
  assert.equal(throttle.isUnthrottled(), false)
  assert.deepEqual(normal.calls, [true, false, true])
})

test('one of two fullscreen Wayland windows leaving does not re-arm throttling', () => {
  const timers = makeTimers()
  const throttle = createStreamThrottle(timers, 5_000, { keepFullscreenPainting: true })
  const first = makeFullscreenableWindow()
  const second = makeFullscreenableWindow()
  throttle.register(first)
  throttle.register(second)

  first.goFullscreen(true)
  second.goFullscreen(true)
  first.goFullscreen(false)

  assert.equal(timers.pendingCount, 0)
  assert.equal(throttle.isUnthrottled(), true)

  second.goFullscreen(false)
  assert.equal(timers.pendingCount, 1)
})

test('leaving Wayland fullscreen during active work keeps the fleet unthrottled', () => {
  const timers = makeTimers()
  const throttle = createStreamThrottle(timers, 5_000, { keepFullscreenPainting: true })
  const win = makeFullscreenableWindow()
  throttle.register(win)

  throttle.update(true)
  win.goFullscreen(true)
  win.goFullscreen(false)

  assert.equal(timers.pendingCount, 0)
  assert.equal(throttle.isUnthrottled(), true)

  throttle.update(false)
  assert.equal(timers.pendingCount, 1)
})

test('overlapping native work wakes only its owner and releases only after the last hold', () => {
  const throttle = createStreamThrottle(makeTimers())
  const owner = makeWindow()
  const peer = makeWindow()
  throttle.register(owner)
  throttle.register(peer)
  const first = throttle.holdWindow(owner)
  const second = throttle.holdWindow(owner)
  assert.equal(owner.calls.at(-1), false)
  assert.equal(peer.calls.at(-1), true)
  first()
  first()
  assert.equal(owner.calls.at(-1), false)
  second()
  assert.equal(owner.calls.at(-1), true)
  assert.equal(peer.calls.at(-1), true)
})

test('native work and stream transitions cannot restore stale scheduling policy', () => {
  const timers = makeTimers()
  const throttle = createStreamThrottle(timers)
  const owner = makeWindow()
  const peer = makeWindow()
  throttle.register(owner)
  throttle.register(peer)
  const release = throttle.holdWindow(owner)
  throttle.update(true)
  throttle.update(false)
  timers.fire()
  assert.equal(owner.calls.at(-1), false)
  assert.equal(peer.calls.at(-1), true)
  throttle.update(true)
  release()
  assert.equal(owner.calls.at(-1), false)
  throttle.update(false)
  timers.fire()
  assert.equal(owner.calls.at(-1), true)
  assert.equal(peer.calls.at(-1), true)
})

test('closing a held window retires its holds and rejects further work', () => {
  const throttle = createStreamThrottle(makeTimers())
  const owner = makeWindow()
  const peer = makeWindow()
  throttle.register(owner)
  throttle.register(peer)
  const release = throttle.holdWindow(owner)
  owner.close()
  const callsAtClose = [...owner.calls]
  release()
  assert.deepEqual(owner.calls, callsAtClose)
  assert.equal(peer.calls.at(-1), true)
  assert.throws(() => throttle.holdWindow(owner), /live registered/)
  assert.throws(() => throttle.holdWindow(makeWindow()), /live registered/)
})
