import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  describeRendererLifecycleEvent,
  installWindowRendererLifecycle,
  pruneReloadTimes,
  pushReloadTime,
  shouldReloadAfterRendererGone
} from './window-renderer-lifecycle'

// Fake Electron surface — real listener wiring, no Electron import. Mirrors
// how the rest of electron/*.test.ts exercises Electron-free modules.
function makeFakeWindow(overrides: { destroyed?: boolean } = {}) {
  const listeners = new Map<string, ((...args: any[]) => void)[]>()
  const reloadCalls: number[] = []
  let destroyed = overrides.destroyed ?? false

  const win = {
    isDestroyed: () => destroyed,
    setDestroyed: (value: boolean) => {
      destroyed = value
    },
    webContents: {
      on: (event: string, listener: (...args: any[]) => void) => {
        const list = listeners.get(event) ?? []

        list.push(listener)
        listeners.set(event, list)
      },
      removeListener: (event: string, listener: (...args: any[]) => void) => {
        const list = listeners.get(event) ?? []

        listeners.set(
          event,
          list.filter(candidate => candidate !== listener)
        )
      },
      emit: (event: string, ...args: unknown[]) => {
        for (const listener of listeners.get(event) ?? []) {
          listener(...args)
        }
      },
      reload: () => {
        reloadCalls.push(1)
      },
      listenerCount: (event: string) => (listeners.get(event) ?? []).length
    },
    reloadCalls
  }

  return win
}

function makeOptions(win: ReturnType<typeof makeFakeWindow>, kind = 'secondary', extra: Record<string, unknown> = {}) {
  const logs: string[] = []

  const options = {
    kind,
    callbacks: {
      log: (message: string) => {
        logs.push(message)
      },
      reload: () => {
        win.webContents.reload()
      }
    },
    ...extra
  }

  return { logs, options }
}

// The reload fires on setImmediate (never from inside the event handler), so
// tests flush the deferred queue before asserting.
function flushDeferred(): Promise<void> {
  return new Promise(resolve => setImmediate(resolve))
}

test('pruneReloadTimes drops timestamps outside the rolling window', () => {
  const now = 100_000

  assert.deepEqual(pruneReloadTimes([100_000, 90_000, 39_999], now, 60_000), [100_000, 90_000])
  assert.deepEqual(pruneReloadTimes([], now, 60_000), [])
})

test('pushReloadTime records the timestamp', () => {
  const times: number[] = []

  assert.deepEqual(pushReloadTime(times, 42), [42])
})

test('shouldReloadAfterRendererGone reloads crashed/oom on a live window', () => {
  assert.deepEqual(shouldReloadAfterRendererGone({ reason: 'crashed', isDestroyed: false, recentReloadTimes: [] }), {
    reload: true
  })
  assert.deepEqual(shouldReloadAfterRendererGone({ reason: 'oom', isDestroyed: false, recentReloadTimes: [] }), {
    reload: true
  })
})

test('shouldReloadAfterRendererGone never reloads expected teardown or unknown reasons', () => {
  // A window the user closed reports reason 'killed' — reloading would pop it
  // back up after close.
  assert.deepEqual(shouldReloadAfterRendererGone({ reason: 'killed', isDestroyed: true, recentReloadTimes: [] }), {
    reload: false,
    suppressedReason: 'expected-teardown'
  })
  // Killed on a live window is a process-initiated loss (e.g. OS reclaim);
  // the primary window never reloaded it, so peers don't either.
  assert.deepEqual(shouldReloadAfterRendererGone({ reason: 'killed', isDestroyed: false, recentReloadTimes: [] }), {
    reload: false,
    suppressedReason: 'unrecoverable-reason'
  })
  assert.deepEqual(
    shouldReloadAfterRendererGone({ reason: 'launch-failed', isDestroyed: false, recentReloadTimes: [] }),
    {
      reload: false,
      suppressedReason: 'unrecoverable-reason'
    }
  )
  assert.deepEqual(
    shouldReloadAfterRendererGone({ reason: 'unknown-reason', isDestroyed: false, recentReloadTimes: [] }),
    {
      reload: false,
      suppressedReason: 'unrecoverable-reason'
    }
  )
  assert.deepEqual(shouldReloadAfterRendererGone({ reason: undefined, isDestroyed: false, recentReloadTimes: [] }), {
    reload: false,
    suppressedReason: 'unrecoverable-reason'
  })
})

test('shouldReloadAfterRendererGone suppresses past the shared crash-loop budget', () => {
  const recentReloadTimes = [100, 50, 10]

  assert.deepEqual(
    shouldReloadAfterRendererGone({
      reason: 'crashed',
      isDestroyed: false,
      recentReloadTimes,
      reloadWindowMs: 60_000,
      reloadMax: 3,
      now: () => 200
    }),
    { reload: false, suppressedReason: 'crash-loop' }
  )

  // An expired budget entry frees a reload slot.
  const stale = [100, 50, 10]

  assert.deepEqual(
    shouldReloadAfterRendererGone({
      reason: 'crashed',
      isDestroyed: false,
      recentReloadTimes: stale,
      reloadWindowMs: 60_000,
      reloadMax: 3,
      now: () => 100_000
    }),
    { reload: true }
  )
})

test('installWindowRendererLifecycle logs and reloads a crashed secondary window', async () => {
  const win = makeFakeWindow()

  const { logs, options } = makeOptions(win, 'secondary', {
    callbacks: {
      log: (message: string) => {
        logs.push(message)
      },
      reload: () => {
        win.webContents.reload()
      }
    }
  })

  installWindowRendererLifecycle(win, options)
  win.webContents.emit('render-process-gone', {}, { reason: 'crashed', exitCode: 3 })
  await flushDeferred()

  assert.equal(win.reloadCalls.length, 1)
  assert.match(logs[0], /\[renderer:secondary\] render-process-gone reason=crashed exitCode=3/)
})

test('installWindowRendererLifecycle logs expected teardown without reloading', () => {
  const win = makeFakeWindow()
  const { logs, options } = makeOptions(win, 'secondary')

  installWindowRendererLifecycle(win, options)
  win.setDestroyed(true)
  win.webContents.emit('render-process-gone', {}, { reason: 'killed', exitCode: 1 })

  assert.equal(win.reloadCalls.length, 0)
  assert.match(logs[0], /render-process-gone reason=killed exitCode=1 \(expected teardown\)/)
})

test('installWindowRendererLifecycle suppresses a peer crash loop after the budget', async () => {
  const win = makeFakeWindow()

  const { logs, options } = makeOptions(win, 'instance', {
    reloadWindowMs: 60_000,
    reloadMax: 3,
    now: () => 1000
  })

  installWindowRendererLifecycle(win, options)

  for (let index = 0; index < 3; index += 1) {
    win.webContents.emit('render-process-gone', {}, { reason: 'crashed', exitCode: 3 })
  }

  await flushDeferred()
  assert.equal(win.reloadCalls.length, 3)

  win.webContents.emit('render-process-gone', {}, { reason: 'crashed', exitCode: 3 })
  await flushDeferred()

  assert.equal(win.reloadCalls.length, 3)
  assert.match(logs[logs.length - 1], /suppressing reload: 3 crashes within 60000ms/)
})

test('windows share one crash-loop budget via recentReloadTimesRef', async () => {
  const shared = { current: [] as number[] }
  const main = makeFakeWindow()
  const secondary = makeFakeWindow()

  const { logs: mainLogs, options: mainOptions } = makeOptions(main, 'main', {
    reloadWindowMs: 60_000,
    reloadMax: 3,
    now: () => 1000,
    recentReloadTimesRef: shared
  })

  const { logs: secondaryLogs, options: secondaryOptions } = makeOptions(secondary, 'secondary', {
    reloadWindowMs: 60_000,
    reloadMax: 3,
    now: () => 1000,
    recentReloadTimesRef: shared
  })

  installWindowRendererLifecycle(main, mainOptions)
  installWindowRendererLifecycle(secondary, secondaryOptions)

  for (let index = 0; index < 2; index += 1) {
    main.webContents.emit('render-process-gone', {}, { reason: 'crashed', exitCode: 3 })
  }

  // The secondary window's crash spends the last budget slot.
  secondary.webContents.emit('render-process-gone', {}, { reason: 'crashed', exitCode: 3 })
  await flushDeferred()
  assert.equal(secondary.reloadCalls.length, 1)

  // A fourth crash anywhere is suppressed.
  main.webContents.emit('render-process-gone', {}, { reason: 'crashed', exitCode: 3 })
  await flushDeferred()
  assert.equal(main.reloadCalls.length, 2)
  assert.match(mainLogs[mainLogs.length - 1], /suppressing reload/)
  assert.equal(secondaryLogs.length, 1)
})

test('log-only mode never reloads', () => {
  const win = makeFakeWindow()
  const { logs, options } = makeOptions(win, 'overlay')

  installWindowRendererLifecycle(win, options)
  win.webContents.emit('render-process-gone', {}, { reason: 'crashed', exitCode: 3 })

  assert.equal(win.reloadCalls.length, 0)
  assert.match(logs[0], /\[renderer:overlay\] render-process-gone reason=crashed exitCode=3/)
})

test('unresponsive is logged, never reloaded', () => {
  const win = makeFakeWindow()
  const { logs, options } = makeOptions(win, 'secondary')

  installWindowRendererLifecycle(win, options)
  win.webContents.emit('unresponsive')

  assert.equal(win.reloadCalls.length, 0)
  assert.equal(logs[0], '[renderer:secondary] webContents became unresponsive')
})

test('did-fail-load on the main frame is logged, not reloaded', () => {
  const win = makeFakeWindow()
  const { logs, options } = makeOptions(win, 'instance')

  installWindowRendererLifecycle(win, options)
  win.webContents.emit('did-fail-load', {}, -3, 'ERR_ABORTED', 'file:///index.html', true)

  assert.equal(win.reloadCalls.length, 0)
  assert.match(logs[0], /\[renderer:instance\] did-fail-load code=-3 url=file:\/\/\/index\.html/)

  // Sub-frame failures are noise; the primary window never logged them.
  win.webContents.emit('did-fail-load', {}, -3, 'ERR_ABORTED', 'https://example.com/asset.js', false)
  assert.equal(logs.length, 1)
})

test('console-message events are NOT handled here (renderer-log.ts is the single owner)', () => {
  const win = makeFakeWindow()
  const { logs, options } = makeOptions(win, 'secondary')

  installWindowRendererLifecycle(win, options)

  // OAuth/portal windows install this helper for process events; their pages
  // must not be able to spill console output (tokens/PII) into desktop.log.
  win.webContents.emit(
    'console-message',
    {},
    { level: 3, message: 'boom', sourceUrl: 'file:///app.js', lineNumber: 42 }
  )

  assert.equal(win.webContents.listenerCount('console-message'), 0)
  assert.equal(logs.length, 0)
})

test('onCrashLoopSuppressed fires when the budget trips (main sandbox-relaunch hook)', async () => {
  const win = makeFakeWindow()
  const suppressed: Array<{ reason?: string; exitCode?: number }> = []

  const { logs, options } = makeOptions(win, 'main', {
    reloadWindowMs: 60_000,
    reloadMax: 1,
    now: () => 1000,
    callbacks: {
      log: (message: string) => {
        logs.push(message)
      },
      reload: () => {
        win.webContents.reload()
      },
      onCrashLoopSuppressed: details => {
        suppressed.push({ reason: details?.reason, exitCode: details?.exitCode })
      }
    }
  })

  installWindowRendererLifecycle(win, options)

  win.webContents.emit('render-process-gone', {}, { reason: 'crashed', exitCode: 3 })
  await flushDeferred()
  assert.equal(win.reloadCalls.length, 1)
  assert.equal(suppressed.length, 0)

  win.webContents.emit('render-process-gone', {}, { reason: 'crashed', exitCode: 3 })
  await flushDeferred()

  assert.equal(win.reloadCalls.length, 1)
  assert.equal(suppressed.length, 1)
  assert.deepEqual(suppressed[0], { reason: 'crashed', exitCode: 3 })
  assert.match(logs[logs.length - 1], /suppressing reload/)

  // Expected teardown and non-recoverable reasons never trip the hook.
  win.setDestroyed(true)
  win.webContents.emit('render-process-gone', {}, { reason: 'killed', exitCode: 1 })
  win.webContents.emit('render-process-gone', {}, { reason: 'launch-failed', exitCode: 7 })
  assert.equal(suppressed.length, 1)
})

test('dispose removes every listener (no stacking on window recreation)', () => {
  const win = makeFakeWindow()
  const { logs, options } = makeOptions(win, 'secondary')

  const dispose = installWindowRendererLifecycle(win, options)
  const before = win.webContents.listenerCount('render-process-gone')

  dispose()
  win.webContents.emit('render-process-gone', {}, { reason: 'crashed', exitCode: 3 })

  assert.equal(win.reloadCalls.length, 0)
  assert.equal(logs.length, 0)
  assert.equal(win.webContents.listenerCount('render-process-gone'), before - 1)
})

test('describeRendererLifecycleEvent sanitizes unknown fields', () => {
  assert.equal(
    describeRendererLifecycleEvent({ kind: 'secondary', event: 'render-process-gone' }),
    '[renderer:secondary] render-process-gone reason=? exitCode=?'
  )
  assert.equal(
    describeRendererLifecycleEvent({
      kind: 'secondary',
      event: 'render-process-gone',
      reason: 'crashed',
      exitCode: undefined
    }),
    '[renderer:secondary] render-process-gone reason=crashed exitCode=?'
  )
  assert.equal(
    describeRendererLifecycleEvent({
      kind: 'main',
      event: 'render-process-gone',
      reason: 'killed',
      exitCode: 1,
      isDestroyed: true
    }),
    '[renderer:main] render-process-gone reason=killed exitCode=1 (expected teardown)'
  )
})
