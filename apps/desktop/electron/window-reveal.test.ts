import assert from 'node:assert/strict'

import { test } from 'vitest'

import { createWindowRevealController, wireWindowReveal } from './window-reveal'

function createHarness({ visible = false }: { visible?: boolean } = {}) {
  let destroyed = false
  let isVisible = visible
  let revealCalls = 0
  let showCalls = 0
  let scheduledCallback: (() => void) | null = null
  let scheduledDelay: number | null = null
  let clearCalls = 0

  const controller = createWindowRevealController(
    {
      isDestroyed: () => destroyed,
      isVisible: () => isVisible,
      show: () => {
        showCalls += 1
        isVisible = true
      }
    },
    {
      onRevealed: () => {
        revealCalls += 1
      },
      setTimer: (callback, delay) => {
        scheduledCallback = callback
        scheduledDelay = delay

        return 1 as unknown as ReturnType<typeof setTimeout>
      },
      clearTimer: () => {
        clearCalls += 1
      }
    }
  )

  return {
    controller,
    destroy: () => {
      destroyed = true
    },
    get clearCalls() {
      return clearCalls
    },
    get revealCalls() {
      return revealCalls
    },
    get scheduledCallback() {
      return scheduledCallback
    },
    get scheduledDelay() {
      return scheduledDelay
    },
    get showCalls() {
      return showCalls
    }
  }
}

test('reveals immediately when Electron emits ready-to-show', () => {
  const harness = createHarness()

  assert.equal(harness.controller.reveal(), true)
  assert.equal(harness.showCalls, 1)
  assert.equal(harness.revealCalls, 1)
  assert.equal(harness.scheduledCallback, null)
})

test('reveals through the fallback when ready-to-show never arrives', () => {
  const harness = createHarness()

  harness.controller.scheduleFallback()

  assert.equal(harness.scheduledDelay, 4_000)
  assert.ok(harness.scheduledCallback)
  harness.scheduledCallback()

  assert.equal(harness.showCalls, 1)
  assert.equal(harness.revealCalls, 1)
})

test('ready-to-show cancels the fallback and settles only once', () => {
  const harness = createHarness()

  harness.controller.scheduleFallback()
  const scheduledCallback = harness.scheduledCallback

  assert.equal(harness.controller.reveal(), true)
  assert.equal(harness.clearCalls, 1)
  scheduledCallback?.()

  assert.equal(harness.showCalls, 1)
  assert.equal(harness.revealCalls, 1)
  assert.equal(harness.controller.reveal(), false)
})

test('finalizes an already visible window without showing it again', () => {
  const harness = createHarness({ visible: true })

  assert.equal(harness.controller.reveal(), true)
  assert.equal(harness.showCalls, 0)
  assert.equal(harness.revealCalls, 1)
})

test('dispose cancels a pending fallback and prevents a late reveal', () => {
  const harness = createHarness()

  harness.controller.scheduleFallback()
  const scheduledCallback = harness.scheduledCallback
  harness.controller.dispose()
  scheduledCallback?.()

  assert.equal(harness.clearCalls, 1)
  assert.equal(harness.showCalls, 0)
  assert.equal(harness.revealCalls, 0)
})

test('does not reveal a destroyed window', () => {
  const harness = createHarness()

  harness.destroy()

  assert.equal(harness.controller.reveal(), false)
  harness.controller.scheduleFallback()
  assert.equal(harness.scheduledCallback, null)
})

// The pet overlay reveals with showInactive() and the HUD with show() + focus(),
// so the fallback has to run the caller's action rather than a plain show().
test('the fallback runs the caller reveal action, not a plain show', () => {
  const actions: string[] = []
  let scheduled: (() => void) | null = null

  const controller = createWindowRevealController(
    {
      isDestroyed: () => false,
      isVisible: () => false,
      show: () => actions.push('showInactive')
    },
    {
      onRevealed: () => actions.push('revealed'),
      setTimer: callback => {
        scheduled = callback

        return 1 as unknown as ReturnType<typeof setTimeout>
      },
      clearTimer: () => {}
    }
  )

  controller.scheduleFallback()
  scheduled?.()

  assert.deepEqual(actions, ['showInactive', 'revealed'])
})

// The HUD hides the main window from onRevealed — whichever path wins, that
// side effect has to happen exactly once.
test('onRevealed side effects run once when both paths fire', () => {
  const harness = createHarness()

  harness.controller.scheduleFallback()
  harness.controller.reveal()
  harness.scheduledCallback?.()
  harness.controller.reveal()

  assert.equal(harness.revealCalls, 1)
})

// Session and instance windows have no post-visible work to do.
test('reveals without an onRevealed callback', () => {
  let shown = false

  const controller = createWindowRevealController({
    isDestroyed: () => false,
    isVisible: () => shown,
    show: () => {
      shown = true
    }
  })

  assert.equal(controller.reveal(), true)
  assert.equal(shown, true)
})

// ── Reveal failure branch (#108230) ─────────────────────────────────────────
//
// A window created `show: false` is only revealed by success-shaped events
// (ready-to-show / did-finish-load). When the main frame fails to load or the
// render process dies before first paint, none of those fire — the fallback
// timer is never even scheduled — so the window stays hidden forever while
// callers keep claiming it is open. The failure branch hands such windows to
// `onRevealFailed` exactly once and disarms every other path.

type FailureHarness = {
  emit(event: 'closed' | 'did-finish-load' | 'did-fail-load' | 'render-process-gone', ...args: unknown[]): void
  emitReadyToShow(): void
  failures: string[]
  revealCalls: number
  scheduledCallback: (() => void) | null
  showCalls: number
  visible: () => boolean
}

function createFailureHarness(): FailureHarness {
  const windowListeners = new Map<string, Array<(...args: any[]) => void>>()
  const webContentsListeners = new Map<string, Array<(...args: any[]) => void>>()
  let visible = false
  const failures: string[] = []
  let revealCalls = 0
  let scheduledCallback: (() => void) | null = null

  const win = {
    isDestroyed: () => false,
    isVisible: () => visible,
    show: () => {
      visible = true
    },
    once: (event: string, listener: (...args: any[]) => void) => {
      windowListeners.set(`once:${event}`, [...(windowListeners.get(`once:${event}`) ?? []), listener])
    },
    on: (event: string, listener: (...args: any[]) => void) => {
      windowListeners.set(`on:${event}`, [...(windowListeners.get(`on:${event}`) ?? []), listener])
    },
    webContents: {
      once: (event: string, listener: (...args: any[]) => void) => {
        webContentsListeners.set(`once:${event}`, [...(webContentsListeners.get(`once:${event}`) ?? []), listener])
      },
      on: (event: string, listener: (...args: any[]) => void) => {
        webContentsListeners.set(`on:${event}`, [...(webContentsListeners.get(`on:${event}`) ?? []), listener])
      }
    }
  }

  wireWindowReveal(win, {
    onRevealed: () => {
      revealCalls += 1
    },
    onRevealFailed: reason => {
      failures.push(reason)
    },
    setTimer: callback => {
      scheduledCallback = callback

      return 1 as unknown as ReturnType<typeof setTimeout>
    },
    clearTimer: () => {}
  })

  const emit = (
    target: 'window' | 'webContents',
    kind: 'once' | 'on',
    event: string,
    ...args: unknown[]
  ) => {
    const key = `${kind}:${event}`
    const map = target === 'window' ? windowListeners : webContentsListeners

    for (const listener of [...(map.get(key) ?? [])]) {
      listener(...args)
    }
  }

  return {
    emit: (event, ...args) => {
      if (event === 'closed') {
        emit('window', 'on', event, ...args)
      } else if (event === 'did-finish-load') {
        emit('webContents', 'once', event, ...args)
      } else {
        emit('webContents', 'on', event, ...args)
      }
    },
    emitReadyToShow: () => emit('window', 'once', 'ready-to-show'),
    failures,
    get revealCalls() {
      return revealCalls
    },
    get scheduledCallback() {
      return scheduledCallback
    },
    showCalls: 0,
    visible: () => visible
  }
}

test('a main-frame load failure before reveal fires onRevealFailed once and disarms the reveal', () => {
  const harness = createFailureHarness()

  // The main frame failed: did-finish-load never fires, so no fallback is
  // scheduled — without the failure branch nothing would ever happen again.
  harness.emit('did-fail-load', {}, -3, 'ABORTED', 'file:///hud', true)

  assert.equal(harness.failures.length, 1)
  assert.match(harness.failures[0]!, /main frame/)

  // Disarmed: a late ready-to-show must not show a dead window, and a second
  // failure must not re-fire the teardown.
  harness.emitReadyToShow()
  assert.equal(harness.visible(), false)
  assert.equal(harness.revealCalls, 0)

  harness.emit('render-process-gone', {}, { reason: 'oom' })
  assert.equal(harness.failures.length, 1)
})

test('a subframe load failure never tears the window down', () => {
  const harness = createFailureHarness()

  harness.emit('did-fail-load', {}, -3, 'ABORTED', 'file:///subframe.js', false)
  harness.emitReadyToShow()

  assert.deepEqual(harness.failures, [])
  assert.equal(harness.visible(), true)
})

test('a render-process crash before first paint fires onRevealFailed', () => {
  const harness = createFailureHarness()

  harness.emit('render-process-gone', {}, { reason: 'crashed' })

  assert.equal(harness.failures.length, 1)
  assert.match(harness.failures[0]!, /render process gone/)
})

test('a crash or load failure AFTER reveal leaves the window alone', () => {
  const harness = createFailureHarness()

  harness.emitReadyToShow()
  assert.equal(harness.visible(), true)

  // Post-reveal, the window's own lifecycle owns recovery — the reveal
  // controller must not resurrect its failure branch for a shown window.
  harness.emit('render-process-gone', {}, { reason: 'crashed' })
  harness.emit('did-fail-load', {}, -3, 'ABORTED', 'file:///hud', true)

  assert.deepEqual(harness.failures, [])
  assert.equal(harness.revealCalls, 1)
})

test('a scheduled fallback is cancelled when the load fails first', () => {
  const harness = createFailureHarness()

  // did-finish-load arrived (fallback scheduled at 4s), then the process died
  // before first paint: the timer must not later reveal a dead window.
  harness.emit('did-finish-load')
  assert.ok(harness.scheduledCallback)

  harness.emit('render-process-gone', {}, { reason: 'kill' })
  harness.scheduledCallback?.()

  assert.equal(harness.failures.length, 1)
  assert.equal(harness.visible(), false)
  assert.equal(harness.revealCalls, 0)
})
