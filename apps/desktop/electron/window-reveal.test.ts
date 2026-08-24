import assert from 'node:assert/strict'

import { test } from 'vitest'

import { createWindowRevealController } from './window-reveal'

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
