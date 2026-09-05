import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  coversDisplay,
  detectFullscreenApp,
  findFullscreenAppAnywhere,
  gameOverlayStateFor,
  INACTIVE_GAME_OVERLAY,
  isShellWindow,
  startHudGameOverlayWatch
} from './hud-game-overlay'
import type { GameOverlayState } from './hud-game-overlay'
import type { EnumeratedWindow } from './window-below'

const DISPLAY = { x: 0, y: 0, width: 2560, height: 1440 }
const SELF_PID = 4242

const win = (over: Partial<EnumeratedWindow>): EnumeratedWindow => ({
  app: 'App',
  bounds: { x: 0, y: 0, width: 100, height: 100 },
  id: 1,
  pid: 1000,
  title: '',
  ...over
})

const fullscreenGame = win({ app: 'Balatro', bounds: { ...DISPLAY }, id: 7, pid: 2000 })
const hudWindow = win({ app: 'Hermes', bounds: { x: 970, y: 1048, width: 620, height: 320 }, id: 2, pid: SELF_PID })
const desktopShell = win({ app: 'Windows Explorer', bounds: { ...DISPLAY }, id: 3, pid: 900 })

// ─── coversDisplay ───────────────────────────────────────────────────────

test('display-sized bounds cover the display', () => {
  assert.equal(coversDisplay({ ...DISPLAY }, DISPLAY), true)
})

test('DPI rounding and the 1px-oversize trick stay within the epsilon', () => {
  assert.equal(coversDisplay({ x: -1, y: -1, width: 2562, height: 1442 }, DISPLAY), true)
  assert.equal(coversDisplay({ x: 1, y: 1, width: 2558, height: 1438 }, DISPLAY), true)
})

test('a maximized window stopping at the taskbar does not cover', () => {
  assert.equal(coversDisplay({ x: 0, y: 0, width: 2560, height: 1392 }, DISPLAY), false)
})

test('a fullscreen window on ANOTHER display does not cover this one', () => {
  assert.equal(coversDisplay({ x: 2560, y: 0, width: 2560, height: 1440 }, DISPLAY), false)
})

// ─── isShellWindow ───────────────────────────────────────────────────────

test('desktop shells are recognized, real apps are not', () => {
  for (const app of ['Windows Explorer', 'explorer.exe', 'Program Manager', 'Finder', 'Dock', 'gnome-shell']) {
    assert.equal(isShellWindow(app), true, app)
  }

  for (const app of ['Balatro', 'Explorer of the Deep', 'Firefox', '']) {
    assert.equal(isShellWindow(app), false, app)
  }
})

// ─── detectFullscreenApp ─────────────────────────────────────────────────

test('fullscreen game under the HUD is detected', () => {
  const found = detectFullscreenApp([hudWindow, fullscreenGame, desktopShell], SELF_PID, DISPLAY)

  assert.equal(found?.app, 'Balatro')
})

test('own windows never count, wherever they sit in the z-order', () => {
  const selfFullscreen = win({ app: 'Hermes', bounds: { ...DISPLAY }, pid: SELF_PID })

  assert.equal(detectFullscreenApp([selfFullscreen, desktopShell], SELF_PID, DISPLAY), null)
})

test('the desktop shell reporting display bounds is not a game', () => {
  assert.equal(detectFullscreenApp([hudWindow, desktopShell], SELF_PID, DISPLAY), null)
})

test('a windowed app in FRONT of the game vetoes the overlay', () => {
  const editor = win({ app: 'VS Code', bounds: { x: 200, y: 100, width: 1400, height: 900 }, id: 9, pid: 3000 })

  assert.equal(detectFullscreenApp([hudWindow, editor, fullscreenGame], SELF_PID, DISPLAY), null)
})

test('zero-area rows (minimized) neither decide nor veto', () => {
  const minimized = win({ app: 'Spotify', bounds: { x: 0, y: 0, width: 0, height: 0 }, id: 11, pid: 3100 })

  assert.equal(detectFullscreenApp([minimized, fullscreenGame], SELF_PID, DISPLAY)?.app, 'Balatro')
})

test('a window intersecting only another display does not veto', () => {
  const otherDisplayWin = win({ app: 'Slack', bounds: { x: 3000, y: 0, width: 800, height: 600 }, id: 12, pid: 3200 })

  assert.equal(detectFullscreenApp([otherDisplayWin, fullscreenGame], SELF_PID, DISPLAY)?.app, 'Balatro')
})

test('empty desktop: nothing to detect', () => {
  assert.equal(detectFullscreenApp([hudWindow], SELF_PID, DISPLAY), null)
})

test('gameOverlayStateFor maps detection to the pushed state shape', () => {
  assert.deepEqual(gameOverlayStateFor([fullscreenGame], SELF_PID, DISPLAY), { active: true, app: 'Balatro' })
  assert.deepEqual(gameOverlayStateFor([hudWindow], SELF_PID, DISPLAY), INACTIVE_GAME_OVERLAY)
})

// ─── hysteresis: enter strictly, stay while the game is still there ──────

test('a windowed app on top vetoes ENTERING but not STAYING', () => {
  // Clicking the HUD to type de-foregrounds the game and floats every other
  // open window above it. Entering must respect that veto; staying must not,
  // or the treatment drops the moment the user engages with it.
  const editor = win({ app: 'VS Code', bounds: { x: 200, y: 100, width: 1400, height: 900 }, id: 9, pid: 3000 })
  const stack = [editor, fullscreenGame]

  assert.deepEqual(gameOverlayStateFor(stack, SELF_PID, DISPLAY, false), INACTIVE_GAME_OVERLAY)
  assert.deepEqual(gameOverlayStateFor(stack, SELF_PID, DISPLAY, true), { active: true, app: 'Balatro' })
})

test('the game actually closing ends overlay mode even when it was active', () => {
  assert.deepEqual(gameOverlayStateFor([hudWindow, desktopShell], SELF_PID, DISPLAY, true), INACTIVE_GAME_OVERLAY)
})

test('findFullscreenAppAnywhere ignores z-order but keeps every other guard', () => {
  const selfFullscreen = win({ app: 'Hermes', bounds: { ...DISPLAY }, pid: SELF_PID })

  assert.equal(findFullscreenAppAnywhere([selfFullscreen], SELF_PID, DISPLAY), null)
  assert.equal(findFullscreenAppAnywhere([desktopShell], SELF_PID, DISPLAY), null)
  assert.equal(findFullscreenAppAnywhere([hudWindow, fullscreenGame], SELF_PID, DISPLAY)?.app, 'Balatro')
})

// ─── startHudGameOverlayWatch ────────────────────────────────────────────

interface FakeClock {
  fire: () => Promise<void>
  cleared: () => boolean
  setIntervalFn: typeof setInterval
  clearIntervalFn: typeof clearInterval
}

const fakeClock = (): FakeClock => {
  let handler: (() => void) | null = null
  let cleared = false

  return {
    fire: async () => {
      handler?.()
      // The tick is async; let its microtasks drain.
      await Promise.resolve()
      await Promise.resolve()
    },
    cleared: () => cleared,
    setIntervalFn: ((fn: () => void) => {
      handler = fn

      return 1 as unknown as ReturnType<typeof setInterval>
    }) as typeof setInterval,
    clearIntervalFn: (() => {
      cleared = true
    }) as typeof clearInterval
  }
}

const flush = async () => {
  await Promise.resolve()
  await Promise.resolve()
}

test('watch pushes on change only, never on a repeat answer', async () => {
  const clock = fakeClock()
  const pushed: Array<{ active: boolean; app: string }> = []
  let windows: EnumeratedWindow[] = [hudWindow]

  const dispose = startHudGameOverlayWatch({
    enumerate: () => Promise.resolve(windows),
    displayBounds: () => DISPLAY,
    selfPid: SELF_PID,
    send: state => pushed.push(state),
    setIntervalFn: clock.setIntervalFn,
    clearIntervalFn: clock.clearIntervalFn
  })

  await flush() // the immediate first tick
  assert.deepEqual(pushed, [{ active: false, app: '' }])

  await clock.fire() // same answer — no push
  assert.equal(pushed.length, 1)

  windows = [fullscreenGame]
  await clock.fire()
  assert.deepEqual(pushed[1], { active: true, app: 'Balatro' })

  windows = [hudWindow]
  await clock.fire()
  assert.deepEqual(pushed[2], { active: false, app: '' })

  dispose()
  assert.equal(clock.cleared(), true)
})

test('watch settles inactive and stops after repeated enumeration failures', async () => {
  const clock = fakeClock()
  const pushed: Array<{ active: boolean }> = []

  startHudGameOverlayWatch({
    enumerate: () => Promise.resolve(null),
    displayBounds: () => DISPLAY,
    selfPid: SELF_PID,
    send: state => pushed.push(state),
    setIntervalFn: clock.setIntervalFn,
    clearIntervalFn: clock.clearIntervalFn
  })

  await flush() // failure 1 — no conclusion yet
  assert.equal(pushed.length, 0)
  assert.equal(clock.cleared(), false)

  await clock.fire() // failure 2 — settle and stop
  assert.deepEqual(pushed, [{ active: false, app: '' }])
  assert.equal(clock.cleared(), true)
})

test('a single transient failure does not give up the watch', async () => {
  const clock = fakeClock()
  const pushed: Array<{ active: boolean }> = []
  let fail = true

  startHudGameOverlayWatch({
    enumerate: () => (fail ? Promise.resolve(null) : Promise.resolve([fullscreenGame])),
    displayBounds: () => DISPLAY,
    selfPid: SELF_PID,
    send: state => pushed.push(state),
    setIntervalFn: clock.setIntervalFn,
    clearIntervalFn: clock.clearIntervalFn
  })

  await flush() // failure 1
  fail = false
  await clock.fire() // recovers

  assert.deepEqual(pushed, [{ active: true, app: 'Balatro' }])
  assert.equal(clock.cleared(), false)

  // …and the failure counter reset: one more null is a transient again.
  fail = true
  await clock.fire()
  assert.equal(clock.cleared(), false)
})

test('disposed watch stops pushing even with a tick in flight', async () => {
  const clock = fakeClock()
  const pushed: unknown[] = []

  let release: (v: EnumeratedWindow[]) => void = () => {}

  const dispose = startHudGameOverlayWatch({
    enumerate: () =>
      new Promise<EnumeratedWindow[]>(resolve => {
        release = resolve
      }),
    displayBounds: () => DISPLAY,
    selfPid: SELF_PID,
    send: state => pushed.push(state),
    setIntervalFn: clock.setIntervalFn,
    clearIntervalFn: clock.clearIntervalFn
  })

  dispose()
  release([fullscreenGame])
  await flush()

  assert.equal(pushed.length, 0)
})

test('the watch carries its own last answer into the next tick', async () => {
  // The hysteresis is only correct if `wasActive` comes from what was last
  // PUBLISHED; a watch that always passed false would re-veto every tick.
  const clock = fakeClock()
  const pushed: GameOverlayState[] = []
  const editor = win({ app: 'VS Code', bounds: { x: 200, y: 100, width: 1400, height: 900 }, id: 9, pid: 3000 })
  let windows: EnumeratedWindow[] = [fullscreenGame]

  startHudGameOverlayWatch({
    enumerate: () => Promise.resolve(windows),
    displayBounds: () => DISPLAY,
    selfPid: SELF_PID,
    send: state => pushed.push(state),
    setIntervalFn: clock.setIntervalFn,
    clearIntervalFn: clock.clearIntervalFn
  })

  await flush()
  assert.deepEqual(pushed, [{ active: true, app: 'Balatro' }])

  // The user clicks the HUD: the editor is now above the game. Still active, so
  // nothing new is published.
  windows = [editor, fullscreenGame]
  await clock.fire()
  assert.equal(pushed.length, 1)

  // The game closes for real.
  windows = [editor]
  await clock.fire()
  assert.deepEqual(pushed[1], INACTIVE_GAME_OVERLAY)
})
