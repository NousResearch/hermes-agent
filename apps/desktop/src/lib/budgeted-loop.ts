/**
 * Budgeted renderer loop — the one way to run a decorative rAF animation.
 *
 * Desktop has repeatedly shipped the same bug class: an animation loop that
 * never sleeps (bots face clock #88543, pixel egg #88406, diffusion
 * placeholder #88564, status pulse / spinner in the #77651 wave). Each fix
 * re-implemented the same four behaviors by hand. This helper owns them:
 *
 *  - **fps budget** — paints are skipped until `1000 / fps` elapsed; the loop
 *    still rides rAF so Chromium can align and pause it natively.
 *  - **observability pause** — wired to `createRendererLoopPauseController`:
 *    the loop cancels while the window is hidden, minimized, or unfocused and
 *    resumes when observable.
 *  - **idle dormancy** — when `idleWhen()` returns true after a draw, the loop
 *    parks (zero pending frames, zero timers) until `wake()` is called. This
 *    is the piece every hand-rolled loop forgot: "nothing visible to animate"
 *    must cost nothing.
 *  - **teardown** — `dispose()` cancels the frame, disposes the pause
 *    controller, and makes every later `wake()` a no-op. Callers MUST call it
 *    from their effect cleanup / plugin disposer.
 *
 * Usage:
 *
 * ```ts
 * const loop = createBudgetedLoop(now => paint(now), {
 *   fps: 15,
 *   idleWhen: () => visibleThings.size === 0
 * })
 * // something became animatable again:
 * loop.wake()
 * // effect cleanup / plugin disable:
 * loop.dispose()
 * ```
 */

import { createRendererLoopPauseController } from '@/lib/renderer-loop-pause'

export interface BudgetedLoopOptions {
  /** Paint budget in frames per second. Default 15 — visually smooth for
   *  decorative/avatar-scale animation at a quarter of display cadence. */
  fps?: number
  /** Return true when there is nothing worth animating (no visible targets,
   *  animation finished). Checked after every draw; when true the loop parks
   *  until `wake()`. Omit for loops that should run whenever observable. */
  idleWhen?: () => boolean
  /** Forwarded to the pause controller. Default true (pause on blur). */
  pauseWhenUnfocused?: boolean
}

export interface BudgetedLoop {
  /** Cancel everything and detach listeners. Idempotent; wake() after this is
   *  a no-op. Call from effect cleanup or the plugin disposer. */
  dispose: () => void
  /** True while parked by `idleWhen` (diagnostics/tests). */
  isDormant: () => boolean
  /** Resume a loop parked by `idleWhen` (e.g. a target mounted or scrolled
   *  into view). Safe to call redundantly; no-op while running or disposed. */
  wake: () => void
}

export function createBudgetedLoop(
  draw: (now: number) => void,
  { fps = 15, idleWhen, pauseWhenUnfocused = true }: BudgetedLoopOptions = {}
): BudgetedLoop {
  const frameInterval = 1000 / fps
  let frame = 0
  let lastDraw = -Infinity
  let dormant = false
  let disposed = false

  const cancelFrame = () => {
    if (frame !== 0) {
      window.cancelAnimationFrame(frame)
      frame = 0
    }
  }

  const scheduleFrame = () => {
    if (disposed || dormant || pauseController.isPaused() || frame !== 0) {
      return
    }

    frame = window.requestAnimationFrame(tick)
  }

  const tick = (now: number) => {
    frame = 0

    if (disposed || pauseController.isPaused()) {
      return
    }

    if (now - lastDraw >= frameInterval) {
      draw(now)
      lastDraw = now
    }

    // Idle dormancy: nothing to animate -> park with zero pending work.
    // The owner wakes us when a target (re)appears.
    if (idleWhen?.()) {
      dormant = true

      return
    }

    scheduleFrame()
  }

  const handlePauseChange = () => {
    cancelFrame()
    scheduleFrame()
  }

  const pauseController = createRendererLoopPauseController(handlePauseChange, { pauseWhenUnfocused })

  const wake = () => {
    if (disposed || !dormant) {
      return
    }

    dormant = false
    scheduleFrame()
  }

  scheduleFrame()

  return {
    dispose: () => {
      if (disposed) {
        return
      }

      disposed = true
      cancelFrame()
      pauseController.dispose()
    },
    isDormant: () => dormant,
    wake
  }
}
