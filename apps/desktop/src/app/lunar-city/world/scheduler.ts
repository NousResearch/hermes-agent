import { createRendererLoopPauseController } from '@/lib/renderer-loop-pause'

export interface FrameRenderer {
  stopRenderLoop?(): void
}

export interface FrameTick {
  elapsedMs: number
  now: number
  targetFps: 0 | 15 | 30
}

export type FramePauseReason = 'context-lost' | 'document-hidden' | 'window-minimized'

export interface FrameSchedulerOptions {
  captureMetrics?: boolean
  cancelFrame?: (handle: number) => void
  clearTimer?: (handle: number) => void
  now?: () => number
  onFrame(frame: FrameTick): boolean | void
  renderer: FrameRenderer
  requestFrame?: (callback: FrameRequestCallback) => number
  setTimer?: (callback: () => void, delayMs: number) => number
}

function defaultNow(): number {
  return typeof performance === 'undefined' ? Date.now() : performance.now()
}

function defaultRequestFrame(callback: FrameRequestCallback): number | undefined {
  return typeof requestAnimationFrame === 'undefined' ? undefined : requestAnimationFrame(callback)
}

function defaultCancelFrame(handle: number): void {
  if (typeof cancelAnimationFrame !== 'undefined') {
    cancelAnimationFrame(handle)
  }
}

function defaultSetTimer(callback: () => void, delayMs: number): number | undefined {
  return typeof window === 'undefined' ? undefined : window.setTimeout(callback, delayMs)
}

function defaultClearTimer(handle: number): void {
  if (typeof window !== 'undefined') {
    window.clearTimeout(handle)
  }
}

/**
 * The sole frame authority for Lunar City.  It deliberately uses no Babylon
 * render loop: camera, navigation, animation, occlusion, and Scene.render are
 * all invoked by the one injected frame callback.
 */
export function createFrameScheduler(options: FrameSchedulerOptions) {
  const clock = options.now ?? defaultNow
  const requestFrame = options.requestFrame ?? defaultRequestFrame
  const cancelFrame = options.cancelFrame ?? defaultCancelFrame
  const setTimer = options.setTimer ?? defaultSetTimer
  const clearTimer = options.clearTimer ?? defaultClearTimer
  const pauseReasons = new Set<FramePauseReason>()
  let disposed = false
  let dirty = false
  let interactiveUntil = Number.NEGATIVE_INFINITY
  let lastFrameAt: number | undefined
  let visualDeadline: number | undefined
  let frameHandle: number | undefined
  let timerHandle: number | undefined
  let timerDeadline: number | undefined
  let throttleUntil: number | undefined
  let releaseVisibility: (() => void) | undefined
  let renderFrames = 0
  const frameTimestampsMs: number[] | undefined = options.captureMetrics ? [] : undefined

  const targetFpsAt = (now: number): 15 | 30 => (now < interactiveUntil ? 30 : 15)
  const canRender = (): boolean => !disposed && pauseReasons.size === 0

  const cancelPending = (): void => {
    if (frameHandle !== undefined) {
      cancelFrame(frameHandle)
      frameHandle = undefined
    }

    if (timerHandle !== undefined) {
      clearTimer(timerHandle)
      timerHandle = undefined
      timerDeadline = undefined
    }
  }

  const ensureTimer = (deadline: number, now: number): void => {
    if (deadline <= now || (timerHandle !== undefined && timerDeadline !== undefined && timerDeadline <= deadline)) {
      return
    }

    if (timerHandle !== undefined) {
      clearTimer(timerHandle)
      timerHandle = undefined
      timerDeadline = undefined
    }

    const handle = setTimer(() => {
      timerHandle = undefined
      timerDeadline = undefined
      wake()
    }, deadline - now)

    if (handle !== undefined) {
      timerHandle = handle
      timerDeadline = deadline
    }
  }

  const wake = (): void => {
    if (!canRender()) {
      return
    }

    const now = clock()

    const deadline = visualDeadline
    const throttleDeadline = throttleUntil

    if (!dirty && deadline === undefined) {
      return
    }

    if (throttleDeadline !== undefined && throttleDeadline > now) {
      ensureTimer(throttleDeadline, now)

      return
    }

    throttleUntil = undefined

    if (!dirty && deadline !== undefined && deadline > now) {
      ensureTimer(deadline, now)

      return
    }

    if (frameHandle !== undefined) {
      return
    }

    const handle = requestFrame(nowFromFrame => {
      frameHandle = undefined
      tick(nowFromFrame)
    })

    if (handle !== undefined) {
      frameHandle = handle
    }
  }

  const scheduleAt = (deadline: number): void => {
    if (visualDeadline === undefined || deadline < visualDeadline) {
      visualDeadline = deadline
    }

    if (canRender()) {
      ensureTimer(visualDeadline!, clock())
    }

    wake()
  }

  const tick = (now: number): boolean => {
    if (!canRender()) {
      return false
    }

    const targetFps = targetFpsAt(now)
    const interval = 1_000 / targetFps
    const due = dirty || (visualDeadline !== undefined && now >= visualDeadline)

    if (!due) {
      wake()

      return false
    }

    if (lastFrameAt !== undefined && now - lastFrameAt < interval) {
      throttleUntil = lastFrameAt + interval
      ensureTimer(throttleUntil, now)

      return false
    }

    throttleUntil = undefined
    const elapsedMs = lastFrameAt === undefined ? interval : Math.max(0, now - lastFrameAt)
    dirty = false
    visualDeadline = undefined
    lastFrameAt = now
    const needsAnotherFrame = options.onFrame({ elapsedMs, now, targetFps }) === true
    renderFrames += 1
    frameTimestampsMs?.push(now)

    if (needsAnotherFrame || now < interactiveUntil) {
      throttleUntil = now + interval
      scheduleAt(throttleUntil)
    }

    return true
  }

  const setPauseReason = (reason: FramePauseReason, paused: boolean): void => {
    if (disposed || pauseReasons.has(reason) === paused) {
      return
    }

    const wasRunnable = canRender()

    if (paused) {
      pauseReasons.add(reason)
    } else {
      pauseReasons.delete(reason)
    }

    if (wasRunnable && !canRender()) {
      dirty = false
      visualDeadline = undefined
      throttleUntil = undefined
      cancelPending()
      options.renderer.stopRenderLoop?.()

      return
    }

    if (!wasRunnable && canRender()) {
      // A resumed world receives one up-to-date snapshot frame, rather than
      // replaying an animation backlog accumulated while it was paused.
      dirty = true
      wake()
    }
  }

  const setVisible = (nextVisible: boolean): void => setPauseReason('context-lost', !nextVisible)

  return {
    bindRendererPauseState(): () => void {
      if (typeof document === 'undefined' || releaseVisibility) {
        return () => undefined
      }

      let controller: ReturnType<typeof createRendererLoopPauseController>
      controller = createRendererLoopPauseController(
        () => {
          setPauseReason('document-hidden', controller.isPaused())
        },
        { pauseWhenUnfocused: false }
      )
      releaseVisibility = () => controller.dispose()
      setPauseReason('document-hidden', controller.isPaused())

      return () => {
        releaseVisibility?.()
        releaseVisibility = undefined
      }
    },
    dispose(): void {
      if (disposed) {
        return
      }

      disposed = true
      cancelPending()
      releaseVisibility?.()
      releaseVisibility = undefined
    },
    getMetrics() {
      return {
        frameTimestampsMs: frameTimestampsMs ? [...frameTimestampsMs] : [],
        listeners:
          releaseVisibility && typeof window !== 'undefined'
            ? 3 + (typeof window.hermesDesktop?.onWindowStateChanged === 'function' ? 1 : 0)
            : 0,
        rafs: frameHandle === undefined ? 0 : 1,
        renderFrames,
        targetFps: canRender() ? targetFpsAt(clock()) : (0 as const),
        timers: timerHandle === undefined ? 0 : 1
      }
    },
    noteInteraction(now: number): void {
      if (disposed) {
        return
      }

      interactiveUntil = Math.max(interactiveUntil, now + 5_000)
      dirty = true
      wake()
    },
    requestRender(): void {
      if (disposed) {
        return
      }

      dirty = true
      wake()
    },
    requestVisualAt(deadline: number): void {
      if (disposed || !Number.isFinite(deadline)) {
        return
      }

      scheduleAt(deadline)
    },
    setVisible,
    setPauseReason,
    tick,
    targetFps(now = clock()): 0 | 15 | 30 {
      return canRender() ? targetFpsAt(now) : 0
    }
  }
}
