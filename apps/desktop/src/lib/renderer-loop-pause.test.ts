import { afterEach, describe, expect, it, vi } from 'vitest'

import { installWindowStateBridge } from '../test/window-state'

import {
  createRendererLoopPauseController,
  installRendererAnimationPauseState,
  RENDERER_ANIMATIONS_PAUSED_ATTRIBUTE
} from './renderer-loop-pause'

describe('installRendererAnimationPauseState', () => {
  afterEach(() => {
    document.documentElement.removeAttribute(RENDERER_ANIMATIONS_PAUSED_ATTRIBUTE)
    vi.restoreAllMocks()
  })

  it('keeps visible animations running across blur and focus, and cleans up its root state', () => {
    let focused = true
    vi.spyOn(document, 'hasFocus').mockImplementation(() => focused)

    const dispose = installRendererAnimationPauseState()
    expect(document.documentElement.hasAttribute(RENDERER_ANIMATIONS_PAUSED_ATTRIBUTE)).toBe(false)

    focused = false
    window.dispatchEvent(new Event('blur'))
    expect(document.documentElement.hasAttribute(RENDERER_ANIMATIONS_PAUSED_ATTRIBUTE)).toBe(false)

    focused = true
    window.dispatchEvent(new Event('focus'))
    expect(document.documentElement.hasAttribute(RENDERER_ANIMATIONS_PAUSED_ATTRIBUTE)).toBe(false)

    focused = false
    window.dispatchEvent(new Event('blur'))
    dispose()
    expect(document.documentElement.hasAttribute(RENDERER_ANIMATIONS_PAUSED_ATTRIBUTE)).toBe(false)
  })
})

// #127647 follow-up: the window-state bridge reports CHANGES only, and only
// the main window receives a seed push after load. A window created (or
// restored) already minimized therefore never pushes, and a controller seeded
// only by pushes believes it is shown — hidden animations keep running. The
// controller now pulls the current snapshot on first subscribe. A live push
// that already arrived beats the pull's snapshot, and a pull that never lands
// (an older preload) or fails leaves the push-only behaviour intact.

describe('createRendererLoopPauseController — initial window-state pull', () => {
  const flush = () => new Promise<void>(resolve => setTimeout(resolve, 0))

  afterEach(() => {
    Reflect.deleteProperty(window, 'hermesDesktop')
    vi.restoreAllMocks()
  })

  it('a window that started minimized pauses as soon as the pull lands', async () => {
    const bridge = installWindowStateBridge()
    const onChange = vi.fn()
    const controller = createRendererLoopPauseController(onChange)

    expect(controller.isPaused()).toBe(false)

    bridge.resolvePull({ isMinimized: true })
    await flush()

    expect(controller.isPaused()).toBe(true)
    expect(onChange).toHaveBeenCalled()
    controller.dispose()
  })

  it('a window that started invisible seeds as paused too', async () => {
    const bridge = installWindowStateBridge()
    const controller = createRendererLoopPauseController(() => undefined)

    bridge.resolvePull({ isVisible: false })
    await flush()

    expect(controller.isPaused()).toBe(true)
    controller.dispose()
  })

  it('a live push beats the pull when both land', async () => {
    const bridge = installWindowStateBridge()
    const onChange = vi.fn()
    const controller = createRendererLoopPauseController(onChange)

    bridge.emit({ isMinimized: true })
    bridge.emit({ isMinimized: false })
    onChange.mockClear()

    bridge.resolvePull({ isMinimized: true })
    await flush()

    expect(controller.isPaused()).toBe(false)
    expect(onChange).not.toHaveBeenCalled()
    controller.dispose()
  })

  it('a failing pull leaves the push-only behaviour intact', async () => {
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { getWindowState: vi.fn(() => Promise.reject(new Error('no snapshot'))) }
    })

    const controller = createRendererLoopPauseController(() => undefined)
    await flush()

    expect(controller.isPaused()).toBe(false)
    controller.dispose()
  })

  it('a pull that lands after dispose changes nothing', async () => {
    const bridge = installWindowStateBridge()
    const onChange = vi.fn()
    const controller = createRendererLoopPauseController(onChange)

    controller.dispose()
    bridge.resolvePull({ isMinimized: true })
    await flush()

    expect(onChange).not.toHaveBeenCalled()
  })
})
