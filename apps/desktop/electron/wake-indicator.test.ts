import { beforeEach, describe, expect, it, vi } from 'vitest'

import { normalizeWakeIndicatorState, selectWakeIndicatorDisplay, wakeIndicatorWindowBounds } from './wake-indicator'
import { createWakeIndicatorWindowController } from './wake-indicator-window'

const electronMock = vi.hoisted(() => {
  type Listener = () => void

  class FakeBrowserWindow {
    private destroyed = false
    private readonly listeners = new Map<string, Listener[]>()

    readonly close = vi.fn(() => {
      this.destroyed = true
      this.emit('closed')
    })
    readonly hide = vi.fn()
    readonly setAlwaysOnTop = vi.fn()
    readonly setBounds = vi.fn()
    readonly setHiddenInMissionControl = vi.fn()
    readonly setIgnoreMouseEvents = vi.fn()
    readonly setVisibleOnAllWorkspaces = vi.fn()
    readonly showInactive = vi.fn()
    readonly webContents = {
      on: vi.fn(),
      send: vi.fn()
    }

    constructor(readonly options: unknown) {
      windows.push(this)
    }

    isDestroyed() {
      return this.destroyed
    }

    on(event: string, listener: Listener) {
      const listeners = this.listeners.get(event) ?? []
      listeners.push(listener)
      this.listeners.set(event, listeners)

      return this
    }

    once(event: string, listener: Listener) {
      return this.on(event, listener)
    }

    private emit(event: string) {
      for (const listener of this.listeners.get(event) ?? []) {
        listener()
      }
    }
  }

  const windows: FakeBrowserWindow[] = []

  const display = {
    bounds: { height: 982, width: 1512, x: 0, y: 0 },
    id: 'internal',
    internal: true
  }

  return {
    BrowserWindow: FakeBrowserWindow,
    screen: {
      getAllDisplays: () => [display],
      getPrimaryDisplay: () => display
    },
    windows
  }
})

vi.mock('electron', () => ({
  BrowserWindow: electronMock.BrowserWindow,
  screen: electronMock.screen
}))

beforeEach(() => {
  electronMock.windows.length = 0
})

describe('wake indicator window', () => {
  it('centers the helper window at the top of the selected display', () => {
    expect(
      wakeIndicatorWindowBounds({
        bounds: { height: 982, width: 1512, x: -120, y: 40 }
      })
    ).toEqual({
      height: 52,
      width: 176,
      x: 548,
      y: 40
    })
  })

  it('prefers the internal display and falls back to the primary display', () => {
    const primary = {
      bounds: { height: 1080, width: 1920, x: 0, y: 0 },
      id: 'primary',
      internal: false
    }

    const internal = {
      bounds: { height: 982, width: 1512, x: 1920, y: 0 },
      id: 'internal',
      internal: true
    }

    expect(selectWakeIndicatorDisplay([primary, internal], primary)).toBe(internal)

    expect(selectWakeIndicatorDisplay([primary], primary)).toBe(primary)
  })

  it('rejects unknown renderer states', () => {
    expect(normalizeWakeIndicatorState('capturing')).toBe('capturing')
    expect(normalizeWakeIndicatorState('other')).toBe('hidden')
    expect(normalizeWakeIndicatorState(null)).toBe('hidden')
  })
})

describe('wake indicator window controller', () => {
  it('closes an active helper window and resets its state', () => {
    const controller = createWakeIndicatorWindowController({
      isMac: true,
      loadWindowUrl: vi.fn(),
      log: () => {},
      preloadPath: '/tmp/preload.cjs',
      rendererIndex: () => '/tmp/index.html',
      wireWindow: vi.fn()
    })

    controller.setState('detected')

    expect(electronMock.windows).toHaveLength(1)
    expect(controller.getState()).toBe('detected')

    const [window] = electronMock.windows
    controller.close()

    expect(window.close).toHaveBeenCalledOnce()
    expect(controller.getState()).toBe('hidden')
  })
})
