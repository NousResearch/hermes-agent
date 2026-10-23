import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { host } from '@/sdk'
import { $translucency, setTranslucencyMode } from '@/store/translucency'

const installBridge = (bridge: Record<string, unknown>): void => {
  Object.defineProperty(window, 'hermesDesktop', { value: bridge, configurable: true, writable: true })
}

const removeBridge = (): void => {
  Reflect.deleteProperty(window as unknown as Record<string, unknown>, 'hermesDesktop')
}

describe('host.window', () => {
  afterEach(() => {
    removeBridge()
    vi.restoreAllMocks()
  })

  describe('without a desktop bridge', () => {
    beforeEach(removeBridge)

    it('reports the window channels unavailable instead of throwing', async () => {
      expect(host.window.zoom.available()).toBe(false)
      expect(await host.window.zoom.get()).toBeNull()
      expect(() => host.window.zoom.setPercent(125)).not.toThrow()
      expect(host.window.zoom.subscribe(() => {})).toBeTypeOf('function')

      expect(host.window.minimizeToTray.available()).toBe(false)
      expect(await host.window.minimizeToTray.get()).toBeNull()
      expect(await host.window.minimizeToTray.set(true)).toBeNull()
      expect(host.window.minimizeToTray.subscribe(() => {})).toBeTypeOf('function')
    })

    it('still serves translucency, which is renderer-owned', () => {
      const state = host.window.translucency.get()

      expect(state).toHaveProperty('mode')
      expect(state).toHaveProperty('intensity')
    })
  })

  describe('with a desktop bridge', () => {
    const zoomGet = vi.fn(async () => ({ level: 1, percent: 125 }))
    const zoomSetPercent = vi.fn()
    const zoomDispose = vi.fn()
    let zoomListener: ((payload: { level: number; percent: number }) => void) | undefined

    const trayGet = vi.fn(async () => ({ enabled: false, available: true }))
    const traySet = vi.fn(async (on: boolean) => ({ enabled: on, available: true }))
    const trayDispose = vi.fn()
    let trayListener: ((status: { enabled: boolean; available: boolean }) => void) | undefined

    beforeEach(() => {
      zoomListener = undefined
      trayListener = undefined
      zoomGet.mockClear()
      zoomSetPercent.mockClear()
      zoomDispose.mockClear()
      trayGet.mockClear()
      traySet.mockClear()
      trayDispose.mockClear()

      installBridge({
        zoom: {
          get: zoomGet,
          setPercent: zoomSetPercent,
          onChanged: (cb: typeof zoomListener) => {
            zoomListener = cb

            return zoomDispose
          }
        },
        minimizeToTray: {
          get: trayGet,
          set: traySet,
          onChanged: (cb: typeof trayListener) => {
            trayListener = cb

            return trayDispose
          }
        }
      })
    })

    it('forwards zoom reads, writes and changes', async () => {
      expect(host.window.zoom.available()).toBe(true)
      expect(await host.window.zoom.get()).toEqual({ level: 1, percent: 125 })

      host.window.zoom.setPercent(150)
      expect(zoomSetPercent).toHaveBeenCalledWith(150)

      const listener = vi.fn()
      const dispose = host.window.zoom.subscribe(listener)

      zoomListener?.({ level: 2, percent: 150 })
      expect(listener).toHaveBeenLastCalledWith({ level: 2, percent: 150 })

      dispose()
      expect(zoomDispose).toHaveBeenCalled()
    })

    it('forwards tray reads, writes and changes', async () => {
      expect(host.window.minimizeToTray.available()).toBe(true)
      expect(await host.window.minimizeToTray.get()).toEqual({ enabled: false, available: true })

      await expect(host.window.minimizeToTray.set(true)).resolves.toEqual({ enabled: true, available: true })
      expect(traySet).toHaveBeenCalledWith(true)

      const listener = vi.fn()
      const dispose = host.window.minimizeToTray.subscribe(listener)

      trayListener?.({ enabled: true, available: true })
      expect(listener).toHaveBeenLastCalledWith({ enabled: true, available: true })

      dispose()
      expect(trayDispose).toHaveBeenCalled()
    })

    it('keeps translucency on the renderer store, not the bridge', () => {
      const before = $translucency.get()

      host.window.translucency.set({ intensity: 40 })
      expect($translucency.get().intensity).toBe(40)
      expect($translucency.get().mode).toBe(before.mode)

      host.window.translucency.set({ fade: 10 })
      expect($translucency.get().fade).toBe(10)
      expect($translucency.get().intensity).toBe(40)

      const listener = vi.fn()
      const dispose = host.window.translucency.subscribe(listener)

      expect(listener).toHaveBeenCalled()

      host.window.translucency.set({ intensity: 20 })
      expect(listener).toHaveBeenCalledWith(expect.objectContaining({ intensity: 20 }))

      dispose()
      setTranslucencyMode('clear')
    })
  })
})
