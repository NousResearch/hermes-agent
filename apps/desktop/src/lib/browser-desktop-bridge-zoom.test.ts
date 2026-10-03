import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { installBrowserDesktopBridge } from './browser-desktop-bridge'
import { createBrowserZoom } from './browser-zoom'

type BootstrapWindow = Window & { __HERMES_SESSION_TOKEN__?: string }
const win = window as unknown as BootstrapWindow

beforeEach(() => {
  win.__HERMES_SESSION_TOKEN__ = 'served-token'
  localStorage.clear()
  vi.stubGlobal('devicePixelRatio', 1.5)
})

afterEach(() => {
  Reflect.deleteProperty(win, 'hermesDesktop')
  delete win.__HERMES_SESSION_TOKEN__
  document.documentElement.style.removeProperty('zoom')
  delete document.documentElement.dataset.hermesDesktopHost
  localStorage.clear()
  vi.unstubAllGlobals()
})

describe('browser UI scale', () => {
  it('reports app scale independently of display density and applies preset changes before notifying', async () => {
    expect(installBrowserDesktopBridge()).toBe(true)
    const zoom = win.hermesDesktop!.zoom!
    expect((await zoom.get()).percent).toBe(100)

    const onChanged = vi.fn(({ percent }: { percent: number }) => {
      expect(Number(document.documentElement.style.zoom)).toBe(percent / 100)
    })

    const stop = zoom.onChanged(onChanged)
    zoom.setPercent(150)

    expect((await zoom.get()).percent).toBe(150)
    expect(document.documentElement.style.zoom).toBe('1.5')
    expect(onChanged).toHaveBeenCalledWith(expect.objectContaining({ percent: 150 }))
    stop()
    zoom.setPercent(125)
    expect(onChanged).toHaveBeenCalledTimes(1)
  })

  it('restores app scale on reload without sharing it across hosted base paths', async () => {
    createBrowserZoom('/first').setPercent(150)
    const restored = createBrowserZoom('/first')
    expect((await restored.get()).percent).toBe(150)
    expect(document.documentElement.style.zoom).toBe('1.5')
    expect(restored.factor!()).toBe(1.5)
    expect((await createBrowserZoom('/second').get()).percent).toBe(100)
  })

  it('keeps scaling usable when storage is blocked and rejects invalid scale inputs', async () => {
    vi.spyOn(localStorage, 'getItem').mockImplementation(() => {
      throw new Error('storage denied')
    })
    vi.spyOn(localStorage, 'setItem').mockImplementation(() => {
      throw new Error('storage denied')
    })
    const zoom = createBrowserZoom('')
    const resized = vi.fn()
    window.addEventListener('resize', resized)
    zoom.setPercent(175)
    expect(document.documentElement.style.zoom).toBe('1.75')
    expect((await zoom.get()).percent).toBe(175)
    expect(resized).toHaveBeenCalledTimes(1)
    window.removeEventListener('resize', resized)
    zoom.setPercent(Number.NaN)
    expect((await zoom.get()).percent).toBe(100)
    zoom.setPercent(-10)
    expect((await zoom.get()).percent).toBe(100)
    vi.restoreAllMocks()
  })

  it('does not apply browser scale when the Electron bridge is installed', () => {
    const existing = { zoom: { setPercent: vi.fn() } } as unknown as Window['hermesDesktop']
    win.hermesDesktop = existing
    expect(installBrowserDesktopBridge()).toBe(false)
    expect(win.hermesDesktop).toBe(existing)
    expect(document.documentElement.style.zoom).toBe('')
  })
})
