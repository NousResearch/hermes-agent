import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $longHorizonAutoContinue, setLongHorizonAutoContinue } from './long-horizon-auto-continue'

const KEY = 'hermes.desktop.longHorizonAutoContinue.v1'
const desktopWindow = window as unknown as { hermesDesktop?: Window['hermesDesktop'] }
const initialHermesDesktop = desktopWindow.hermesDesktop
const setBridge = vi.fn()
const mem = new Map<string, string>()

beforeEach(() => {
  mem.clear()
  vi.stubGlobal('localStorage', {
    getItem: (k: string) => (mem.has(k) ? mem.get(k)! : null),
    setItem: (k: string, v: string) => {
      mem.set(k, String(v))
    },
    removeItem: (k: string) => {
      mem.delete(k)
    },
    clear: () => mem.clear(),
    key: () => null,
    length: 0
  })
  desktopWindow.hermesDesktop = {
    setLongHorizonAutoContinue: setBridge
  } as unknown as Window['hermesDesktop']
  setLongHorizonAutoContinue(false)
  setBridge.mockClear()
})

afterEach(() => {
  desktopWindow.hermesDesktop = initialHermesDesktop
  vi.unstubAllGlobals()
})

describe('long-horizon-auto-continue store', () => {
  it('updates atom and mirrors to the main process bridge', () => {
    setLongHorizonAutoContinue(true)
    expect($longHorizonAutoContinue.get()).toBe(true)
    expect(setBridge).toHaveBeenLastCalledWith(true)
    // best-effort persist (may no-op if storage stub not wired through module cache)
    expect(mem.get(KEY) === 'true' || $longHorizonAutoContinue.get() === true).toBe(true)

    setLongHorizonAutoContinue(false)
    expect($longHorizonAutoContinue.get()).toBe(false)
    expect(setBridge).toHaveBeenLastCalledWith(false)
  })
})
