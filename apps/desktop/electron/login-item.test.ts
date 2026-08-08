import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { registerLoginItemHandlers } from './login-item'

type LoginItemSettings = { openAtLogin?: boolean; openAsHidden?: boolean }

describe('login-item IPC handlers', () => {
  const handlers = new Map<string, (...args: any[]) => unknown>()
  const getLoginItemSettings = vi.fn<() => LoginItemSettings>(() => ({ openAtLogin: false, openAsHidden: false }))
  const setLoginItemSettings = vi.fn()
  const ipcMain = {
    handle: vi.fn((channel: string, handler: (...args: any[]) => unknown) => {
      handlers.set(channel, handler)
    })
  }
  const app = { getLoginItemSettings, setLoginItemSettings } as any
  const originalPlatform = process.platform

  beforeEach(() => {
    handlers.clear()
    getLoginItemSettings.mockReset()
    getLoginItemSettings.mockReturnValue({ openAtLogin: false, openAsHidden: false })
    setLoginItemSettings.mockReset()
    registerLoginItemHandlers(app, ipcMain as any)
  })

  afterEach(() => {
    Object.defineProperty(process, 'platform', { value: originalPlatform })
  })

  function setPlatform(platform: NodeJS.Platform) {
    Object.defineProperty(process, 'platform', { value: platform })
  }

  it('returns the Electron login-item state on a supported platform', () => {
    setPlatform('win32')
    registerLoginItemHandlers(app, ipcMain as any)
    getLoginItemSettings.mockReturnValue({ openAtLogin: true, openAsHidden: false })

    expect(handlers.get('hermes:login-item:get')?.()).toEqual({ openAtLogin: true, supported: true })
    expect(getLoginItemSettings).toHaveBeenCalledOnce()
  })

  it('sets openAtLogin and the current executable path on a supported platform', () => {
    setPlatform('win32')
    registerLoginItemHandlers(app, ipcMain as any)

    handlers.get('hermes:login-item:set')?.({}, { openAtLogin: true })

    expect(setLoginItemSettings).toHaveBeenCalledWith({
      openAtLogin: true,
      openAsHidden: false,
      path: process.execPath,
      args: process.defaultApp && process.argv[1] ? [process.argv[1]] : []
    })
  })

  it('forwards openAsHidden when provided and tolerates an empty Electron state', () => {
    setPlatform('darwin')
    registerLoginItemHandlers(app, ipcMain as any)
    getLoginItemSettings.mockReturnValue({})

    expect(handlers.get('hermes:login-item:get')?.()).toEqual({ openAtLogin: undefined, supported: true })
    expect(() => handlers.get('hermes:login-item:set')?.({}, { openAtLogin: false, openAsHidden: true })).not.toThrow()
    expect(setLoginItemSettings).toHaveBeenCalledWith(expect.objectContaining({ openAsHidden: true }))
  })

  it('reports supported:false and never touches Electron on Linux', () => {
    setPlatform('linux')
    registerLoginItemHandlers(app, ipcMain as any)

    expect(handlers.get('hermes:login-item:get')?.()).toEqual({ openAtLogin: false, supported: false })
    expect(handlers.get('hermes:login-item:set')?.({}, { openAtLogin: true })).toEqual({
      openAtLogin: false,
      supported: false
    })
    expect(getLoginItemSettings).not.toHaveBeenCalled()
    expect(setLoginItemSettings).not.toHaveBeenCalled()
  })

  it('returns Electron authoritative state from set, not the requested value', () => {
    setPlatform('win32')
    registerLoginItemHandlers(app, ipcMain as any)
    // A write that did not land (e.g. policy-blocked) must not be echoed back.
    getLoginItemSettings.mockReturnValue({ openAtLogin: false, openAsHidden: false })

    const result = handlers.get('hermes:login-item:set')?.({}, { openAtLogin: true })

    expect(setLoginItemSettings).toHaveBeenCalled()
    expect(result).toEqual({ openAtLogin: false, supported: true })
  })
})
