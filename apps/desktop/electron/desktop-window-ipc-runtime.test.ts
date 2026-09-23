import { describe, expect, it, vi } from 'vitest'

import { registerDesktopWindowIpcRuntime } from './desktop-window-ipc-runtime'

function registerWindowIpc() {
  const handles = new Map<string, (...args: any[]) => any>()
  const events = new Map<string, (...args: any[]) => any>()
  const createSessionWindow = vi.fn()
  const resolveHermesBackend = vi.fn()
  const setAndPersistZoomLevel = vi.fn()
  const window = { isDestroyed: () => false, webContents: { getZoomLevel: () => 2 } }

  registerDesktopWindowIpcRuntime({
    BrowserWindow: { fromWebContents: () => window },
    DEFAULT_ZOOM_LEVEL: 0,
    HERMES_HOME: 'test-home',
    app: { getPath: () => 'test-user-data' },
    buildTerminalScript: vi.fn(),
    createBrowserWindow: vi.fn(),
    createInstanceWindow: vi.fn(),
    createSessionWindow,
    findOnPath: vi.fn(),
    ipcMain: {
      handle: (name: string, handler: (...args: any[]) => any) => handles.set(name, handler),
      on: (name: string, handler: (...args: any[]) => any) => events.set(name, handler)
    },
    percentToZoomLevel: (percent: number) => percent / 100,
    registerWindowControlIpc: vi.fn(),
    rememberLog: vi.fn(),
    resolveHermesBackend,
    resolveTerminalLaunch: vi.fn(),
    sanitizeWorkspaceCwd: vi.fn(),
    setAndPersistZoomLevel,
    terminalScriptEnv: vi.fn(),
    terminalScriptExtension: vi.fn(),
    tuiResumeArgs: vi.fn(),
    wakeIndicatorController: { getState: () => ({}), setState: vi.fn() },
    zoomLevelToPercent: (level: number) => level * 100
  })

  return { createSessionWindow, events, handles, resolveHermesBackend, setAndPersistZoomLevel }
}

describe('desktop window IPC registration', () => {
  it('validates session IDs before opening a native window', async () => {
    const runtime = registerWindowIpc()
    const open = runtime.handles.get('hermes:window:openSession')!

    expect(await open(null, ' ')).toEqual({ ok: false, error: 'invalid-session-id' })
    expect(runtime.createSessionWindow).not.toHaveBeenCalled()
    expect(await open(null, ' session ', { profile: 'work', watch: true })).toEqual({ ok: true })
    expect(runtime.createSessionWindow).toHaveBeenCalledWith('session', { profile: 'work', watch: true })
  })

  it('keeps terminal handoff and zoom routed through their original native channels', async () => {
    const runtime = registerWindowIpc()

    expect(await runtime.handles.get('hermes:window:openInTerminal')!(null, '')).toEqual({
      ok: false,
      error: 'invalid-session-id'
    })
    expect(runtime.resolveHermesBackend).not.toHaveBeenCalled()
    expect(runtime.handles.get('hermes:zoom:get')!({ sender: {} })).toEqual({ level: 2, percent: 200 })

    runtime.events.get('hermes:zoom:set-percent')!({ sender: {} }, 125)

    expect(runtime.setAndPersistZoomLevel).toHaveBeenCalledWith(expect.anything(), 1.25)
  })
})
