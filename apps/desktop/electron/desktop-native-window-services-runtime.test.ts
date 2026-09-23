import { describe, expect, it, vi } from 'vitest'

import { createDesktopNativeWindowServicesRuntime } from './desktop-native-window-services-runtime'

describe('desktop native window services', () => {
  it('registers microphone and below-window metadata at the late IPC seam', async () => {
    const handlers = new Map<string, (...args: any[]) => any>()
    const readWindowBelow = vi.fn(() => ({ app: 'Neighbor' }))
    const askForMediaAccess = vi.fn(async () => true)

    const window = {
      getPosition: () => [10, 20],
      getSize: () => [300, 400],
      isDestroyed: () => false
    }

    const runtime = createDesktopNativeWindowServicesRuntime({
      APP_ICON_PATHS: ['icon.png'],
      BrowserWindow: { fromWebContents: () => window },
      DESKTOP_WINDOW_STATE_PATH: 'test-window-state.json',
      IS_MAC: true,
      app: { getPath: () => 'test-user-data' },
      debounce: (callback: any) => callback,
      getMainWindow: () => window as any,
      ipcMain: { handle: (name: string, handler: (...args: any[]) => any) => handlers.set(name, handler) },
      readWindowBelow,
      rememberLog: vi.fn(),
      resolveAppIcon: () => 'icon.png',
      sanitizeWindowState: value => value,
      systemPreferences: { askForMediaAccess, getMediaAccessStatus: () => 'denied' },
      writeFileAtomic: vi.fn()
    })

    expect(runtime.getAppIconPath()).toBe('icon.png')
    expect(handlers.size).toBe(0)

    runtime.registerNativeWindowServicesIpc()

    expect(await handlers.get('hermes:requestMicrophoneAccess')!()).toBe(true)
    expect(askForMediaAccess).toHaveBeenCalledWith('microphone')
    expect(await handlers.get('hermes:window:readBelow')!({ sender: {} })).toEqual({ app: 'Neighbor' })
    expect(readWindowBelow).toHaveBeenCalledWith(process.pid, { x: 10, y: 20, width: 300, height: 400 }, false)
  })
})
