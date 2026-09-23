import path from 'node:path'

import { beforeEach, expect, test, vi } from 'vitest'

const native = vi.hoisted(() => ({
  app: { getPath: (name: string) => (name === 'downloads' ? 'C:/downloads' : 'C:/user-data') },
  downloadHandler: null as null | ((event: unknown, item: any) => void),
  permissionCheck: null as null | ((contents: unknown, permission: string) => boolean),
  permissionRequest: null as
    null | ((contents: unknown, permission: string, callback: (allowed: boolean) => void, details: unknown) => void)
}))

vi.mock('electron', () => ({
  app: native.app,
  Menu: { buildFromTemplate: (template: unknown) => template },
  session: {
    defaultSession: {
      on: (_event: string, handler: typeof native.downloadHandler) => {
        native.downloadHandler = handler
      },
      setPermissionCheckHandler: (handler: typeof native.permissionCheck) => {
        native.permissionCheck = handler
      },
      setPermissionRequestHandler: (handler: typeof native.permissionRequest) => {
        native.permissionRequest = handler
      }
    }
  }
}))

import { createDesktopNativeChromeRuntime } from './desktop-native-chrome-runtime'

function makeWindow() {
  const handlers = new Map<string, (...args: any[]) => void>()

  return {
    handlers,
    isDestroyed: () => false,
    webContents: {
      executeJavaScript: vi.fn(() => Promise.resolve()),
      getZoomLevel: () => 0,
      isDestroyed: () => false,
      on: (event: string, handler: (...args: any[]) => void) => handlers.set(event, handler),
      send: vi.fn(),
      setZoomLevel: vi.fn()
    }
  }
}

function makeRuntime(options: { isMac?: boolean } = {}) {
  let mainWindow: ReturnType<typeof makeWindow> | null = null
  let hudWindow: ReturnType<typeof makeWindow> | null = null
  let createInstanceWindow = vi.fn()
  const closeHudWindow = vi.fn()
  const sendClosePreviewRequested = vi.fn()
  const writeZoomState = vi.fn()

  const runtime = createDesktopNativeChromeRuntime({
    APP_NAME: 'Hermes',
    IS_MAC: options.isMac ?? true,
    closeHudWindow,
    extensionForMimeType: () => '.png',
    getCreateInstanceWindow: () => createInstanceWindow,
    getF12Blocked: () => false,
    getHudWindow: () => hudWindow as never,
    getMainWindow: () => mainWindow as never,
    readZoomState: () => null,
    rememberLog: vi.fn(),
    sendClosePreviewRequested,
    sendOpenFolderRequested: vi.fn(),
    sendOpenUpdatesRequested: vi.fn(),
    sendPreviewNavCommand: vi.fn(),
    showAboutPanelFresh: vi.fn(),
    writeZoomState
  })

  return {
    closeHudWindow,
    runtime,
    sendClosePreviewRequested,
    setCreateInstanceWindow: (next: typeof createInstanceWindow) => (createInstanceWindow = next),
    setHudWindow: (next: ReturnType<typeof makeWindow>) => (hudWindow = next),
    setMainWindow: (next: ReturnType<typeof makeWindow>) => (mainWindow = next),
    writeZoomState
  }
}

beforeEach(() => {
  native.downloadHandler = null
  native.permissionCheck = null
  native.permissionRequest = null
})

test('menu actions resolve late-created windows when the user clicks', () => {
  const { runtime, setCreateInstanceWindow, setMainWindow, writeZoomState } = makeRuntime()
  const window = makeWindow()
  const openPeer = vi.fn()
  setCreateInstanceWindow(openPeer)
  setMainWindow(window)

  const menu = runtime.buildApplicationMenu() as unknown as any[]
  const file = menu.find(item => item.label === 'File')
  const view = menu.find(item => item.label === 'View')
  file.submenu.find((item: any) => item.label === 'New Window').click()
  view.submenu.find((item: any) => item.label === 'Actual Size').click()

  expect(openPeer).toHaveBeenCalledTimes(1)
  expect(writeZoomState).toHaveBeenCalledTimes(1)
  expect(window.webContents.executeJavaScript).toHaveBeenCalledTimes(1)
})

test('Command-W leaves HUD through its own close path and closes ordinary tabs through the renderer', () => {
  const { closeHudWindow, runtime, sendClosePreviewRequested, setHudWindow } = makeRuntime()
  const hud = makeWindow()
  const normal = makeWindow()
  setHudWindow(hud)
  runtime.installPreviewShortcut(hud as never)
  runtime.installPreviewShortcut(normal as never)

  const event = { preventDefault: vi.fn() }
  const input = { alt: false, key: 'w', meta: true, shift: false }
  hud.handlers.get('before-input-event')?.(event, input)
  normal.handlers.get('before-input-event')?.(event, input)

  expect(closeHudWindow).toHaveBeenCalledTimes(1)
  expect(sendClosePreviewRequested).toHaveBeenCalledTimes(1)
  expect(event.preventDefault).toHaveBeenCalledTimes(2)
})

test('media permission request and check paths admit capture without Windows mediaTypes metadata', () => {
  const { runtime } = makeRuntime()
  runtime.installMediaPermissions()
  const reply = vi.fn()

  native.permissionRequest?.({}, 'media', reply, { mediaTypes: [] })
  expect(reply).toHaveBeenCalledWith(true)
  expect(native.permissionCheck?.({}, 'media')).toBe(true)
  expect(native.permissionCheck?.({}, 'geolocation')).toBe(false)
})

test('Chromium downloads default to Downloads with a MIME extension', () => {
  const { runtime } = makeRuntime()
  runtime.installDownloadHandling()

  const item = {
    getFilename: () => 'image',
    getMimeType: () => 'image/png',
    setSaveDialogOptions: vi.fn()
  }

  native.downloadHandler?.({}, item)

  expect(item.setSaveDialogOptions).toHaveBeenCalledWith(
    expect.objectContaining({ defaultPath: path.join('C:/downloads', 'image.png'), title: 'Save File' })
  )
})
