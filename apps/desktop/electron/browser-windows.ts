// Popped-out in-app Browser windows. Same query-before-hash contract as
// session-windows / hud-url: `?win=browser` MUST sit in the search string
// before the '#', or HashRouter swallows it as part of the route.

import { pathToFileURL } from 'node:url'

import type { BrowserWindow, BrowserWindowConstructorOptions, IpcMain, IpcMainInvokeEvent, WebContents } from 'electron'

import { safeViewerUrl } from './plugin-viewer-policy'
import { createSessionWindowRegistry } from './session-windows'

export interface PluginViewerRequest {
  id: string
  url: string
  title: string
}

const safeSlug = (value: unknown): value is string =>
  typeof value === 'string' && /^[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}$/.test(value)

/** The browser popup's dimensions/registry, but remote content has NO app preload.
 * Electron has no per-window Linux WM_CLASS/app_id API. A locked title supplies
 * instance identity without changing the identity of every window in the process.
 */
export function createPluginViewerWindows(factory: (options: BrowserWindowConstructorOptions) => BrowserWindow) {
  const registry = createSessionWindowRegistry()
  const loadGenerations = new WeakMap<BrowserWindow, number>()
  const owned = new Map<string, { owner: number; plugin: string; id: string; origin: string; win: BrowserWindow }>()
  const keyFor = (owner: number, plugin: string, id: string) => JSON.stringify([owner, plugin, id])

  async function open(owner: number, plugin: string, raw: unknown): Promise<boolean> {
    if (!safeSlug(plugin) || !raw || typeof raw !== 'object') {
      return false
    }

    const input = raw as PluginViewerRequest

    if (
      Object.keys(input).some(key => !['id', 'url', 'title'].includes(key)) ||
      !safeSlug(input.id) ||
      !safeViewerUrl(input.url) ||
      typeof input.title !== 'string' ||
      input.title.length > 120 ||
      [...input.title].some(character => character.charCodeAt(0) < 32 || character.charCodeAt(0) === 127)
    ) {
      return false
    }

    const key = keyFor(owner, plugin, input.id)
    const title = `Hermes Viewer [${plugin}/${input.id}] — ${input.title}`
    let win = registry.get(key) as BrowserWindow | undefined

    if (!win || win.isDestroyed()) {
      win = registry.openOrFocus(key, () => {
        const created = factory({
          width: BROWSER_WINDOW_WIDTH,
          height: BROWSER_WINDOW_HEIGHT,
          minWidth: BROWSER_WINDOW_MIN_WIDTH,
          minHeight: BROWSER_WINDOW_MIN_HEIGHT,
          title,
          show: false,
          webPreferences: {
            nodeIntegration: false,
            contextIsolation: true,
            sandbox: true,
            webviewTag: false,
            focusOnNavigation: false,
            partition: `hermes-plugin-viewer:${owner}:${plugin}:${input.id}`
          }
        })

        created.setMenu(null)
        created.once('ready-to-show', () => {
          if (!created.isDestroyed()) {
            created.showInactive()
          }
        })
        created.on('page-title-updated', event => event.preventDefault())
        created.webContents.setWindowOpenHandler(() => ({ action: 'deny' }))
        const origin = new URL(input.url).origin

        const navigation = (event: { preventDefault(): void }, url: string) => {
          if (!safeViewerUrl(url) || new URL(url).origin !== origin) {
            event.preventDefault()
          }
        }

        created.webContents.on('will-navigate', navigation)
        created.webContents.on('will-redirect', navigation)
        const session = created.webContents.session
        session.setPermissionRequestHandler((_contents, _permission, callback) => callback(false))
        session.setPermissionCheckHandler(() => false)
        session.on('will-download', event => event.preventDefault())
        owned.set(key, { owner, plugin, id: input.id, origin, win: created })
        created.on('closed', () => {
          if (owned.get(key)?.win === created) {
            owned.delete(key)
          }
        })

        return created
      })
    }

    // Same id cannot retarget a live viewer to a different origin. Close it
    // explicitly before changing server, so its navigation policy stays exact.
    if (!win || owned.get(key)?.origin !== new URL(input.url).origin) {
      return false
    }

    win.setTitle(title)
    const generation = (loadGenerations.get(win) ?? 0) + 1
    loadGenerations.set(win, generation)

    try {
      await win.loadURL(input.url)

      return !win.isDestroyed()
    } catch {
      // A newer loadURL can abort this one while reusing the same window.
      if (loadGenerations.get(win) === generation && !win.isDestroyed()) {
        win.destroy()
      }

      return false
    }
  }

  function close(owner: number, plugin: string, id?: string): boolean {
    let closed = false

    for (const entry of [...owned.values()]) {
      if (entry.owner === owner && entry.plugin === plugin && (id === undefined || entry.id === id)) {
        if (!entry.win.isDestroyed()) {
          entry.win.destroy()
        }

        closed = true
      }
    }

    return closed
  }

  function closeOwner(owner: number) {
    for (const entry of [...owned.values()]) {
      if (entry.owner === owner) {
        close(owner, entry.plugin)
      }
    }
  }

  return { open, close, closeOwner }
}

export function registerPluginViewerIpc(
  ipc: Pick<IpcMain, 'handle'>,
  factory: (options: BrowserWindowConstructorOptions) => BrowserWindow,
  rendererUrl: () => string
) {
  const viewers = createPluginViewerWindows(factory)
  const owners = new WeakSet<WebContents>()

  const trusted = (event: IpcMainInvokeEvent) => {
    if (event.senderFrame !== event.sender.mainFrame) {
      return false
    }

    try {
      const source = new URL(event.senderFrame.url)
      const app = new URL(rendererUrl())

      return source.protocol === app.protocol && source.origin === app.origin && source.pathname === app.pathname
    } catch {
      return false
    }
  }

  ipc.handle('hermes:window:openPluginViewer', async (event, plugin, input) => {
    if (!trusted(event)) {
      return false
    }

    const sender = event.sender

    if (!owners.has(sender)) {
      owners.add(sender)
      sender.once('destroyed', () => viewers.closeOwner(sender.id))
      sender.on('render-process-gone', () => viewers.closeOwner(sender.id))
      sender.on('did-start-navigation', (_event, _url, inPlace, mainFrame) => {
        if (mainFrame && !inPlace) {
          viewers.closeOwner(sender.id)
        }
      })
    }

    return viewers.open(sender.id, plugin, input)
  })
  ipc.handle('hermes:window:closePluginViewer', (event, plugin, id) =>
    trusted(event) ? viewers.close(event.sender.id, plugin, id) : false
  )
}

export const BROWSER_WINDOW_WIDTH = 960
export const BROWSER_WINDOW_HEIGHT = 720
export const BROWSER_WINDOW_MIN_WIDTH = 480
export const BROWSER_WINDOW_MIN_HEIGHT = 400

/**
 * Renderer URL for a popped-out Browser. `tab` is the `$previewTabs` id the
 * window should show — the tab stays in storage so closing the window can
 * dock it again. Absent/blank tab is still a valid Browser window (blank page).
 */
export function buildBrowserWindowUrl(
  tabId: null | string | undefined,
  { devServer, rendererIndexPath }: { devServer?: null | string; rendererIndexPath?: string } = {}
): string {
  const tab = typeof tabId === 'string' ? tabId.trim() : ''
  const query = `?win=browser${tab ? `&tab=${encodeURIComponent(tab)}` : ''}`

  if (devServer) {
    const base = devServer.endsWith('/') ? devServer.slice(0, -1) : devServer

    return `${base}/${query}#/`
  }

  return `${pathToFileURL(rendererIndexPath!).toString()}${query}#/`
}
