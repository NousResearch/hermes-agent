// Popped-out canvas windows: one per canvas provider (pen today), hosting the
// same provider pane the docked tile renders, in its own OS window. Same
// query-before-hash contract as browser-windows: `?win=canvas` MUST sit in the
// search string before the '#', or HashRouter swallows it as part of the route.

import { pathToFileURL } from 'node:url'

import type { BrowserWindow, IpcMain } from 'electron'

export const CANVAS_WINDOW_SIZE = { width: 1200, height: 800, minWidth: 640, minHeight: 480 }

/** The docked tab, verbatim — the window is just another host for it. */
export interface CanvasWindowTab {
  provider: string
  docId: string
  title: string
  url: string
}

export function isCanvasWindowTab(value: unknown): value is CanvasWindowTab {
  if (!value || typeof value !== 'object') {
    return false
  }

  const tab = value as Record<string, unknown>

  return (
    typeof tab.provider === 'string' &&
    tab.provider.trim() !== '' &&
    typeof tab.docId === 'string' &&
    typeof tab.title === 'string' &&
    typeof tab.url === 'string'
  )
}

/**
 * Renderer URL for a popped-out canvas. The tab rides the query so the new
 * renderer can seat it without asking anyone — it has no layout tree and no
 * `$canvasTabs` of its own until it reads this.
 */
export function buildCanvasWindowUrl(
  tab: CanvasWindowTab,
  { devServer, rendererIndexPath }: { devServer?: null | string; rendererIndexPath?: string } = {}
): string {
  const params = new URLSearchParams({
    win: 'canvas',
    provider: tab.provider,
    doc: tab.docId,
    title: tab.title,
    url: tab.url
  })

  const query = `?${params.toString()}`

  if (devServer) {
    const base = devServer.endsWith('/') ? devServer.slice(0, -1) : devServer

    return `${base}/${query}#/`
  }

  return `${pathToFileURL(rendererIndexPath!).toString()}${query}#/`
}

interface CanvasWindowRegistry {
  get: (key: string) => BrowserWindow | null | undefined
  openOrFocus: (key: string, spawn: () => BrowserWindow) => BrowserWindow | null | undefined
}

/**
 * One canvas window per provider. Re-opening while it is up focuses it and
 * hands it the tab (the live document can change under a popped canvas — an
 * agent opens another .pen, the user switches chats), so the window follows
 * instead of a second guest appearing in the docked tree. `spawn` builds the
 * window (main owns window chrome); this owns the registry and the IPC doors.
 */
export function wireCanvasWindows({
  ipcMain,
  registry,
  spawn
}: {
  ipcMain: IpcMain
  registry: CanvasWindowRegistry
  spawn: (tab: CanvasWindowTab) => BrowserWindow
}): void {
  ipcMain.handle('hermes:window:openCanvas', async (_event, tab) => {
    if (!isCanvasWindowTab(tab)) {
      return { ok: false, error: 'invalid-canvas-tab' }
    }

    const existing = registry.get(tab.provider)
    const win = registry.openOrFocus(tab.provider, () => spawn(tab))

    if (win && win === existing && !win.isDestroyed()) {
      win.webContents.send('hermes:canvas-popout:tab', tab)
    }

    return { ok: true }
  })

  ipcMain.handle('hermes:window:closeCanvas', async (_event, provider) => {
    const win = typeof provider === 'string' && provider.trim() ? registry.get(provider.trim()) : null

    if (win && !win.isDestroyed()) {
      win.close()
    }
  })
}
