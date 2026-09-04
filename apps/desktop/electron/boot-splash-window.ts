// Pre-renderer boot splash window (#102419).
//
// The main window is created `show: false` and only appears once its renderer
// has painted its first themed frame (wireWindowReveal). On macOS first
// launch Chromium's system-keychain walk can delay that first paint by
// minutes, so the user would otherwise stare at the Dock's frozen "Opening…"
// with no feedback. See boot-splash-html.ts for the full rationale.
//
// This module owns the tiny frameless splash BrowserWindow:
//
//  1. createBootSplashWindow() opens a HIDDEN splash before the main window
//     even starts its renderer load, so its own frame paints first and is
//     ready to show instantly later.
//  2. gateBootSplash() waits BOOT_SPLASH_SHOW_AFTER_MS: if the main window is
//     visible by then (a normal launch) the splash closes silently — no flash;
//     otherwise the splash takes over the screen and its status line is fed
//     live from the existing boot-progress state until the main window
//     finally shows, then it closes for good.
//
// Skipped under Playwright (TEST_WORKER_INDEX) in main.ts because the reveal
// is forced there and an extra window would confuse the suite.

import { BrowserWindow } from 'electron'

import {
  BOOT_SPLASH_SHOW_AFTER_MS,
  BOOT_SPLASH_WATCH_MS,
  bootSplashStatusScript,
  buildBootSplashHtml,
  type BootSplashMeta
} from './boot-splash-html'

function splashDataUrl(meta: BootSplashMeta): string {
  return `data:text/html;charset=utf-8,${encodeURIComponent(buildBootSplashHtml(meta))}`
}

export function createBootSplashWindow(meta: BootSplashMeta): BrowserWindow {
  const splash = new BrowserWindow({
    width: 470,
    height: 210,
    show: false,
    frame: false,
    resizable: false,
    minimizable: false,
    maximizable: false,
    fullscreenable: false,
    skipTaskbar: true,
    alwaysOnTop: true,
    center: true,
    title: 'Hermes',
    backgroundColor: '#0e1116',
    // Deliberately no preload and no node integration: this page is static
    // HTML updated only by the main process via executeJavaScript.
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      devTools: false,
      spellcheck: false
    }
  })

  splash.loadURL(splashDataUrl(meta)).catch(() => {
    // A data: URL load failing is essentially impossible; if it ever does,
    // the splash stays hidden and startup proceeds exactly as before.
  })

  return splash
}

export interface BootSplashGateOptions {
  splash: BrowserWindow
  getMainWindow: () => BrowserWindow | null
  getStatusMessage: () => string
}

/**
 * Reveal gate described above. Timers are the only moving parts; they are
 * cancelled on dispose so a closed splash never leaks a tick.
 */
export function gateBootSplash({ splash, getMainWindow, getStatusMessage }: BootSplashGateOptions): () => void {
  let disposed = false
  let shown = false
  let lastStatus: string | null = null

  const dispose = () => {
    if (disposed) {
      return
    }

    disposed = true
    clearTimeout(showTimer)
    clearInterval(watchTimer)
  }

  const splashClosed = () => {
    // Covers manual close (e.g. Cmd+W on macOS) and app quit.
    dispose()
  }

  const pushStatus = () => {
    if (disposed || shown === false || splash.isDestroyed() || splash.webContents.isLoadingMainFrame()) {
      return
    }

    const message = getStatusMessage()

    if (message !== lastStatus) {
      lastStatus = message
      void splash.webContents
        .executeJavaScript(bootSplashStatusScript(message))
        .catch(() => {
          // The page can be torn down between the isDestroyed check and the
          // eval landing; the next watch tick catches up or gives up.
        })
    }
  }

  const showTimer = setTimeout(() => {
    if (disposed) {
      return
    }

    const mainWindow = getMainWindow()

    // Normal launch won the race — the splash was never needed.
    if (mainWindow && !mainWindow.isDestroyed() && mainWindow.isVisible()) {
      dispose()
      splash.close()
      return
    }

    if (splash.isDestroyed()) {
      dispose()
      return
    }

    shown = true
    splash.show()
    pushStatus()
  }, BOOT_SPLASH_SHOW_AFTER_MS)

  const watchTimer = setInterval(() => {
    if (disposed) {
      return
    }

    const mainWindow = getMainWindow()

    // The main window appeared (or was destroyed, e.g. crash-relaunch) —
    // hand over to its own boot UI and get out of the way.
    if (mainWindow && !mainWindow.isDestroyed() && mainWindow.isVisible()) {
      dispose()
      splash.close()
      return
    }

    pushStatus()
  }, BOOT_SPLASH_WATCH_MS)

  splash.once('closed', splashClosed)

  return dispose
}
