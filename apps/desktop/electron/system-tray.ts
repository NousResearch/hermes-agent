import fs from 'node:fs'
import path from 'node:path'

import {
  app,
  BrowserWindow,
  Menu,
  nativeImage,
  Tray
} from 'electron'

import { appIconCandidates, resolveAppIcon } from './app-icon'

// ── Module state ────────────────────────────────────────────────
let tray: Tray | null = null

/** Whether the tray has been initialized this session. */
let initialized = false

/** The main window reference, set by the integration call. */
let mainWindowRef: BrowserWindow | null = null

// ── Helpers ─────────────────────────────────────────────────────

/**
 * Resolve a 16×16 or 22×22 tray icon from the app's icon candidates.
 * Falls back to a 16×16 empty image so the tray still appears.
 */
function resolveTrayIcon(): Electron.NativeImage {
  const candidates = appIconCandidates()
  const iconPath = resolveAppIcon(candidates, '16x16')

  if (iconPath) {
    try {
      const img = nativeImage.createFromPath(iconPath)
      if (!img.isEmpty()) return img
    } catch {
      // fall through
    }
  }

  // Generate a 16×16 image from a larger icon
  const largeIcon = resolveAppIcon(candidates)
  if (largeIcon) {
    try {
      const img = nativeImage.createFromPath(largeIcon)
      if (!img.isEmpty()) return img.resize({ width: 16, height: 16 })
    } catch {
      // fall through
    }
  }

  // Last resort: blank 16×16
  return nativeImage.createEmpty()
}

// ── Context menu ────────────────────────────────────────────────

function buildContextMenu(): Electron.Menu {
  return Menu.buildFromTemplate([
    {
      label: 'Show Hermes',
      click: () => {
        if (mainWindowRef && !mainWindowRef.isDestroyed()) {
          if (mainWindowRef.isMinimized()) mainWindowRef.restore()
          mainWindowRef.show()
          mainWindowRef.focus()
        }
      }
    },
    { type: 'separator' },
    {
      label: 'Quit Hermes',
      click: () => {
        app.quit()
      }
    }
  ])
}

// ── Public API ──────────────────────────────────────────────────

/**
 * Initialise the system tray icon and its context menu.
 * Safe to call multiple times — only the first call takes effect.
 *
 * @param mainWindow - The primary BrowserWindow to show/hide via the tray.
 */
export function initSystemTray(mainWindow: BrowserWindow | null): void {
  if (initialized) return
  initialized = true

  mainWindowRef = mainWindow

  // Don't create a tray in tests or headless environments
  if (app.isPackaged === false && process.env.ELECTRON_RUN_AS_NODE) return
  if (!app.requestSingleInstanceLock?.()) return

  try {
    const icon = resolveTrayIcon()
    tray = new Tray(icon)
    tray.setToolTip('Hermes')
    tray.setContextMenu(buildContextMenu())

    // Left-click (or single-click on some platforms) shows the window
    tray.on('click', () => {
      if (mainWindowRef && !mainWindowRef.isDestroyed()) {
        if (mainWindowRef.isMinimized()) mainWindowRef.restore()
        if (mainWindowRef.isVisible()) {
          mainWindowRef.focus()
        } else {
          mainWindowRef.show()
          mainWindowRef.focus()
        }
      }
    })

    tray.on('double-click', () => {
      if (mainWindowRef && !mainWindowRef.isDestroyed()) {
        if (mainWindowRef.isMinimized()) mainWindowRef.restore()
        mainWindowRef.show()
        mainWindowRef.focus()
      }
    })

    // Update icon when the app icon changes (theme-aware)
    app.on('will-quit', () => {
      destroySystemTray()
    })
  } catch {
    // Tray creation failed silently (possible in non-DBus Linux envs)
    tray = null
  }
}

/**
 * Destroy the system tray. Safe to call when none exists.
 */
export function destroySystemTray(): void {
  if (tray) {
    tray.destroy()
    tray = null
  }
  initialized = false
  mainWindowRef = null
}

/**
 * Update the tray icon (e.g. after a theme change).
 */
export function updateTrayIcon(): void {
  if (!tray) return
  try {
    tray.setImage(resolveTrayIcon())
  } catch {
    // ignore
  }
}

/**
 * Whether the system tray is active.
 */
export function hasSystemTray(): boolean {
  return tray !== null
}