// System-tray support: keep Hermes alive in the tray when the main window
// closes, instead of quitting. The enabled flag persists to a tiny JSON file
// in userData so the choice survives restarts; the module is pure state +
// Electron calls so main.ts only wires IPC and the close handler.

import * as fs from 'node:fs'
import * as path from 'node:path'

import { app, Menu, nativeImage, Tray } from 'electron'

let tray: Tray | null = null
let enabled = false
let onShow: (() => void) | null = null
let onQuit: (() => void) | null = null

function prefsPath(): string {
  return path.join(app.getPath('userData'), 'tray-prefs.json')
}

export function loadTrayPrefs(): boolean {
  try {
    const raw = fs.readFileSync(prefsPath(), 'utf8')
    const parsed = JSON.parse(raw)
    enabled = parsed?.closeToTray === true
  } catch {
    enabled = false
  }

  return enabled
}

function persist(): void {
  try {
    fs.writeFileSync(prefsPath(), JSON.stringify({ closeToTray: enabled }))
  } catch {
    // non-fatal: tray just won't remember the choice this run
  }
}

export function isTrayEnabled(): boolean {
  return enabled
}

function iconPath(): string {
  // Packaged: resources/assets; dev: apps/desktop/assets.
  const candidates = [
    path.join(process.resourcesPath ?? '', 'assets', 'icon.png'),
    path.join(app.getAppPath(), 'assets', 'icon.png'),
    path.join(app.getAppPath(), '..', 'assets', 'icon.png')
  ]

  for (const candidate of candidates) {
    if (candidate && fs.existsSync(candidate)) {
      return candidate
    }
  }

  return candidates[1]
}

function ensureTray(): void {
  if (tray) {
    return
  }
  if (!app.isReady()) {
    // Tray before app-ready throws and would take down the main process;
    // callers on startup paths must defer via app.whenReady().
    throw new Error('ensureTray called before app ready')
  }

  const image = nativeImage.createFromPath(iconPath())
  tray = new Tray(image.isEmpty() ? nativeImage.createEmpty() : image)
  tray.setToolTip('Hermes')
  tray.setContextMenu(
    Menu.buildFromTemplate([
      { label: 'Open Hermes', click: () => onShow?.() },
      { type: 'separator' },
      { label: 'Quit', click: () => onQuit?.() }
    ])
  )
  tray.on('click', () => onShow?.())
}

export function setTrayEnabled(
  next: boolean,
  callbacks: { onShow: () => void; onQuit: () => void }
): boolean {
  enabled = next
  onShow = callbacks.onShow
  onQuit = callbacks.onQuit
  persist()

  if (enabled) {
    ensureTray()
  } else if (tray) {
    tray.destroy()
    tray = null
  }

  return enabled
}

/** Call on app quit so the icon doesn't linger after the process exits. */
export function destroyTray(): void {
  if (tray) {
    tray.destroy()
    tray = null
  }
}
