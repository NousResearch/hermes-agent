/**
 * Close-to-system-tray (Windows only).
 *
 * When the user clicks the window's close (X), the main window is hidden to the
 * system tray instead of being destroyed, so the agent keeps running. The tray
 * icon restores it; right-click → "Exit" (or the in-app quit) ends the app.
 *
 * The decision — "should this close become a hide?" — is a single pure function
 * so it can be unit-tested without a live Tray or BrowserWindow. Invariants:
 *  - macOS never minimizes to tray on close (its convention is the Dock); the
 *    app quits normally there.
 *  - A hand-off relaunch (update / swap / uninstall) must exit, never hide.
 *  - Only the *main* window hides; secondary session windows always close.
 *  - A real quit (isQuitting) must close the window even with the setting on.
 */

import { type Menu as MenuType, type NativeImage, type Tray as TrayType } from 'electron'

export interface ShouldMinimizeParams {
  isEnabled: boolean
  isQuittingForHandoff: boolean
  isQuitting: boolean
  isMainWindow: boolean
  isWindows: boolean
  /** null when this is a programmatic query (no event to intercept). */
  event: null | { preventDefault: () => void }
}

/** True when the close should be swallowed and the window hidden to the tray. */
export function shouldMinimizeToTray({
  isEnabled,
  isQuittingForHandoff,
  isQuitting,
  isMainWindow,
  isWindows,
  event
}: ShouldMinimizeParams): boolean {
  if (!isWindows || !isEnabled || isQuittingForHandoff || isQuitting || !isMainWindow) {
    return false
  }

  event?.preventDefault()

  return true
}

export interface TrayMenuItem {
  click?: () => void
  label?: string
  type?: 'separator' | 'normal' | 'submenu' | 'checkbox' | 'radio'
}

export interface TrayBuildOptions {
  iconPath: null | string
  onRestore: () => void
  onQuit: () => void
}

export interface TrayController {
  /** Create (or recreate) the tray icon. No-op store for non-Windows. */
  build: (options: TrayBuildOptions) => void
  /** Remove the tray icon (setting turned off, or on a real quit). */
  destroy: () => void
  isActive: () => boolean
}

/**
 * Owns the single tray icon. `makeTray`, `makeIcon`, and `makeMenu` are
 * injected so the controller can be exercised without a live Electron
 * (Tray / nativeImage / Menu).
 */
export function createTrayController(
  platformIsWindows: boolean,
  makeTray: (icon: NativeImage) => TrayType,
  makeIcon: (iconPath: null | string) => NativeImage,
  makeMenu: (template: TrayMenuItem[]) => MenuType
): TrayController {
  let tray: null | TrayType = null

  const destroy = () => {
    if (tray) {
      tray.destroy()
      tray = null
    }
  }

  const build = ({ iconPath, onRestore, onQuit }: TrayBuildOptions) => {
    destroy()

    if (!platformIsWindows) {
      return
    }

    const icon = makeIcon(iconPath)
    const created = makeTray(icon)

    created.setToolTip('Hermes')
    created.setContextMenu(
      makeMenu([
        { click: () => onRestore(), label: 'Open Hermes' },
        { type: 'separator' },
        { click: () => onQuit(), label: 'Exit' }
      ])
    )
    // Left-click also restores — Windows tray apps commonly do this.
    created.on('click', () => onRestore())

    tray = created
  }

  return { build, destroy, isActive: () => tray !== null }
}
