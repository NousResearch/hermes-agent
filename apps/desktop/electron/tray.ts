import { type BrowserWindow, Menu, nativeImage, Tray } from 'electron'

let tray: Tray | null = null

function showWindow(window: BrowserWindow | null) {
  if (!window || window.isDestroyed()) return
  if (!window.isVisible()) window.show()
  if (!window.isFocused()) window.focus()
}

export interface TrayOptions {
  window: BrowserWindow | null
  iconPath: string | undefined
  onQuit: () => void
}

/**
 * Create a system-tray icon for Hermes if one does not already exist.
 *
 * Clicking/double-clicking the icon restores the main window; the right-click
 * menu offers "Show" and "Quit". The tray is only created when needed (the
 * first time the user closes the main window) so that users who never close
 * the window are not surprised by a tray icon.
 *
 * Call `destroyTray()` during teardown to remove the icon.
 */
export function ensureTray({ window, iconPath, onQuit }: TrayOptions): void {
  if (tray || !window || window.isDestroyed()) return
  if (!iconPath) return

  const icon = nativeImage.createFromPath(iconPath)
  if (icon.isEmpty()) return

  tray = new Tray(icon)
  tray.setToolTip('Hermes')
  tray.setContextMenu(
    Menu.buildFromTemplate([
      {
        label: 'Show Hermes',
        click: () => showWindow(window)
      },
      { type: 'separator' },
      {
        label: 'Quit Hermes',
        click: () => {
          destroyTray()
          onQuit()
        }
      }
    ])
  )
  tray.on('click', () => showWindow(window))
  tray.on('double-click', () => showWindow(window))
}

/** Remove the Hermes tray icon and clean up its listeners. */
export function destroyTray(): void {
  if (!tray) return
  tray.destroy()
  tray = null
}
