import type { MenuItemConstructorOptions } from 'electron'

interface WindowMenuItem extends MenuItemConstructorOptions {
  submenu: MenuItemConstructorOptions[]
}

interface WindowMenuLabels {
  bringAllToFront: string
  close: string
  minimize: string
  window: string
  zoom: string
}

const ENGLISH_WINDOW_MENU_LABELS: WindowMenuLabels = {
  bringAllToFront: 'Bring All to Front',
  close: 'Close',
  minimize: 'Minimize',
  window: 'Window',
  zoom: 'Zoom'
}

function windowMenuTemplate(isMac: boolean, labels: WindowMenuLabels = ENGLISH_WINDOW_MENU_LABELS): WindowMenuItem {
  return {
    label: labels.window,
    // Mark the macOS menu as AppKit's standard Window menu. Without this role,
    // macOS still exposes tiling from the green button but does not attach its
    // Move & Resize commands (and their keyboard shortcuts) to the app menu.
    ...(isMac ? { role: 'windowMenu' } : {}),
    submenu: isMac
      ? [
          { label: labels.minimize, role: 'minimize' },
          { label: labels.zoom, role: 'zoom' },
          { label: labels.bringAllToFront, role: 'front' }
        ]
      : // Click-only Close: the `close` role would register its default
        // CommandOrControl+W accelerator, claiming the chord before the
        // before-input-event run that routes a terminal-focused Ctrl+W to the
        // shell's word erase (#65457). The menu item still closes the focused
        // window when clicked.
        [
          { label: labels.minimize, role: 'minimize' },
          {
            click: (_menuItem, window) => window?.close(),
            label: labels.close,
            registerAccelerator: false
          }
        ]
  }
}

export { windowMenuTemplate }
export type { WindowMenuLabels }
