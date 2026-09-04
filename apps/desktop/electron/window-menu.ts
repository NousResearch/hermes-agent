interface WindowMenuItem {
  label?: string
  role?: string
  submenu: Array<{ role: string }>
}

function windowMenuTemplate(isMac: boolean): WindowMenuItem {
  return {
    label: 'Window',
    // Mark the macOS menu as AppKit's standard Window menu. Without this role,
    // macOS still exposes tiling from the green button but does not attach its
    // Move & Resize commands (and their keyboard shortcuts) to the app menu.
    ...(isMac ? { role: 'windowMenu' } : {}),
    submenu: isMac
      ? [{ role: 'minimize' }, { role: 'zoom' }, { role: 'front' }]
      : [{ role: 'minimize' }, { role: 'close' }]
  }
}

export { windowMenuTemplate }
