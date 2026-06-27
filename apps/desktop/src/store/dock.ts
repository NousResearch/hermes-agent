import { atom } from 'nanostores'

/**
 * Dock mode: the title of the app window Hermes is currently docked beside, or
 * null. When set, the selected app is tiled on the left and the Hermes window is
 * snapped to a narrow panel on the right (see electron hermes:dockToWindow).
 */
export const $dockedWindow = atom<string | null>(null)

/** Tile the app and snap Hermes beside it. Returns true on success. */
export async function dockWindow(title: string): Promise<boolean> {
  const dock = window.hermesDesktop?.dockToWindow

  if (!dock) {
    return false
  }

  const result = await dock(title).catch(() => ({ ok: false }))

  if (result?.ok) {
    $dockedWindow.set(title)

    return true
  }

  return false
}

/** Restore Hermes's pre-dock geometry and exit dock mode. */
export async function undockWindow(): Promise<void> {
  await window.hermesDesktop?.undockWindow?.().catch(() => undefined)
  $dockedWindow.set(null)
}
