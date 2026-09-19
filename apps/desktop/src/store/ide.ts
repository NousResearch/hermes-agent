// The Hermes IDE — renderer-side store for the dedicated IDE window.
//
// The IDE is a window kind (`?win=ide`), not a route: opening focuses the live
// window (main owns creation — see electron/ide-window.ts for the URL contract
// and electron/main.ts spawnIdeWindow for construction). This module owns the
// renderer half: the open bridge call the entry points share.

import { notifyError } from './notifications'
import { canOpenIdeWindow } from './windows'

/**
 * Open (or focus) the Hermes IDE window. No-ops gracefully outside Electron so
 * callers can wire it unconditionally. The opener's workspace root rides along
 * as the IDE's initial seed; the backend route is resolved in main from the
 * calling window, mirroring "New Window inherits its opener's device/profile".
 */
export async function openIdeWindow(): Promise<boolean> {
  if (!canOpenIdeWindow()) {
    return false
  }

  // Lazy import: `./session` drags the session store graph in, and this module
  // is imported by keybind/palette surfaces that should stay light.
  const { $currentCwd } = await import('./session')
  const cwd = $currentCwd.get().trim() || null
  const request = cwd ? { cwd } : undefined

  try {
    const result = await window.hermesDesktop.ide?.open(request)

    if (!result?.ok) {
      notifyError(new Error(result?.error || 'unknown error'), 'Could not open the Hermes IDE')

      return false
    }

    return true
  } catch (err) {
    notifyError(err, 'Could not open the Hermes IDE')

    return false
  }
}
