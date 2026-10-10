/**
 * HUD overlay adapters.
 *
 * Electron `alwaysOnTop` is the generic ask. Some compositors ignore it
 * (Hyprland tiles the toplevel; COSMIC drops z-order). Each adapter speaks
 * that compositor's dialect; `promoteHudOverlay` is the one call site.
 */

import { promoteHudOnHyprland } from './hud-hyprland'

export interface HudElectronOverlayWindow {
  setAlwaysOnTop(flag: boolean, level?: string): void
  setVisibleOnAllWorkspaces?(
    visible: boolean,
    options?: { skipTransformProcessType?: boolean; visibleOnFullScreen?: boolean }
  ): void
}

/** Chrome Electron itself can honour. Compositor IPC is `promoteHudOverlay`. */
export function applyHudElectronOverlay(win: HudElectronOverlayWindow, platform: string): void {
  win.setAlwaysOnTop(true, platform === 'darwin' ? 'floating' : 'screen-saver')

  if (platform !== 'darwin') {
    return
  }

  try {
    win.setVisibleOnAllWorkspaces?.(true, { visibleOnFullScreen: true, skipTransformProcessType: true })
  } catch {
    // Not supported everywhere — best effort.
  }
}

/**
 * Pin or unpin the HUD above the user's other windows — the Mini Assistant's
 * "always on top" preference (Settings → Appearance → Window layout, and the
 * pin button in the bar itself).
 *
 * Unpinning hands the window back to the compositor's ordinary z-order: no
 * level, and no all-workspaces claim either, because a window that no longer
 * floats must not keep striding across virtual desktops. Re-pinning restores
 * the same levels `applyHudElectronOverlay` chose at spawn, so a toggled HUD
 * and a born-pinned HUD end up identical.
 */
export function setHudAlwaysOnTop(win: HudElectronOverlayWindow, on: boolean, platform: string): void {
  if (!on) {
    win.setAlwaysOnTop(false)

    if (platform === 'darwin') {
      try {
        win.setVisibleOnAllWorkspaces?.(false)
      } catch {
        // Best effort, same as the set.
      }
    }

    return
  }

  applyHudElectronOverlay(win, platform)
}

/**
 * Ask the running compositor to treat the HUD as an overlay. Hyprland is the
 * first adapter (float + pin). Sway/niri hang off this same function later.
 * Returns true when an adapter applied; false is "nothing to do / not that WM".
 */
export async function promoteHudOverlay(options: { title: string }): Promise<boolean> {
  return promoteHudOnHyprland(options)
}
