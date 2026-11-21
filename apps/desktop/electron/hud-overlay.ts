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

/**
 * Electron's vibrancy update on a transparent HUD terminates the main process
 * on macOS 27 (Darwin 26). Keep the HUD usable there until Electron can make
 * that native transition safe again. Unknown and later releases retain frost.
 */
export function hudVibrancySupported(darwinMajor: number): boolean {
  return darwinMajor !== 26
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
 * Ask the running compositor to treat the HUD as an overlay. Hyprland is the
 * first adapter (float + pin). Sway/niri hang off this same function later.
 * Returns true when an adapter applied; false is "nothing to do / not that WM".
 */
export async function promoteHudOverlay(options: { title: string }): Promise<boolean> {
  return promoteHudOnHyprland(options)
}
