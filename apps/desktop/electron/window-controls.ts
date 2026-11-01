export type WindowControlAction = 'close' | 'minimize' | 'toggle-maximize'

interface CustomWindowControlsOptions {
  env?: NodeJS.ProcessEnv
  kernelRelease?: string | null
  platform?: NodeJS.Platform
}

export const RENDERER_WINDOW_CONTROLS_ENV = 'HERMES_DESKTOP_RENDERER_WINDOW_CONTROLS'

const RENDERER_WINDOW_CONTROLS_ON = new Set(['1', 'true', 'yes', 'on'])

/**
 * Plain-Linux escape hatch for the stuck native overlay Button (#133423,
 * #124255): Electron paints the overlay min/max/close as views::Buttons, and
 * their compositor-animation observer can latch "active" forever on some
 * Chromium/Electron builds ("CompositorAnimationObserver is active for too
 * long … Button@ui/views/controls/button/button.cc"), pinning 1-5 cores of
 * frame compositing with the window idle and visible. Setting
 * HERMES_DESKTOP_RENDERER_WINDOW_CONTROLS=1 swaps every window to the
 * renderer-painted controls (the WSLg path), so no views Button exists to
 * stick. WSL already paints its own controls, so the env is a no-op there.
 */
export function rendererWindowControlsForced(env: NodeJS.ProcessEnv = process.env): boolean {
  const override = String(env[RENDERER_WINDOW_CONTROLS_ENV] || '')
    .trim()
    .toLowerCase()

  return RENDERER_WINDOW_CONTROLS_ON.has(override)
}

// WSL detection kept fs-free on purpose: this module is bundled into the
// sandboxed preload (sandbox: true), where importing node:fs — even
// transitively via bootstrap-platform — throws when the preload module loads
// and tears down the whole `window.hermesDesktop` bridge. The preload only
// needs the env-var signal (WSLg always sets WSL_INTEROP/WSL_DISTRO_NAME), and
// the authoritative flag still reaches the renderer through the main process's
// getWindowState (IS_WSL, which keeps the /proc kernel-release fallback). The
// kernelRelease branch mirrors bootstrap-platform's isWslEnvironment so tests
// and the main process agree; window-controls.test.ts guards that parity.
export function customWindowControlsEnabled(options: CustomWindowControlsOptions = {}): boolean {
  const platform = options.platform ?? process.platform

  if (platform !== 'linux') {
    return false
  }

  const env = options.env ?? process.env

  if (rendererWindowControlsForced(env)) {
    return true
  }

  if (env.WSL_DISTRO_NAME || env.WSL_INTEROP) {
    return true
  }

  return options.kernelRelease ? /microsoft|wsl/i.test(options.kernelRelease) : false
}

interface ControllableWindow {
  close?: () => void
  focus?: () => void
  isDestroyed?: () => boolean
  isMaximized?: () => boolean
  maximize?: () => void
  minimize?: () => void
  unmaximize?: () => void
}

export function performWindowControl(win: ControllableWindow | null | undefined, action: string): boolean {
  if (!win || win.isDestroyed?.()) {
    return false
  }

  if (action === 'minimize' && win.minimize) {
    win.minimize()

    return true
  }

  if (action === 'toggle-maximize' && win.maximize && win.unmaximize) {
    if (win.isMaximized?.()) {
      win.unmaximize()
    } else {
      win.maximize()
    }

    // WSLg's RAIL host can keep pointer activation while dropping the keyboard
    // focus after a renderer-owned maximize/restore button invokes Electron.
    // Reassert the BrowserWindow immediately so typing and Ctrl+V continue to
    // reach the existing focused editor instead of requiring an app restart.
    win.focus?.()

    return true
  }

  if (action === 'close' && win.close) {
    win.close()

    return true
  }

  return false
}

export function windowControlState(win: ControllableWindow | null | undefined, customWindowControls: boolean) {
  return {
    customWindowControls,
    isMaximized: Boolean(win && !win.isDestroyed?.() && win.isMaximized?.())
  }
}

interface WindowControlIpc {
  on(channel: string, listener: (event: { sender: Electron.WebContents }, action: unknown) => void): unknown
}

// Registers the renderer → main channel that the WSLg window-control buttons
// send on. Kept here (not inline in main.ts) so the registration + dispatch is
// unit-testable without importing the electron entry module.
export function registerWindowControlIpc(
  ipcMain: WindowControlIpc,
  resolveWindow: (sender: Electron.WebContents) => ControllableWindow | null | undefined
): void {
  ipcMain.on('hermes:window-control', (event, action) => {
    performWindowControl(resolveWindow(event.sender), typeof action === 'string' ? action : '')
  })
}
