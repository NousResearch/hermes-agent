import { isWslEnvironment } from './bootstrap-platform'

export type WindowControlAction = 'close' | 'minimize' | 'toggle-maximize'

interface CustomWindowControlsOptions {
  env?: NodeJS.ProcessEnv
  kernelRelease?: string | null
  platform?: NodeJS.Platform
}

export function customWindowControlsEnabled(options: CustomWindowControlsOptions = {}): boolean {
  return isWslEnvironment(options.env, options.platform, options.kernelRelease)
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
