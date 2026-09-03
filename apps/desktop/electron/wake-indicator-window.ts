import { pathToFileURL } from 'node:url'

import { BrowserWindow, screen } from 'electron'

import { attachRendererConsoleCapture } from './renderer-log'
import {
  normalizeWakeIndicatorState,
  selectWakeIndicatorDisplay,
  WAKE_INDICATOR_FADE_MS,
  type WakeIndicatorState,
  wakeIndicatorWindowBounds
} from './wake-indicator'
import { installWindowRendererLifecycle } from './window-renderer-lifecycle'

interface WakeIndicatorWindowOptions {
  devServer?: string
  isMac: boolean
  loadWindowUrl: (window: BrowserWindow, url: string, label: string) => void
  log: (message: string) => void
  preloadPath: string
  rendererIndex: () => string
  wireWindow: (window: BrowserWindow) => void
}

export function createWakeIndicatorWindowController({
  devServer,
  isMac,
  loadWindowUrl,
  log,
  preloadPath,
  rendererIndex,
  wireWindow
}: WakeIndicatorWindowOptions) {
  let hideTimer: NodeJS.Timeout | null = null
  let state: WakeIndicatorState = 'hidden'
  let window: BrowserWindow | null = null

  const url = () => {
    if (devServer) {
      return `${devServer.endsWith('/') ? devServer.slice(0, -1) : devServer}/?win=wake#/`
    }

    return `${pathToFileURL(rendererIndex()).toString()}?win=wake#/`
  }

  const selectedDisplay = () => selectWakeIndicatorDisplay(screen.getAllDisplays(), screen.getPrimaryDisplay())

  const reposition = () => {
    if (!window || window.isDestroyed()) {
      return
    }

    window.setBounds(wakeIndicatorWindowBounds(selectedDisplay()))
  }

  const sendState = () => {
    if (!window || window.isDestroyed()) {
      return
    }

    window.webContents.send('hermes:wake-indicator:state', state)
  }

  const spawn = () => {
    const next = new BrowserWindow({
      ...wakeIndicatorWindowBounds(selectedDisplay()),
      alwaysOnTop: true,
      backgroundColor: '#00000000',
      focusable: false,
      frame: false,
      fullscreenable: false,
      hasShadow: false,
      hiddenInMissionControl: true,
      maximizable: false,
      minimizable: false,
      movable: false,
      resizable: false,
      show: false,
      skipTaskbar: false,
      transparent: true,
      type: 'panel',
      webPreferences: {
        backgroundThrottling: false,
        contextIsolation: true,
        devTools: true,
        nodeIntegration: false,
        preload: preloadPath,
        sandbox: true
      }
    })

    next.setAlwaysOnTop(true, 'floating')
    next.setHiddenInMissionControl?.(true)
    next.setIgnoreMouseEvents(true, { forward: true })

    try {
      next.setVisibleOnAllWorkspaces(true, {
        skipTransformProcessType: true,
        visibleOnFullScreen: true
      })
    } catch {
      // Best effort on older Electron/macOS combinations.
    }

    wireWindow(next)

    // Log-only renderer lifecycle (#81290): the wake cue is ambient and
    // macOS-only; its loss belongs in desktop.log, never resurrected.
    installWindowRendererLifecycle(next, { kind: 'wake', callbacks: { log } })
    // Console errors go through the shared capture (renderer-log.ts owns
    // console-message; the lifecycle helper deliberately does not).
    attachRendererConsoleCapture(next, 'wake', log)

    next.webContents.on('did-finish-load', sendState)
    next.once('ready-to-show', () => {
      if (!next.isDestroyed() && state !== 'hidden') {
        next.showInactive()
      }
    })
    next.on('closed', () => {
      if (window === next) {
        window = null
      }
    })

    loadWindowUrl(next, url(), 'Wake indicator')

    return next
  }

  const setState = (value: unknown) => {
    if (!isMac) {
      return
    }

    state = normalizeWakeIndicatorState(value)

    if (hideTimer) {
      clearTimeout(hideTimer)
      hideTimer = null
    }

    if (state === 'hidden') {
      sendState()
      hideTimer = setTimeout(() => {
        hideTimer = null

        if (state === 'hidden' && window && !window.isDestroyed()) {
          window.hide()
        }
      }, WAKE_INDICATOR_FADE_MS)

      return
    }

    if (!window || window.isDestroyed()) {
      window = spawn()
    } else {
      reposition()
      sendState()
      window.showInactive()
    }
  }

  const close = () => {
    if (hideTimer) {
      clearTimeout(hideTimer)
      hideTimer = null
    }

    if (window && !window.isDestroyed()) {
      window.close()
    }

    window = null
    state = 'hidden'
  }

  return {
    close,
    getState: () => state,
    reposition,
    setState
  }
}
