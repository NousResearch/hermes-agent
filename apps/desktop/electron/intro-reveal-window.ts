import { pathToFileURL } from 'node:url'

import { BrowserWindow, ipcMain, screen } from 'electron'

import { attachRendererConsoleCapture } from './renderer-log'
import { chatWindowWebPreferences } from './session-windows'
import { installWindowRendererLifecycle } from './window-renderer-lifecycle'
import { createWindowRevealController } from './window-reveal'

// Longer than the renderer's INTRO_DEADMAN_MS, so the main process closes the overlay if the renderer clock stalls.
export const INTRO_REVEAL_WATCHDOG_MS = 34_000
const INTRO_FROST_IN_MS = 500
const INTRO_FROST_OUT_MS = 600

interface IntroRevealOpenPayload {
  hideMain?: boolean
}

interface IntroRevealClosePayload {
  showMain?: boolean
}

interface IntroRevealWindowOptions {
  devServer?: string
  enabled: boolean
  isMac: boolean
  loadWindowUrl: (window: BrowserWindow, url: string, label: string) => void
  log: (message: string) => void
  mainWindow: () => BrowserWindow | null
  preloadPath: string
  rendererIndex: () => string
  showMain: () => void
  wireWindow: (window: BrowserWindow) => void
}

export function createIntroRevealWindowController({
  devServer,
  enabled,
  isMac,
  loadWindowUrl,
  log,
  mainWindow,
  preloadPath,
  rendererIndex,
  showMain,
  wireWindow
}: IntroRevealWindowOptions) {
  let introRevealWindow: BrowserWindow | null = null
  let introRevealWatchdog: ReturnType<typeof setTimeout> | null = null
  let introRevealShow: ReturnType<typeof createWindowRevealController> | null = null
  let frostTimer: ReturnType<typeof setTimeout> | null = null
  let mainFadeTimer: ReturnType<typeof setInterval> | null = null
  let onboardingFlowHidMain = false

  function clearIntroRevealWatchdog() {
    if (introRevealWatchdog) {
      clearTimeout(introRevealWatchdog)
      introRevealWatchdog = null
    }
  }

  function introRevealUrl() {
    if (devServer) {
      return `${devServer.endsWith('/') ? devServer.slice(0, -1) : devServer}/?win=intro#/`
    }

    return `${pathToFileURL(rendererIndex()).toString()}?win=intro#/`
  }

  function showMainAfterOnboarding() {
    onboardingFlowHidMain = false
    const main = mainWindow()

    if (!main || main.isDestroyed()) {
      return
    }

    if (mainFadeTimer) {
      clearInterval(mainFadeTimer)
    }

    const started = Date.now()

    main.setOpacity(0)
    showMain()
    mainFadeTimer = setInterval(() => {
      if (main.isDestroyed()) {
        clearInterval(mainFadeTimer)
        mainFadeTimer = null

        return
      }

      const t = Math.min(1, (Date.now() - started) / 450)

      main.setOpacity(t * (2 - t))

      if (t >= 1) {
        clearInterval(mainFadeTimer)
        mainFadeTimer = null
      }
    }, 16)
  }

  function armIntroRevealShow(win: BrowserWindow) {
    introRevealShow?.dispose()
    introRevealShow = createWindowRevealController({
      isDestroyed: () => win.isDestroyed(),
      isVisible: () => win.isVisible(),
      show: () => {
        win.show()

        if (isMac) {
          win.setVibrancy('hud', { animationDuration: INTRO_FROST_IN_MS })
        }
      }
    })
    // ready-to-show fires on the empty shell; the renderer signals after paint.
    win.webContents.once('did-finish-load', introRevealShow.scheduleFallback)
  }

  function spawnIntroRevealWindow() {
    const win = new BrowserWindow({
      ...screen.getPrimaryDisplay().bounds,
      alwaysOnTop: true,
      backgroundColor: '#00000000',
      focusable: true,
      frame: false,
      fullscreenable: false,
      hasShadow: false,
      hiddenInMissionControl: isMac,
      maximizable: false,
      minimizable: false,
      movable: false,
      resizable: false,
      show: false,
      skipTaskbar: !isMac,
      transparent: true,
      type: isMac ? 'panel' : undefined,
      visualEffectState: isMac ? 'active' : undefined,
      webPreferences: { ...chatWindowWebPreferences(preloadPath), backgroundThrottling: false }
    })

    win.setAlwaysOnTop(true, 'screen-saver')

    if (isMac) {
      win.setVisibleOnAllWorkspaces(true, { visibleOnFullScreen: true, skipTransformProcessType: true })
    }

    wireWindow(win)
    armIntroRevealShow(win)
    installWindowRendererLifecycle(win, { kind: 'overlay', callbacks: { log } })
    attachRendererConsoleCapture(win, 'intro-reveal', log)
    win.on('closed', () => {
      introRevealShow?.dispose()
      introRevealShow = null
      introRevealWindow = null
      clearIntroRevealWatchdog()

      if (onboardingFlowHidMain) {
        showMainAfterOnboarding()
      }

      const main = mainWindow()

      if (main && !main.isDestroyed()) {
        main.webContents.send('hermes:intro-reveal:closed')
      }
    })
    loadWindowUrl(win, introRevealUrl(), 'Intro reveal')

    return win
  }

  function openIntroReveal(payload: IntroRevealOpenPayload = {}) {
    if (!enabled) {
      return { ok: false }
    }

    if (introRevealWindow && !introRevealWindow.isDestroyed()) {
      return { ok: true }
    }

    introRevealWindow = spawnIntroRevealWindow()
    introRevealWatchdog = setTimeout(() => closeIntroReveal({ showMain: true }), INTRO_REVEAL_WATCHDOG_MS)
    const main = mainWindow()

    if (payload.hideMain === true && main && !main.isDestroyed()) {
      // Set before the overlay's first paint, so a skip during load still shows the main window again.
      onboardingFlowHidMain = true
      main.hide()
    }

    return { ok: true }
  }

  function closeIntroReveal(payload: IntroRevealClosePayload = {}) {
    clearIntroRevealWatchdog()
    introRevealShow?.dispose()
    const win = introRevealWindow

    if (win && !win.isDestroyed() && !frostTimer) {
      if (isMac) {
        win.setVibrancy(null, { animationDuration: INTRO_FROST_OUT_MS })
      }

      frostTimer = setTimeout(() => {
        frostTimer = null

        if (!win.isDestroyed()) {
          win.close()
        }
      }, INTRO_FROST_OUT_MS)
    }

    if (payload.showMain === true && onboardingFlowHidMain) {
      showMainAfterOnboarding()
    }

    return { ok: true }
  }

  function destroy() {
    clearIntroRevealWatchdog()
    introRevealShow?.dispose()

    if (frostTimer) {
      clearTimeout(frostTimer)
      frostTimer = null
    }

    if (mainFadeTimer) {
      clearInterval(mainFadeTimer)
      mainFadeTimer = null
    }

    // Cleared before destroy, so the 'closed' handler does not show the main window while the app quits.
    onboardingFlowHidMain = false
    introRevealWindow?.destroy()
    introRevealWindow = null
  }

  ipcMain.handle('hermes:intro-reveal:open', (_event, payload?: IntroRevealOpenPayload) => openIntroReveal(payload))
  ipcMain.handle('hermes:intro-reveal:close', (_event, payload?: IntroRevealClosePayload) => closeIntroReveal(payload))
  ipcMain.on('hermes:intro-reveal:ready', event => {
    if (event.sender === introRevealWindow?.webContents) {
      introRevealShow?.reveal()
    }
  })
  ipcMain.on('hermes:intro-reveal:skip', event => {
    const main = mainWindow()

    if (event.sender === introRevealWindow?.webContents && main && !main.isDestroyed()) {
      main.webContents.send('hermes:intro-reveal:skip')
    }
  })

  return { destroy, showMainAfterOnboarding }
}
