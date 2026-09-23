import crypto from 'node:crypto'
import fs from 'node:fs'
import path from 'node:path'

import { app, BrowserWindow, contentTracing, webContents as electronWebContents, ipcMain, screen } from 'electron'

import {
  type ChromiumMemoryInfraCaptureConfig,
  createChromiumMemoryInfraGpuProbe,
  createLunarCityPerfMainController,
  resolveLunarCityPerfLaunch
} from './lunar-city-perf-main'

/** Packaged, nonce-bound measurement surface. Ordinary renderer sessions get no authority. */
export function installLunarCityPerfBridge(options: { buildStamp: unknown; getMainWindow(): BrowserWindow | null }) {
  const lunarCityPerfLaunch = resolveLunarCityPerfLaunch({
    buildStamp: options.buildStamp,
    env: process.env,
    // The development compatibility override used elsewhere is deliberately
    // not accepted here: this is a packaged acceptance-only surface.
    isPackaged: app.isPackaged
  })

  let lunarCityTraceCapture = Promise.resolve()

  async function captureLunarCityGpuTrace(config: ChromiumMemoryInfraCaptureConfig) {
    let release: () => void
    const previous = lunarCityTraceCapture
    lunarCityTraceCapture = new Promise(resolve => {
      release = resolve
    })

    await previous

    const tracePath = path.join(app.getPath('temp'), `hermes-lunar-city-memory-${crypto.randomUUID()}.json`)
    let recording = false

    try {
      await contentTracing.startRecording({
        included_categories: [config.category],
        memory_dump_config: {
          allowed_dump_modes: [config.dumpMode],
          triggers: [{ mode: config.dumpMode, periodic_interval_ms: config.periodicIntervalMs }]
        }
      })
      recording = true
      await new Promise(resolve => setTimeout(resolve, config.periodicIntervalMs + 100))
      const completedPath = await contentTracing.stopRecording(tracePath)
      recording = false

      return JSON.parse(await fs.promises.readFile(completedPath, 'utf8'))
    } finally {
      if (recording) {
        await contentTracing.stopRecording().catch(() => undefined)
      }

      await fs.promises.rm(tracePath, { force: true }).catch(() => undefined)
      release()
    }
  }

  const lunarCityPerfController = lunarCityPerfLaunch
    ? createLunarCityPerfMainController({
        appMetrics: () => app.getAppMetrics(),
        environmentSnapshot: async sender => {
          const contents = electronWebContents.fromId(sender.id)
          const win = contents ? BrowserWindow.fromWebContents(contents) : null
          const gpuFeatureStatus = app.getGPUFeatureStatus()
          const gpuInfo = await app.getGPUInfo('complete').catch(() => undefined)
          const bounds = win && !win.isDestroyed() ? win.getBounds() : undefined
          const displayScaleFactor = bounds ? screen.getDisplayMatching(bounds).scaleFactor : undefined

          return {
            chromiumVersion: process.versions.chrome,
            displayScaleFactor,
            electronMode: 'packaged',
            electronVersion: process.versions.electron,
            gpuEnabled:
              gpuFeatureStatus.gpu_compositing !== 'disabled_software' &&
              gpuFeatureStatus.gpu_compositing !== 'disabled_off',
            gpuFeatureStatus,
            gpuInfo,
            windowBounds: bounds,
            windowState:
              win && !win.isDestroyed()
                ? { minimized: win.isMinimized(), visible: win.isVisible() }
                : { minimized: false, visible: false }
          }
        },
        gpuSnapshot: createChromiumMemoryInfraGpuProbe(captureLunarCityGpuTrace),
        launch: lunarCityPerfLaunch,
        mainPid: process.pid,
        now: () => Date.now(),
        ownsSender: sender =>
          Boolean(
            options.getMainWindow() &&
            !options.getMainWindow()!.isDestroyed() &&
            options.getMainWindow()!.webContents.id === sender.id
          ),
        scenarioWindowAction: async (sender, action) => {
          const contents = electronWebContents.fromId(sender.id)
          const win = contents ? BrowserWindow.fromWebContents(contents) : null

          if (!win || win.isDestroyed()) {
            throw new Error('Lunar City performance window is unavailable')
          }

          const before = { minimized: win.isMinimized(), visible: win.isVisible() }

          if (
            (action === 'window-hidden' && !before.visible) ||
            (action === 'window-minimized' && before.minimized) ||
            (action === 'window-visible-cycle' && (!before.visible || before.minimized))
          ) {
            throw new Error(`Lunar City performance window action has no nonzero transition: ${action}`)
          }

          const windowTrace = [before]

          if (action === 'window-hidden') {
            win.hide()
          } else if (action === 'window-minimized') {
            win.minimize()
          } else {
            win.hide()
            await new Promise(resolve => setImmediate(resolve))
            const hidden = { minimized: win.isMinimized(), visible: win.isVisible() }

            if (hidden.visible) {
              throw new Error('Lunar City performance visible-window hide transition was not observed')
            }

            windowTrace.push(hidden)
            win.show()
          }

          await new Promise(resolve => setImmediate(resolve))
          const windowState = { minimized: win.isMinimized(), visible: win.isVisible() }

          const observed =
            action === 'window-hidden'
              ? !windowState.visible
              : action === 'window-minimized'
                ? windowState.minimized
                : windowState.visible && !windowState.minimized

          if (!observed) {
            throw new Error(`Lunar City performance window transition was not observed: ${action}`)
          }

          windowTrace.push(windowState)

          return { action, proof: 1, windowState, windowTrace }
        }
      })
    : undefined

  const lunarCityPerfEvent = (event: Electron.IpcMainEvent | Electron.IpcMainInvokeEvent) => ({
    frameId: event.senderFrame?.routingId ?? -1,
    sender: event.sender
  })

  const channels = {
    bootstrap: 'hermes:lunar-city-perf:bootstrap',
    register: 'hermes:lunar-city-perf:register-responder',
    activate: 'hermes:lunar-city-perf:activate',
    metrics: 'hermes:lunar-city-perf:process-metrics',
    request: 'hermes:lunar-city-perf:renderer-request',
    response: 'hermes:lunar-city-perf:response'
  }

  const bootstrap = (event: Electron.IpcMainEvent) => {
    event.returnValue = lunarCityPerfController?.bootstrap(lunarCityPerfEvent(event))
  }

  const register = (event: Electron.IpcMainEvent, handshake: unknown) => {
    event.returnValue = lunarCityPerfController?.registerResponder(lunarCityPerfEvent(event), handshake) === true
  }

  const activate = (event: Electron.IpcMainEvent, handshake: unknown) => {
    event.returnValue = lunarCityPerfController?.activate(lunarCityPerfEvent(event), handshake) === true
  }

  const response = (event: Electron.IpcMainEvent, value: unknown) => {
    lunarCityPerfController?.resolveRendererResponse(lunarCityPerfEvent(event), value)
  }

  ipcMain.on(channels.bootstrap, bootstrap)
  ipcMain.on(channels.register, register)
  ipcMain.on(channels.activate, activate)
  ipcMain.on(channels.response, response)
  ipcMain.handle(channels.metrics, event => lunarCityPerfController?.processMetrics(lunarCityPerfEvent(event)))
  ipcMain.handle(channels.request, (event, action, payload) =>
    lunarCityPerfController?.requestRenderer(lunarCityPerfEvent(event), action, payload)
  )
  const windowReleases = new Set<() => void>()

  return {
    attachWindow(window: BrowserWindow) {
      if (!lunarCityPerfController) {return}
      const contents = window.webContents

      const navigation = (_event: Electron.Event, _url: string, _inPlace: boolean, mainFrame: boolean) => {
        if (mainFrame) {lunarCityPerfController.invalidateRenderer(contents, 'navigation')}
      }

      const gone = () => lunarCityPerfController.invalidateRenderer(contents, 'render-process-gone')

      const destroyed = () => {
        lunarCityPerfController.invalidateRenderer(contents, 'destroyed')
        release()
      }

      const release = () => {
        contents.removeListener('did-start-navigation', navigation)
        contents.removeListener('render-process-gone', gone)
        contents.removeListener('destroyed', destroyed)
        windowReleases.delete(release)
      }

      contents.on('did-start-navigation', navigation)
      contents.on('render-process-gone', gone)
      contents.on('destroyed', destroyed)
      windowReleases.add(release)
    },
    dispose() {
      lunarCityPerfController?.dispose()

      for (const release of windowReleases) {release()}
      ipcMain.removeListener(channels.bootstrap, bootstrap)
      ipcMain.removeListener(channels.register, register)
      ipcMain.removeListener(channels.activate, activate)
      ipcMain.removeListener(channels.response, response)
      ipcMain.removeHandler(channels.metrics)
      ipcMain.removeHandler(channels.request)
    }
  }
}
