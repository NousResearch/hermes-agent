import fs from 'node:fs'
import path from 'node:path'
import { pathToFileURL } from 'node:url'

import { app, BrowserWindow, desktopCapturer, type Rectangle, screen } from 'electron'

import { attachRendererConsoleCapture } from './renderer-log'
import {
  normalizeScreenTutorAnnotations,
  normalizeScreenTutorPoint,
  SCREEN_TUTOR_POINT_TTL_MS,
  type ScreenTutorAnnotationsPayload,
  screenTutorDisplayForPoint,
  type ScreenTutorPointPayload,
  screenTutorThumbnailSize,
  selectScreenTutorSource
} from './screen-tutor'

interface ScreenTutorWindowOptions {
  devServer?: string
  getWindowsToHide: () => Array<BrowserWindow | null>
  loadWindowUrl: (window: BrowserWindow, url: string, label: string) => void
  log: (message: string) => void
  preloadPath: string
  rendererIndex: () => string
  wireWindow: (window: BrowserWindow) => void
}

const captureDelay = () => new Promise(resolve => setTimeout(resolve, 90))

export function createScreenTutorWindowController({
  devServer,
  getWindowsToHide,
  loadWindowUrl,
  log,
  preloadPath,
  rendererIndex,
  wireWindow
}: ScreenTutorWindowOptions) {
  let hideTimer: NodeJS.Timeout | null = null
  let activeAnnotations: ScreenTutorAnnotationsPayload | null = null
  let window: BrowserWindow | null = null

  const broadcastState = (visible: boolean) => {
    const state = {
      count: activeAnnotations?.annotations.length ?? 0,
      frozen: activeAnnotations?.frozen ?? false,
      guide: activeAnnotations?.guide,
      visible
    }

    for (const candidate of BrowserWindow.getAllWindows()) {
      if (!candidate.isDestroyed() && candidate !== window) {
        candidate.webContents.send('hermes:screen-tutor:state', state)
      }
    }
  }

  const url = () => {
    if (devServer) {
      return `${devServer.endsWith('/') ? devServer.slice(0, -1) : devServer}/?win=screen-tutor#/`
    }

    return `${pathToFileURL(rendererIndex()).toString()}?win=screen-tutor#/`
  }

  const dismiss = () => {
    if (hideTimer) {
      clearTimeout(hideTimer)
      hideTimer = null
    }

    if (window && !window.isDestroyed()) {
      window.hide()
    }

    activeAnnotations = null
    broadcastState(false)
  }

  const spawn = (bounds: Rectangle) => {
    const next = new BrowserWindow({
      ...bounds,
      alwaysOnTop: true,
      backgroundColor: '#00000000',
      focusable: false,
      frame: false,
      fullscreenable: false,
      hasShadow: false,
      maximizable: false,
      minimizable: false,
      movable: false,
      resizable: false,
      show: false,
      skipTaskbar: true,
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

    next.setAlwaysOnTop(true, 'screen-saver')
    next.setIgnoreMouseEvents(true, { forward: true })
    next.setVisibleOnAllWorkspaces(true, { skipTransformProcessType: true, visibleOnFullScreen: true })
    wireWindow(next)
    attachRendererConsoleCapture(next, 'screen-tutor', log)
    next.on('closed', () => {
      if (window === next) {
        window = null
      }
    })
    loadWindowUrl(next, url(), 'Screen tutor overlay')

    return next
  }

  const showPoint = (value: unknown): boolean => {
    const point = normalizeScreenTutorPoint(value)

    if (!point) {
      return false
    }

    const display = screenTutorDisplayForPoint(screen.getAllDisplays(), point)

    if (!display) {
      return false
    }

    if (!window || window.isDestroyed()) {
      window = spawn(display.bounds)
    } else {
      window.setBounds(display.bounds)
    }

    const send = () => {
      if (!window || window.isDestroyed()) {
        return
      }

      window.webContents.send('hermes:screen-tutor:point', point)
      window.showInactive()
    }

    if (window.webContents.isLoading()) {
      window.webContents.once('did-finish-load', send)
    } else {
      send()
    }

    if (hideTimer) {
      clearTimeout(hideTimer)
    }

    hideTimer = setTimeout(dismiss, SCREEN_TUTOR_POINT_TTL_MS)

    return true
  }

  const showAnnotations = (value: unknown): boolean => {
    const payload = normalizeScreenTutorAnnotations(value)

    if (!payload) {
      return false
    }

    const display = screenTutorDisplayForPoint(screen.getAllDisplays(), payload)

    if (!display) {
      return false
    }

    if (!window || window.isDestroyed()) {
      window = spawn(display.bounds)
    } else {
      window.setBounds(display.bounds)
    }

    activeAnnotations = payload

    const send = () => {
      if (!window || window.isDestroyed()) {
        return
      }

      window.webContents.send('hermes:screen-tutor:annotations', payload)
      window.showInactive()
      broadcastState(true)
    }

    if (window.webContents.isLoading()) {
      window.webContents.once('did-finish-load', send)
    } else {
      send()
    }

    if (hideTimer) {
      clearTimeout(hideTimer)
    }

    hideTimer = payload.frozen ? null : setTimeout(dismiss, payload.ttlMs)

    return true
  }

  const setFrozen = (frozen: boolean): boolean => {
    if (!activeAnnotations || !window || window.isDestroyed() || !window.isVisible()) {
      return false
    }

    activeAnnotations = { ...activeAnnotations, frozen }

    if (hideTimer) {
      clearTimeout(hideTimer)
      hideTimer = null
    }

    if (!frozen) {
      hideTimer = setTimeout(dismiss, activeAnnotations.ttlMs)
    }

    broadcastState(true)

    return true
  }

  const capture = async () => {
    const cursor = screen.getCursorScreenPoint()
    const display = screen.getDisplayNearestPoint(cursor)

    const hidden = getWindowsToHide().filter((candidate): candidate is BrowserWindow =>
      Boolean(candidate && !candidate.isDestroyed() && candidate.isVisible())
    )

    for (const candidate of hidden) {
      candidate.hide()
    }

    try {
      await captureDelay()
      const thumbnailSize = screenTutorThumbnailSize(display.bounds, display.scaleFactor)
      const sources = await desktopCapturer.getSources({ fetchWindowIcons: false, thumbnailSize, types: ['screen'] })
      const source = selectScreenTutorSource(sources, String(display.id))

      if (!source || source.thumbnail.isEmpty()) {
        throw new Error(`No screen capture source matched display ${display.id}.`)
      }

      const cacheDir = path.join(app.getPath('userData'), 'screen-tutor-cache')
      fs.mkdirSync(cacheDir, { recursive: true })

      const now = Date.now()

      for (const entry of fs.readdirSync(cacheDir, { withFileTypes: true })) {
        if (entry.isFile()) {
          const candidate = path.join(cacheDir, entry.name)

          try {
            if (now - fs.statSync(candidate).mtimeMs > 60 * 60 * 1000) {
              fs.unlinkSync(candidate)
            }
          } catch {
            // Best effort transient-cache cleanup.
          }
        }
      }

      const capturePath = path.join(cacheDir, `capture-${now}-${String(display.id).replace(/[^a-zA-Z0-9_-]/g, '')}.png`)
      fs.writeFileSync(capturePath, source.thumbnail.toPNG(), { mode: 0o600 })
      const imageSize = source.thumbnail.getSize()

      return {
        display: {
          bounds: display.bounds,
          id: String(display.id),
          label: display.label || `Display ${display.id}`,
          scaleFactor: display.scaleFactor
        },
        image: imageSize,
        path: capturePath
      }
    } finally {
      for (const candidate of hidden) {
        if (!candidate.isDestroyed()) {
          candidate.showInactive()
        }
      }
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
  }

  return { capture, close, dismiss, setFrozen, showAnnotations, showPoint }
}

export type { ScreenTutorPointPayload }
