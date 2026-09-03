// subtitle-capture-session.ts — the live-subtitle session (Electron main).
//
// One session at a time: periodic window snapshots (not a persistent display
// stream — that blanks DRM video in the player's own window), crop the
// subtitle band, and ship changed crops to the backend (`/api/subtitles/process`).
// Main paints the translation over the original line through the
// screen-annotation overlay's `subtitles` channel. The agent starts and stops
// the session; nothing in the per-line path touches a model conversation.
//
// All geometry lives in subtitle-capture.ts. The target window is re-resolved
// on a slow poll so the band follows a moved or resized player, and the
// session self-stops when the window disappears or the backend stays broken.

import fs from 'node:fs'
import path from 'node:path'

import { app, BrowserWindow, desktopCapturer, ipcMain, screen } from 'electron'

import { type AnnotationBounds, resolveAnnotationWindow } from './screen-annotations'
import type { ScreenAnnotationsController } from './screen-annotations-window'
import {
  bandFractions,
  brightMaskHash,
  clampBandFraction,
  clampSampleHz,
  cropRectFor,
  hammingDistance,
  matchCapturerWindow,
  SAME_TEXT_MAX_DISTANCE,
  shipSize,
  SUBTITLE_KEEPALIVE_MS,
  subtitleBand,
  type SubtitleBandFractions,
  subtitleShapes,
  type SubtitleTextBox
} from './subtitle-capture'
import { type EnumeratedWindow, enumerateWindowsFrontToBack, enumerationFailed, enumerationFailureNote } from './window-below'

interface SubtitleCaptureOptions {
  annotations: ScreenAnnotationsController
  log: (message: string) => void
  /** Authenticated POST to the backend that owns OCR + translation. */
  postToBackend: (path: string, body: Record<string, unknown>) => Promise<Record<string, unknown>>
  titlesAvailable: () => boolean
}

export interface SubtitleCaptureController {
  close(): void
  control(request: unknown, senderBounds: AnnotationBounds | null): Promise<Record<string, unknown>>
}

interface SubtitleSession {
  band: AnnotationBounds
  bandFraction: number
  consecutiveBackendErrors: number
  debugFrameSaved: boolean
  display: AnnotationBounds
  displayId: number
  epoch: number
  fractions: SubtitleBandFractions
  language: string
  lastHash: string
  lastLatencyMs: number
  lastSentAt: number
  lastShapesAt: number
  lastSourceText: string
  linesDrawn: number
  sampleHz: number
  startedAt: number
  streamId: string
  targetSpec?: string
  windowBounds: AnnotationBounds
  windowId: number
  windowRef: { app: string; title: string }
}

const FOLLOW_INTERVAL_MS = 5000
const FOLLOW_MISSES_BEFORE_STOP = 2
const BACKEND_ERRORS_BEFORE_STOP = 5
const SUPPORTED_ACTIONS = ['start', 'status', 'stop'] as const

const asBounds = (value: unknown): AnnotationBounds | null => {
  const raw = (value ?? {}) as Record<string, unknown>

  const nums = [raw.x, raw.y, raw.width, raw.height].map(entry =>
    typeof entry === 'number' && Number.isFinite(entry) ? entry : null
  )

  return nums.every(entry => entry !== null)
    ? { height: nums[3] as number, width: nums[2] as number, x: nums[0] as number, y: nums[1] as number }
    : null
}

/** Chromium NativeImage.toBitmap() is BGRA on little-endian; the hash wants RGBA. */
const bgraToRgba = (bgra: Buffer, width: number, height: number): Uint8ClampedArray => {
  const rgba = new Uint8ClampedArray(width * height * 4)

  for (let index = 0; index < width * height; index += 1) {
    const offset = index * 4

    rgba[offset] = bgra[offset + 2]
    rgba[offset + 1] = bgra[offset + 1]
    rgba[offset + 2] = bgra[offset]
    rgba[offset + 3] = bgra[offset + 3]
  }

  return rgba
}

export function createSubtitleCaptureController(options: SubtitleCaptureOptions): SubtitleCaptureController {
  const { annotations, log, postToBackend, titlesAvailable } = options

  let current: SubtitleSession | null = null
  let followTimer: NodeJS.Timeout | null = null
  let sampleTimer: NodeJS.Timeout | null = null
  let followMisses = 0
  let followInFlight = false
  let sampleInFlight = false
  let frameGeneration = 0
  let lastDrawn: { display: AnnotationBounds; shapes: ReturnType<typeof subtitleShapes> } | null = null

  const resolveTarget = async (
    spec: string | undefined,
    senderBounds: AnnotationBounds | null
  ): Promise<{ error: string; window?: undefined } | { error?: undefined; window: EnumeratedWindow }> => {
    const windows = await enumerateWindowsFrontToBack(process.pid, titlesAvailable())

    if (enumerationFailed(windows)) {
      return { error: enumerationFailureNote(process.platform, process.env, windows.reason) }
    }

    return resolveAnnotationWindow(windows, process.pid, senderBounds, spec)
  }

  const stopSampling = () => {
    if (sampleTimer) {
      clearInterval(sampleTimer)
      sampleTimer = null
    }
  }

  const stop = (reason?: string): Record<string, unknown> => {
    const wasRunning = current !== null

    const stats = current
      ? { lines_translated: current.linesDrawn, ran_seconds: Math.round((Date.now() - current.startedAt) / 1000) }
      : {}

    if (followTimer) {
      clearInterval(followTimer)
      followTimer = null
    }

    stopSampling()
    followMisses = 0
    frameGeneration += 1
    current = null
    annotations.clearChannel('subtitles')

    if (reason) {
      log(`subtitles: stopped — ${reason}`)
    }

    return { stopped: wasRunning, success: true, ...(reason ? { reason } : {}), ...stats }
  }

  const followTick = async () => {
    if (!current || followInFlight) {
      return
    }

    followInFlight = true

    try {
      const windows = await enumerateWindowsFrontToBack(process.pid, titlesAvailable())

      if (!current || enumerationFailed(windows)) {
        return
      }

      const match =
        windows.find(win => win.id === current!.windowId) ??
        windows.find(
          win =>
            win.app === current!.windowRef.app &&
            win.title === current!.windowRef.title &&
            win.bounds.width > 0 &&
            win.bounds.height > 0
        )

      if (!match || match.bounds.width <= 0 || match.bounds.height <= 0) {
        followMisses += 1

        if (followMisses >= FOLLOW_MISSES_BEFORE_STOP) {
          stop('the target window is gone')
        }

        return
      }

      followMisses = 0

      const display = screen.getDisplayMatching(match.bounds)
      const band = subtitleBand(match.bounds, display.bounds, current.bandFraction)

      if (!band) {
        return
      }

      const moved =
        band.x !== current.band.x ||
        band.y !== current.band.y ||
        band.width !== current.band.width ||
        band.height !== current.band.height ||
        display.id !== current.displayId

      if (!moved) {
        return
      }

      current.band = band
      current.display = display.bounds
      current.displayId = display.id
      current.fractions = bandFractions(band, match.bounds)
      current.windowBounds = match.bounds
      current.windowId = match.id
      current.epoch += 1
      current.lastHash = ''
    } catch (error) {
      log(`subtitles: follow tick failed: ${error instanceof Error ? error.message : String(error)}`)
    } finally {
      followInFlight = false
    }
  }

  const saveDebugFrame = (dataUrl: string) => {
    try {
      const base64 = dataUrl.slice(dataUrl.indexOf(',') + 1)
      const file = path.join(app.getPath('temp'), 'hermes-subtitles-first-frame.png')

      fs.writeFileSync(file, Buffer.from(base64, 'base64'))
      log(`subtitles: first captured frame saved to ${file}`)
    } catch {
      // Diagnostics only — never fail the pipeline over it.
    }
  }

  const asBox = (value: unknown): SubtitleTextBox | null => {
    const bounds = asBounds(value)

    return bounds && bounds.width > 0 && bounds.height > 0 ? bounds : null
  }

  const redrawLast = () => {
    if (lastDrawn) {
      annotations.setChannelShapes('subtitles', lastDrawn.shapes, lastDrawn.display)
    }
  }

  const processCrop = (dataUrl: string, cropWidth: number, cropHeight: number, epoch: number): void => {
    if (!current || epoch !== current.epoch) {
      return
    }

    if (!current.debugFrameSaved) {
      current.debugFrameSaved = true
      saveDebugFrame(dataUrl)
    }

    const generation = ++frameGeneration
    const startedAt = Date.now()
    const sessionAtSubmit = current

    void postToBackend('/api/subtitles/process', {
      image_data_url: dataUrl,
      language: sessionAtSubmit.language,
      prev_text: sessionAtSubmit.lastSourceText,
      stream_id: sessionAtSubmit.streamId
    })
      .then(result => {
        if (!current || current !== sessionAtSubmit || generation !== frameGeneration) {
          return
        }

        current.lastLatencyMs = Date.now() - startedAt

        if (!result || result.ok !== true) {
          throw new Error(typeof result?.detail === 'string' ? result.detail : 'backend returned no result')
        }

        current.consecutiveBackendErrors = 0

        if (result.unchanged === true) {
          if (current.lastShapesAt > 0) {
            redrawLast()
          }

          return
        }

        const sourceText = typeof result.source_text === 'string' ? result.source_text : ''
        const text = typeof result.text === 'string' ? result.text.trim() : ''
        const box = asBox(result.box)

        current.lastSourceText = sourceText

        if (!text || !box) {
          lastDrawn = null
          current.lastShapesAt = 0
          annotations.clearChannel('subtitles')

          return
        }

        const shapes = subtitleShapes({
          band: current.band,
          box,
          cropHeight,
          cropWidth,
          display: current.display,
          text
        })

        if (shapes.length === 0) {
          return
        }

        lastDrawn = { display: current.display, shapes }
        current.lastShapesAt = Date.now()
        current.linesDrawn += 1
        annotations.setChannelShapes('subtitles', shapes, current.display)
      })
      .catch(error => {
        if (!current || current !== sessionAtSubmit) {
          return
        }

        current.consecutiveBackendErrors += 1
        log(
          `subtitles: process failed (${current.consecutiveBackendErrors}): ${error instanceof Error ? error.message : String(error)}`
        )

        if (current.consecutiveBackendErrors >= BACKEND_ERRORS_BEFORE_STOP) {
          stop('the backend kept failing to OCR/translate frames')
        }
      })
  }

  const sampleTick = async () => {
    if (!current || sampleInFlight) {
      return
    }

    sampleInFlight = true
    const sessionAtSample = current

    try {
      const display = screen.getDisplayMatching(sessionAtSample.windowBounds)
      const scale = display.scaleFactor || 1

      const sources = await desktopCapturer.getSources({
        fetchWindowIcons: false,
        thumbnailSize: {
          height: Math.max(8, Math.round(sessionAtSample.windowBounds.height * scale)),
          width: Math.max(8, Math.round(sessionAtSample.windowBounds.width * scale))
        },
        types: ['window']
      })

      if (!current || current !== sessionAtSample) {
        return
      }

      const hit = matchCapturerWindow(sources, sessionAtSample.windowId, sessionAtSample.windowRef)
      const source = hit ? sources.find(entry => entry.id === hit.id) : undefined

      if (!source || source.thumbnail.isEmpty()) {
        return
      }

      const size = source.thumbnail.getSize()
      const crop = cropRectFor(sessionAtSample.fractions, size.width, size.height)

      if (!crop) {
        return
      }

      const cropped = source.thumbnail.crop(crop)
      const shipped = shipSize(crop)
      const sized = cropped.getSize().width === shipped.width ? cropped : cropped.resize(shipped)
      const shippedSize = sized.getSize()
      const rgba = bgraToRgba(sized.toBitmap(), shippedSize.width, shippedSize.height)
      const hash = brightMaskHash(rgba, shippedSize.width, shippedSize.height)
      const now = Date.now()
      const changed = hammingDistance(hash, sessionAtSample.lastHash) > SAME_TEXT_MAX_DISTANCE

      if (!changed && now - sessionAtSample.lastSentAt < SUBTITLE_KEEPALIVE_MS) {
        return
      }

      sessionAtSample.lastHash = hash
      sessionAtSample.lastSentAt = now
      const dataUrl = `data:image/png;base64,${sized.toPNG().toString('base64')}`

      processCrop(dataUrl, shippedSize.width, shippedSize.height, sessionAtSample.epoch)
    } catch (error) {
      log(`subtitles: snapshot failed: ${error instanceof Error ? error.message : String(error)}`)
    } finally {
      sampleInFlight = false
    }
  }

  const startSampling = (hz: number) => {
    stopSampling()
    sampleTimer = setInterval(() => void sampleTick(), Math.round(1000 / Math.max(1, hz)))
    void sampleTick()
  }

  const start = async (req: Record<string, unknown>, senderBounds: AnnotationBounds | null) => {
    const language = typeof req.language === 'string' ? req.language.trim() : ''

    if (!language) {
      return { error: "start needs a language (e.g. 'pt', 'es', 'Portuguese').", success: false }
    }

    if (current) {
      stop('replaced by a new start')
    }

    const spec = typeof req.target === 'string' && req.target.trim() ? req.target.trim() : undefined
    const resolved = await resolveTarget(spec, senderBounds)

    if (resolved.error !== undefined) {
      return { error: resolved.error, success: false }
    }

    const target = resolved.window
    const display = screen.getDisplayMatching(target.bounds)
    const fraction = clampBandFraction(typeof req.band_fraction === 'number' ? req.band_fraction : undefined)
    const band = subtitleBand(target.bounds, display.bounds, fraction)

    if (!band) {
      return { error: `The "${target.app}" window is not visibly on any display.`, success: false }
    }

    const sampleHz = clampSampleHz(typeof req.sample_hz === 'number' ? req.sample_hz : undefined)

    current = {
      band,
      bandFraction: fraction,
      consecutiveBackendErrors: 0,
      debugFrameSaved: false,
      display: display.bounds,
      displayId: display.id,
      epoch: 1,
      fractions: bandFractions(band, target.bounds),
      language,
      lastHash: '',
      lastLatencyMs: 0,
      lastSentAt: 0,
      lastShapesAt: 0,
      lastSourceText: '',
      linesDrawn: 0,
      sampleHz,
      startedAt: Date.now(),
      streamId: `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`,
      targetSpec: spec,
      windowBounds: target.bounds,
      windowId: target.id,
      windowRef: { app: target.app, title: target.title }
    }

    followMisses = 0
    followTimer = setInterval(() => void followTick(), FOLLOW_INTERVAL_MS)
    startSampling(sampleHz)

    return {
      band,
      language,
      success: true,
      target: { app: target.app, title: target.title },
      watching: `bottom ${Math.round(fraction * 100)}% of the ${target.app} window`
    }
  }

  const status = (): Record<string, unknown> => {
    if (!current) {
      return { running: false, success: true }
    }

    return {
      band: current.band,
      language: current.language,
      last_latency_ms: current.lastLatencyMs,
      lines_translated: current.linesDrawn,
      running: true,
      running_seconds: Math.round((Date.now() - current.startedAt) / 1000),
      success: true,
      target: current.windowRef
    }
  }

  const control = async (request: unknown, senderBounds: AnnotationBounds | null): Promise<Record<string, unknown>> => {
    const req = (request && typeof request === 'object' ? request : {}) as Record<string, unknown>
    const action = typeof req.action === 'string' ? req.action.trim().toLowerCase() : ''

    if (action === 'start') {
      return start(req, senderBounds)
    }

    if (action === 'stop') {
      return stop()
    }

    if (action === 'status') {
      return status()
    }

    return { error: `action must be one of: ${SUPPORTED_ACTIONS.join(', ')}.`, success: false }
  }

  const close = () => {
    stop()
  }

  return { close, control }
}

export function registerSubtitleCaptureIpc(controller: SubtitleCaptureController): void {
  ipcMain.handle('hermes:subtitles:control', async (event, request) => {
    const sender = BrowserWindow.fromWebContents(event.sender)
    const senderBounds = sender && !sender.isDestroyed() ? sender.getBounds() : null

    try {
      return await controller.control(request, senderBounds)
    } catch (error) {
      return { error: error instanceof Error ? error.message : String(error), success: false }
    }
  })
}
