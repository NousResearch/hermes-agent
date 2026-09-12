import { type BrowserWindow, desktopCapturer, screen } from 'electron'

import { detectHorizontalContrastSurface } from './pet-overlay-contrast'

export interface PetOverlayVisualLedge {
  left: number
  right: number
  y: number
}

export interface PetOverlayVisualCapture {
  ledges: PetOverlayVisualLedge[]
  revision: number
}

export interface PetOverlayVisualFrame {
  bitmap: Uint8Array
  displayBounds: { height: number; width: number; x: number; y: number }
  height: number
  width: number
}

export interface PetOverlayVisualProbe {
  motionProbe?: boolean
  overlayBounds: { height: number; width: number; x: number; y: number }
  petHeight?: unknown
  petWidth: unknown
  scanMode?: 'destination' | 'landing' | 'support'
  workArea: { height: number; width: number; x: number; y: number }
}

const PET_FOOT_INSET_PX = 24
const SCAN_ABOVE_FEET_PX = 6
const SCAN_BELOW_FEET_PX = 220
const SUPPORT_SCAN_BELOW_FEET_PX = 16
const CAPTURE_TIMEOUT_MS = 550
const DEFAULT_REUSED_CAPTURE_MAX_AGE_MS = 750

let screenCaptureInFlight: ReturnType<typeof desktopCapturer.getSources> | undefined

interface CachedVisualFrame extends PetOverlayVisualFrame {
  capturedAt: number
  displayId: string
  revision: number
}

let cachedVisualFrame: CachedVisualFrame | undefined
let visualCaptureRevision = 0

async function screenSources(width: number, height: number) {
  if (!screenCaptureInFlight) {
    const capture = desktopCapturer.getSources({
      fetchWindowIcons: false,
      thumbnailSize: { height, width },
      types: ['screen']
    })

    const trackedCapture = capture.finally(() => {
      if (screenCaptureInFlight === trackedCapture) {
        screenCaptureInFlight = undefined
      }
    })

    screenCaptureInFlight = trackedCapture
  }

  const activeCapture = screenCaptureInFlight
  let timeout: ReturnType<typeof setTimeout> | undefined

  try {
    return await Promise.race([
      activeCapture,
      new Promise<never>((_resolve, reject) => {
        timeout = setTimeout(() => {
          if (screenCaptureInFlight === activeCapture) {
            screenCaptureInFlight = undefined
          }

          reject(new Error('Pet overlay screen capture timed out'))
        }, CAPTURE_TIMEOUT_MS)
      })
    ])
  } finally {
    if (timeout) {
      clearTimeout(timeout)
    }
  }
}

/** Analyze an already captured frame. No Electron state or platform check is involved. */
export function detectPetOverlayVisualLedges(
  frame: PetOverlayVisualFrame,
  {
    motionProbe = false,
    overlayBounds,
    petWidth: requestedPetWidth,
    scanMode = 'landing',
    workArea
  }: PetOverlayVisualProbe
): PetOverlayVisualLedge[] {
  const footX = overlayBounds.x + overlayBounds.width / 2
  const footY = overlayBounds.y + overlayBounds.height - PET_FOOT_INSET_PX
  const localFootX = footX - frame.displayBounds.x
  const localFootY = footY - frame.displayBounds.y
  const localPetTopY = overlayBounds.y - frame.displayBounds.y
  const workAreaTop = workArea.y - frame.displayBounds.y
  const workAreaBottom = workArea.y + workArea.height - frame.displayBounds.y
  const petWidth = Math.max(24, Math.min(256, Number(requestedPetWidth) || 64))
  const motionMinimumSpan = Math.max(120, Math.round(petWidth * 2.25))
  const motionScanWidth = Math.max(192, Math.round(petWidth * 3))
  const destinationScanWidth = Math.max(36, Math.round(petWidth * 1.5))
  const destinationMinimumSpan = destinationScanWidth
  const scanBelow = motionProbe && scanMode === 'support' ? SUPPORT_SCAN_BELOW_FEET_PX : SCAN_BELOW_FEET_PX
  const bitmap = { data: frame.bitmap, height: frame.height, width: frame.width }

  const toScreenLedge = (surface: { left: number; right: number; y: number }): PetOverlayVisualLedge => ({
    left: frame.displayBounds.x + surface.left,
    right: frame.displayBounds.x + surface.right,
    y: frame.displayBounds.y + surface.y
  })

  const traceSelectedSurface = (surface: { left: number; right: number; y: number }) =>
    detectHorizontalContrastSurface(bitmap, {
      centerX: localFootX,
      footY: localFootY,
      fromY: surface.y,
      maximumScanWidth: frame.width,
      minimumSpan: destinationMinimumSpan,
      petTopY: localPetTopY,
      petWidth,
      toY: surface.y
    }) ?? surface

  if (scanMode === 'destination') {
    const sharedProbe = {
      centerX: localFootX,
      footY: localFootY,
      maximumScanWidth: destinationScanWidth,
      minimumSpan: destinationMinimumSpan,
      petTopY: localPetTopY,
      petWidth
    }

    const support = detectHorizontalContrastSurface(bitmap, {
      ...sharedProbe,
      fromY: localFootY - SCAN_ABOVE_FEET_PX,
      toY: Math.min(localFootY + SUPPORT_SCAN_BELOW_FEET_PX, workAreaBottom - 3)
    })

    const scanEndY = Math.min(frame.height - 3, workAreaBottom - 3)
    let nextY = Math.max(2, workAreaTop + 2)
    const destinations: NonNullable<ReturnType<typeof detectHorizontalContrastSurface>>[] = []
    const maximumSurfaceCount = Math.max(1, Math.ceil((scanEndY - nextY + 1) / 4))

    for (let index = 0; index < maximumSurfaceCount && nextY <= scanEndY; index += 1) {
      const surface = detectHorizontalContrastSurface(bitmap, {
        ...sharedProbe,
        fromY: nextY,
        toY: scanEndY
      })

      if (!surface) {
        break
      }

      destinations.push(surface)
      nextY = Math.max(nextY + 1, surface.y + 4)
    }

    const tracedSupport = support ? traceSelectedSurface(support) : null
    const surfaces = tracedSupport ? [tracedSupport] : []

    for (const destination of destinations) {
      if (!tracedSupport || Math.abs(destination.y - tracedSupport.y) > 4) {
        surfaces.push(destination)
      }
    }

    return surfaces.map(toScreenLedge)
  }

  const surface = detectHorizontalContrastSurface(bitmap, {
    centerX: localFootX,
    footY: localFootY,
    fromY: localFootY - SCAN_ABOVE_FEET_PX,
    maximumScanWidth: motionProbe ? motionScanWidth : undefined,
    minimumSpan: motionProbe ? motionMinimumSpan : undefined,
    petTopY: localPetTopY,
    petWidth,
    toY: Math.min(localFootY + scanBelow, workAreaBottom - 3)
  })

  return surface ? [toScreenLedge(surface)] : []
}

const captureMaxAge = (requested: unknown): number => {
  const value = Number(requested)

  return Number.isFinite(value) && value >= 0 ? Math.min(2000, Math.round(value)) : DEFAULT_REUSED_CAPTURE_MAX_AGE_MS
}

/**
 * Local-only visual fallback for Windows. The screenshot stays in memory and
 * only the detected line coordinates leave this function; no pixels, titles,
 * or application identities are persisted or sent to the renderer.
 */
export async function captureVisualLedgeBelow(
  petOverlayWindow: BrowserWindow,
  requestedPetWidth: unknown,
  reuseCapture = false,
  scanMode: 'destination' | 'landing' | 'support' = 'landing',
  requestedPetHeight?: unknown,
  requestedMaxCaptureAgeMs?: unknown
): Promise<PetOverlayVisualCapture> {
  if (process.platform !== 'win32') {
    return { ledges: [], revision: 0 }
  }

  try {
    const overlayBounds = petOverlayWindow.getBounds()
    const display = screen.getDisplayMatching(overlayBounds)
    const width = Math.max(1, Math.round(display.bounds.width))
    const height = Math.max(1, Math.round(display.bounds.height))
    const displayId = String(display.id)
    const maximumAgeMs = captureMaxAge(requestedMaxCaptureAgeMs)

    const cachedFrameMatches =
      cachedVisualFrame?.displayId === displayId &&
      cachedVisualFrame.width === width &&
      cachedVisualFrame.height === height &&
      cachedVisualFrame.displayBounds.x === display.bounds.x &&
      cachedVisualFrame.displayBounds.y === display.bounds.y

    const cacheIsFresh = reuseCapture && cachedFrameMatches && Date.now() - cachedVisualFrame.capturedAt <= maximumAgeMs

    if (!cacheIsFresh) {
      // A failed refresh must never expose the previous scene to a later motion
      // probe: clear it before awaiting the native capture.
      cachedVisualFrame = undefined
      const sources = await screenSources(width, height)

      const source =
        sources.find(candidate => candidate.display_id === displayId) ?? (sources.length === 1 ? sources[0] : undefined)

      if (!source || source.thumbnail.isEmpty()) {
        return { ledges: [], revision: ++visualCaptureRevision }
      }

      const image = source.thumbnail.resize({ height, quality: 'good', width })
      const size = image.getSize()
      const bitmap = image.toBitmap()

      if (size.width !== width || size.height !== height || bitmap.length < width * height * 4) {
        return { ledges: [], revision: ++visualCaptureRevision }
      }

      cachedVisualFrame = {
        bitmap,
        capturedAt: Date.now(),
        displayBounds: { ...display.bounds },
        displayId,
        height,
        revision: ++visualCaptureRevision,
        width
      }
    }

    const frame = cachedVisualFrame

    if (!frame) {
      return { ledges: [], revision: visualCaptureRevision }
    }

    return {
      ledges: detectPetOverlayVisualLedges(frame, {
        motionProbe: reuseCapture,
        overlayBounds,
        petHeight: requestedPetHeight,
        petWidth: requestedPetWidth,
        scanMode,
        workArea: display.workArea
      }),
      revision: frame.revision
    }
  } catch {
    cachedVisualFrame = undefined

    return { ledges: [], revision: ++visualCaptureRevision }
  }
}
