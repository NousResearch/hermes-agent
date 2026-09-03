// subtitle-capture.ts — pure geometry and shape-building for live subtitles.
//
// Backs the live-subtitle session (subtitle-capture-session.ts): a hidden
// renderer samples the subtitle band of the target window, the backend OCRs
// and translates changed frames, and main paints the translation over the
// original line through the screen-annotation overlay's `subtitles` channel.
// Everything Electron-free lives here so band placement, crop mapping and the
// painted shapes are unit-testable without booting Electron.

import type { AnnotationBounds, MappedAnnotationShape } from './screen-annotations'

// Sampling bounds. The floor keeps the loop from being pointless (subtitle
// lines last ~2s); the ceiling keeps a misconfigured session from burning a
// core drawing crops nobody reads faster than lines can change.
export const SUBTITLE_SAMPLE_HZ_DEFAULT = 4
export const SUBTITLE_SAMPLE_HZ_MIN = 1
export const SUBTITLE_SAMPLE_HZ_MAX = 8

// How much of the target window's height the subtitle band covers, measured
// up from its bottom edge. Streaming players draw captions in the bottom
// quarter; the default adds margin for players that float them a little higher.
export const SUBTITLE_BAND_FRACTION_DEFAULT = 0.28
export const SUBTITLE_BAND_FRACTION_MIN = 0.1
export const SUBTITLE_BAND_FRACTION_MAX = 0.5

export function clampSampleHz(raw: unknown): number {
  const value = typeof raw === 'number' && Number.isFinite(raw) ? raw : SUBTITLE_SAMPLE_HZ_DEFAULT

  return Math.min(SUBTITLE_SAMPLE_HZ_MAX, Math.max(SUBTITLE_SAMPLE_HZ_MIN, value))
}

export function clampBandFraction(raw: unknown): number {
  const value = typeof raw === 'number' && Number.isFinite(raw) ? raw : SUBTITLE_BAND_FRACTION_DEFAULT

  return Math.min(SUBTITLE_BAND_FRACTION_MAX, Math.max(SUBTITLE_BAND_FRACTION_MIN, value))
}

/**
 * The screen region (DIP) the capture loop watches: the bottom `fraction` of
 * the target window, clipped to the display it sits on — a window hanging off
 * the display edge must not produce crop coordinates outside the stream.
 * Returns null when the clipped band has no area (window off-screen).
 */
export function subtitleBand(
  windowBounds: AnnotationBounds,
  display: AnnotationBounds,
  fraction: number
): AnnotationBounds | null {
  const bandHeight = Math.max(1, Math.round(windowBounds.height * clampBandFraction(fraction)))
  const raw = {
    height: bandHeight,
    width: windowBounds.width,
    x: windowBounds.x,
    y: windowBounds.y + windowBounds.height - bandHeight
  }

  const left = Math.max(raw.x, display.x)
  const top = Math.max(raw.y, display.y)
  const right = Math.min(raw.x + raw.width, display.x + display.width)
  const bottom = Math.min(raw.y + raw.height, display.y + display.height)

  if (right - left < 8 || bottom - top < 8) {
    return null
  }

  return { height: bottom - top, width: right - left, x: left, y: top }
}

/** The band expressed as fractions of its display. The capture renderer
 *  multiplies these by the live video dimensions, so the same config works
 *  whatever pixel size the display stream arrives at (Retina or not). */
export interface SubtitleBandFractions {
  height: number
  left: number
  top: number
  width: number
}

export function bandFractions(band: AnnotationBounds, display: AnnotationBounds): SubtitleBandFractions {
  return {
    height: band.height / display.height,
    left: (band.x - display.x) / display.width,
    top: (band.y - display.y) / display.height,
    width: band.width / display.width
  }
}

/** Union bounding box of the OCR'd subtitle lines, in crop pixels. */
export interface SubtitleTextBox {
  height: number
  width: number
  x: number
  y: number
}

const SUBTITLE_FONT_MIN = 14
const SUBTITLE_FONT_MAX = 64
const COVER_PAD = 10
const LINE_HEIGHT = 1.3

export interface SubtitleShapeParams {
  /** The OCR'd text's union box, in submitted-crop pixels. */
  box: SubtitleTextBox
  /** Band position on screen, in DIP. */
  band: AnnotationBounds
  /** Display origin the overlay window covers (overlay-local = screen − origin). */
  display: AnnotationBounds
  /** Pixel size of the crop the box coordinates live in. */
  cropWidth: number
  cropHeight: number
  /** Translated text; newlines are line breaks. */
  text: string
}

/**
 * The painted result: an opaque cover over the original line and the
 * translation centered on it, both `steady` (no entrance/pulse animation —
 * these replace each other every couple of seconds).
 *
 * Font size derives from the ORIGINAL text's rendered height, so the
 * translation visually matches the player's own caption size; the cover pads
 * beyond the box so antialiased fringes of the original don't peek out.
 */
export function subtitleShapes(params: SubtitleShapeParams): MappedAnnotationShape[] {
  const { band, box, cropHeight, cropWidth, display } = params
  const text = params.text.trim()

  if (!text || cropWidth <= 0 || cropHeight <= 0 || box.width <= 0 || box.height <= 0) {
    return []
  }

  const scaleX = band.width / cropWidth
  const scaleY = band.height / cropHeight

  const cover = {
    height: Math.round(box.height * scaleY) + COVER_PAD * 2,
    width: Math.round(box.width * scaleX) + COVER_PAD * 2,
    x: Math.round(band.x + box.x * scaleX - display.x) - COVER_PAD,
    y: Math.round(band.y + box.y * scaleY - display.y) - COVER_PAD
  }

  const lines = text.split('\n').filter(line => line.trim().length > 0)
  const lineCount = Math.max(1, lines.length)
  const originalLineHeight = (box.height * scaleY) / lineCount
  const fontSize = Math.min(SUBTITLE_FONT_MAX, Math.max(SUBTITLE_FONT_MIN, Math.round(originalLineHeight * 0.78)))

  // Center the text block vertically inside the cover. SVG text anchors at the
  // baseline, so the first line's y is the block top plus most of a line.
  const blockHeight = fontSize * (1 + LINE_HEIGHT * (lineCount - 1))
  const firstBaseline = Math.round(cover.y + (cover.height - blockHeight) / 2 + fontSize * 0.85)

  return [
    { color: 'black', fill: true, kind: 'rect', steady: true, ...cover },
    {
      color: 'white',
      fontSize,
      kind: 'label',
      steady: true,
      text: lines.join('\n'),
      x: Math.round(cover.x + cover.width / 2),
      y: firstBaseline
    }
  ]
}

/** Hash grid: 16x8 cells = 128 bits, packed into 32 hex chars. */
export const HASH_WIDTH = 16
export const HASH_HEIGHT = 8

/** Luma threshold (0-255) above which a pixel counts as "bright" — subtitle
 *  text is white or near-white in every mainstream player. */
export const BRIGHT_LUMA = 200

const CELL_ON_SHARE = 0.08

/** Hamming distance at or below this is "same text" — antialiasing wobble and
 *  compression shimmer flip a few cells frame to frame. */
export const SAME_TEXT_MAX_DISTANCE = 6

/** Successive identical-hash frames still re-send occasionally so the overlay
 *  hold refreshes (a paused movie must keep its translated line up). */
export const SUBTITLE_KEEPALIVE_MS = 8000

/** Bright-mask hash of an RGBA buffer (any size). Cells are averaged over
 *  their pixel block, so callers can pass the full-resolution band directly. */
export function brightMaskHash(rgba: Uint8ClampedArray, width: number, height: number): string {
  const bits: number[] = []

  for (let cellY = 0; cellY < HASH_HEIGHT; cellY += 1) {
    for (let cellX = 0; cellX < HASH_WIDTH; cellX += 1) {
      const x0 = Math.floor((cellX * width) / HASH_WIDTH)
      const x1 = Math.max(x0 + 1, Math.floor(((cellX + 1) * width) / HASH_WIDTH))
      const y0 = Math.floor((cellY * height) / HASH_HEIGHT)
      const y1 = Math.max(y0 + 1, Math.floor(((cellY + 1) * height) / HASH_HEIGHT))

      let bright = 0
      let total = 0

      for (let y = y0; y < y1; y += 1) {
        for (let x = x0; x < x1; x += 1) {
          const offset = (y * width + x) * 4
          const luma = 0.299 * rgba[offset] + 0.587 * rgba[offset + 1] + 0.114 * rgba[offset + 2]

          if (luma >= BRIGHT_LUMA) {
            bright += 1
          }

          total += 1
        }
      }

      bits.push(bright / total >= CELL_ON_SHARE ? 1 : 0)
    }
  }

  let hex = ''

  for (let index = 0; index < bits.length; index += 4) {
    hex += ((bits[index] << 3) | (bits[index + 1] << 2) | (bits[index + 2] << 1) | bits[index + 3]).toString(16)
  }

  return hex
}

export function hammingDistance(a: string, b: string): number {
  if (a.length !== b.length) {
    return Math.max(a.length, b.length) * 4
  }

  let distance = 0

  for (let index = 0; index < a.length; index += 1) {
    let xor = parseInt(a[index], 16) ^ parseInt(b[index], 16)

    while (xor) {
      distance += xor & 1
      xor >>= 1
    }
  }

  return distance
}

export interface PixelRect {
  height: number
  width: number
  x: number
  y: number
}

/** Band fractions × live video dimensions → integer crop rect, clamped inside
 *  the frame. Returns null when the result has no usable area. */
export function cropRectFor(fractions: SubtitleBandFractions, videoWidth: number, videoHeight: number): PixelRect | null {
  if (videoWidth <= 0 || videoHeight <= 0) {
    return null
  }

  const x = Math.max(0, Math.round(fractions.left * videoWidth))
  const y = Math.max(0, Math.round(fractions.top * videoHeight))
  const width = Math.min(videoWidth - x, Math.round(fractions.width * videoWidth))
  const height = Math.min(videoHeight - y, Math.round(fractions.height * videoHeight))

  if (width < 8 || height < 8) {
    return null
  }

  return { height, width, x, y }
}

export const SHIP_MAX_WIDTH = 1280

export function shipSize(crop: PixelRect): { height: number; width: number } {
  if (crop.width <= SHIP_MAX_WIDTH) {
    return { height: crop.height, width: crop.width }
  }

  const scale = SHIP_MAX_WIDTH / crop.width

  return { height: Math.max(8, Math.round(crop.height * scale)), width: SHIP_MAX_WIDTH }
}

export interface CapturerSourceHit {
  id: string
  name: string
}

/** Pick the desktopCapturer window source for the session's target. Chromium
 *  ids look like `window:<nativeId>:0`; names are the OS title (often
 *  "Netflix - Google Chrome"). */
export function matchCapturerWindow(
  sources: CapturerSourceHit[],
  windowId: number,
  ref: { app: string; title: string }
): CapturerSourceHit | null {
  const prefix = `window:${windowId}:`
  const byId = sources.find(source => source.id === `${prefix}0` || source.id.startsWith(prefix))

  if (byId) {
    return byId
  }

  const title = ref.title.trim().toLowerCase()
  const app = ref.app.trim().toLowerCase()

  if (title) {
    const exact = sources.find(source => source.name.trim().toLowerCase() === title)

    if (exact) {
      return exact
    }

    const named = sources.find(source => source.name.toLowerCase().includes(title))

    if (named) {
      return named
    }
  }

  if (app) {
    return sources.find(source => source.name.toLowerCase().includes(app)) ?? null
  }

  return null
}
