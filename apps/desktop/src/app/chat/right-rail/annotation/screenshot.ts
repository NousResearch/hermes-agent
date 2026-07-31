/**
 * Screenshot helpers for the preview annotation layer.
 *
 * Rect math lives here (pure, unit-testable); the actual capture goes through
 * the preview <webview>'s `capturePage(rect)`, which resolves to a NativeImage
 * in Electron. We convert that to a PNG data URL for the composer.
 */

export interface ViewportSize {
  pageHeight: number
  pageWidth: number
}

export interface SourceRect {
  height: number
  width: number
  x: number
  y: number
}

export interface CaptureRect {
  height: number
  width: number
  x: number
  y: number
}

export const DEFAULT_PADDING_PX = 20

/**
 * Pads `rect` by `padding` on every side, then clamps to the page bounds and
 * rounds to integers (Electron's capturePage rejects fractional values and
 * out-of-bounds rects).
 */
export function computeCaptureRect(
  rect: SourceRect,
  page: ViewportSize,
  padding = DEFAULT_PADDING_PX
): CaptureRect {
  const x = Math.max(0, Math.floor(rect.x - padding))
  const y = Math.max(0, Math.floor(rect.y - padding))
  const right = Math.min(page.pageWidth, Math.ceil(rect.x + rect.width + padding))
  const bottom = Math.min(page.pageHeight, Math.ceil(rect.y + rect.height + padding))

  return {
    height: Math.max(1, bottom - y),
    width: Math.max(1, right - x),
    x,
    y
  }
}

interface CapturableWebview {
  capturePage?: (rect?: CaptureRect) => Promise<{ toDataURL: () => string }>
}

/**
 * Captures the region around `rect` from the given webview and returns a PNG
 * data URL. Returns null when the webview doesn't expose capturePage (older
 * Electron, non-webview target) — callers treat a missing screenshot as an
 * optional enhancement, never a failure.
 */
export async function captureRegionDataUrl(
  webview: CapturableWebview | null,
  rect: SourceRect,
  page: ViewportSize,
  padding = DEFAULT_PADDING_PX
): Promise<string | null> {
  if (!webview || typeof webview.capturePage !== 'function') {
    return null
  }

  const captureRect = computeCaptureRect(rect, page, padding)

  try {
    const image = await webview.capturePage(captureRect)
    return image.toDataURL()
  } catch {
    return null
  }
}
