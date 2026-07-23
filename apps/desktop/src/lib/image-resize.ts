/**
 * Downscale a data URL for preview rendering.
 *
 * Large images (Retina screenshots, 6000×4000+) cause the Chromium main thread
 * to block during <img> decode because macOS ImageIO/vImage decodes synchronously.
 * This utility uses createImageBitmap + OffscreenCanvas to resize *before* the
 * data URL is assigned to an <img> element, keeping the preview fast.
 *
 * Full-resolution bytes are still saved to disk for the model — only the preview
 * thumbnail is downscaled.
 *
 * @param dataUrl  The full-resolution data URL (data:image/...;base64,...)
 * @param maxLongEdge  Maximum pixel dimension on the longest side (default 2048)
 * @returns  A downscaled data URL (PNG), the original if already small enough,
 *           or a 1×1 transparent PNG placeholder if downscaling fails.
 */

/** 1×1 transparent PNG — used when downscaling fails so the UI never receives a multi-MB data URL. */
const FALLBACK_PLACEHOLDER =
  'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+GkZcAAAAASUVORK5CYII='

export async function downscaleDataUrlForPreview(
  dataUrl: string,
  maxLongEdge = 2048
): Promise<string> {
  // Guard: createImageBitmap and OffscreenCanvas are not available in jsdom
  // (test environment) or very old Chromium builds. Return the original if missing.
  if (typeof createImageBitmap !== 'function' || typeof OffscreenCanvas !== 'function') {
    return dataUrl
  }

  const commaIndex = dataUrl.indexOf(',')

  if (commaIndex === -1) {
    return dataUrl
  }

  try {
    // fetch(dataUrl) decodes base64 natively in C++ — avoids the O(n) atob()
    // + charCodeAt loop that creates main-thread jank for multi-MB images.
    const blob = await fetch(dataUrl).then(r => r.blob())

    // createImageBitmap decodes off the main thread in Chromium, but the
    // operation may still be costly for >5 MB retina screenshots; we then
    // resize quickly with OffscreenCanvas.
    const bitmap = await createImageBitmap(blob)

    try {
      const { width, height } = bitmap
      const longEdge = Math.max(width, height)

      if (longEdge <= maxLongEdge) {
        return dataUrl
      }

      const scale = maxLongEdge / longEdge
      const newWidth = Math.round(width * scale)
      const newHeight = Math.round(height * scale)

      const canvas = new OffscreenCanvas(newWidth, newHeight)
      const ctx = canvas.getContext('2d')

      if (!ctx) {
        return FALLBACK_PLACEHOLDER
      }

      ctx.drawImage(bitmap, 0, 0, newWidth, newHeight)

      const resultBlob = await canvas.convertToBlob({ type: 'image/png' })
      const reader = new FileReader()

      return new Promise<string>(resolve => {
        reader.onloadend = () => {
          if (typeof reader.result === 'string') {
            resolve(reader.result)
          } else {
            resolve(FALLBACK_PLACEHOLDER)
          }
        }
        reader.onerror = () => resolve(FALLBACK_PLACEHOLDER)
        reader.readAsDataURL(resultBlob)
      })
    } finally {
      bitmap.close()
    }
  } catch {
    // Downscaling failed (unsupported format, OOM, etc.) — return a tiny
    // placeholder rather than the original multi-MB data URL, which would
    // re-introduce the main-thread decode freeze this function exists to prevent.
    return FALLBACK_PLACEHOLDER
  }
}
