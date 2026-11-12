/**
 * Downscale a data URL for preview rendering.
 *
 * Large images (Retina screenshots, 6000×4000+) cause Chromium to build very
 * large paint operations when the original is assigned to an <img>. This
 * utility uses createImageBitmap + OffscreenCanvas to resize before display.
 *
 * Calls share a one-at-a-time queue. Multi-image attach flows must not keep
 * dozens of decoded full-resolution bitmaps alive concurrently while their
 * 512px thumbnails are produced.
 *
 * @param dataUrl  The full-resolution data URL (data:image/...;base64,...)
 * @param maxLongEdge  Maximum pixel dimension on the longest side (default 512)
 * @returns A downscaled PNG data URL, the original if already small enough, or
 *          a 1×1 transparent PNG placeholder if downscaling fails.
 */

/** 1×1 transparent PNG — fail closed so the pill never receives the original. */
export const FALLBACK_PLACEHOLDER =
  'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+GkZcAAAAASUVORK5CYII='

// Composer pills render at 32 CSS pixels. A 512px source stays sharp on
// high-density displays while keeping a square RGBA decode to 1 MiB. A 2048px
// thumbnail can decode to 16 MiB and reproduce Chromium's paint-op pressure.
const DEFAULT_MAX_LONG_EDGE = 512

let resizeQueue = Promise.resolve()

async function downscaleDataUrl(dataUrl: string, maxLongEdge: number): Promise<string> {
  try {
    // fetch(data:) decodes base64 natively in C++ and avoids an O(n) atob loop.
    const blob = await fetch(dataUrl).then(response => response.blob())
    const bitmap = await createImageBitmap(blob)

    try {
      const { width, height } = bitmap
      const longEdge = Math.max(width, height)

      if (longEdge <= maxLongEdge) {
        return dataUrl
      }

      const scale = maxLongEdge / longEdge
      const newWidth = Math.max(1, Math.round(width * scale))
      const newHeight = Math.max(1, Math.round(height * scale))
      const canvas = new OffscreenCanvas(newWidth, newHeight)
      const ctx = canvas.getContext('2d')

      if (!ctx) {
        return FALLBACK_PLACEHOLDER
      }

      ctx.drawImage(bitmap, 0, 0, newWidth, newHeight)

      const resultBlob = await canvas.convertToBlob({ type: 'image/png' })
      const reader = new FileReader()

      return await new Promise<string>(resolve => {
        reader.onloadend = () => resolve(typeof reader.result === 'string' ? reader.result : FALLBACK_PLACEHOLDER)
        reader.onabort = () => resolve(FALLBACK_PLACEHOLDER)
        reader.onerror = () => resolve(FALLBACK_PLACEHOLDER)
        reader.readAsDataURL(resultBlob)
      })
    } finally {
      bitmap.close()
    }
  } catch {
    return FALLBACK_PLACEHOLDER
  }
}

export async function downscaleDataUrlForPreview(
  dataUrl: string,
  maxLongEdge = DEFAULT_MAX_LONG_EDGE
): Promise<string> {
  if (!dataUrl.startsWith('data:image/') || dataUrl.indexOf(',') === -1) {
    return dataUrl
  }

  // Never hand a full-resolution image to the pill if resize support is absent.
  // A tiny placeholder is preferable to recreating the renderer failure this
  // helper exists to prevent.
  if (typeof createImageBitmap !== 'function' || typeof OffscreenCanvas !== 'function') {
    return FALLBACK_PLACEHOLDER
  }

  const task = resizeQueue.then(() => downscaleDataUrl(dataUrl, maxLongEdge))

  resizeQueue = task.then(
    () => undefined,
    () => undefined
  )

  return task
}
