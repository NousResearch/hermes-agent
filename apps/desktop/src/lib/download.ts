const MIME_EXTENSIONS: Record<string, string> = {
  'image/bmp': '.bmp',
  'image/gif': '.gif',
  'image/jpeg': '.jpg',
  'image/png': '.png',
  'image/svg+xml': '.svg',
  'image/webp': '.webp'
}

const KNOWN_IMAGE_EXTENSION_RE = /\.(?:apng|avif|bmp|gif|ico|jpe?g|png|svg|tiff?|webp)$/i

// The download manager may resolve the blob URL after click() returns, so
// release it only once the transfer has had ample time to start.
const BLOB_URL_TTL_MS = 60_000

export function imageFilename(src?: string): string {
  if (!src) {
    return 'image'
  }

  try {
    return new URL(src, window.location.href).pathname.split('/').filter(Boolean).pop() || 'image'
  } catch {
    return src.split(/[\\/]/).filter(Boolean).pop() || 'image'
  }
}

/** Filename for a browser-anchor download. Generated-image URLs (fal.media
 *  etc.) often end in an extensionless content hash — without an extension the
 *  OS save dialog shows "All Files" and the saved file won't open by
 *  double-click, so append one derived from the blob's MIME type. */
export function downloadFilename(src: string, mimeType?: string): string {
  const base = imageFilename(src)

  if (KNOWN_IMAGE_EXTENSION_RE.test(base)) {
    return base
  }

  const type = String(mimeType || '')
    .split(';')[0]
    .trim()
    .toLowerCase()

  return `${base}${MIME_EXTENSIONS[type] || '.png'}`
}

/** Hand `href` to the browser's download manager. An empty `filename` lets the
 *  response's Content-Disposition (or the URL) name the file. */
export function clickDownloadLink(href: string, filename: string): void {
  const link = document.createElement('a')
  link.href = href
  link.download = filename
  link.rel = 'noopener noreferrer'
  link.style.display = 'none'
  document.body.append(link)
  link.click()
  link.remove()
}

/** Save in-memory bytes as `filename`. Electron's default will-download
 *  behavior shows the OS save dialog, so this needs no dedicated IPC handler. */
export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob)
  clickDownloadLink(url, filename)
  window.setTimeout(() => URL.revokeObjectURL(url), BLOB_URL_TTL_MS)
}
