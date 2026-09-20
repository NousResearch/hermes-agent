import { useCallback, useState } from 'react'

import { useI18n } from '@/i18n'
import { notify, notifyError } from '@/store/notifications'

const MIME_EXTENSIONS: Record<string, string> = {
  'image/bmp': '.bmp',
  'image/gif': '.gif',
  'image/jpeg': '.jpg',
  'image/png': '.png',
  'image/svg+xml': '.svg',
  'image/webp': '.webp'
}

const KNOWN_IMAGE_EXTENSION_RE = /\.(?:apng|avif|bmp|gif|ico|jpe?g|png|svg|tiff?|webp)$/i

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

function isMissingIpcHandler(error: unknown): boolean {
  const message = error instanceof Error ? error.message : typeof error === 'string' ? error : ''

  return message.includes("No handler registered for 'hermes:saveImageFromUrl'")
}

async function startBrowserDownload(src: string) {
  const response = await fetch(src)

  if (!response.ok) {
    throw new Error(`Could not fetch image: ${response.status}`)
  }

  const blob = await response.blob()
  const blobUrl = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = blobUrl
  link.download = downloadFilename(src, blob.type)
  link.rel = 'noopener noreferrer'
  document.body.appendChild(link)
  link.click()
  link.remove()
  window.setTimeout(() => URL.revokeObjectURL(blobUrl), 30_000)
}

/** Save an image to disk via the desktop IPC bridge, falling back to a browser
 *  download when the handler is unavailable (older shell / web preview). */
export function useImageDownload(src?: string) {
  const { t } = useI18n()
  const copy = t.desktop
  const [saving, setSaving] = useState(false)

  const download = useCallback(async () => {
    if (!src || saving) {
      return
    }

    setSaving(true)

    try {
      if (window.hermesDesktop?.saveImageFromUrl) {
        if (await window.hermesDesktop.saveImageFromUrl(src)) {
          notify({ kind: 'success', title: copy.imageSaved, message: imageFilename(src) })
        }

        return
      }

      await startBrowserDownload(src)
    } catch (error) {
      if (isMissingIpcHandler(error)) {
        try {
          await startBrowserDownload(src)
          notify({ kind: 'info', title: copy.downloadStarted, message: copy.restartToUseSaveImage })
        } catch (fallbackError) {
          notifyError(fallbackError, copy.restartToSaveImages)
        }

        return
      }

      notifyError(error, copy.imageDownloadFailed)
    } finally {
      setSaving(false)
    }
  }, [copy, saving, src])

  return { download, saving }
}
