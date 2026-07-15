import { useCallback, useState } from 'react'

import { useI18n } from '@/i18n'
import { notify, notifyError } from '@/store/notifications'

export function imageFilename(src?: string, suggestedFilename?: string): string {
  if (suggestedFilename) {
    return suggestedFilename.split(/[\\/]/).filter(Boolean).pop() || 'image'
  }

  if (!src || /^data:/i.test(src.trim())) {
    return 'image'
  }

  try {
    const basename = new URL(src.trim(), window.location.href).pathname.split('/').filter(Boolean).pop() || 'image'

    try {
      return decodeURIComponent(basename)
    } catch {
      return basename
    }
  } catch {
    return src.split(/[\\/]/).filter(Boolean).pop() || 'image'
  }
}

function isMissingIpcHandler(error: unknown): boolean {
  const message = error instanceof Error ? error.message : typeof error === 'string' ? error : ''

  return message.includes("No handler registered for 'hermes:saveImageFromUrl'")
}

async function startBrowserDownload(src: string, suggestedFilename?: string) {
  const response = await fetch(src)

  if (!response.ok) {
    throw new Error(`Could not fetch image: ${response.status}`)
  }

  const blobUrl = URL.createObjectURL(await response.blob())
  const link = document.createElement('a')
  link.href = blobUrl
  link.download = imageFilename(src, suggestedFilename)
  link.rel = 'noopener noreferrer'
  document.body.appendChild(link)
  link.click()
  link.remove()
  window.setTimeout(() => URL.revokeObjectURL(blobUrl), 30_000)
}

/** Save an image to disk via the desktop IPC bridge, falling back to a browser
 *  download when the handler is unavailable (older shell / web preview). */
export function useImageDownload(src?: string, suggestedFilename?: string) {
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
        if (await window.hermesDesktop.saveImageFromUrl(src, suggestedFilename)) {
          notify({ kind: 'success', title: copy.imageSaved, message: imageFilename(src, suggestedFilename) })
        }

        return
      }

      await startBrowserDownload(src, suggestedFilename)
    } catch (error) {
      if (isMissingIpcHandler(error)) {
        try {
          await startBrowserDownload(src, suggestedFilename)
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
  }, [copy, saving, src, suggestedFilename])

  return { download, saving }
}
