import { useCallback, useState } from 'react'

import { useI18n } from '@/i18n'
import { notify, notifyError } from '@/store/notifications'

import { imageFilename } from './use-image-download'

function isMissingIpcHandler(error: unknown): boolean {
  const message = error instanceof Error ? error.message : typeof error === 'string' ? error : ''

  return message.includes("No handler registered for 'hermes:copyImageFromUrl'")
}

async function copyImageWithBrowserClipboard(src: string) {
  if (!navigator.clipboard?.write || typeof ClipboardItem === 'undefined') {
    throw new Error('Image clipboard API is not available')
  }

  const response = await fetch(src)

  if (!response.ok) {
    throw new Error(`Could not fetch image: ${response.status}`)
  }

  const blob = await response.blob()
  const type = blob.type || 'image/png'
  await navigator.clipboard.write([new ClipboardItem({ [type]: blob })])
}

/** Copy an image to the OS clipboard via the desktop IPC bridge, falling back
 *  to the browser ClipboardItem API when the handler is unavailable. */
export function useImageCopy(src?: string) {
  const { t } = useI18n()
  const copy = t.desktop
  const [copying, setCopying] = useState(false)

  const copyImage = useCallback(async () => {
    if (!src || copying) {
      return
    }

    setCopying(true)

    try {
      if (window.hermesDesktop?.copyImageFromUrl) {
        if (await window.hermesDesktop.copyImageFromUrl(src)) {
          notify({ kind: 'success', title: copy.imageCopied, message: imageFilename(src) })
        }

        return
      }

      await copyImageWithBrowserClipboard(src)
      notify({ kind: 'success', title: copy.imageCopied, message: imageFilename(src) })
    } catch (error) {
      if (isMissingIpcHandler(error)) {
        try {
          await copyImageWithBrowserClipboard(src)
          notify({ kind: 'info', title: copy.imageCopied, message: copy.restartToCopyImages })
        } catch (fallbackError) {
          notifyError(fallbackError, copy.restartToCopyImages)
        }

        return
      }

      notifyError(error, copy.imageCopyFailed)
    } finally {
      setCopying(false)
    }
  }, [copy, copying, src])

  return { copyImage, copying }
}
