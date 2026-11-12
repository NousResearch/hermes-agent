import { translateNow } from '@/i18n'
import { fetchBrowserImage } from '@/lib/browser-image-download'
import { notifyError } from '@/store/notifications'

function blobToPng(blob: Blob): Promise<Blob> {
  const objectUrl = URL.createObjectURL(blob)

  return new Promise((resolve, reject) => {
    const image = new Image()

    image.onload = () => {
      URL.revokeObjectURL(objectUrl)

      try {
        const isSvg = blob.type === 'image/svg+xml'
        const width = image.naturalWidth || image.width || (isSvg ? 800 : 0)
        const height = image.naturalHeight || image.height || (isSvg ? 600 : 0)

        if (!width || !height) {
          reject(new Error('Image has no dimensions to copy'))

          return
        }

        const canvas = document.createElement('canvas')
        canvas.width = width
        canvas.height = height

        const context = canvas.getContext('2d')

        if (!context) {
          reject(new Error('Canvas 2D context is unavailable'))

          return
        }

        context.drawImage(image, 0, 0, width, height)
        canvas.toBlob(png => {
          if (png) {
            resolve(png)
          } else {
            reject(new Error('Failed to encode image as PNG'))
          }
        }, 'image/png')
      } catch (error) {
        reject(error)
      }
    }

    image.onerror = () => {
      URL.revokeObjectURL(objectUrl)
      reject(new Error('Failed to decode image for clipboard copy'))
    }

    image.src = objectUrl
  })
}

async function fetchImageAsPngBlob(url: string): Promise<Blob> {
  const target = new URL(url, window.location.href)
  let blob: Blob

  if (target.origin !== window.location.origin && !['blob:', 'data:'].includes(target.protocol)) {
    blob = await fetchBrowserImage(target)
  } else {
    const response = await fetch(target.href)

    if (!response.ok) {
      throw new Error(`Image fetch failed (${response.status})`)
    }

    blob = await response.blob()
  }

  return blob.type === 'image/png' ? blob : blobToPng(blob)
}

/** Copy an image URL into the system clipboard as PNG in browser-hosted mode. */
export async function copyImageFromUrl(url: string): Promise<boolean> {
  if (!url) {
    return false
  }

  if (!navigator.clipboard?.write || typeof ClipboardItem === 'undefined') {
    notifyError(
      new Error('Copying images to the clipboard requires a secure context (HTTPS)'),
      translateNow('common.copyFailed')
    )

    return false
  }

  try {
    // Pass the async fetch/conversion as a Promise inside ClipboardItem so
    // navigator.clipboard.write runs synchronously inside the click gesture
    // (required by Safari/WebKit user-activation checks).
    const pngBlobPromise = fetchImageAsPngBlob(url)

    // Observe rejections independently in case navigator.clipboard.write rejects
    // before consuming the ClipboardItem promise.
    void pngBlobPromise.catch(() => undefined)

    await navigator.clipboard.write([new ClipboardItem({ 'image/png': pngBlobPromise })])

    return true
  } catch (error) {
    notifyError(error, translateNow('common.copyFailed'))

    return false
  }
}
