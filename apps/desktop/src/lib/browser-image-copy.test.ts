import { afterEach, describe, expect, it, vi } from 'vitest'

import * as notifications from '@/store/notifications'

import { copyImageFromUrl } from './browser-image-copy'
import * as browserImageDownload from './browser-image-download'

const originalClipboard = navigator.clipboard
const originalClipboardItem = globalThis.ClipboardItem
const originalFetch = globalThis.fetch
const originalImage = globalThis.Image
const originalCreateObjectURL = URL.createObjectURL
const originalRevokeObjectURL = URL.revokeObjectURL

class FakeClipboardItem {
  readonly items: Record<string, Blob | Promise<Blob>>
  constructor(items: Record<string, Blob | Promise<Blob>>) {
    this.items = items
  }
}

function stubCanvasAndImage({
  drawThrows,
  encodedBlob = new Blob(['converted-png'], { type: 'image/png' }),
  errorOnLoad = false,
  height = 120,
  width = 200
}: {
  drawThrows?: Error
  encodedBlob?: Blob | null
  errorOnLoad?: boolean
  height?: number
  width?: number
} = {}) {
  const drawImage = vi.fn(() => {
    if (drawThrows) {
      throw drawThrows
    }
  })

  const createElement = document.createElement.bind(document)

  vi.spyOn(document, 'createElement').mockImplementation(((tagName: string, options?: ElementCreationOptions) => {
    if (tagName.toLowerCase() === 'canvas') {
      return {
        getContext: () => ({ drawImage }),
        height: 0,
        toBlob: (callback: BlobCallback) => callback(encodedBlob),
        width: 0
      } as unknown as HTMLCanvasElement
    }

    return createElement(tagName, options)
  }) as typeof document.createElement)

  URL.createObjectURL = vi.fn(() => 'blob:fake-image')
  URL.revokeObjectURL = vi.fn()

  class FakeImage {
    height = height
    naturalHeight = height
    naturalWidth = width
    onerror: (() => void) | null = null
    onload: (() => void) | null = null
    width = width

    set src(_value: string) {
      queueMicrotask(() => {
        if (errorOnLoad) {
          this.onerror?.()
        } else {
          this.onload?.()
        }
      })
    }
  }

  globalThis.Image = FakeImage as unknown as typeof Image

  return { drawImage }
}

afterEach(() => {
  vi.restoreAllMocks()
  Object.defineProperty(navigator, 'clipboard', { configurable: true, value: originalClipboard })
  Object.defineProperty(globalThis, 'ClipboardItem', { configurable: true, value: originalClipboardItem })
  globalThis.fetch = originalFetch
  globalThis.Image = originalImage
  URL.createObjectURL = originalCreateObjectURL
  URL.revokeObjectURL = originalRevokeObjectURL
})

describe('copyImageFromUrl', () => {
  it('writes same-origin PNG blobs directly to the clipboard', async () => {
    const pngBlob = new Blob(['png-bytes'], { type: 'image/png' })
    let writtenBlob: Blob | undefined

    const write = vi.fn(async (items: FakeClipboardItem[]) => {
      writtenBlob = await items[0].items['image/png']
    })

    Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { write } })
    Object.defineProperty(globalThis, 'ClipboardItem', { configurable: true, value: FakeClipboardItem })
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response('png-bytes', { headers: { 'Content-Type': 'image/png' }, status: 200 })
    )

    const ok = await copyImageFromUrl('/api/fs/stream?path=chart.png')

    expect(ok).toBe(true)
    expect(write).toHaveBeenCalledTimes(1)
    expect(writtenBlob?.type).toBe('image/png')
    expect(writtenBlob?.size).toBe(pngBlob.size)
  })

  it('converts non-PNG images to PNG via canvas before writing to the clipboard', async () => {
    const convertedPng = new Blob(['converted-png-bytes'], { type: 'image/png' })
    const { drawImage } = stubCanvasAndImage({ encodedBlob: convertedPng })
    let writtenBlob: Blob | undefined

    const write = vi.fn(async (items: FakeClipboardItem[]) => {
      writtenBlob = await items[0].items['image/png']
    })

    Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { write } })
    Object.defineProperty(globalThis, 'ClipboardItem', { configurable: true, value: FakeClipboardItem })
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response('jpeg-bytes', { headers: { 'Content-Type': 'image/jpeg' }, status: 200 })
    )

    const ok = await copyImageFromUrl('/api/fs/stream?path=photo.jpg')

    expect(ok).toBe(true)
    expect(drawImage).toHaveBeenCalledTimes(1)
    expect(writtenBlob).toBe(convertedPng)
  })

  it('reports an error when canvas conversion throws (e.g. tainted SVG with foreignObject)', async () => {
    const notifySpy = vi.spyOn(notifications, 'notifyError').mockImplementation(() => '')

    stubCanvasAndImage({ drawThrows: new DOMException('Tainted canvases may not be exported', 'SecurityError') })

    const write = vi.fn(async (items: FakeClipboardItem[]) => {
      await items[0].items['image/png']
    })

    Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { write } })
    Object.defineProperty(globalThis, 'ClipboardItem', { configurable: true, value: FakeClipboardItem })
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response('<svg></svg>', { headers: { 'Content-Type': 'image/svg+xml' }, status: 200 })
    )

    const ok = await copyImageFromUrl('/api/fs/stream?path=tainted.svg')

    expect(ok).toBe(false)
    expect(notifySpy).toHaveBeenCalledTimes(1)
  })

  it('reports an error when image decoding or PNG encoding fails', async () => {
    const notifySpy = vi.spyOn(notifications, 'notifyError').mockImplementation(() => '')

    stubCanvasAndImage({ errorOnLoad: true })

    const write = vi.fn(async (items: FakeClipboardItem[]) => {
      await items[0].items['image/png']
    })

    Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { write } })
    Object.defineProperty(globalThis, 'ClipboardItem', { configurable: true, value: FakeClipboardItem })
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response('bad-jpeg', { headers: { 'Content-Type': 'image/jpeg' }, status: 200 })
    )

    expect(await copyImageFromUrl('/api/fs/stream?path=broken.jpg')).toBe(false)
    expect(notifySpy).toHaveBeenCalledTimes(1)

    stubCanvasAndImage({ encodedBlob: null })
    expect(await copyImageFromUrl('/api/fs/stream?path=encode-fail.jpg')).toBe(false)
    expect(notifySpy).toHaveBeenCalledTimes(2)
  })

  it('fetches cross-origin images with CORS and copies PNG bytes', async () => {
    const pngBlob = new Blob(['remote-png'], { type: 'image/png' })
    const fetchSpy = vi.spyOn(browserImageDownload, 'fetchBrowserImage').mockResolvedValue(pngBlob)
    let writtenBlob: Blob | undefined

    const write = vi.fn(async (items: FakeClipboardItem[]) => {
      writtenBlob = await items[0].items['image/png']
    })

    Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { write } })
    Object.defineProperty(globalThis, 'ClipboardItem', { configurable: true, value: FakeClipboardItem })

    const ok = await copyImageFromUrl('https://cdn.example.com/pic.png')

    expect(ok).toBe(true)
    expect(fetchSpy).toHaveBeenCalledTimes(1)
    expect(writtenBlob).toBe(pngBlob)
  })

  it('reports an error when the Clipboard image API is unavailable', async () => {
    const notifySpy = vi.spyOn(notifications, 'notifyError').mockImplementation(() => '')

    Object.defineProperty(navigator, 'clipboard', { configurable: true, value: undefined })

    const ok = await copyImageFromUrl('/api/fs/stream?path=chart.png')

    expect(ok).toBe(false)
    expect(notifySpy).toHaveBeenCalledTimes(1)
    expect(notifySpy).toHaveBeenCalledWith(expect.any(Error), 'Copy failed')
  })

  it('reports an error when fetching the image fails', async () => {
    const notifySpy = vi.spyOn(notifications, 'notifyError').mockImplementation(() => '')

    const write = vi.fn(async (items: FakeClipboardItem[]) => {
      await items[0].items['image/png']
    })

    Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { write } })
    Object.defineProperty(globalThis, 'ClipboardItem', { configurable: true, value: FakeClipboardItem })
    globalThis.fetch = vi.fn().mockResolvedValue(new Response(null, { status: 404 }))

    const ok = await copyImageFromUrl('/api/fs/stream?path=missing.png')

    expect(ok).toBe(false)
    expect(notifySpy).toHaveBeenCalledTimes(1)
    expect(notifySpy).toHaveBeenCalledWith(expect.any(Error), 'Copy failed')
  })

  it('does not leave an unhandled rejection if clipboard.write rejects before fetch settles', async () => {
    const notifySpy = vi.spyOn(notifications, 'notifyError').mockImplementation(() => '')
    const unhandled = vi.fn()

    process.on('unhandledRejection', unhandled)

    try {
      const write = vi.fn().mockRejectedValue(new DOMException('Write permission denied', 'NotAllowedError'))

      Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { write } })
      Object.defineProperty(globalThis, 'ClipboardItem', { configurable: true, value: FakeClipboardItem })
      globalThis.fetch = vi.fn().mockImplementation(
        () => new Promise((_, reject) => setTimeout(() => reject(new Error('late network failure')), 10))
      )

      const ok = await copyImageFromUrl('/api/fs/stream?path=slow.png')

      expect(ok).toBe(false)
      expect(notifySpy).toHaveBeenCalledTimes(1)

      await new Promise(resolve => setTimeout(resolve, 30))
      expect(unhandled).not.toHaveBeenCalled()
    } finally {
      process.off('unhandledRejection', unhandled)
    }
  })
})
