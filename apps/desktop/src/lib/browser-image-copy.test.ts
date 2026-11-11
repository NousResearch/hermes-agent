import { afterEach, describe, expect, it, vi } from 'vitest'

import * as notifications from '@/store/notifications'

import { copyImageFromUrl } from './browser-image-copy'
import * as browserImageDownload from './browser-image-download'

const originalClipboard = navigator.clipboard
const originalClipboardItem = globalThis.ClipboardItem
const originalFetch = globalThis.fetch

class FakeClipboardItem {
  readonly items: Record<string, Blob | Promise<Blob>>
  constructor(items: Record<string, Blob | Promise<Blob>>) {
    this.items = items
  }
}

afterEach(() => {
  vi.restoreAllMocks()
  Object.defineProperty(navigator, 'clipboard', { configurable: true, value: originalClipboard })
  Object.defineProperty(globalThis, 'ClipboardItem', { configurable: true, value: originalClipboardItem })
  globalThis.fetch = originalFetch
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

  it('fetches cross-origin images with CORS and copies PNG bytes', async () => {
    const pngBlob = new Blob(['remote-png'], { type: 'image/png' })
    const fetchSpy = vi.spyOn(browserImageDownload, 'fetchBrowserImage').mockResolvedValue(pngBlob)

    const write = vi.fn(async (items: FakeClipboardItem[]) => {
      const resolved = await items[0].items['image/png']
      expect(resolved).toBe(pngBlob)
    })

    Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { write } })
    Object.defineProperty(globalThis, 'ClipboardItem', { configurable: true, value: FakeClipboardItem })

    const ok = await copyImageFromUrl('https://cdn.example.com/pic.png')

    expect(ok).toBe(true)
    expect(fetchSpy).toHaveBeenCalledTimes(1)
    expect(write).toHaveBeenCalledTimes(1)
  })

  it('reports an error when the Clipboard image API is unavailable', async () => {
    const notifySpy = vi.spyOn(notifications, 'notifyError').mockImplementation(() => '')

    Object.defineProperty(navigator, 'clipboard', { configurable: true, value: undefined })

    const ok = await copyImageFromUrl('/api/fs/stream?path=chart.png')

    expect(ok).toBe(false)
    expect(notifySpy).toHaveBeenCalledTimes(1)
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
  })
})
