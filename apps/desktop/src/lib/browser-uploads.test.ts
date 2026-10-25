import { afterEach, describe, expect, it, vi } from 'vitest'

import { $connection } from '@/store/session'
import { clearWebappSession, seedWebappSession } from '@/test/webapp-session'

import { installBrowserDesktopBridge } from './browser-desktop-bridge'
import { BROWSER_UPLOAD_MAX_BYTES } from './browser-uploads'

afterEach(() => {
  clearWebappSession()
  Reflect.deleteProperty(window, 'hermesDesktop')
  document.documentElement.removeAttribute('data-hermes-desktop-host')
  $connection.set(null)
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

function png(size: number): Uint8Array {
  const bytes = new Uint8Array(size)
  bytes.set([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])

  return bytes
}

describe('browser image paste size', () => {
  // A pasted or dropped image is staged on the host, and sending the message reads it back through
  // /api/fs/read-data-url, which stops at 16 MiB. An image over that limit is refused at the paste,
  // with the limit named, instead of attaching an image the message can never send.
  it('refuses an image larger than the browser can send, before uploading it', async () => {
    seedWebappSession('served-token')
    const fetchMock = vi.fn()
    vi.stubGlobal('fetch', fetchMock)
    expect(installBrowserDesktopBridge()).toBe(true)

    await expect(window.hermesDesktop!.saveImageBuffer(png(BROWSER_UPLOAD_MAX_BYTES + 1), '.png')).rejects.toThrow(
      'The image is larger than 16 MiB, the largest image the browser can send.'
    )
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it('stages an image of exactly the limit', async () => {
    seedWebappSession('served-token')

    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ path: '/srv/hermes/cache/images/upload.png' }), { status: 200 })
    )

    vi.stubGlobal('fetch', fetchMock)
    expect(installBrowserDesktopBridge()).toBe(true)

    await expect(window.hermesDesktop!.saveImageBuffer(png(BROWSER_UPLOAD_MAX_BYTES), '.png')).resolves.toBe(
      '/srv/hermes/cache/images/upload.png'
    )
    expect((fetchMock.mock.calls[0] as [URL])[0].pathname).toBe('/api/chat/image-upload')
  })
})
