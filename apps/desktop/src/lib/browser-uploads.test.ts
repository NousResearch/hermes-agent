import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { useComposerActions } from '@/app/chat/hooks/use-composer-actions'
import { $notifications, clearNotifications } from '@/store/notifications'
import { $connection } from '@/store/session'
import { clearWebappSession, seedWebappSession } from '@/test/webapp-session'

import { installBrowserDesktopBridge } from './browser-desktop-bridge'
import { BROWSER_UPLOAD_MAX_BYTES } from './browser-uploads'

afterEach(() => {
  cleanup()
  clearNotifications()
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

describe('browser file stage size', () => {
  // A picked or dropped file over the upload limit gets the server's 413 "Upload request is too large",
  // which doesn't say what the limit is, and a refused image pick used to show nothing at all.
  const tooLarge = (name: string, type: string) => new File([new Uint8Array(BROWSER_UPLOAD_MAX_BYTES + 1)], name, { type })
  const limitMessage = 'The file is larger than 16 MiB, the largest file the browser can attach.'

  async function installBrowser() {
    seedWebappSession('served-token')

    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = new URL(String(input), window.location.origin)

      if (url.pathname === '/api/chat/file-upload') {
        return new Response(JSON.stringify({ detail: 'Upload request is too large' }), { status: 413 })
      }

      throw new Error(`Unexpected fetch: ${url}`)
    })

    vi.stubGlobal('fetch', fetchMock)
    expect(installBrowserDesktopBridge()).toBe(true)
    $connection.set(await window.hermesDesktop.getConnection('owner'))

    const uploads = () => fetchMock.mock.calls.filter(([input]) => String(input).includes('/api/chat/file-upload'))

    return { uploads }
  }

  const renderComposer = () =>
    renderHook(() => useComposerActions({ activeSessionId: null, currentCwd: '/workspace', requestGateway: vi.fn() }))

  it('refuses a picked image over the limit, names the limit, and uploads nothing', async () => {
    const { uploads } = await installBrowser()
    const { result } = renderComposer()

    vi.spyOn(HTMLInputElement.prototype, 'click').mockImplementation(function (this: HTMLInputElement) {
      Object.defineProperty(this, 'files', { configurable: true, value: [tooLarge('huge.png', 'image/png')] })
      this.dispatchEvent(new Event('change'))
    })

    await act(async () => {
      await result.current.pickImages()
    })

    expect(uploads()).toHaveLength(0)
    expect($notifications.get()).toContainEqual(expect.objectContaining({ kind: 'error', message: limitMessage }))
  })

  // Attach → Files… used to leave the refused stage as an unhandled rejection, with nothing shown.
  it('refuses a picked file over the limit with one notice naming the limit, and uploads nothing', async () => {
    const { uploads } = await installBrowser()
    const { result } = renderComposer()

    vi.spyOn(HTMLInputElement.prototype, 'click').mockImplementation(function (this: HTMLInputElement) {
      Object.defineProperty(this, 'files', { configurable: true, value: [tooLarge('huge.zip', 'application/zip')] })
      this.dispatchEvent(new Event('change'))
    })

    await act(async () => {
      await result.current.pickContextPaths('file')
    })

    expect(uploads()).toHaveLength(0)
    expect($notifications.get()).toEqual([
      expect.objectContaining({ kind: 'error', message: limitMessage, title: 'Attach failed' })
    ])
  })

  it('names the limit when a dropped file over it cannot be attached', async () => {
    const { uploads } = await installBrowser()
    const { result } = renderComposer()
    let attached = true

    await act(async () => {
      attached = await result.current.attachDroppedItems([{ file: tooLarge('huge.zip', 'application/zip'), path: '' }])
    })

    expect(attached).toBe(false)
    expect(uploads()).toHaveLength(0)
    expect($notifications.get()).toContainEqual(
      expect.objectContaining({ kind: 'warning', message: `Could not attach huge.zip: ${limitMessage}` })
    )
  })

  // The reason is the same for any name: a long one used to push it past readableError's length cap.
  it('still names the limit when the dropped file has a long name', async () => {
    await installBrowser()
    const { result } = renderComposer()
    const name = `${'a'.repeat(150)}.zip`

    await act(async () => {
      await result.current.attachDroppedItems([{ file: tooLarge(name, 'application/zip'), path: '' }])
    })

    const warning = $notifications.get().find(notice => notice.kind === 'warning')

    expect(warning?.message).toMatch(/^Could not attach a+/)
    expect(warning?.message).toContain(limitMessage)
  })
})
