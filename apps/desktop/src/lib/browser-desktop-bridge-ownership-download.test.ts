import { afterEach, expect, it, vi } from 'vitest'

import { setApiRequestConnection, setApiRequestProfile } from '@/api/client'
import { $notifications, clearNotifications } from '@/store/notifications'
import { $connection } from '@/store/session'

import { installBrowserDesktopBridge } from './browser-desktop-bridge'
import { BROWSER_IMAGE_DOWNLOAD_MAX_BYTES, BROWSER_IMAGE_DOWNLOAD_TIMEOUT_MS } from './browser-image-download'
import { captureGatewayFileDownload } from './media'

it('keeps a captured download on its owning browser profile after a foreground switch', async () => {
  win.__HERMES_SESSION_TOKEN__ = 'served-token'

  const fetchMock = vi.fn(async (input: URL, _init?: RequestInit) =>
    input.pathname === '/api/files/ticket'
      ? new Response(JSON.stringify({ ticket: 'file-ticket' }), { status: 200 })
      : new Response(null, { status: 200 })
  )

  vi.stubGlobal('fetch', fetchMock)
  const downloads: HTMLAnchorElement[] = []
  vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (this: HTMLAnchorElement) {
    downloads.push(this)
  })
  expect(installBrowserDesktopBridge()).toBe(true)
  setApiRequestConnection('local')
  setApiRequestProfile('file-owner')
  const download = captureGatewayFileDownload()
  setApiRequestProfile('foreground')
  $connection.set({ connectionId: 'local', mode: 'remote', profile: 'foreground' } as never)

  await download('file://nas/share/report.pdf', 'report.pdf')

  const [requestUrl, init] = fetchMock.mock.calls[0] as [URL, RequestInit]
  expect(init.method).toBe('HEAD')
  expect(requestUrl.searchParams.get('profile')).toBe('file-owner')
  expect(requestUrl.searchParams.get('path')).toBe('file://nas/share/report.pdf')
  expect(downloads).toHaveLength(1)
  const url = new URL(downloads[0].href)
  expect(url.searchParams.get('profile')).toBe('file-owner')
  expect(url.searchParams.get('path')).toBe('file://nas/share/report.pdf')
  expect(url.searchParams.get('ticket')).toBe('file-ticket')
  expect(JSON.parse(String(fetchMock.mock.calls[1][1]?.body))).toMatchObject({ profile: 'file-owner' })
  expect(downloads[0].download).toBe('report.pdf')
})

it.each([
  { name: 'exact byte ceiling', oversized: false, declaredLength: null },
  { name: 'oversized without Content-Length', oversized: true, declaredLength: null },
  { name: 'oversized with understated Content-Length', oversized: true, declaredLength: '1' }
])('bounds external image streams: $name', async ({ oversized, declaredLength }) => {
  vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] })
  win.__HERMES_SESSION_TOKEN__ = 'served-token'
  const limit = BROWSER_IMAGE_DOWNLOAD_MAX_BYTES
  const total = oversized ? 48 * 1024 * 1024 : limit
  let produced = 0
  const cancel = vi.fn()

  const body = new ReadableStream<Uint8Array>({
    pull(controller) {
      if (produced === total) {
        controller.close()

        return
      }

      const size = Math.min(total - produced, produced === limit ? 1 : 64 * 1024)
      controller.enqueue(new Uint8Array(size).fill(produced === 0 ? 7 : 9))
      produced += size
    },
    cancel
  }, { highWaterMark: 0 })

  const headers = new Headers({ 'content-type': 'image/png' })

  if (declaredLength !== null) {headers.set('content-length', declaredLength)}
  const response = new Response(body, { headers })
  const fetchMock = vi.fn().mockResolvedValue(response)
  vi.stubGlobal('fetch', fetchMock)
  const objectUrl = `blob:${window.location.origin}/download`
  const create = vi.spyOn(URL, 'createObjectURL').mockReturnValue(objectUrl)
  const revoke = vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => undefined)
  const clicked: string[] = []
  vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (this: HTMLAnchorElement) {
    clicked.push(this.href)
  })
  expect(installBrowserDesktopBridge()).toBe(true)
  await expect(win.hermesDesktop.saveImageFromUrl('https://images.example/image.png')).resolves.toBe(!oversized)
  const [url, init] = fetchMock.mock.calls[0] as [URL, RequestInit]
  expect(String(url)).toBe('https://images.example/image.png')
  expect(init.credentials).toBe('omit')
  expect(init.mode).toBe('cors')
  expect([...new Headers(init.headers)]).toEqual([])
  expect(body.locked).toBe(false)

  if (oversized) {
    expect(produced).toBe(limit + 1)
    expect(init.signal?.aborted).toBe(true)
    expect(cancel).toHaveBeenCalledOnce()
    expect(create).not.toHaveBeenCalled()
    expect(clicked).toEqual([])
    expect($notifications.get().at(-1)).toMatchObject({
      kind: 'error', title: 'Download failed', message: expect.stringMatching(/exceeds.*MiB/i)
    })
  } else {
    expect(produced).toBe(limit)
    expect(init.signal?.aborted).toBe(false)
    expect(cancel).not.toHaveBeenCalled()
    expect(clicked).toEqual([objectUrl])
    const image = create.mock.calls[0][0] as Blob
    expect(image.size).toBe(limit)
    expect(image.type).toBe('image/png')

    const bytes = await new Promise<ArrayBuffer>((resolve, reject) => {
      const reader = new FileReader()
      reader.onload = () => resolve(reader.result as ArrayBuffer)
      reader.onerror = () => reject(reader.error)
      reader.readAsArrayBuffer(image)
    })

    const content = new Uint8Array(bytes)
    expect(content[0]).toBe(7)
    expect(content[64 * 1024]).toBe(9)
    expect(content.at(-1)).toBe(9)
    await vi.runOnlyPendingTimersAsync()
    expect(init.signal?.aborted).toBe(false)
    expect(revoke).toHaveBeenCalledWith(objectUrl)
  }

  const directUrls = [`${window.location.origin}/same-origin.png`, 'blob:https://images.example/local', 'data:image/png;base64,AA==']

  for (const directUrl of directUrls) {
    await expect(win.hermesDesktop.saveImageFromUrl(directUrl)).resolves.toBe(true)
  }

  expect(clicked.slice(-directUrls.length)).toEqual(directUrls)
  expect(fetchMock).toHaveBeenCalledTimes(1)
})

it.each(['headers', 'body'])('times out stalled image %s within one whole-download deadline', async phase => {
  vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] })
  win.__HERMES_SESSION_TOKEN__ = 'served-token'
  // Cancellation itself must not extend the deadline.
  const cancel = vi.fn(() => new Promise<void>(() => undefined))

  const body = new ReadableStream<Uint8Array>({
    start(controller) {controller.enqueue(new Uint8Array([7]))},
    cancel
  })

  const response = new Response(body, { headers: { 'content-type': 'image/png' } })

  const fetchMock = vi.fn((_url: URL, init: RequestInit) => new Promise<Response>((resolve, reject) => {
    init.signal?.addEventListener('abort', () => reject(init.signal?.reason), { once: true })

    if (phase === 'body') {
      window.setTimeout(() => resolve(response), BROWSER_IMAGE_DOWNLOAD_TIMEOUT_MS / 2)
    }
  }))

  vi.stubGlobal('fetch', fetchMock)
  const click = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => undefined)
  const create = vi.spyOn(URL, 'createObjectURL')
  expect(installBrowserDesktopBridge()).toBe(true)
  let saved: boolean | undefined
  void win.hermesDesktop.saveImageFromUrl('https://images.example/slow.png').then(result => {saved = result})
  const signal = fetchMock.mock.calls[0][1].signal
  await vi.advanceTimersByTimeAsync(BROWSER_IMAGE_DOWNLOAD_TIMEOUT_MS - 1)
  expect(saved).toBeUndefined()
  await vi.advanceTimersByTimeAsync(1)
  expect(saved).toBe(false)
  expect(signal?.aborted).toBe(true)
  expect(body.locked).toBe(false)
  expect(cancel).toHaveBeenCalledTimes(phase === 'body' ? 1 : 0)
  expect($notifications.get().at(-1)).toMatchObject({
    kind: 'error', title: 'Download failed', message: expect.stringMatching(/timed out/i)
  })
  expect(click).not.toHaveBeenCalled()
  expect(create).not.toHaveBeenCalled()
})

it('surfaces CORS and HTTP download failures without clicking a navigation link', async () => {
  win.__HERMES_SESSION_TOKEN__ = 'served-token'

  const cancel = vi.fn()
  const deniedBody = new ReadableStream<Uint8Array>({ cancel })

  const fetchMock = vi.fn().mockRejectedValueOnce(new TypeError('Failed to fetch'))
    .mockResolvedValueOnce(new Response(deniedBody, { status: 403 }))

  vi.stubGlobal('fetch', fetchMock)
  const click = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => undefined)
  expect(installBrowserDesktopBridge()).toBe(true)

  for (const suffix of ['cors.png', 'denied.png']) {
    await expect(win.hermesDesktop.saveImageFromUrl(`https://images.example/${suffix}`)).resolves.toBe(false)
    expect($notifications.get().at(-1)).toMatchObject({ kind: 'error', title: 'Download failed' })
  }

  expect(click).not.toHaveBeenCalled()
  expect(cancel).toHaveBeenCalledOnce()
  expect(deniedBody.locked).toBe(false)

  for (const [, init] of fetchMock.mock.calls as [URL, RequestInit][]) {
    expect(init.signal?.aborted).toBe(true)
  }

  clearNotifications()
})

const win = window as Window & { __HERMES_SESSION_TOKEN__?: string }

afterEach(() => {
  delete win.__HERMES_SESSION_TOKEN__
  Reflect.deleteProperty(win, 'hermesDesktop')
  document.documentElement.removeAttribute('data-hermes-desktop-host')
  window.history.replaceState(null, '', '/#/')
  setApiRequestConnection(null)
  setApiRequestProfile(null)
  $connection.set(null)
  clearNotifications()
  window.dispatchEvent(new Event('beforeunload'))
  vi.clearAllTimers()
  vi.useRealTimers()
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

it('reconnects the window owner rather than the active secondary profile', async () => {
  win.__HERMES_SESSION_TOKEN__ = 'served-token'
  window.history.replaceState(null, '', '/?profile=window-owner#/session')
  expect(installBrowserDesktopBridge()).toBe(true)
  const initial = await win.hermesDesktop.getConnection('window-owner')
  $connection.set({ profile: 'active-secondary' } as never)
  const reconnect = await win.hermesDesktop.getConnection()
  expect(reconnect.profile).toBe(initial.profile)
  expect(new URL(reconnect.wsUrl).searchParams.get('profile')).toBe('window-owner')
  const explicitDefault = await win.hermesDesktop.getConnection(null)
  expect(new URL(explicitDefault.wsUrl).searchParams.has('profile')).toBe(false)
  const namedDefault = await win.hermesDesktop.getConnection('default')
  expect(new URL(namedDefault.wsUrl).searchParams.get('profile')).toBe('default')
})
