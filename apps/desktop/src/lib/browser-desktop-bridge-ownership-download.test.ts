import { afterEach, expect, it, vi } from 'vitest'

import { $notifications, clearNotifications } from '@/store/notifications'
import { $connection } from '@/store/session'

import { installBrowserDesktopBridge } from './browser-desktop-bridge'

it('downloads external images through a local blob without navigating the app', async () => {
  win.__HERMES_SESSION_TOKEN__ = 'served-token'
  const image = new Blob(['image bytes'], { type: 'image/png' })
  const fetchMock = vi.fn().mockResolvedValue({ ok: true, blob: async () => image })
  vi.stubGlobal('fetch', fetchMock)
  const objectUrl = `blob:${window.location.origin}/download`
  const create = vi.spyOn(URL, 'createObjectURL').mockReturnValue(objectUrl)
  const clicked: string[] = []
  vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (this: HTMLAnchorElement) {
    clicked.push(this.href)
  })
  expect(installBrowserDesktopBridge()).toBe(true)
  await expect(win.hermesDesktop.saveImageFromUrl('https://images.example/image.png')).resolves.toBe(true)
  expect(clicked).toEqual([objectUrl])
  expect(create).toHaveBeenCalledWith(image)
  const [url, init] = fetchMock.mock.calls[0] as [URL, RequestInit]
  expect(String(url)).toBe('https://images.example/image.png')
  expect(init.credentials).toBe('omit')
  expect(new Headers(init.headers).has('X-Hermes-Session-Token')).toBe(false)
  await win.hermesDesktop.saveImageFromUrl('/same-origin.png')
  expect(clicked[1]).toBe(`${window.location.origin}/same-origin.png`)
  expect(fetchMock).toHaveBeenCalledTimes(1)
})

it('surfaces CORS and HTTP download failures without clicking a navigation link', async () => {
  win.__HERMES_SESSION_TOKEN__ = 'served-token'

  const fetchMock = vi.fn().mockRejectedValueOnce(new TypeError('Failed to fetch'))
    .mockResolvedValueOnce({ ok: false, status: 403 })

  vi.stubGlobal('fetch', fetchMock)
  const click = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => undefined)
  expect(installBrowserDesktopBridge()).toBe(true)

  for (const suffix of ['cors.png', 'denied.png']) {
    await expect(win.hermesDesktop.saveImageFromUrl(`https://images.example/${suffix}`)).resolves.toBe(false)
    expect($notifications.get().at(-1)).toMatchObject({ kind: 'error', title: 'Download failed' })
  }

  expect(click).not.toHaveBeenCalled()
  clearNotifications()
})

const win = window as Window & { __HERMES_SESSION_TOKEN__?: string }

afterEach(() => {
  delete win.__HERMES_SESSION_TOKEN__
  Reflect.deleteProperty(win, 'hermesDesktop')
  document.documentElement.removeAttribute('data-hermes-desktop-host')
  window.history.replaceState(null, '', '/#/')
  $connection.set(null)
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
