import { afterEach, describe, expect, it, vi } from 'vitest'

import { $notifications } from '@/store/notifications'
import { openNewWindow } from '@/store/windows'
import { browserOpensWebappWindows, stubBroadcastChannels } from '@/test/webapp-windows'

import { installBrowserDesktopBridge } from './browser-desktop-bridge'
import { announceWebappWindow, createBrowserWindowOpener } from './browser-window'

const POPUP_BLOCKED = 'Webapp window did not open. Allow popups for this site and try again.'
const bootstrap = window as Window & { __HERMES_AUTH_REQUIRED__?: boolean }

afterEach(() => {
  vi.useRealTimers()
  Reflect.deleteProperty(window, 'hermesDesktop')
  delete bootstrap.__HERMES_AUTH_REQUIRED__
  window.document.documentElement.removeAttribute('data-hermes-desktop-host')
  window.history.replaceState(null, '', '/')
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
  $notifications.set([])
})

function opener(privateSession: boolean, api = vi.fn()) {
  return createBrowserWindowOpener({ api, basePath: '/one', privateSession })
}

function target(): URL {
  return new URL('/one/?profile=coder&win=secondary#/session-1', window.location.origin)
}

/** 'opened' or the rejection message once the pop-out settles, so a test can look while it is pending. */
function outcome(pending: Promise<unknown>): () => string | undefined {
  let result: string | undefined
  pending.then(() => { result = 'opened' }, (error: Error) => { result = error.message })

  return () => result
}

/** `/webapp/window` as the gateway serves it: says ready on the channel its fragment names, done once it has a ticket. */
function handoffPage(href: string): unknown[] {
  const id = new URLSearchParams(new URL(href).hash.slice(1)).get('channel')
  const channel = new BroadcastChannel(`hermes.webapp.window:${id}`)
  const received: unknown[] = []

  channel.onmessage = event => {
    received.push(event.data)

    if (event.data?.ticket) {
      channel.postMessage({ type: 'done' })
      channel.close()
    }
  }

  channel.postMessage({ type: 'ready' })

  return received
}

/** The toasts showing, by message. */
function toasts(): string[] {
  return $notifications.get().map(({ message }) => message)
}

/** The window opens that address late and boots the Webapp, which says its hello. */
function bootLate(href: string) {
  const opener = window.location.href
  window.history.replaceState(null, '', href)
  announceWebappWindow()
  window.history.replaceState(null, '', opener)
}

// A window that boots after the deadline (a slow machine, or a gated window landing on the login
// page or a 503) opened after all: its late `ready` takes the "Allow popups" toast back down.
describe('a window that says ready after the deadline', () => {
  it.each([
    ['gated host', false],
    ['launch-link', true]
  ])('takes the blocked-popup toast down on a %s pop-out', async (_name, privateSession) => {
    vi.useFakeTimers()
    const ticket = 'd'.repeat(43)
    const channels = stubBroadcastChannels()
    const open = vi.spyOn(window, 'open').mockReturnValue(null)
    const api = vi.fn().mockResolvedValue({ ticket })
    Reflect.set(window, 'hermesDesktop', { openWindow: () => opener(privateSession, api)(target()) })

    const opening = openNewWindow()
    await vi.advanceTimersByTimeAsync(3_000)
    await opening
    expect(toasts()).toEqual([POPUP_BLOCKED])

    await vi.advanceTimersByTimeAsync(1_000)
    const href = String(open.mock.calls[0]?.[0])
    const received = privateSession ? handoffPage(href) : (bootLate(href), [])
    await vi.advanceTimersByTimeAsync(0)

    expect(toasts()).toEqual([])
    expect(received).toEqual(privateSession ? [{ ticket }] : [])
    expect(api).toHaveBeenCalledTimes(privateSession ? 1 : 0)
    expect(channels.size).toBe(0)
  })
})

describe('gated host pop-out', () => {
  it('reports a window the browser blocked within seconds and stops listening for a late one', async () => {
    vi.useFakeTimers()
    const { channels, open } = browserOpensWebappWindows({ blocked: true })
    const result = outcome(opener(false)(target()))

    await vi.advanceTimersByTimeAsync(0)
    expect(result()).toBeUndefined()
    await vi.advanceTimersByTimeAsync(5_000)
    expect(result()).toBe(POPUP_BLOCKED)
    expect(open.mock.calls[0]?.[2]).toBe('noopener,noreferrer')
    // Still listening: a late window could yet say ready.
    expect(channels.size).toBe(1)
    await vi.advanceTimersByTimeAsync(25_000)
    expect(channels.size).toBe(0)
  })

  it('resolves once the window boots, which drops the hello from its address', async () => {
    vi.useFakeTimers()
    // A gated host may be served over plain HTTP, where crypto.randomUUID does not exist.
    const { getRandomValues } = crypto
    vi.stubGlobal('crypto', { getRandomValues: getRandomValues.bind(crypto) })
    const channels = stubBroadcastChannels()
    const open = vi.spyOn(window, 'open').mockReturnValue(null)
    const result = outcome(opener(false)(target()))
    const [href, name, features] = open.mock.calls[0]
    expect([name, features]).toEqual(['_blank', 'noopener,noreferrer'])

    // The window loads that address and boots the Webapp.
    bootstrap.__HERMES_AUTH_REQUIRED__ = true
    window.history.replaceState(null, '', String(href))
    expect(installBrowserDesktopBridge()).toBe(true)
    await vi.advanceTimersByTimeAsync(5_000)

    expect(result()).toBe('opened')
    const booted = new URL(window.location.href)
    expect([...booted.searchParams.keys()].sort()).toEqual(['profile', 'win'])
    expect(booted.hash).toBe('#/session-1')
    expect(channels.size).toBe(0)
  })
})

describe('launch-link pop-out', () => {
  it('reports a window that has not said ready within seconds and stops listening when the handoff expires', async () => {
    vi.useFakeTimers()
    const { channels, open } = browserOpensWebappWindows({ blocked: true })
    const api = vi.fn()
    const result = outcome(opener(true, api)(target()))

    await vi.advanceTimersByTimeAsync(0)
    expect(result()).toBeUndefined()
    await vi.advanceTimersByTimeAsync(5_000)
    expect(result()).toBe(POPUP_BLOCKED)
    // Still listening: a late window could yet say ready.
    expect(channels.size).toBe(1)
    await vi.advanceTimersByTimeAsync(25_000)
    expect(channels.size).toBe(0)
    expect(api).not.toHaveBeenCalled()
    expect(open.mock.calls[0]?.[2]).toBe('noopener,noreferrer')
  })

  it('signs in a blocked window the user opens from the blocked-popup prompt before the handoff expires', async () => {
    vi.useFakeTimers()
    const ticket = 'b'.repeat(43)
    const { channels, open } = browserOpensWebappWindows({ blocked: true })
    const api = vi.fn().mockResolvedValue({ ticket })
    const result = outcome(opener(true, api)(target()))

    await vi.advanceTimersByTimeAsync(5_000)
    expect(result()).toBe(POPUP_BLOCKED)
    expect(api).not.toHaveBeenCalled()

    // The browser's "Pop-ups blocked" prompt opens the address it stopped.
    await vi.advanceTimersByTimeAsync(10_000)
    const received = handoffPage(String(open.mock.calls[0]?.[0]))
    await vi.advanceTimersByTimeAsync(0)

    expect(api).toHaveBeenCalledOnce()
    expect(received).toEqual([{ ticket }])
    expect(channels.size).toBe(0)
    expect(result()).toBe(POPUP_BLOCKED)
  })

  it('signs in a window opened from a plain-HTTP tab, which has no crypto.randomUUID', async () => {
    vi.useFakeTimers()
    const { getRandomValues } = crypto
    vi.stubGlobal('crypto', { getRandomValues: getRandomValues.bind(crypto) })
    const ticket = 'c'.repeat(43)
    const channels = stubBroadcastChannels()
    const open = vi.spyOn(window, 'open').mockReturnValue(null)
    const api = vi.fn().mockResolvedValue({ ticket })
    const result = outcome(opener(true, api)(target()))

    const received = handoffPage(String(open.mock.calls[0]?.[0]))
    await vi.advanceTimersByTimeAsync(0)

    expect(api).toHaveBeenCalledOnce()
    expect(received).toEqual([{ ticket }])
    expect(channels.size).toBe(0)
    expect(result()).toBe('opened')
  })
})
