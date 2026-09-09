/**
 * Tests for electron/embed-referer.ts — stamping a YouTube Referer header so
 * the inline embed iframe doesn't fail with error 153 (see #106596).
 *
 * Run with: vitest run --project electron embed-referer
 */

import { describe, expect, it, vi } from 'vitest'

import { applyRemoteRequestHeaders } from './remote-ws-headers'

function fakeSession() {
  let handler: ((details: unknown, callback: (result: unknown) => void) => void) | undefined

  return {
    webRequest: {
      onBeforeSendHeaders: vi.fn((fn: typeof handler) => {
        handler = fn
      })
    },
    sendHeaders: (url: string, requestHeaders: Record<string, string> = {}) => {
      let result: { requestHeaders?: Record<string, string> } | undefined

      handler!({ url, requestHeaders }, (r: unknown) => {
        result = r as { requestHeaders?: Record<string, string> }
      })

      return result!.requestHeaders
    }
  }
}

const defaultSession = fakeSession()
const embedPartitionSession = fakeSession()

vi.mock('electron', () => ({
  session: {
    defaultSession,
    fromPartition: vi.fn(() => embedPartitionSession)
  }
}))

const { installEmbedReferer, withEmbedReferer } = await import('./embed-referer')

describe('withEmbedReferer', () => {
  it('stamps a Referer for known YouTube hosts', () => {
    const headers = withEmbedReferer('https://www.youtube-nocookie.com/embed/abc123', {})

    expect(headers.Referer).toBe('https://www.youtube.com/')
  })

  it('does not override an existing Referer', () => {
    const headers = withEmbedReferer('https://www.youtube.com/watch?v=abc123', { Referer: 'https://custom/' })

    expect(headers.Referer).toBe('https://custom/')
  })

  it('leaves non-YouTube requests untouched', () => {
    const headers = withEmbedReferer('https://example.com/thing', { 'X-Foo': 'bar' })

    expect(headers).toEqual({ 'X-Foo': 'bar' })
  })
})

describe('installEmbedReferer', () => {
  it('stamps the embed webview partition session', () => {
    installEmbedReferer()

    const headers = embedPartitionSession.sendHeaders('https://www.youtube-nocookie.com/embed/abc123')

    expect(headers!.Referer).toBe('https://www.youtube.com/')
  })

  it('does not register a handler on the default session', () => {
    defaultSession.webRequest.onBeforeSendHeaders.mockClear()

    installEmbedReferer()

    // Composed into main.ts's installRemoteHeaderRules() listener instead —
    // Electron only allows a single onBeforeSendHeaders listener per session,
    // so a second listener registered here would silently replace, or be
    // replaced by, that one.
    expect(defaultSession.webRequest.onBeforeSendHeaders).not.toHaveBeenCalled()
  })
})

describe('default session composition (mirrors main.ts installRemoteHeaderRules)', () => {
  // Reproduces the exact registration main.ts performs: a single
  // onBeforeSendHeaders listener on session.defaultSession that runs the
  // remote-gateway header logic and then layers the embed Referer on top,
  // since registering installEmbedReferer's own listener there as well
  // would just be discarded by whichever of the two registers last.
  function installComposedDefaultSessionHandler(headersForRemoteRequest: (url: string) => Record<string, string>) {
    defaultSession.webRequest.onBeforeSendHeaders((details: any, callback: any) => {
      applyRemoteRequestHeaders(
        details,
        (result: { requestHeaders?: Record<string, string> }) => {
          const requestHeaders = withEmbedReferer(details.url, result.requestHeaders ?? details.requestHeaders)

          callback({ requestHeaders })
        },
        headersForRemoteRequest
      )
    })
  }

  it('still stamps the YouTube Referer when no remote headers apply (the common case)', () => {
    installComposedDefaultSessionHandler(() => ({}))

    const headers = defaultSession.sendHeaders('https://www.youtube-nocookie.com/embed/abc123')

    expect(headers!.Referer).toBe('https://www.youtube.com/')
  })

  it('stamps the Referer alongside remote-gateway headers when in remote mode', () => {
    installComposedDefaultSessionHandler(() => ({ Authorization: 'Bearer token' }))

    const headers = defaultSession.sendHeaders('https://www.youtube-nocookie.com/embed/abc123')

    expect(headers!.Referer).toBe('https://www.youtube.com/')
    expect(headers!.Authorization).toBe('Bearer token')
  })

  it('leaves non-YouTube requests governed only by the remote-gateway headers', () => {
    installComposedDefaultSessionHandler(() => ({ Authorization: 'Bearer token' }))

    const headers = defaultSession.sendHeaders('https://example.com/thing')

    expect(headers!.Referer).toBeUndefined()
    expect(headers!.Authorization).toBe('Bearer token')
  })
})
