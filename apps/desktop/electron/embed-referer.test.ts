/**
 * Tests for electron/embed-referer.ts — stamping a YouTube Referer header so
 * the inline embed iframe doesn't fail with error 153 (see #106596).
 *
 * Run with: vitest run --project electron embed-referer
 */

import { describe, expect, it, vi } from 'vitest'

function fakeSession() {
  let handler: ((details: unknown, callback: (result: unknown) => void) => void) | undefined

  return {
    webRequest: {
      onBeforeSendHeaders: vi.fn((fn: typeof handler) => {
        handler = fn
      })
    },
    sendHeaders: (url: string, requestHeaders: Record<string, string> = {}) => {
      let result: { requestHeaders: Record<string, string> } | undefined

      handler!({ url, requestHeaders }, (r: unknown) => {
        result = r as { requestHeaders: Record<string, string> }
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

const { installEmbedReferer } = await import('./embed-referer')

describe('installEmbedReferer', () => {
  it('stamps a YouTube Referer on the default session, not just the embed partition', () => {
    installEmbedReferer()

    const headers = defaultSession.sendHeaders('https://www.youtube-nocookie.com/embed/abc123')

    expect(headers.Referer).toBe('https://www.youtube.com/')
  })

  it('still stamps the embed webview partition session', () => {
    installEmbedReferer()

    const headers = embedPartitionSession.sendHeaders('https://www.youtube-nocookie.com/embed/abc123')

    expect(headers.Referer).toBe('https://www.youtube.com/')
  })

  it('leaves non-YouTube requests untouched', () => {
    installEmbedReferer()

    const headers = defaultSession.sendHeaders('https://example.com/thing')

    expect(headers.Referer).toBeUndefined()
  })
})
