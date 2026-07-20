import type { WebFrameMain } from 'electron'
import { describe, expect, it } from 'vitest'

import { createSandboxedHtmlNetworkGuard, SANDBOXED_HTML_FRAME_PREFIX } from './sandboxed-html-network'

function frame(frameTreeNodeId: number, name = '', frames: WebFrameMain[] = []): WebFrameMain {
  return { detached: false, frameTreeNodeId, frames, name } as WebFrameMain
}

describe('sandboxed HTML network guard', () => {
  it('protects the exact direct child by immutable frame id', () => {
    const guard = createSandboxedHtmlNetworkGuard()
    const name = `${SANDBOXED_HTML_FRAME_PREFIX}preview-1`
    const child = frame(42, name)
    const sender = frame(1, '', [child])

    expect(guard.register(7, sender, name)).toBe(true)
    expect(guard.shouldBlock(child, 'https://example.com/data')).toBe(true)
    expect(guard.shouldBlock(frame(43, name), 'https://example.com/data')).toBe(false)

    Object.defineProperty(child, 'name', { value: 'authored-name' })
    expect(guard.shouldBlock(child, 'wss://example.com/socket')).toBe(true)
    expect(guard.shouldBlock(child, 'data:image/png;base64,AAAA')).toBe(false)
    expect(guard.shouldBlock(child, 'blob:null/preview')).toBe(false)
    expect(guard.shouldBlock(child, 'about:srcdoc')).toBe(false)
  })

  it('rejects invalid, missing, and non-direct frames', () => {
    const guard = createSandboxedHtmlNetworkGuard()
    const name = `${SANDBOXED_HTML_FRAME_PREFIX}nested`
    const nested = frame(9, name)
    const sender = frame(1, '', [frame(2, 'parent', [nested])])

    expect(guard.register(7, sender, 'untrusted')).toBe(false)
    expect(guard.register(7, sender, name)).toBe(false)
    expect(guard.shouldBlock(nested, 'https://example.com')).toBe(false)
  })

  it('allows requests again only when the owning renderer unregisters', () => {
    const guard = createSandboxedHtmlNetworkGuard()
    const name = `${SANDBOXED_HTML_FRAME_PREFIX}preview-2`
    const child = frame(84, name)
    const sender = frame(1, '', [child])

    expect(guard.register(7, sender, name)).toBe(true)
    expect(guard.unregister(8, name)).toBe(false)
    expect(guard.shouldBlock(child, 'https://example.com')).toBe(true)
    expect(guard.unregister(7, name)).toBe(true)
    expect(guard.shouldBlock(child, 'https://example.com')).toBe(false)
  })

  it('clears every protected frame owned by a destroyed renderer', () => {
    const guard = createSandboxedHtmlNetworkGuard()
    const firstName = `${SANDBOXED_HTML_FRAME_PREFIX}first`
    const secondName = `${SANDBOXED_HTML_FRAME_PREFIX}second`
    const first = frame(11, firstName)
    const second = frame(12, secondName)
    const sender = frame(1, '', [first, second])

    guard.register(7, sender, firstName)
    guard.register(7, sender, secondName)
    guard.unregisterOwner(7)

    expect(guard.shouldBlock(first, 'https://example.com')).toBe(false)
    expect(guard.shouldBlock(second, 'https://example.com')).toBe(false)
  })
})
