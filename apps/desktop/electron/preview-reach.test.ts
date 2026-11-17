import { describe, expect, it, vi } from 'vitest'

import {
  loopbackTarget,
  openPreviewReach,
  PREVIEW_REACH_LEASE_MS,
  PreviewReachRegistry,
  rewriteToLocalPort
} from './preview-reach'

function deps(overrides: Partial<Parameters<typeof openPreviewReach>[1]> = {}) {
  return {
    cancel: vi.fn(async () => {}),
    forward: vi.fn(async () => {}),
    isCurrent: () => true,
    pickLocalPort: vi.fn(async () => 45_173),
    ...overrides
  }
}

describe('loopbackTarget', () => {
  it.each([
    ['http://localhost:5173/', 5173],
    ['http://127.0.0.1:3000/app', 3000],
    ['http://0.0.0.0:8080', 8080],
    ['http://[::1]:4321/x', 4321]
  ])('recognizes %s', (url, port) => {
    expect(loopbackTarget(url)?.port).toBe(port)
  })

  // An allowlist of "known" dev ports just means the next framework's default
  // silently fails; the transport is the security boundary, not the port.
  it('accepts any port, not a curated list', () => {
    expect(loopbackTarget('http://localhost:61234/')?.port).toBe(61_234)
  })

  it('defaults the port by scheme', () => {
    expect(loopbackTarget('http://localhost/')?.port).toBe(80)
    expect(loopbackTarget('https://localhost/')?.port).toBe(443)
  })

  it.each([
    ['a public host', 'https://example.com/x'],
    ['a lookalike subdomain', 'http://localhost.evil.com/'],
    ['a non-web scheme', 'file:///etc/passwd'],
    ['junk', 'not a url']
  ])('rejects %s', (_name, url) => {
    expect(loopbackTarget(url)).toBeNull()
  })
})

describe('rewriteToLocalPort', () => {
  it('keeps path, query and hash — they are what make the URL useful', () => {
    expect(rewriteToLocalPort('http://localhost:5173/a/b?q=1#f', 42)).toBe('http://127.0.0.1:42/a/b?q=1#f')
  })

  // The forward carries plain TCP to a dev server that is almost never
  // TLS-terminated; keeping https would fail the handshake.
  it('forces http', () => {
    expect(rewriteToLocalPort('https://localhost:5173/', 42)).toBe('http://127.0.0.1:42/')
  })
})

describe('openPreviewReach', () => {
  it('forwards the remote port and returns the local URL', async () => {
    const d = deps()
    const lease = await openPreviewReach('http://localhost:5173/app', d)

    expect(d.forward).toHaveBeenCalledWith(45_173, 5173, '127.0.0.1')
    expect(lease?.url).toBe('http://127.0.0.1:45173/app')
    expect(lease?.expiresAt).toBeGreaterThan(Date.now())
    expect(lease?.expiresAt).toBeLessThanOrEqual(Date.now() + PREVIEW_REACH_LEASE_MS)
  })

  it('ignores a non-loopback URL without opening anything', async () => {
    const d = deps()

    expect(await openPreviewReach('https://example.com/', d)).toBeNull()
    expect(d.forward).not.toHaveBeenCalled()
  })

  // The connection can die between picking a port and using it; forwarding
  // then would tunnel into whatever host replaced it.
  it('bails when the authorizing connection went away mid-open', async () => {
    const d = deps({ isCurrent: () => false })

    expect(await openPreviewReach('http://localhost:5173/', d)).toBeNull()
    expect(d.forward).not.toHaveBeenCalled()
  })

  it('surfaces a failed forward instead of pretending it worked', async () => {
    const d = deps({
      forward: vi.fn(async () => {
        throw new Error('ssh exited 255')
      })
    })

    await expect(openPreviewReach('http://localhost:5173/', d)).rejects.toThrow('ssh exited 255')
  })

  it('cancels exactly once however many times close is called', async () => {
    const d = deps()
    const lease = await openPreviewReach('http://localhost:5173/', d)

    await lease!.close()
    await lease!.close()

    expect(d.cancel).toHaveBeenCalledExactlyOnceWith(45_173, 5173)
  })

  it('closes itself when the lease expires', async () => {
    vi.useFakeTimers()

    try {
      const d = deps()

      await openPreviewReach('http://localhost:5173/', d)
      expect(d.cancel).not.toHaveBeenCalled()

      await vi.advanceTimersByTimeAsync(PREVIEW_REACH_LEASE_MS + 1)
      expect(d.cancel).toHaveBeenCalledWith(45_173, 5173)
    } finally {
      vi.useRealTimers()
    }
  })
})

describe('PreviewReachRegistry', () => {
  // Navigating a dev server is the same tunnel; a lease per page would leak a
  // socket per click.
  it('reuses one lease per remote port across pages', async () => {
    const registry = new PreviewReachRegistry()
    const d = deps()

    const first = await registry.resolve('http://localhost:5173/', d)
    const second = await registry.resolve('http://localhost:5173/about?x=1', d)

    expect(d.forward).toHaveBeenCalledOnce()
    expect(registry.size).toBe(1)
    expect(first).toBe('http://127.0.0.1:45173/')
    expect(second).toBe('http://127.0.0.1:45173/about?x=1')
  })

  it('opens a separate lease per distinct remote port', async () => {
    const registry = new PreviewReachRegistry()
    let next = 46_000
    const d = deps({ pickLocalPort: vi.fn(async () => (next += 1)) })

    await registry.resolve('http://localhost:5173/', d)
    await registry.resolve('http://localhost:3000/', d)

    expect(registry.size).toBe(2)
    expect(d.forward).toHaveBeenCalledTimes(2)
  })

  it('replaces an expired lease rather than serving a dead port', async () => {
    vi.useFakeTimers()

    try {
      const registry = new PreviewReachRegistry()
      let next = 47_000
      const d = deps({ pickLocalPort: vi.fn(async () => (next += 1)) })

      const first = await registry.resolve('http://localhost:5173/', d)

      await vi.advanceTimersByTimeAsync(PREVIEW_REACH_LEASE_MS + 1)

      const second = await registry.resolve('http://localhost:5173/', d)

      expect(second).not.toBe(first)
      expect(d.forward).toHaveBeenCalledTimes(2)
      expect(registry.size).toBe(1)
    } finally {
      vi.useRealTimers()
    }
  })

  it('closeAll tears down every forward — the connection changed hosts', async () => {
    const registry = new PreviewReachRegistry()
    let next = 48_000
    const d = deps({ pickLocalPort: vi.fn(async () => (next += 1)) })

    await registry.resolve('http://localhost:5173/', d)
    await registry.resolve('http://localhost:3000/', d)
    await registry.closeAll()

    expect(d.cancel).toHaveBeenCalledTimes(2)
    expect(registry.size).toBe(0)
  })

  it('leaves a non-loopback URL alone', async () => {
    const registry = new PreviewReachRegistry()
    const d = deps()

    expect(await registry.resolve('https://example.com/', d)).toBeNull()
    expect(registry.size).toBe(0)
  })
})
