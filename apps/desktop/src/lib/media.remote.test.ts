import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $connection } from '@/store/session'

import {
  filePathFromMediaPath,
  gatewayMediaDataUrl,
  isRemoteGateway,
  mediaExternalUrl,
  shouldRouteMediaViaGateway
} from './media'

describe('isRemoteGateway', () => {
  afterEach(() => {
    $connection.set(null)
  })

  it('is false with no connection', () => {
    $connection.set(null)
    expect(isRemoteGateway()).toBe(false)
  })

  it('is false in local mode', () => {
    $connection.set({ mode: 'local' } as never)
    expect(isRemoteGateway()).toBe(false)
  })

  it('is true in remote mode', () => {
    $connection.set({ mode: 'remote' } as never)
    expect(isRemoteGateway()).toBe(true)
  })
})

describe('filePathFromMediaPath', () => {
  it('passes through a plain path', () => {
    expect(filePathFromMediaPath('/home/u/.hermes/images/a.png')).toBe('/home/u/.hermes/images/a.png')
  })

  it('decodes a file:// URL with encoded characters', () => {
    expect(filePathFromMediaPath('file:///tmp/a%20b.png')).toBe('/tmp/a b.png')
  })
})

describe('mediaExternalUrl', () => {
  afterEach(() => {
    $connection.set(null)
  })

  it('passes through http(s) URLs untouched', () => {
    $connection.set({ mode: 'remote', baseUrl: 'https://gw', token: 't' } as never)
    expect(mediaExternalUrl('https://example.com/a.png')).toBe('https://example.com/a.png')
  })

  it('keeps file:// form in local mode', () => {
    $connection.set({ mode: 'local' } as never)
    expect(mediaExternalUrl('/tmp/a.png')).toBe('file:///tmp/a.png')
    expect(mediaExternalUrl('file:///tmp/a.png')).toBe('file:///tmp/a.png')
  })

  it('rewrites gateway-local paths to an authenticated download URL', () => {
    $connection.set({ mode: 'remote', baseUrl: 'https://gw', token: 's e/cret' } as never)
    expect(mediaExternalUrl('file:///tmp/a b.png')).toBe(
      'https://gw/api/files/download?path=%2Ftmp%2Fa%20b.png&token=s%20e%2Fcret'
    )
    expect(mediaExternalUrl('/tmp/a b.png')).toBe(
      'https://gw/api/files/download?path=%2Ftmp%2Fa%20b.png&token=s%20e%2Fcret'
    )
  })

  it('falls back to file:// when remote connection lacks a token', () => {
    $connection.set({ mode: 'remote', baseUrl: 'https://gw' } as never)
    expect(mediaExternalUrl('/tmp/a.png')).toBe('file:///tmp/a.png')
  })
})

describe('gatewayMediaDataUrl', () => {
  const api = vi.fn(async () => ({ data_url: 'data:image/png;base64,ZHVtbXk=' }))

  beforeEach(() => {
    api.mockClear()
    vi.stubGlobal('window', { hermesDesktop: { api } })
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('requests the encoded gateway path and returns the data URL', async () => {
    const url = await gatewayMediaDataUrl('/home/u/.hermes/images/a b.png')

    expect(url).toBe('data:image/png;base64,ZHVtbXk=')
    expect(api).toHaveBeenCalledWith({
      path: '/api/media?path=%2Fhome%2Fu%2F.hermes%2Fimages%2Fa%20b.png'
    })
  })
})

describe('shouldRouteMediaViaGateway (#63669)', () => {
  // Regression tests for #63669: Windows+WSL local-mode image rendering.
  // The previous decision used `isRemoteGateway()` alone, which misses
  // the WSL+local case where the image lives on the gateway machine's
  // WSL Linux filesystem but the connection mode is still 'local'. The
  // Windows client cannot read WSL paths via Electron IPC; the gateway
  // (which runs on the WSL side) must serve the data URL.

  afterEach(() => {
    $connection.set(null)
    vi.unstubAllGlobals()
  })

  function stubNavigatorPlatform(value: string) {
    vi.stubGlobal('navigator', { platform: value })
  }

  it('returns true in remote mode regardless of platform', () => {
    $connection.set({ mode: 'remote' } as never)
    stubNavigatorPlatform('Win32')
    expect(shouldRouteMediaViaGateway('/home/u/.hermes/images/a.png')).toBe(true)
  })

  it('returns true for WSL POSIX path on Windows in local mode (#63669)', () => {
    $connection.set({ mode: 'local' } as never)
    stubNavigatorPlatform('Win32')
    expect(shouldRouteMediaViaGateway('/home/u/.hermes/cache/images/a.png')).toBe(true)
  })

  it('returns false for plain Windows path in local mode', () => {
    $connection.set({ mode: 'local' } as never)
    stubNavigatorPlatform('Win32')
    expect(shouldRouteMediaViaGateway('C:\\Users\\u\\.hermes\\images\\a.png')).toBe(false)
  })

  it('returns false for POSIX path on non-Windows in local mode', () => {
    // On macOS/Linux local mode, /home/... IS on this machine's disk —
    // Electron IPC can read it directly. Don't route via gateway.
    $connection.set({ mode: 'local' } as never)
    stubNavigatorPlatform('MacIntel')
    expect(shouldRouteMediaViaGateway('/home/u/.hermes/cache/images/a.png')).toBe(false)
  })

  it('returns false when no connection is established yet', () => {
    $connection.set(null)
    stubNavigatorPlatform('Win32')
    // Cold start: $connection is null, mode unknown. Without the path
    // shape heuristic we cannot know if WSL is in play. The safe default
    // is to NOT route (the existing readFileDataUrl path will be tried,
    // which is what the unfixed code does). If the user is on Windows,
    // they'll see a stuck "Loading" — same as today. This documents the
    // tradeoff: the fix only kicks in when $connection has been hydrated.
    expect(shouldRouteMediaViaGateway('/home/u/.hermes/cache/images/a.png')).toBe(false)
  })

  it('returns true for any POSIX absolute path under /home on Windows', () => {
    // /home/<user> is the canonical WSL home. /tmp and /var are also
    // typical WSL filesystem paths that Windows can't read.
    $connection.set({ mode: 'local' } as never)
    stubNavigatorPlatform('Win32')
    for (const p of [
      '/home/u/file.png',
      '/tmp/scratch/image.png',
      '/var/log/thing.png',
    ]) {
      expect(shouldRouteMediaViaGateway(p)).toBe(true)
    }
  })
})
