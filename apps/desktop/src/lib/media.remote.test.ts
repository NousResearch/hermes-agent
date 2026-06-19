import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $connection } from '@/store/session'

import { filePathFromMediaPath, gatewayMediaDataUrl, isRemoteGateway, mediaExternalUrl, toLocalFileUrl } from './media'

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

describe('toLocalFileUrl', () => {
  // Round-trip every produced file:// URL through the same WHATWG parser the
  // Electron main process uses (`new URL` + fileURLToPath), asserting the path
  // survives intact. This is the regression guard for the "chat file link is a
  // silent dead end" bug.
  const roundTrip = (fileUrl: string): string => {
    const u = new URL(fileUrl)
    // Mirror Node's fileURLToPath for the POSIX cases under test.
    return decodeURIComponent(u.pathname)
  }

  it('encodes a plain absolute path (unchanged for simple paths)', () => {
    expect(toLocalFileUrl('/tmp/a.png')).toBe('file:///tmp/a.png')
    expect(roundTrip('file:///tmp/a.png')).toBe('/tmp/a.png')
  })

  it('preserves spaces in an absolute path (Application Support case)', () => {
    const p = '/Users/cary/Library/Application Support/Hermes/x.png'
    const url = toLocalFileUrl(p)

    expect(url).toBe('file:///Users/cary/Library/Application%20Support/Hermes/x.png')
    expect(roundTrip(url)).toBe(p)
  })

  it('preserves URL-special characters instead of truncating at # or ?', () => {
    // Old `file://${path}` concat truncated this to `/tmp/weird` (the rest was
    // parsed as fragment/query) — the shell then opened the wrong file.
    const p = '/tmp/weird#name?.png'
    const url = toLocalFileUrl(p)

    expect(url).toBe('file:///tmp/weird%23name%3F.png')
    expect(roundTrip(url)).toBe(p)
  })

  it('preserves unicode characters', () => {
    const p = '/tmp/café.png'
    expect(roundTrip(toLocalFileUrl(p))).toBe(p)
  })

  it('passes a file:// URL through unchanged', () => {
    expect(toLocalFileUrl('file:///tmp/a.png')).toBe('file:///tmp/a.png')
  })

  it('returns a RELATIVE path raw (main process resolves it, never file://<host>)', () => {
    // The screenshot case: `file://hermes-support-slack.png` would parse the
    // filename as a host and get rejected. Raw passthrough lets the main
    // process resolve it against the session cwd.
    expect(toLocalFileUrl('hermes-support-slack.png')).toBe('hermes-support-slack.png')
    expect(toLocalFileUrl('out/report.png')).toBe('out/report.png')
  })

  it('returns a ~ path raw (main process expands the home dir)', () => {
    expect(toLocalFileUrl('~/Desktop/foo.png')).toBe('~/Desktop/foo.png')
  })

  it('builds a canonical file URL for a Windows drive path', () => {
    expect(toLocalFileUrl('C:\\Users\\me\\a b.png')).toBe('file:///C:/Users/me/a%20b.png')
  })
})
