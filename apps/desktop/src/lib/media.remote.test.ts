import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $connection } from '@/store/session'

import {
  downloadGatewayMediaFile,
  filePathFromMediaPath,
  gatewayMediaDataUrl,
  isRemoteGateway,
  mediaExternalUrl
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
  const api = vi.fn(async ({ path }: { path: string }) => {
    if (path.startsWith('/api/fs/read-data-url?')) {
      return { dataUrl: 'data:image/png;base64,ZHVtbXk=' }
    }

    throw new Error(`unexpected path ${path}`)
  })

  beforeEach(() => {
    api.mockClear()
    vi.stubGlobal('window', { hermesDesktop: { api } })
    $connection.set({ mode: 'remote' } as never)
  })

  afterEach(() => {
    vi.unstubAllGlobals()
    $connection.set(null)
  })

  it('reads gateway media through the desktop fs bridge instead of /api/media roots', async () => {
    const url = await gatewayMediaDataUrl('/home/u/.hermes/skills/demo/images/a b.png')

    expect(url).toBe('data:image/png;base64,ZHVtbXk=')
    expect(api).toHaveBeenCalledWith({
      path: '/api/fs/read-data-url?path=%2Fhome%2Fu%2F.hermes%2Fskills%2Fdemo%2Fimages%2Fa%20b.png'
    })
  })
})

describe('downloadGatewayMediaFile', () => {
  const saveGatewayFile = vi.fn(async () => ({ path: '/Users/me/Downloads/report.md', saved: true }))

  beforeEach(() => {
    saveGatewayFile.mockClear()
    vi.stubGlobal('window', { hermesDesktop: { saveGatewayFile } })
    $connection.set({ mode: 'remote', profile: 'docker-gw' } as never)
  })

  afterEach(() => {
    vi.unstubAllGlobals()
    $connection.set(null)
  })

  it('downloads gateway files through the native desktop save bridge', async () => {
    await expect(downloadGatewayMediaFile('file:///Users/me/project/a%20b.md')).resolves.toEqual({
      path: '/Users/me/Downloads/report.md',
      saved: true
    })

    expect(saveGatewayFile).toHaveBeenCalledWith({
      path: '/Users/me/project/a b.md',
      profile: 'docker-gw',
      suggestedName: 'a b.md'
    })
  })

  it('rejects when the desktop bridge is unavailable', async () => {
    vi.stubGlobal('window', { hermesDesktop: {} })

    await expect(downloadGatewayMediaFile('/Users/me/project/report.md')).rejects.toThrow(
      'Desktop file download bridge'
    )
  })
})
