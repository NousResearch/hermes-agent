import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $connection } from '@/store/session'

import {
  filePathFromMediaPath,
  gatewayMediaDataUrl,
  isRemoteGateway,
  mediaExternalUrl,
  mediaPathFromMarkdownImageSrc
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

describe('mediaPathFromMarkdownImageSrc', () => {
  it('routes gateway-local markdown image paths through the media pipeline', () => {
    expect(mediaPathFromMarkdownImageSrc('/home/u/Documents/monday-artifacts/chart.png')).toBe(
      '/home/u/Documents/monday-artifacts/chart.png'
    )
    expect(mediaPathFromMarkdownImageSrc('~/Documents/monday-artifacts/chart.png')).toBe(
      '~/Documents/monday-artifacts/chart.png'
    )
    expect(mediaPathFromMarkdownImageSrc('file:///tmp/a%20b.png')).toBe('file:///tmp/a%20b.png')
    expect(mediaPathFromMarkdownImageSrc('#media:%2Ftmp%2Fa%20b.png')).toBe('/tmp/a b.png')
  })

  it('leaves fetchable and relative image sources on the normal image renderer', () => {
    expect(mediaPathFromMarkdownImageSrc('https://example.com/a.png')).toBeNull()
    expect(mediaPathFromMarkdownImageSrc('data:image/png;base64,AAAA')).toBeNull()
    expect(mediaPathFromMarkdownImageSrc('//cdn.example.com/a.png')).toBeNull()
    expect(mediaPathFromMarkdownImageSrc('assets/chart.png')).toBeNull()
    expect(mediaPathFromMarkdownImageSrc(undefined)).toBeNull()
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

  it('falls back to managed file reads for images outside dedicated media roots', async () => {
    api.mockRejectedValueOnce(new Error('403: {"detail":"Path outside media roots"}'))
    api.mockResolvedValueOnce({ data_url: 'data:image/png;base64,b3V0c2lkZQ==' })

    const url = await gatewayMediaDataUrl('/home/u/Documents/monday-artifacts/a b.png')

    expect(url).toBe('data:image/png;base64,b3V0c2lkZQ==')
    expect(api).toHaveBeenNthCalledWith(1, {
      path: '/api/media?path=%2Fhome%2Fu%2FDocuments%2Fmonday-artifacts%2Fa%20b.png'
    })
    expect(api).toHaveBeenNthCalledWith(2, {
      path: '/api/files/read?path=%2Fhome%2Fu%2FDocuments%2Fmonday-artifacts%2Fa%20b.png'
    })
  })

  it('does not fall back to managed file reads for auth or non-media failures', async () => {
    api.mockRejectedValueOnce(new Error('401: {"detail":"Invalid session token"}'))

    await expect(gatewayMediaDataUrl('/home/u/Documents/monday-artifacts/a.png')).rejects.toThrow(
      'Invalid session token'
    )
    expect(api).toHaveBeenCalledTimes(1)
  })

  it('rejects fallback payloads that are not image data URLs', async () => {
    api.mockRejectedValueOnce(new Error('403: {"detail":"Path outside media roots"}'))
    api.mockResolvedValueOnce({ data_url: 'data:text/plain;base64,c2VjcmV0' })

    await expect(gatewayMediaDataUrl('/home/u/Documents/monday-artifacts/a.png')).rejects.toThrow(
      'Expected image data'
    )
  })
})
