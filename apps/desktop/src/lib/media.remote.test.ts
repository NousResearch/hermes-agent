import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $connection } from '@/store/session'

import {
  filePathFromMediaPath,
  gatewayMediaDataUrl,
  isRemoteGateway,
  mediaExternalUrl,
  pathFromRemoteGatewayFileUrl
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

  it('routes gateway-local paths through the bytes-to-temp opener even when a token exists', () => {
    $connection.set({ mode: 'remote', baseUrl: 'https://gw', token: 's e/cret' } as never)

    const fromFile = mediaExternalUrl('file:///tmp/a b.png')
    const fromPath = mediaExternalUrl('/tmp/a b.png')

    expect(fromFile.startsWith('hermes-gateway-file://open?')).toBe(true)
    expect(fromPath.startsWith('hermes-gateway-file://open?')).toBe(true)
    expect(fromFile).not.toContain('/api/files/download')
    expect(fromFile).not.toContain('token=')
    expect(pathFromRemoteGatewayFileUrl(fromFile)).toEqual({ path: '/tmp/a b.png' })
    expect(pathFromRemoteGatewayFileUrl(fromPath)).toEqual({ path: '/tmp/a b.png' })
  })

  it('never falls back to file:// when a remote connection lacks a token', () => {
    $connection.set({ mode: 'remote', baseUrl: 'https://gw', profile: 'mbp' } as never)

    const url = mediaExternalUrl('/tmp/a.png')

    expect(url).toBe('hermes-gateway-file://open?path=%2Ftmp%2Fa.png&profile=mbp')
    expect(url).not.toMatch(/^file:/)
    expect(pathFromRemoteGatewayFileUrl(url)).toEqual({ path: '/tmp/a.png', profile: 'mbp' })
  })
})

describe('gatewayMediaDataUrl', () => {
  const api = vi.fn(async () => ({ data_url: 'data:image/png;base64,ZHVtbXk=' }))

  beforeEach(() => {
    api.mockClear()
    $connection.set(null)
    vi.stubGlobal('window', { hermesDesktop: { api } })
  })

  afterEach(() => {
    $connection.set(null)
    vi.unstubAllGlobals()
  })

  it('requests the encoded gateway path and returns the data URL', async () => {
    const url = await gatewayMediaDataUrl('/home/u/.hermes/images/a b.png')

    expect(url).toBe('data:image/png;base64,ZHVtbXk=')
    expect(api).toHaveBeenCalledWith({
      path: '/api/media?path=%2Fhome%2Fu%2F.hermes%2Fimages%2Fa%20b.png'
    })
  })

  it('passes the active remote profile to the gateway API bridge', async () => {
    $connection.set({ mode: 'remote', profile: 'mbp' } as never)

    await gatewayMediaDataUrl('/home/u/.hermes/images/a.png')

    expect(api).toHaveBeenCalledWith({
      path: '/api/media?path=%2Fhome%2Fu%2F.hermes%2Fimages%2Fa.png',
      profile: 'mbp'
    })
  })
})
