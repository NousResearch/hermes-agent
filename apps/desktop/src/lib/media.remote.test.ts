// @vitest-environment jsdom
// downloadGatewayMediaFile drives an <a download> click, so these need a DOM.
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

  it.each([
    ['/tmp/a #?%.png', 'file:///tmp/a%20%23%3F%25.png'],
    ['C:\\Users\\alice\\a #?%.png', 'file:///C:/Users/alice/a%20%23%3F%25.png'],
    ['\\\\server\\share\\a #?%.png', 'file://server/share/a%20%23%3F%25.png']
  ])('encodes reserved characters in local path %s', (input, expected) => {
    $connection.set({ mode: 'local' } as never)

    const url = mediaExternalUrl(input)

    expect(url).toBe(expected)
    expect(new URL(url).hash).toBe('')
    expect(new URL(url).search).toBe('')
  })

  it('normalizes file URLs and malformed Unicode without throwing', () => {
    $connection.set({ mode: 'local' } as never)

    expect(mediaExternalUrl('file:///tmp/a%20%23.png')).toBe('file:///tmp/a%20%23.png')
    expect(mediaExternalUrl('/tmp/bad-\ud800.png')).toBe('file:///tmp/bad-%EF%BF%BD.png')
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
  const api = vi.fn(async ({ path }: { path: string }) => {
    if (path.startsWith('/api/fs/read-data-url?')) {
      return { dataUrl: 'data:text/markdown;base64,IyByZXBvcnQ=' }
    }

    throw new Error(`unexpected path ${path}`)
  })

  let clickSpy: ReturnType<typeof vi.spyOn>

  beforeEach(() => {
    api.mockClear()
    vi.stubGlobal('window', { hermesDesktop: { api }, setTimeout: vi.fn() })
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({ blob: async () => new Blob(['# report'], { type: 'text/markdown' }) }))
    )
    URL.createObjectURL = vi.fn(() => 'blob:remote-artifact')
    URL.revokeObjectURL = vi.fn()
    clickSpy = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {})
    $connection.set({ mode: 'remote' } as never)
  })

  afterEach(() => {
    vi.unstubAllGlobals()
    vi.clearAllMocks()
    clickSpy.mockRestore()
    $connection.set(null)
  })

  it('downloads gateway files through the desktop fs bridge', async () => {
    await downloadGatewayMediaFile('file:///Users/me/project/report.md')

    expect(api).toHaveBeenCalledWith({
      path: '/api/fs/read-data-url?path=%2FUsers%2Fme%2Fproject%2Freport.md'
    })
    expect(clickSpy).toHaveBeenCalledOnce()
  })

  it('rejects when the gateway refuses the file read', async () => {
    api.mockRejectedValueOnce(new Error('403 File is not readable'))

    await expect(downloadGatewayMediaFile('/Users/me/project/report.md')).rejects.toThrow('403')
    expect(clickSpy).not.toHaveBeenCalled()
  })
})
