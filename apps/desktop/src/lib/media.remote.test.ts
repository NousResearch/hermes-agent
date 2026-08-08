// @vitest-environment jsdom
// downloadGatewayMediaFile drives an <a download> click, so these need a DOM.
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $connection } from '@/store/session'

import {
  downloadGatewayMediaFile,
  filePathFromMediaPath,
  gatewayMediaDataUrl,
  isInlineMediaSrc,
  isRemoteGateway,
  mediaExternalUrl,
  resolveMediaDisplaySrc,
  resolveMediaPlaybackSrc
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

  it('normalizes Windows absolute paths to file:/// form (regression: c: scheme)', () => {
    $connection.set({ mode: 'local' } as never)
    expect(mediaExternalUrl('C:\\Work\\a.md')).toBe('file:///C:/Work/a.md')
    expect(mediaExternalUrl('C:/Work/a.md')).toBe('file:///C:/Work/a.md')
    expect(mediaExternalUrl('d:\\data\\file.txt')).toBe('file:///d:/data/file.txt')
  })

  it('encodes # in Windows paths so URL parsing cannot treat it as a fragment', () => {
    $connection.set({ mode: 'local' } as never)
    expect(mediaExternalUrl('C:\\Work\\a#b.md')).toBe('file:///C:/Work/a%23b.md')
  })

  it('encodes % in Windows paths so %25 is not double-decoded', () => {
    $connection.set({ mode: 'local' } as never)
    expect(mediaExternalUrl('C:\\Work\\100%25.md')).toBe('file:///C:/Work/100%2525.md')
  })

  it('leaves spaces raw in the emitted URL (WHATWG parses them as %20)', () => {
    $connection.set({ mode: 'local' } as never)
    expect(mediaExternalUrl('C:\\Work\\My Docs\\a b.md')).toBe('file:///C:/Work/My Docs/a b.md')
    // Behavior contract: URL parsing must recover the exact filesystem path.
    expect(new URL(mediaExternalUrl('C:\\Work\\My Docs\\a b.md')).pathname).toBe('/C:/Work/My%20Docs/a%20b.md')
  })

  it('handles a drive root', () => {
    $connection.set({ mode: 'local' } as never)
    expect(mediaExternalUrl('C:\\')).toBe('file:///C:/')
  })

  // UNC paths and the legacy `file://C:` form are out of scope for this PR
  // (declared in the PR description). These tests lock in current behavior so
  // a future change has to consciously extend the drive-letter branch.
  it('does not touch UNC paths (out of scope, documented)', () => {
    $connection.set({ mode: 'local' } as never)
    expect(mediaExternalUrl('\\\\server\\share\\a.md')).toBe('file://\\\\server\\share\\a.md')
  })

  it('passes legacy file://C: URLs through unchanged (out of scope, documented)', () => {
    $connection.set({ mode: 'local' } as never)
    expect(mediaExternalUrl('file://C:/x/a.md')).toBe('file://C:/x/a.md')
  })

  it('rewrites Windows paths through the authenticated download URL in remote mode', () => {
    $connection.set({ mode: 'remote', baseUrl: 'https://gw', token: 't' } as never)
    expect(mediaExternalUrl('C:\\Work\\a.md')).toBe('https://gw/api/files/download?path=C%3A%5CWork%5Ca.md&token=t')
  })

  it('URL-encodes special characters in the remote download query', () => {
    $connection.set({ mode: 'remote', baseUrl: 'https://gw', token: 't' } as never)
    expect(mediaExternalUrl('C:\\a b#c.md')).toBe('https://gw/api/files/download?path=C%3A%5Ca%20b%23c.md&token=t')
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

describe('resolveMediaDisplaySrc', () => {
  const api = vi.fn(async ({ path }: { path: string }) => {
    if (path.startsWith('/api/fs/read-data-url?')) {
      return { dataUrl: 'data:image/png;base64,ZHVtbXk=' }
    }

    throw new Error(`unexpected path ${path}`)
  })

  beforeEach(() => {
    api.mockClear()
  })

  afterEach(() => {
    vi.unstubAllGlobals()
    $connection.set(null)
  })

  it('recognizes inline image URLs', () => {
    expect(isInlineMediaSrc('https://example.com/a.png')).toBe(true)
    expect(isInlineMediaSrc('data:image/png;base64,ZHVtbXk=')).toBe(true)
    expect(isInlineMediaSrc('/Users/me/a.png')).toBe(false)
  })

  it('leaves web, data, and relative markdown image sources unchanged', async () => {
    vi.stubGlobal('window', { hermesDesktop: { api } })
    $connection.set({ mode: 'remote', profile: 'remote-work' } as never)

    await expect(resolveMediaDisplaySrc('https://example.com/a.png')).resolves.toBe('https://example.com/a.png')
    await expect(resolveMediaDisplaySrc('data:image/png;base64,ZHVtbXk=')).resolves.toBe(
      'data:image/png;base64,ZHVtbXk='
    )
    await expect(resolveMediaDisplaySrc('images/a.png')).resolves.toBe('images/a.png')
    await expect(resolveMediaDisplaySrc('./images/a.png')).resolves.toBe('./images/a.png')
    await expect(resolveMediaDisplaySrc('../images/a.png')).resolves.toBe('../images/a.png')
    expect(api).not.toHaveBeenCalled()
  })

  it('reads remote gateway-local file paths through the desktop fs bridge', async () => {
    vi.stubGlobal('window', { hermesDesktop: { api } })
    $connection.set({ mode: 'remote', profile: 'remote-work' } as never)

    await expect(resolveMediaDisplaySrc('/Users/me/project/a b.png')).resolves.toBe('data:image/png;base64,ZHVtbXk=')
    expect(api).toHaveBeenCalledWith({
      path: '/api/fs/read-data-url?path=%2FUsers%2Fme%2Fproject%2Fa%20b.png',
      profile: 'remote-work'
    })
  })

  it('reads local desktop file paths from the local desktop shell', async () => {
    const readFileDataUrl = vi.fn(async () => 'data:image/png;base64,bG9jYWw=')

    vi.stubGlobal('window', { hermesDesktop: { readFileDataUrl } })
    $connection.set({ mode: 'local' } as never)

    await expect(resolveMediaDisplaySrc('file:///Users/me/project/a%20b.png')).resolves.toBe(
      'data:image/png;base64,bG9jYWw='
    )
    expect(readFileDataUrl).toHaveBeenCalledWith('/Users/me/project/a b.png')
  })
})

describe('resolveMediaPlaybackSrc', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
    $connection.set(null)
  })

  it('keeps a remote HTTPS video URL unchanged', async () => {
    vi.stubGlobal('window', { hermesDesktop: { api: vi.fn() } })
    $connection.set({ mode: 'remote', baseUrl: 'https://gateway.test', token: 'secret' } as never)

    await expect(resolveMediaPlaybackSrc('https://cdn.example.com/render.mp4')).resolves.toBe(
      'https://cdn.example.com/render.mp4'
    )
  })

  it('routes gateway-local video through the authenticated download endpoint', async () => {
    vi.stubGlobal('window', { hermesDesktop: { api: vi.fn() } })
    $connection.set({ mode: 'remote', baseUrl: 'https://gateway.test', token: 's e/cret' } as never)

    await expect(resolveMediaPlaybackSrc('/root/outputs/render.mp4')).resolves.toBe(
      'https://gateway.test/api/files/download?path=%2Froot%2Foutputs%2Frender.mp4&token=s%20e%2Fcret'
    )
  })

  it('uses the Electron streaming protocol for local desktop video', async () => {
    vi.stubGlobal('window', { hermesDesktop: { api: vi.fn() } })
    $connection.set({ mode: 'local' } as never)

    await expect(resolveMediaPlaybackSrc('C:\\renders\\demo.mp4')).resolves.toBe(
      'hermes-media://stream/C%3A%5Crenders%5Cdemo.mp4'
    )
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
