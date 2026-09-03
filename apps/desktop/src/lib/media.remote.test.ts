import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $connection } from '@/store/session'

import {
  downloadGatewayMediaFile,
  filePathFromMediaPath,
  gatewayMediaDataUrl,
  gatewayMediaDataUrlLocalFallback,
  isInlineMediaSrc,
  isRemoteGateway,
  mediaExternalUrl,
  mediaGatewayStreamUrl,
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

describe('mediaGatewayStreamUrl', () => {
  afterEach(() => {
    $connection.set(null)
  })

  it('rewrites gateway-local media to the main-process remote stream proxy', () => {
    $connection.set({ mode: 'remote', baseUrl: 'https://gw', token: 's e/cret' } as never)
    expect(mediaGatewayStreamUrl('file:///tmp/a b.mp4')).toBe('hermes-media://remote/%2Ftmp%2Fa%20b.mp4')
  })

  it('supports OAuth remotes with no renderer-visible token and scopes pool profiles', () => {
    $connection.set({ authMode: 'oauth', mode: 'remote', profile: 'voice reviewer', token: null } as never)
    expect(mediaGatewayStreamUrl('/tmp/a.mp4')).toBe('hermes-media://remote/%2Ftmp%2Fa.mp4?profile=voice%20reviewer')
  })

  it('pins remote streams to their registered connection and profile', () => {
    $connection.set({
      connectionId: 'studio-ssh',
      mode: 'remote',
      profile: 'voice reviewer',
      remoteKind: 'ssh'
    } as never)

    expect(mediaGatewayStreamUrl('/tmp/a.mp4')).toBe(
      'hermes-media://remote/%2Ftmp%2Fa.mp4?connectionId=studio-ssh&profile=voice%20reviewer'
    )
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

  it('routes OAuth gateway-local video through the authenticated main-process proxy', async () => {
    vi.stubGlobal('window', { hermesDesktop: { api: vi.fn() } })
    $connection.set({ authMode: 'oauth', mode: 'remote', profile: 'default', token: null } as never)

    await expect(resolveMediaPlaybackSrc('/root/outputs/render.mp4')).resolves.toBe(
      'hermes-media://remote/%2Froot%2Foutputs%2Frender.mp4?profile=default'
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
  const saveGatewayFile = vi.fn(async () => ({ path: '/Users/me/Downloads/report.md', saved: true }))

  beforeEach(() => {
    saveGatewayFile.mockClear()
    vi.stubGlobal('window', { hermesDesktop: { saveGatewayFile } })
    $connection.set({ connectionId: 'work-ssh', mode: 'remote', profile: 'docker-gw' } as never)
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
      connectionId: 'work-ssh',
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

  // A preview screenshot is written by THIS shell's Electron main process into
  // userData/composer-images, never on the gateway. Fetching it from the
  // gateway can only 404 ("missing, unreadable, or too large").
  it('opens a locally produced artifact from this disk instead of the gateway', async () => {
    const readFileDataUrl = vi.fn(async () => 'data:image/png;base64,bG9jYWw=')
    const openExternal = vi.fn()

    vi.stubGlobal('window', { hermesDesktop: { openExternal, readFileDataUrl, saveGatewayFile } })

    await expect(
      downloadGatewayMediaFile('/home/alex/.config/Hermes/composer-images/preview-shot_47a88d.png')
    ).resolves.toEqual({
      path: '/home/alex/.config/Hermes/composer-images/preview-shot_47a88d.png',
      saved: true
    })

    expect(saveGatewayFile).not.toHaveBeenCalled()
  })

  it('still uses the gateway when the path is not on this disk', async () => {
    const readFileDataUrl = vi.fn(async () => {
      throw new Error('ENOENT')
    })

    vi.stubGlobal('window', { hermesDesktop: { readFileDataUrl, saveGatewayFile } })

    await expect(downloadGatewayMediaFile('/srv/app/report.md')).resolves.toEqual({
      path: '/Users/me/Downloads/report.md',
      saved: true
    })

    expect(saveGatewayFile).toHaveBeenCalled()
  })
})

describe('gatewayMediaDataUrlLocalFallback', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
    $connection.set(null)
  })

  it('prefers the gateway when the file lives there', async () => {
    const api = vi.fn(async () => ({ dataUrl: 'data:image/png;base64,cmVtb3Rl' }))
    const readFileDataUrl = vi.fn(async () => 'data:image/png;base64,bG9jYWw=')

    vi.stubGlobal('window', { hermesDesktop: { api, readFileDataUrl } })
    $connection.set({ mode: 'remote' } as never)

    await expect(gatewayMediaDataUrlLocalFallback('/srv/app/a.png')).resolves.toBe('data:image/png;base64,cmVtb3Rl')
    expect(readFileDataUrl).not.toHaveBeenCalled()
  })

  it('falls back to this machine for shell-produced screenshots', async () => {
    const api = vi.fn(async () => {
      throw new Error('404 not found')
    })

    const readFileDataUrl = vi.fn(async () => 'data:image/png;base64,bG9jYWw=')

    vi.stubGlobal('window', { hermesDesktop: { api, readFileDataUrl } })
    $connection.set({ mode: 'remote' } as never)

    await expect(
      gatewayMediaDataUrlLocalFallback('/home/alex/.config/Hermes/composer-images/preview-shot_47a88d.png')
    ).resolves.toBe('data:image/png;base64,bG9jYWw=')

    expect(readFileDataUrl).toHaveBeenCalledWith(
      '/home/alex/.config/Hermes/composer-images/preview-shot_47a88d.png'
    )
  })

  it('reports both hosts when the file exists on neither', async () => {
    const api = vi.fn(async () => {
      throw new Error('404 not found')
    })

    vi.stubGlobal('window', { hermesDesktop: { api, readFileDataUrl: vi.fn(async () => '') } })
    $connection.set({ mode: 'remote' } as never)

    await expect(gatewayMediaDataUrlLocalFallback('/nowhere/a.png')).rejects.toThrow(
      'not found on the gateway or this machine'
    )
  })
})
