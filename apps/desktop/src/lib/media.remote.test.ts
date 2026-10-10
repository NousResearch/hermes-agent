import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { setApiRequestConnection, setApiRequestProfile } from '@/hermes'
import { $connection } from '@/store/session'

import {
  downloadGatewayMediaFile,
  filePathFromMediaPath,
  gatewayImageProxyDataUrl,
  gatewayMediaDataUrl,
  isInlineMediaSrc,
  mediaExternalUrl,
  mediaGatewayStreamUrl,
  resolveMediaDisplaySrc,
  resolveMediaPlaybackSrc
} from './media'
import { CONVERSATION_LEASE, resetTtsLeasesForTests, syncTtsLease } from './tts-lease'

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

  // #84361: the raw `file://${path}` concat broke on URL-structural
  // characters — `#`/`?` truncated the path at the fragment/query boundary
  // and a stray `%` made the main process's fileURLToPath throw.
  it('escapes URL-structural characters so the whole path survives the round trip', () => {
    $connection.set({ mode: 'local' } as never)

    const roundTrip = (fileUrl: string): string => {
      const parsed = new URL(fileUrl)

      return decodeURIComponent(parsed.pathname)
    }

    expect(mediaExternalUrl('/tmp/Report #2.pdf')).toBe('file:///tmp/Report %232.pdf')
    expect(mediaExternalUrl('/tmp/a?b.pdf')).toBe('file:///tmp/a%3Fb.pdf')
    expect(mediaExternalUrl('/tmp/100%.pdf')).toBe('file:///tmp/100%25.pdf')

    for (const path of ['/tmp/Report #2.pdf', '/tmp/a?b.pdf', '/tmp/100%.pdf', '/tmp/café.png']) {
      expect(roundTrip(mediaExternalUrl(path))).toBe(path)
    }
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
    setApiRequestConnection(null)
    setApiRequestProfile(null)
    resetTtsLeasesForTests()
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

  it('pins A media delivery and synthesis leases through an A→B→A switch with colliding profile names', async () => {
    const api = vi.fn(async (_request: unknown) => ({ ok: true }))
    const ownerA = { connectionId: 'gateway-a', profile: 'shared' }
    const ownerB = { connectionId: 'gateway-b', profile: 'shared' }

    vi.stubGlobal('window', { hermesDesktop: { api } })
    setApiRequestConnection(ownerA.connectionId)
    setApiRequestProfile(ownerA.profile)
    $connection.set({ ...ownerA, mode: 'remote' } as never)

    await syncTtsLease(CONVERSATION_LEASE, true, ownerA)
    await syncTtsLease(CONVERSATION_LEASE, true, ownerB)

    setApiRequestConnection(ownerB.connectionId)
    setApiRequestProfile(ownerB.profile)
    $connection.set({ ...ownerB, mode: 'remote' } as never)

    const first = await resolveMediaPlaybackSrc('/tmp/reply.mp3', ownerA)
    await syncTtsLease(CONVERSATION_LEASE, false, ownerA)

    setApiRequestConnection(ownerA.connectionId)
    $connection.set({ ...ownerA, mode: 'remote' } as never)
    const again = await resolveMediaPlaybackSrc('/tmp/reply.mp3', ownerA)
    await syncTtsLease(CONVERSATION_LEASE, false, ownerB)

    expect(first).toBe('hermes-media://remote/%2Ftmp%2Freply.mp3?connectionId=gateway-a&profile=shared')
    expect(again).toBe(first)

    const leaseCalls = api.mock.calls.map(
      ([request]) =>
        request as {
          body: { active: boolean; lease: string }
          connectionId?: string
          profile?: string
        }
    )
    expect(leaseCalls.map(({ body, connectionId, profile }) => [connectionId, profile, body.active])).toEqual([
      ['gateway-a', 'shared', true],
      ['gateway-b', 'shared', true],
      ['gateway-a', 'shared', false],
      ['gateway-b', 'shared', false]
    ])
    expect(leaseCalls[0].body.lease).toBe(leaseCalls[2].body.lease)
    expect(leaseCalls[1].body.lease).toBe(leaseCalls[3].body.lease)
    expect(leaseCalls[0].body.lease).not.toBe(leaseCalls[1].body.lease)
    expect(leaseCalls[0].body.lease).not.toBe(CONVERSATION_LEASE)
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

describe('gatewayImageProxyDataUrl (#74564)', () => {
  const api = vi.fn(async ({ path }: { path: string }) => {
    if (path.startsWith('/api/media/proxy?')) {
      return { dataUrl: 'data:image/png;base64,cGRveGllZA==' }
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

  it('fetches a client-unreachable CDN image through the gateway proxy', async () => {
    const url = 'https://v3.fal.media/media/abc123?x=1'

    await expect(gatewayImageProxyDataUrl(url)).resolves.toBe('data:image/png;base64,cGRveGllZA==')
    expect(api).toHaveBeenCalledWith({ path: `/api/media/proxy?url=${encodeURIComponent(url)}` })
  })

  it('pins the request to the owner connection and profile when given', async () => {
    await gatewayImageProxyDataUrl('https://fal.run/img.png', {
      connectionId: 'studio-ssh',
      profile: 'voice reviewer'
    })

    expect(api).toHaveBeenCalledWith({
      connectionId: 'studio-ssh',
      path: '/api/media/proxy?url=https%3A%2F%2Ffal.run%2Fimg.png',
      profile: 'voice reviewer'
    })
  })

  it('returns an empty string for non-http sources and proxy failures', async () => {
    await expect(gatewayImageProxyDataUrl('data:image/png;base64,aGk=')).resolves.toBe('')
    await expect(gatewayImageProxyDataUrl('/local/file.png')).resolves.toBe('')
    expect(api).not.toHaveBeenCalled()

    api.mockRejectedValueOnce(new Error('403 Image host not allowed'))
    await expect(gatewayImageProxyDataUrl('https://fal.media/x.png')).resolves.toBe('')
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
      path: 'file:///Users/me/project/a%20b.md',
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
