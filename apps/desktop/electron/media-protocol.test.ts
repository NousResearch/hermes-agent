import { describe, expect, it, vi } from 'vitest'

import {
  createMediaProtocolHandler,
  isStreamableMediaPath,
  type MediaProtocolDependencies,
  mediaRequestHeaders,
  remoteMediaEndpoint
} from './media-protocol'

function dependencies(overrides: Partial<MediaProtocolDependencies> = {}) {
  return {
    ensureRemoteBearer: vi.fn(async (_baseUrl: string) => null),
    fetchLocal: vi.fn(async (_resolvedPath: string, _headers: Headers) => new Response('local', { status: 206 })),
    fetchRemote: vi.fn(async (_url: string, _headers: Headers) => new Response('remote', { status: 206 })),
    fetchRemoteWithCookies: vi.fn(async (_url: string, _headers: Headers) => new Response('cookie', { status: 206 })),
    resolveLocalFile: vi.fn(async (filePath: string) => filePath),
    resolveRemoteConnection: vi.fn(async (_scope: { connectionId?: string; profile?: string }) => ({
      authMode: 'token' as const,
      baseUrl: 'https://gateway.test',
      mode: 'remote' as const,
      token: 'secret'
    })),
    ...overrides
  }
}

function request(url: string, headers: Record<string, string> = {}, method = 'GET') {
  return { headers: new Headers(headers), method, url }
}

describe('media protocol helpers', () => {
  it('recognises only supported audio/video extensions case-insensitively', () => {
    expect(isStreamableMediaPath('/tmp/render.MP4')).toBe(true)
    expect(isStreamableMediaPath('/tmp/voice.flac')).toBe(true)
    expect(isStreamableMediaPath('/tmp/secrets.txt')).toBe(false)
  })

  it('forwards range/cache negotiation headers but strips renderer credentials', () => {
    const headers = mediaRequestHeaders(
      new Headers({
        Accept: 'video/mp4',
        Authorization: 'Bearer renderer-secret',
        Cookie: 'session=renderer-secret',
        Host: 'attacker.test',
        Range: 'bytes=10-20'
      })
    )

    expect(Object.fromEntries(headers)).toEqual({ accept: 'video/mp4', range: 'bytes=10-20' })
  })

  it('preserves a configured gateway path prefix', () => {
    const endpoint = new URL(remoteMediaEndpoint('https://gateway.test/hermes/', '/tmp/a b.mp4'))

    expect(endpoint.pathname).toBe('/hermes/api/fs/stream')
    expect(endpoint.searchParams.get('path')).toBe('/tmp/a b.mp4')
  })
})

describe('createMediaProtocolHandler', () => {
  it('streams local media through the resolved local-file dependency', async () => {
    const deps = dependencies()

    const response = await createMediaProtocolHandler(deps)(
      request('hermes-media://stream/%2Ftmp%2Fclip.mp4', {
        Authorization: 'Bearer renderer-secret',
        Range: 'bytes=1-3'
      })
    )

    expect(response.status).toBe(206)
    expect(deps.resolveLocalFile).toHaveBeenCalledWith('/tmp/clip.mp4')
    expect(deps.fetchLocal).toHaveBeenCalledOnce()
    const [, headers] = vi.mocked(deps.fetchLocal).mock.calls[0]
    expect(headers.get('range')).toBe('bytes=1-3')
    expect(headers.get('authorization')).toBeNull()
  })

  it('preserves explicit HEAD requests through the local stream fetch', async () => {
    const fetchLocal = vi.fn(async (..._args: unknown[]) => new Response(null, { status: 200 }))

    const deps = dependencies({
      fetchLocal: fetchLocal as MediaProtocolDependencies['fetchLocal']
    })

    const response = await createMediaProtocolHandler(deps)(
      request('hermes-media://stream/%2Ftmp%2Fclip.mp4', {}, 'HEAD')
    )

    expect(response.status).toBe(200)
    expect(fetchLocal).toHaveBeenCalledOnce()
    expect(fetchLocal.mock.calls[0]?.[2]).toBe('HEAD')
    expect(deps.resolveRemoteConnection).not.toHaveBeenCalled()
  })

  it('proxies token-auth remote media without placing the token in the URL', async () => {
    const deps = dependencies({
      resolveRemoteConnection: vi.fn(async () => ({
        authMode: 'token' as const,
        baseUrl: 'https://gateway.test/hermes',
        mode: 'remote' as const,
        token: 's e/cret'
      }))
    })

    const response = await createMediaProtocolHandler(deps)(
      request('hermes-media://remote/%2Froot%2Foutputs%2Frender.mp4?connectionId=work-ssh&profile=reviewer&sessionId=session-7', {
        Range: 'bytes=0-1023'
      })
    )

    expect(response.status).toBe(206)
    expect(deps.resolveRemoteConnection).toHaveBeenCalledWith({ connectionId: 'work-ssh', profile: 'reviewer' })
    expect(deps.fetchRemote).toHaveBeenCalledOnce()
    const [rawUrl, headers] = vi.mocked(deps.fetchRemote).mock.calls[0]
    const url = new URL(rawUrl)
    expect(url.pathname).toBe('/hermes/api/fs/stream')
    expect(url.searchParams.get('path')).toBe('/root/outputs/render.mp4')
    expect(url.searchParams.get('session_id')).toBe('session-7')
    expect(url.searchParams.has('token')).toBe(false)
    expect(headers.get('x-hermes-session-token')).toBe('s e/cret')
    expect(headers.get('range')).toBe('bytes=0-1023')
  })

  it('adds profile scope when one registry backend serves multiple profiles', async () => {
    const deps = dependencies({
      resolveRemoteConnection: vi.fn(async () => ({
        authMode: 'token' as const,
        baseUrl: 'https://gateway.test',
        mode: 'remote' as const,
        sharedRemote: true,
        token: 'secret'
      }))
    })

    await createMediaProtocolHandler(deps)(
      request(
        'hermes-media://remote/%2Froot%2Foutputs%2Frender.mp4?connectionId=cloud&profile=desktop-alias&targetProfile=research'
      )
    )

    const [rawUrl] = vi.mocked(deps.fetchRemote).mock.calls[0]
    const url = new URL(rawUrl)

    expect(url.searchParams.get('path')).toBe('/root/outputs/render.mp4')
    expect(url.searchParams.get('profile')).toBe('research')
    expect(deps.resolveRemoteConnection).toHaveBeenCalledWith({
      connectionId: 'cloud',
      profile: 'desktop-alias'
    })
  })

  it('falls back to the legacy managed stream only when the new route is missing', async () => {
    const fetchRemote = vi
      .fn()
      .mockResolvedValueOnce(Response.json({ detail: 'Not Found' }, { status: 404 }))
      .mockResolvedValueOnce(new Response('legacy', { status: 206 }))

    const deps = dependencies({ fetchRemote })

    const response = await createMediaProtocolHandler(deps)(
      request('hermes-media://remote/%2Froot%2Foutputs%2Frender.mp4?connectionId=work-ssh')
    )

    expect(response.status).toBe(206)
    expect(fetchRemote).toHaveBeenCalledTimes(2)
    expect(new URL(fetchRemote.mock.calls[0][0]).pathname).toBe('/api/fs/stream')
    expect(new URL(fetchRemote.mock.calls[1][0]).pathname).toBe('/api/files/stream')
  })

  it('uses the same OAuth cookie transport for the legacy 404 fallback', async () => {
    const fetchRemoteWithCookies = vi
      .fn()
      .mockResolvedValueOnce(Response.json({ detail: 'Not Found' }, { status: 404 }))
      .mockResolvedValueOnce(new Response('legacy', { status: 206 }))

    const deps = dependencies({
      fetchRemoteWithCookies,
      resolveRemoteConnection: vi.fn(async () => ({
        authMode: 'oauth' as const,
        baseUrl: 'https://gateway.test',
        mode: 'remote' as const,
        token: null
      }))
    })

    const response = await createMediaProtocolHandler(deps)(
      request('hermes-media://remote/%2Froot%2Foutputs%2Frender.mp4')
    )

    expect(response.status).toBe(206)
    expect(fetchRemoteWithCookies).toHaveBeenCalledTimes(2)
    expect(new URL(fetchRemoteWithCookies.mock.calls[1][0]).pathname).toBe('/api/files/stream')
  })

  it('safely probes a missing HEAD route before using the legacy endpoint', async () => {
    const fetchRemote = vi
      .fn()
      .mockResolvedValueOnce(new Response(null, { status: 404 }))
      .mockResolvedValueOnce(Response.json({ detail: 'Not Found' }, { status: 404 }))
      .mockResolvedValueOnce(new Response(null, { status: 200 }))

    const deps = dependencies({ fetchRemote })

    const response = await createMediaProtocolHandler(deps)(
      request('hermes-media://remote/%2Froot%2Foutputs%2Frender.mp4', {}, 'HEAD')
    )

    expect(response.status).toBe(200)
    expect(fetchRemote).toHaveBeenCalledTimes(3)
    expect(fetchRemote.mock.calls.map(call => call[2])).toEqual(['HEAD', 'GET', 'HEAD'])
    expect((fetchRemote.mock.calls[1][1] as Headers).get('range')).toBe('bytes=0-0')
    expect(new URL(fetchRemote.mock.calls[2][0]).pathname).toBe('/api/files/stream')
  })

  it('does not retry auth or server failures against the legacy endpoint', async () => {
    for (const status of [401, 500]) {
      const fetchRemote = vi.fn(async () => new Response('failed', { status }))
      const deps = dependencies({ fetchRemote })

      const response = await createMediaProtocolHandler(deps)(
        request('hermes-media://remote/%2Froot%2Foutputs%2Frender.mp4')
      )

      expect(response.status).toBe(status)
      expect(fetchRemote).toHaveBeenCalledOnce()
    }
  })

  it('does not turn a file or session 404 into an unscoped legacy request', async () => {
    const fetchRemote = vi.fn(async () => Response.json({ detail: 'Session not found' }, { status: 404 }))
    const deps = dependencies({ fetchRemote })

    const response = await createMediaProtocolHandler(deps)(
      request('hermes-media://remote/%2Froot%2Foutputs%2Frender.mp4?sessionId=unknown')
    )

    expect(response.status).toBe(404)
    expect(fetchRemote).toHaveBeenCalledOnce()
  })

  it('preserves explicit HEAD requests through the token-auth remote proxy', async () => {
    const fetchRemote = vi.fn(async (..._args: unknown[]) => new Response(null, { status: 200 }))

    const deps = dependencies({
      fetchRemote: fetchRemote as MediaProtocolDependencies['fetchRemote']
    })

    const response = await createMediaProtocolHandler(deps)(
      request('hermes-media://remote/%2Froot%2Foutputs%2Frender.mp4', {}, 'HEAD')
    )

    expect(response.status).toBe(200)
    expect(fetchRemote).toHaveBeenCalledOnce()
    expect(fetchRemote.mock.calls[0]?.[2]).toBe('HEAD')
  })

  it('rejects protocol methods other than GET and HEAD', async () => {
    const deps = dependencies()

    const response = await createMediaProtocolHandler(deps)(
      request('hermes-media://remote/%2Froot%2Foutputs%2Frender.mp4', {}, 'POST')
    )

    expect(response.status).toBe(405)
    expect(response.headers.get('allow')).toBe('GET, HEAD')
    expect(deps.resolveRemoteConnection).not.toHaveBeenCalled()
    expect(deps.fetchRemote).not.toHaveBeenCalled()
  })

  it('uses a refreshed native bearer for OAuth remote media when available', async () => {
    const deps = dependencies({
      ensureRemoteBearer: vi.fn(async () => 'native-access-token'),
      resolveRemoteConnection: vi.fn(async () => ({
        authMode: 'oauth' as const,
        baseUrl: 'https://gateway.test',
        mode: 'remote' as const,
        token: null
      }))
    })

    const response = await createMediaProtocolHandler(deps)(request('hermes-media://remote/%2Ftmp%2Fclip.mp4'))

    expect(response.status).toBe(206)
    expect(deps.fetchRemote).toHaveBeenCalledOnce()
    expect(deps.fetchRemoteWithCookies).not.toHaveBeenCalled()
    const [, headers] = vi.mocked(deps.fetchRemote).mock.calls[0]
    expect(headers.get('authorization')).toBe('Bearer native-access-token')
  })

  it('preserves explicit HEAD requests through the native-bearer remote fetch', async () => {
    const fetchRemote = vi.fn(async (..._args: unknown[]) => new Response(null, { status: 200 }))

    const deps = dependencies({
      ensureRemoteBearer: vi.fn(async () => 'native-access-token'),
      fetchRemote: fetchRemote as MediaProtocolDependencies['fetchRemote'],
      resolveRemoteConnection: vi.fn(async () => ({
        authMode: 'oauth' as const,
        baseUrl: 'https://gateway.test',
        mode: 'remote' as const,
        token: null
      }))
    })

    const response = await createMediaProtocolHandler(deps)(
      request('hermes-media://remote/%2Ftmp%2Fclip.mp4', {}, 'HEAD')
    )

    expect(response.status).toBe(200)
    expect(fetchRemote).toHaveBeenCalledOnce()
    expect((fetchRemote.mock.calls[0]?.[1] as Headers).get('authorization')).toBe('Bearer native-access-token')
    expect(fetchRemote.mock.calls[0]?.[2]).toBe('HEAD')
    expect(deps.fetchRemoteWithCookies).not.toHaveBeenCalled()
  })

  it('uses the isolated OAuth cookie session when no native bearer exists', async () => {
    const deps = dependencies({
      resolveRemoteConnection: vi.fn(async () => ({
        authMode: 'oauth' as const,
        baseUrl: 'https://gateway.test',
        mode: 'remote' as const,
        token: null
      }))
    })

    const response = await createMediaProtocolHandler(deps)(request('hermes-media://remote/%2Ftmp%2Fclip.mp4'))

    expect(response.status).toBe(206)
    expect(deps.fetchRemote).not.toHaveBeenCalled()
    expect(deps.fetchRemoteWithCookies).toHaveBeenCalledOnce()
    const [, headers] = vi.mocked(deps.fetchRemoteWithCookies).mock.calls[0]
    expect(headers.get('authorization')).toBeNull()
  })

  it('preserves explicit HEAD requests through the isolated-cookie remote fetch', async () => {
    const fetchRemoteWithCookies = vi.fn(async (..._args: unknown[]) => new Response(null, { status: 200 }))

    const deps = dependencies({
      fetchRemoteWithCookies: fetchRemoteWithCookies as MediaProtocolDependencies['fetchRemoteWithCookies'],
      resolveRemoteConnection: vi.fn(async () => ({
        authMode: 'oauth' as const,
        baseUrl: 'https://gateway.test',
        mode: 'remote' as const,
        token: null
      }))
    })

    const response = await createMediaProtocolHandler(deps)(
      request('hermes-media://remote/%2Ftmp%2Fclip.mp4', {}, 'HEAD')
    )

    expect(response.status).toBe(200)
    expect(fetchRemoteWithCookies).toHaveBeenCalledOnce()
    expect((fetchRemoteWithCookies.mock.calls[0]?.[1] as Headers).get('authorization')).toBeNull()
    expect(fetchRemoteWithCookies.mock.calls[0]?.[2]).toBe('HEAD')
    expect(deps.fetchRemote).not.toHaveBeenCalled()
  })

  it('fails closed for unsupported extensions and missing remote auth', async () => {
    const deps = dependencies({
      resolveRemoteConnection: vi.fn(async () => ({
        authMode: 'token' as const,
        baseUrl: 'https://gateway.test',
        mode: 'remote' as const,
        token: null
      }))
    })

    const handler = createMediaProtocolHandler(deps)

    expect((await handler(request('hermes-media://remote/%2Ftmp%2Fsecret.txt'))).status).toBe(415)
    expect((await handler(request('hermes-media://remote/%2Ftmp%2Fclip.mp4'))).status).toBe(401)
    expect(deps.fetchRemote).not.toHaveBeenCalled()
  })
})
