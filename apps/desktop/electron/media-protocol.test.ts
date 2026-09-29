import { mkdtemp, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'

import { describe, expect, it, vi } from 'vitest'

import {
  createMediaProtocolHandler,
  isPdfStreamUrl,
  isStreamableMediaPath,
  type MediaProtocolDependencies,
  mediaRequestHeaders,
  remoteMediaEndpoint,
  validatePdfPreviewStream
} from './media-protocol'
import { fetchLocalMedia } from './media-range'

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
  it('recognises streamable media and PDF extensions case-insensitively', () => {
    expect(isStreamableMediaPath('/tmp/render.MP4')).toBe(true)
    expect(isStreamableMediaPath('/tmp/voice.flac')).toBe(true)
    expect(isStreamableMediaPath('/tmp/book.PDF')).toBe(true)
    expect(isStreamableMediaPath('/tmp/secrets.txt')).toBe(false)
    expect(isPdfStreamUrl('hermes-media://stream/%2Ftmp%2Fbook.PDF')).toBe(true)
    expect(isPdfStreamUrl('hermes-media://remote/%2Ftmp%2Fbook.pdf?profile=work')).toBe(true)
    expect(isPdfStreamUrl('hermes-media://stream/%2Ftmp%2Fsecrets.txt')).toBe(false)
    expect(isPdfStreamUrl('https://example.com/book.pdf')).toBe(false)
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

    expect(endpoint.pathname).toBe('/hermes/api/files/stream')
    expect(endpoint.searchParams.get('path')).toBe('/tmp/a b.mp4')
  })

  it('uses the desktop filesystem stream route for remote PDFs', () => {
    const endpoint = new URL(remoteMediaEndpoint('https://gateway.test/hermes/', '/tmp/book.pdf', 'work'))

    expect(endpoint.pathname).toBe('/hermes/api/fs/stream')
    expect(endpoint.searchParams.get('path')).toBe('/tmp/book.pdf')
    expect(endpoint.searchParams.get('profile')).toBe('work')
  })
})

describe('createMediaProtocolHandler', () => {
  it('serves a PDF by range without transferring the whole document', async () => {
    const dir = await mkdtemp(path.join(tmpdir(), 'pdf-protocol-range-'))
    const file = path.join(dir, 'book.pdf')
    const bytes = Buffer.from('%PDF-1.7\n')
    await writeFile(file, bytes)

    const deps = dependencies({ fetchLocal: fetchLocalMedia, resolveLocalFile: async () => file })
    const url = `hermes-media://stream/${encodeURIComponent(file)}`
    const head = await createMediaProtocolHandler(deps)(request(url, {}, 'HEAD'))
    const range = await createMediaProtocolHandler(deps)(request(url, { Range: 'bytes=0-4' }))

    expect(head.status).toBe(200)
    expect(head.headers.get('content-type')).toBe('application/pdf')
    expect(head.headers.get('accept-ranges')).toBe('bytes')
    expect(range.status).toBe(206)
    expect(await range.text()).toBe('%PDF-')
  })
  it('recovers native refresh outages through a live cookie without turning an empty jar into auth failure', async () => {
    for (const cookieStatus of [206, 401, 403, 503]) {
      const deps = dependencies({
        ensureRemoteBearer: async () => {
          throw new Error('refresh timed out')
        },
        resolveRemoteConnection: async () => ({ authMode: 'oauth', baseUrl: 'https://gw.test', mode: 'remote' }),
        fetchRemoteWithCookies: async () => new Response('cookie', { status: cookieStatus })
      })

      const response = await createMediaProtocolHandler(deps)(request('hermes-media://remote/%2Ftmp%2Fclip.mp4'))
      expect(response.status).toBe(cookieStatus === 401 || cookieStatus === 403 ? 502 : cookieStatus)
    }
  })

  it('serves a 206 byte range through the production local fetch (seeking)', async () => {
    const dir = await mkdtemp(path.join(tmpdir(), 'media-protocol-range-'))
    const file = path.join(dir, 'clip.mp4')
    const bytes = Buffer.from(Array.from({ length: 100 }, (_, i) => i))

    await writeFile(file, bytes)

    const deps = dependencies({
      fetchLocal: fetchLocalMedia,
      resolveLocalFile: vi.fn(async () => file)
    })

    const response = await createMediaProtocolHandler(deps)(
      request(`hermes-media://stream/${encodeURIComponent(file)}`, { Range: 'bytes=10-19' })
    )

    expect(response.status).toBe(206)
    expect(response.headers.get('content-range')).toBe('bytes 10-19/100')
    expect(response.headers.get('accept-ranges')).toBe('bytes')
    expect(Buffer.from(await response.arrayBuffer()).equals(bytes.subarray(10, 20))).toBe(true)
  })

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
      request('hermes-media://remote/%2Froot%2Foutputs%2Frender.mp4?connectionId=work-ssh&profile=reviewer', {
        Range: 'bytes=0-1023'
      })
    )

    expect(response.status).toBe(206)
    expect(deps.resolveRemoteConnection).toHaveBeenCalledWith({ connectionId: 'work-ssh', profile: 'reviewer' })
    expect(deps.fetchRemote).toHaveBeenCalledOnce()
    const [rawUrl, headers] = vi.mocked(deps.fetchRemote).mock.calls[0]
    const url = new URL(rawUrl)
    expect(url.pathname).toBe('/hermes/api/files/stream')
    expect(url.searchParams.get('path')).toBe('/root/outputs/render.mp4')
    expect(url.searchParams.has('token')).toBe(false)
    expect(headers.get('x-hermes-session-token')).toBe('s e/cret')
    expect(headers.get('range')).toBe('bytes=0-1023')
  })

  it('sends the connection extra gateway headers on remote media without clobbering range negotiation', async () => {
    const deps = dependencies({
      resolveRemoteConnection: vi.fn(async () => ({
        authMode: 'token' as const,
        baseUrl: 'https://gateway.test',
        headers: { 'CF-Access-Client-Id': 'client-id', Range: 'bytes=9-9' },
        mode: 'remote' as const,
        token: 'secret'
      }))
    })

    await createMediaProtocolHandler(deps)(
      request('hermes-media://remote/%2Ftmp%2Fclip.mp4', { Range: 'bytes=0-1023' })
    )

    const [, headers] = vi.mocked(deps.fetchRemote).mock.calls[0]
    expect(headers.get('cf-access-client-id')).toBe('client-id')
    expect(headers.get('range')).toBe('bytes=0-1023')
    expect(headers.get('x-hermes-session-token')).toBe('secret')
  })

  it('sends the connection extra gateway headers on OAuth cookie-session remote media', async () => {
    const deps = dependencies({
      resolveRemoteConnection: vi.fn(async () => ({
        authMode: 'oauth' as const,
        baseUrl: 'https://gateway.test',
        headers: { 'CF-Access-Client-Id': 'client-id' },
        mode: 'remote' as const,
        token: null
      }))
    })

    await createMediaProtocolHandler(deps)(request('hermes-media://remote/%2Ftmp%2Fclip.mp4'))

    const [, headers] = vi.mocked(deps.fetchRemoteWithCookies).mock.calls[0]
    expect(headers.get('cf-access-client-id')).toBe('client-id')
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
      request('hermes-media://remote/%2Froot%2Foutputs%2Frender.mp4?connectionId=cloud&profile=research')
    )

    const [rawUrl] = vi.mocked(deps.fetchRemote).mock.calls[0]
    const url = new URL(rawUrl)

    expect(url.searchParams.get('path')).toBe('/root/outputs/render.mp4')
    expect(url.searchParams.get('profile')).toBe('research')
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

describe('validatePdfPreviewStream', () => {
  const remoteUrl = 'hermes-media://remote/%2Ftmp%2Fbook.pdf'
  const localUrl = 'hermes-media://stream/%2Ftmp%2Fbook.pdf'

  it.each([404, 405, 415])('falls back when an older remote backend answers %i', async status => {
    const handler = vi.fn(async () => new Response('unsupported', { status }))

    await expect(validatePdfPreviewStream(remoteUrl, handler)).resolves.toBeNull()
  })

  it('keeps invalid PDFs and authentication failures fail-closed', async () => {
    await expect(
      validatePdfPreviewStream(localUrl, async () => new Response('bad header', { status: 422 }))
    ).rejects.toThrow('Invalid PDF file header')
    await expect(
      validatePdfPreviewStream(remoteUrl, async () => new Response('unauthorized', { status: 401 }))
    ).rejects.toThrow('PDF preview unavailable (HTTP 401)')
  })

  it('returns a validated PDF stream URL', async () => {
    await expect(
      validatePdfPreviewStream(
        localUrl,
        async () => new Response(null, { headers: { 'Content-Type': 'application/pdf' }, status: 200 })
      )
    ).resolves.toBe(localUrl)
  })
})
