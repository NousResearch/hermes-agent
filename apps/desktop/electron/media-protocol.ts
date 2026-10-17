import { httpStatusError, readStatusCode } from './api-transport'
import { requestWithOauthFallback } from './oauth-rest-request'

const STREAMABLE_MEDIA_EXTENSIONS = [
  '.avi',
  '.flac',
  '.m4a',
  '.mkv',
  '.mov',
  '.mp3',
  '.mp4',
  '.ogg',
  '.opus',
  '.wav',
  '.webm'
] as const

const FORWARDED_MEDIA_REQUEST_HEADERS = ['accept', 'if-modified-since', 'if-none-match', 'if-range', 'range'] as const

export const MEDIA_PROTOCOL = 'hermes-media'

// Remote media rides Chromium's 6-per-host HTTP/1.1 pool together with REST and
// WS-ticket minting. A <video> stops reading once buffered but keeps its request
// open, so piping the gateway stream straight through pins a pooled socket for
// as long as the element lives (or forever, once abandoned); a few clips starve
// the pool and the gateway becomes unreachable. The renderer still gets one
// response for the range it asked for (Chromium treats a short 206 as a
// truncated file), but it is fed from bounded gateway ranges, each read to
// completion and requested only when the renderer reads on.
export const REMOTE_MEDIA_CHUNK_BYTES = 2 * 1024 * 1024

const OPEN_OR_CLOSED_RANGE = /^bytes=(\d+)-(\d*)$/
const CONTENT_RANGE = /^bytes (\d+)-(\d+)\/(\d+)$/

async function remoteMediaInChunks(
  requestedRange: null | string,
  fetchRange: (range: string) => Promise<Response>
): Promise<Response> {
  const requested = OPEN_OR_CLOSED_RANGE.exec((requestedRange ?? 'bytes=0-').trim())

  // Suffix and multi-range requests cannot be split without the file size.
  if (!requested) {
    return fetchRange(requestedRange ?? '')
  }

  const start = Number(requested[1])
  const askedEnd = requested[2] === '' ? Infinity : Number(requested[2])
  const nextRange = (from: number, last: number) => `bytes=${from}-${Math.min(last, from + REMOTE_MEDIA_CHUNK_BYTES - 1)}`
  const first = await fetchRange(nextRange(start, askedEnd))
  const served = CONTENT_RANGE.exec(first.headers.get('content-range') ?? '')

  // A gateway that ignored Range answers 200 with the whole file: stream it.
  if (first.status !== 206 || !served) {
    return first
  }

  const size = Number(served[3])
  const end = Math.min(askedEnd, size - 1)
  const head = new Uint8Array(await first.arrayBuffer())
  let next = Number(served[2]) + 1

  const headers = new Headers(first.headers)
  headers.set('content-range', `bytes ${start}-${end}/${size}`)
  headers.set('content-length', String(end - start + 1))

  if (next > end) {
    return new Response(head, { headers, status: 206 })
  }

  const body = new ReadableStream<Uint8Array>(
    {
      start(controller) {
        controller.enqueue(head)
      },
      async pull(controller) {
        const response = await fetchRange(nextRange(next, end))
        const chunk = CONTENT_RANGE.exec(response.headers.get('content-range') ?? '')

        if (response.status !== 206 || !chunk || Number(chunk[1]) !== next) {
          await response.body?.cancel()
          throw new Error('Remote media range unavailable')
        }

        controller.enqueue(new Uint8Array(await response.arrayBuffer()))
        next = Number(chunk[2]) + 1

        if (next > end) {
          controller.close()
        }
      }
    },
    { highWaterMark: 0 }
  )

  return new Response(body, { headers, status: 206 })
}

type MediaProtocolMode = 'remote' | 'stream'

interface MediaProtocolTarget {
  connectionId?: string
  filePath: string
  mode: MediaProtocolMode
  profile?: string
}

export interface MediaRemoteScope {
  connectionId?: string
  profile?: string
}

export interface MediaRemoteConnection {
  authMode?: 'oauth' | 'token'
  baseUrl: string
  headers?: Record<string, string>
  mode?: 'local' | 'remote'
  token?: null | string
  sharedRemote?: boolean
}

type MediaRequestMethod = 'GET' | 'HEAD'

export interface MediaProtocolDependencies {
  ensureRemoteBearer: (baseUrl: string) => Promise<null | string>
  fetchLocal: (resolvedPath: string, headers: Headers, method: MediaRequestMethod) => Promise<Response>
  fetchRemote: (url: string, headers: Headers, method: MediaRequestMethod) => Promise<Response>
  fetchRemoteWithCookies: (url: string, headers: Headers, method: MediaRequestMethod) => Promise<Response>
  resolveLocalFile: (filePath: string) => Promise<string>
  resolveRemoteConnection: (scope: MediaRemoteScope) => Promise<MediaRemoteConnection>
}

function parseMediaProtocolTarget(rawUrl: string): MediaProtocolTarget {
  const url = new URL(rawUrl)
  const mode = url.hostname as MediaProtocolMode

  if (mode !== 'remote' && mode !== 'stream') {
    throw new Error('Unsupported media protocol target')
  }

  const filePath = decodeURIComponent(url.pathname.replace(/^\/+/, ''))

  if (!filePath) {
    throw new Error('Missing media path')
  }

  const connectionId = url.searchParams.get('connectionId')?.trim() || undefined
  const profile = url.searchParams.get('profile')?.trim() || undefined

  return { connectionId, filePath, mode, profile }
}

export function isStreamableMediaPath(filePath: string): boolean {
  const lower = filePath.toLowerCase()

  return STREAMABLE_MEDIA_EXTENSIONS.some(extension => lower.endsWith(extension))
}

export function mediaRequestHeaders(source: Headers): Headers {
  const forwarded = new Headers()

  for (const name of FORWARDED_MEDIA_REQUEST_HEADERS) {
    const value = source.get(name)

    if (value) {
      forwarded.set(name, value)
    }
  }

  return forwarded
}

export function remoteMediaEndpoint(baseUrl: string, filePath: string, profile?: string): string {
  const normalizedBase = baseUrl.replace(/\/+$/, '')
  const url = new URL(`${normalizedBase}/api/files/stream`)

  if (url.protocol !== 'http:' && url.protocol !== 'https:') {
    throw new Error(`Unsupported Hermes backend URL protocol: ${url.protocol}`)
  }

  url.searchParams.set('path', filePath)

  if (profile) {
    url.searchParams.set('profile', profile)
  }

  return url.toString()
}

export function createMediaProtocolHandler(dependencies: MediaProtocolDependencies) {
  return async (request: Pick<Request, 'headers' | 'method' | 'url'>): Promise<Response> => {
    if (request.method !== 'GET' && request.method !== 'HEAD') {
      return new Response('Method not allowed', {
        headers: { allow: 'GET, HEAD' },
        status: 405
      })
    }

    const method: MediaRequestMethod = request.method
    let target: MediaProtocolTarget

    try {
      target = parseMediaProtocolTarget(request.url)
    } catch {
      return new Response('Media not found', { status: 404 })
    }

    if (!isStreamableMediaPath(target.filePath)) {
      return new Response('Unsupported media type', { status: 415 })
    }

    const headers = mediaRequestHeaders(request.headers)

    if (target.mode === 'stream') {
      try {
        const resolvedPath = await dependencies.resolveLocalFile(target.filePath)

        if (!isStreamableMediaPath(resolvedPath)) {
          return new Response('Unsupported media type', { status: 415 })
        }

        return await dependencies.fetchLocal(resolvedPath, headers, method)
      } catch {
        return new Response('Media not found', { status: 404 })
      }
    }

    try {
      const connection = await dependencies.resolveRemoteConnection({
        connectionId: target.connectionId,
        profile: target.profile
      })

      if (connection.mode !== 'remote') {
        return new Response('Remote media backend unavailable', { status: 404 })
      }

      const endpoint = remoteMediaEndpoint(
        connection.baseUrl,
        target.filePath,
        connection.sharedRemote ? target.profile : undefined
      )

      // The gateway's configured extra headers (access-proxy gates) travel on
      // every remote request; forwarded range/cache negotiation headers win.
      for (const [name, value] of Object.entries(connection.headers ?? {})) {
        if (!headers.has(name)) {
          headers.set(name, value)
        }
      }

      const { token } = connection

      if (connection.authMode !== 'oauth' && !token) {
        return new Response('Remote media authentication unavailable', { status: 401 })
      }

      const fetchUpstream = (): Promise<Response> => {
        if (connection.authMode !== 'oauth') {
          headers.set('x-hermes-session-token', token as string)

          return dependencies.fetchRemote(endpoint, headers, method)
        }

        return requestWithOauthFallback(connection.baseUrl, {
          ensureNativeAccessToken: dependencies.ensureRemoteBearer,
          requestWithBearer: bearer => {
            headers.set('authorization', `Bearer ${bearer}`)

            return dependencies.fetchRemote(endpoint, headers, method)
          },
          requestWithCookie: async () => {
            const response = await dependencies.fetchRemoteWithCookies(endpoint, headers, method)

            // Fetch resolves HTTP errors; translate only the auth verdict so
            // the shared fallback can preserve a failed native refresh.
            if (response.status === 401 || response.status === 403) {
              await response.body?.cancel()
              throw httpStatusError(response.status, 'Remote media authentication unavailable')
            }

            return response
          }
        })
      }

      if (method === 'HEAD') {
        return await fetchUpstream()
      }

      return await remoteMediaInChunks(headers.get('range'), range => {
        headers.set('range', range)

        return fetchUpstream()
      })
    } catch (error) {
      const status = readStatusCode(error)

      return new Response('Remote media unavailable', { status: status === 401 || status === 403 ? status : 502 })
    }
  }
}
