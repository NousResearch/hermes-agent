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

type MediaProtocolMode = 'remote' | 'stream'

interface MediaProtocolTarget {
  connectionId?: string
  filePath: string
  mode: MediaProtocolMode
  profile?: string
  sessionId?: string
  targetProfile?: string
}

export interface MediaRemoteScope {
  connectionId?: string
  profile?: string
}

export interface MediaRemoteConnection {
  authMode?: 'oauth' | 'token'
  baseUrl: string
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
  const sessionId = url.searchParams.get('sessionId')?.trim() || undefined
  const targetProfile = url.searchParams.get('targetProfile')?.trim() || undefined

  return { connectionId, filePath, mode, profile, sessionId, targetProfile }
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

export function remoteMediaEndpoint(baseUrl: string, filePath: string, profile?: string, sessionId?: string): string {
  const normalizedBase = baseUrl.replace(/\/+$/, '')
  const url = new URL(`${normalizedBase}/api/fs/stream`)

  if (url.protocol !== 'http:' && url.protocol !== 'https:') {
    throw new Error(`Unsupported Hermes backend URL protocol: ${url.protocol}`)
  }

  url.searchParams.set('path', filePath)

  if (profile) {
    url.searchParams.set('profile', profile)
  }

  if (sessionId) {
    url.searchParams.set('session_id', sessionId)
  }

  return url.toString()
}

export function legacyRemoteMediaEndpoint(baseUrl: string, filePath: string, profile?: string): string {
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

async function isMissingStreamRoute(response: Response, method: MediaRequestMethod): Promise<boolean> {
  if (method !== 'GET' || response.status !== 404) {
    return false
  }

  try {
    const body = (await response.clone().json()) as { detail?: unknown }

    return body.detail === 'Not Found'
  } catch {
    return false
  }
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
        connection.sharedRemote ? target.targetProfile || target.profile : undefined,
        target.sessionId
      )

      let fetchAuthenticated: (url: string, requestMethod?: MediaRequestMethod, requestHeaders?: Headers) => Promise<Response>

      if (connection.authMode === 'oauth') {
        const bearer = await dependencies.ensureRemoteBearer(connection.baseUrl)

        if (bearer) {
          headers.set('authorization', `Bearer ${bearer}`)
          fetchAuthenticated = (url, requestMethod = method, requestHeaders = headers) =>
            dependencies.fetchRemote(url, requestHeaders, requestMethod)
        } else {
          fetchAuthenticated = (url, requestMethod = method, requestHeaders = headers) =>
            dependencies.fetchRemoteWithCookies(url, requestHeaders, requestMethod)
        }
      } else {
        if (!connection.token) {
          return new Response('Remote media authentication unavailable', { status: 401 })
        }

        headers.set('x-hermes-session-token', connection.token)
        fetchAuthenticated = (url, requestMethod = method, requestHeaders = headers) =>
          dependencies.fetchRemote(url, requestHeaders, requestMethod)
      }

      const response = await fetchAuthenticated(endpoint)
      let missingRoute = await isMissingStreamRoute(response, method)

      if (!missingRoute && method === 'HEAD' && response.status === 404) {
        // HEAD bodies cannot distinguish a missing route from a missing file.
        // Probe the same authenticated endpoint with a one-byte GET; only the
        // canonical FastAPI route-missing body unlocks the legacy fallback.
        const probeHeaders = new Headers(headers)

        probeHeaders.set('range', 'bytes=0-0')
        missingRoute = await isMissingStreamRoute(
          await fetchAuthenticated(endpoint, 'GET', probeHeaders),
          'GET'
        )
      }

      if (!missingRoute) {
        return response
      }

      // Desktop and remote gateways update independently. Retry only a
      // confirmed FastAPI missing-route response against the older
      // managed-root endpoint. File/session 404s must keep failing closed.
      return await fetchAuthenticated(
        legacyRemoteMediaEndpoint(
          connection.baseUrl,
          target.filePath,
          connection.sharedRemote ? target.targetProfile || target.profile : undefined
        )
      )
    } catch {
      return new Response('Remote media unavailable', { status: 502 })
    }
  }
}
