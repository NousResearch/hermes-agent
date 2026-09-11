import { pathWithProfileScope, translateSelfProfileQuery } from './connection-config'

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

type MediaProtocolMode = 'plugin' | 'remote' | 'stream'

interface MediaProtocolTarget {
  connectionId?: string
  filePath: string
  mode: MediaProtocolMode
  pluginId?: string
  profile?: string
}

export interface MediaRemoteScope {
  connectionId?: string
  profile?: string
}

export interface MediaRemoteConnection {
  authMode?: 'oauth' | 'token'
  baseUrl: string
  mode?: 'local' | 'remote'
  remoteProfile?: null | string
  sharedPrimary?: boolean
  sharedRemote?: boolean
  token?: null | string
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

const PLUGIN_MEDIA_QUERY_PARAMS = new Set(['connectionId', 'profile'])

// Keep these segment rules in sync with the defense-in-depth URL minting check
// in apps/desktop/src/api/plugins.ts::assertSafePluginMediaSegment.
function assertSafePluginMediaSegment(segment: string, rawUrl: string): void {
  let inspected = segment

  while (true) {
    if (!inspected || inspected === '.' || inspected === '..' || inspected.includes('/') || inspected.includes('\\')) {
      throw new Error(`Unsafe plugin media URL: ${rawUrl}`)
    }

    // A raw `%252e` becomes `%2e` after the first decode. Inspect nested
    // escapes solely for path syntax so a later URL/server decode cannot turn
    // an accepted segment into traversal. A literal malformed percent is safe
    // once the original URL segment itself decoded successfully.
    if (!/%[0-9a-f]{2}/i.test(inspected)) {
      return
    }

    try {
      inspected = decodeURIComponent(inspected)
    } catch {
      return
    }
  }
}

function decodePluginMediaSegment(rawSegment: string, rawUrl: string): string {
  let segment: string

  try {
    segment = decodeURIComponent(rawSegment)
  } catch {
    throw new Error(`Malformed plugin media URL: ${rawUrl}`)
  }

  assertSafePluginMediaSegment(segment, rawUrl)

  return segment
}

function parsePluginMediaTarget(rawUrl: string, url: URL): MediaProtocolTarget {
  const prefix = `${MEDIA_PROTOCOL}://plugin/`

  if (!rawUrl.startsWith(prefix) || url.hash) {
    throw new Error(`Malformed plugin media URL: ${rawUrl}`)
  }

  const rawPath = rawUrl.slice(prefix.length).split(/[?#]/, 1)[0]
  const [rawPluginId, ...rawPluginPath] = rawPath.split('/')

  if (!rawPluginId || rawPluginPath.length === 0) {
    throw new Error(`Missing plugin media path: ${rawUrl}`)
  }

  const pluginId = decodePluginMediaSegment(rawPluginId, rawUrl)
  const pluginPath = rawPluginPath.map(segment => decodePluginMediaSegment(segment, rawUrl))
  const scope = new Map<string, string>()

  for (const [key, rawValue] of url.searchParams) {
    const value = rawValue.trim()

    if (!PLUGIN_MEDIA_QUERY_PARAMS.has(key) || !value || value !== rawValue || scope.has(key)) {
      throw new Error(`Unsafe plugin media URL: ${rawUrl}`)
    }

    scope.set(key, value)
  }

  return {
    connectionId: scope.get('connectionId'),
    filePath: pluginPath.join('/'),
    mode: 'plugin',
    pluginId,
    profile: scope.get('profile')
  }
}

function parseMediaProtocolTarget(rawUrl: string): MediaProtocolTarget {
  const url = new URL(rawUrl)
  const mode = url.hostname as MediaProtocolMode

  if (url.protocol !== `${MEDIA_PROTOCOL}:` || (mode !== 'plugin' && mode !== 'remote' && mode !== 'stream')) {
    throw new Error('Unsupported media protocol target')
  }

  if (mode === 'plugin') {
    return parsePluginMediaTarget(rawUrl, url)
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

export function pluginMediaEndpoint(baseUrl: string, pluginId: string, filePath: string, profile?: string): string {
  const normalizedBase = baseUrl.replace(/\/+$/, '')
  const pluginPath = filePath.split('/').map(encodeURIComponent).join('/')
  const url = new URL(`${normalizedBase}/api/plugins/${encodeURIComponent(pluginId)}/${pluginPath}`)

  if (url.protocol !== 'http:' && url.protocol !== 'https:') {
    throw new Error(`Unsupported Hermes backend URL protocol: ${url.protocol}`)
  }

  if (profile) {
    url.searchParams.set('profile', profile)
  }

  return url.toString()
}

function profileScopedMediaEndpoint(endpoint: string, connection: MediaRemoteConnection, profile?: string): string {
  const url = new URL(endpoint)
  const requestPath = `${url.pathname}${url.search}`

  const scopedPath =
    connection.sharedPrimary || connection.sharedRemote
      ? pathWithProfileScope(requestPath, profile)
      : translateSelfProfileQuery(requestPath, profile, connection.remoteProfile)

  if (scopedPath === requestPath) {
    return endpoint
  }

  const scoped = new URL(scopedPath, url.origin)

  url.pathname = scoped.pathname
  url.search = scoped.search

  return url.toString()
}

function validatePluginMediaResponse(target: MediaProtocolTarget, response: Response): Response {
  if (target.mode !== 'plugin' || !response.ok) {
    return response
  }

  const contentType = response.headers.get('content-type')?.split(';', 1)[0]?.trim().toLowerCase()

  if (contentType?.startsWith('audio/') || contentType?.startsWith('video/')) {
    return response
  }

  void response.body?.cancel().catch(() => undefined)

  return new Response('Unsupported media type', { status: 415 })
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

    if (target.mode !== 'plugin' && !isStreamableMediaPath(target.filePath)) {
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

      if (target.mode === 'remote' && connection.mode !== 'remote') {
        return new Response('Remote media backend unavailable', { status: 404 })
      }

      const endpoint = profileScopedMediaEndpoint(
        target.mode === 'plugin'
          ? pluginMediaEndpoint(connection.baseUrl, target.pluginId!, target.filePath)
          : remoteMediaEndpoint(connection.baseUrl, target.filePath),
        connection,
        target.profile
      )

      if (connection.authMode === 'oauth') {
        const bearer = await dependencies.ensureRemoteBearer(connection.baseUrl)

        if (bearer) {
          headers.set('authorization', `Bearer ${bearer}`)

          return validatePluginMediaResponse(target, await dependencies.fetchRemote(endpoint, headers, method))
        }

        return validatePluginMediaResponse(target, await dependencies.fetchRemoteWithCookies(endpoint, headers, method))
      }

      if (!connection.token) {
        return new Response('Remote media authentication unavailable', { status: 401 })
      }

      headers.set('x-hermes-session-token', connection.token)

      return validatePluginMediaResponse(target, await dependencies.fetchRemote(endpoint, headers, method))
    } catch {
      return new Response('Remote media unavailable', { status: 502 })
    }
  }
}
