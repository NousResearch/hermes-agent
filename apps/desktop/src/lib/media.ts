import { readDesktopFileDataUrl } from '@/lib/desktop-fs'
import { capitalize } from '@/lib/text'
import { $connectionsRegistry } from '@/store/connection-registry-state'
import { $connection } from '@/store/session'

export type MediaKind = 'audio' | 'image' | 'video' | 'file'

export interface GatewayMediaOrigin {
  connectionId?: string
  mode?: 'local' | 'remote'
  profile?: string
  sessionId?: string
  targetProfile?: string
}

export function gatewayMediaOriginIsRemote(origin?: GatewayMediaOrigin): boolean {
  if (origin?.mode) {
    return origin.mode === 'remote'
  }

  const connection = origin?.connectionId
    ? $connectionsRegistry.get()?.connections.find(candidate => candidate.id === origin.connectionId)
    : undefined

  if (connection) {
    return connection.kind !== 'local'
  }

  const foreground = $connection.get()

  if (origin?.connectionId && origin.connectionId !== foreground?.connectionId) {
    return true
  }

  return foreground?.mode === 'remote'
}

interface MediaInfo {
  kind: MediaKind
  mime: string
}

const MEDIA_BY_EXT: Record<string, MediaInfo> = {
  avi: { kind: 'video', mime: 'video/x-msvideo' },
  bmp: { kind: 'image', mime: 'image/bmp' },
  flac: { kind: 'audio', mime: 'audio/flac' },
  gif: { kind: 'image', mime: 'image/gif' },
  jpeg: { kind: 'image', mime: 'image/jpeg' },
  jpg: { kind: 'image', mime: 'image/jpeg' },
  m4a: { kind: 'audio', mime: 'audio/mp4' },
  mkv: { kind: 'video', mime: 'video/x-matroska' },
  mov: { kind: 'video', mime: 'video/quicktime' },
  mp3: { kind: 'audio', mime: 'audio/mpeg' },
  mp4: { kind: 'video', mime: 'video/mp4' },
  ogg: { kind: 'audio', mime: 'audio/ogg' },
  opus: { kind: 'audio', mime: 'audio/ogg; codecs=opus' },
  png: { kind: 'image', mime: 'image/png' },
  svg: { kind: 'image', mime: 'image/svg+xml' },
  wav: { kind: 'audio', mime: 'audio/wav' },
  webm: { kind: 'video', mime: 'video/webm' },
  webp: { kind: 'image', mime: 'image/webp' }
}

function mediaInfo(path: string): MediaInfo | undefined {
  const ext = path.split(/[?#]/, 1)[0]?.split('.').pop()?.toLowerCase()

  return ext ? MEDIA_BY_EXT[ext] : undefined
}

export function mediaKind(path: string): MediaKind {
  return mediaInfo(path)?.kind ?? 'file'
}

// Markdown is renderable content, not an opaque download: the preview rail
// already knows how to render a `.md` file (rendered/source toggle), so the
// MEDIA delivery path routes these to a preview instead of a download link.
const MARKDOWN_EXTENSIONS = new Set(['md', 'markdown', 'mdown', 'mkd'])

export function isMarkdownDocumentPath(path: string): boolean {
  const ext = path.split(/[?#]/, 1)[0]?.split('.').pop()?.toLowerCase()

  return ext ? MARKDOWN_EXTENSIONS.has(ext) : false
}

export function mediaMime(path: string): string {
  return mediaInfo(path)?.mime ?? 'application/octet-stream'
}

export function mediaName(path: string): string {
  try {
    const url = new URL(path)

    return url.pathname.split('/').filter(Boolean).pop() || path
  } catch {
    return path.split(/[\\/]/).filter(Boolean).pop() || path
  }
}

export function mediaMarkdownHref(path: string): string {
  return `#media:${encodeURIComponent(path)}`
}

export function isInlineMediaSrc(path: string): boolean {
  return /^(?:https?|data):/i.test(path)
}

export function isArtifactFilePath(path: string): boolean {
  return /^(?:file:|\/|[~.][\\/]|\.\.[\\/]|[a-z]:[\\/]|\\\\)/i.test(path)
}

export function isFileMediaPath(path: string): boolean {
  return /^(?:file:|\/|~\/|[a-z]:[\\/]|\\\\)/i.test(path)
}

export async function resolveMediaDisplaySrc(path: string, origin?: GatewayMediaOrigin): Promise<string> {
  if (isInlineMediaSrc(path) || !isFileMediaPath(path)) {
    return path
  }

  if (window.hermesDesktop && gatewayMediaOriginIsRemote(origin)) {
    return gatewayMediaDataUrl(path, origin)
  }

  if (!window.hermesDesktop?.readFileDataUrl) {
    return mediaExternalUrl(path)
  }

  return window.hermesDesktop.readFileDataUrl(filePathFromMediaPath(path))
}

// Audio/video need a seekable source instead of a whole-file data URL. Keep
// remote URLs untouched and route filesystem paths through the Electron media
// protocol. Its main-process handler reads local files directly or proxies a
// remote gateway with the connection's bearer/cookie/token authentication.
export async function resolveMediaPlaybackSrc(path: string, origin?: GatewayMediaOrigin): Promise<string> {
  if (isInlineMediaSrc(path)) {
    return path
  }

  if (window.hermesDesktop && ['audio', 'video'].includes(mediaKind(path))) {
    const remote = gatewayMediaOriginIsRemote(origin)

    return remote ? mediaGatewayStreamUrl(path, origin) : mediaStreamUrl(path)
  }

  return resolveMediaDisplaySrc(path, origin)
}

// Resolve a media path to a URL the shell can open. Remote mode rewrites
// gateway-local paths to an authenticated /api/files/download URL (the file
// lives on the gateway, not this disk); local mode keeps the file:// form.
export function mediaExternalUrl(path: string): string {
  if (/^https?:/i.test(path)) {
    return path
  }

  if (isRemoteGateway()) {
    const conn = $connection.get()

    if (conn?.baseUrl && conn.token) {
      const file = encodeURIComponent(filePathFromMediaPath(path))

      return `${conn.baseUrl}/api/files/download?path=${file}&token=${encodeURIComponent(conn.token)}`
    }
  }

  return /^file:/i.test(path) ? path : `file://${path}`
}

// Remote gateway audio/video is proxied by the Electron main process. OAuth
// connections intentionally expose no static token to the renderer, so a bare
// HTTPS source cannot authenticate reliably. The custom protocol keeps secrets
// out of renderer URLs while forwarding Range requests to /api/fs/stream.
export function mediaGatewayStreamUrl(path: string, origin?: GatewayMediaOrigin): string {
  const conn = $connection.get()

  if (gatewayMediaOriginIsRemote(origin)) {
    const file = encodeURIComponent(filePathFromMediaPath(path))

    const scope = [
      origin?.connectionId || conn?.connectionId
        ? `connectionId=${encodeURIComponent(origin?.connectionId || conn!.connectionId!)}`
        : '',
      origin?.profile || conn?.profile ? `profile=${encodeURIComponent(origin?.profile || conn!.profile!)}` : '',
      origin?.targetProfile ? `targetProfile=${encodeURIComponent(origin.targetProfile)}` : '',
      origin?.sessionId ? `sessionId=${encodeURIComponent(origin.sessionId)}` : ''
    ]
      .filter(Boolean)
      .join('&')

    return `hermes-media://remote/${file}${scope ? `?${scope}` : ''}`
  }

  return mediaExternalUrl(path)
}

// Custom Electron scheme (registered in electron/main.ts) that streams a local
// file with Range support. Used for audio/video so playback bypasses the data
// URL size cap and supports seeking. `path` may be a plain path or `file://…`.
export function mediaStreamUrl(path: string): string {
  return `hermes-media://stream/${encodeURIComponent(filePathFromMediaPath(path))}`
}

export function mediaPathFromMarkdownHref(href?: string): string | null {
  if (!href?.startsWith('#media:')) {
    return null
  }

  try {
    return decodeURIComponent(href.slice('#media:'.length))
  } catch {
    return null
  }
}

export function filePathFromMediaPath(path: string): string {
  if (!path.startsWith('file:')) {
    return path
  }

  try {
    return decodeURIComponent(new URL(path).pathname)
  } catch {
    return path.replace(/^file:\/\//, '')
  }
}

// True when this desktop shell is wired to a remote gateway. Local media paths
// then live on the gateway machine, not this disk, so we fetch them over the API.
export function isRemoteGateway(): boolean {
  return $connection.get()?.mode === 'remote'
}

// Fetch gateway-local media as a data URL via the authenticated desktop FS
// bridge. Remote Desktop artifacts can live anywhere the gateway can read
// (workspace, skills, ~/.hermes/cache, etc.); /api/media is intentionally
// narrower and rejects non-images plus images outside its media roots.
export async function gatewayMediaDataUrl(path: string, origin?: GatewayMediaOrigin): Promise<string> {
  if (origin && window.hermesDesktop?.api) {
    const connection = $connection.get()
    const params = new URLSearchParams({ path: filePathFromMediaPath(path) })

    if (origin.targetProfile) {
      params.set('profile', origin.targetProfile)
    }

    if (origin.sessionId) {
      params.set('session_id', origin.sessionId)
    }

    const result = await window.hermesDesktop.api<string | { dataUrl?: string }>({
      connectionId: origin.connectionId ?? connection?.connectionId,
      path: `/api/fs/read-data-url?${params.toString()}`,
      profile: origin.profile ?? connection?.profile
    })

    return typeof result === 'string' ? result : result.dataUrl || ''
  }

  return readDesktopFileDataUrl(filePathFromMediaPath(path))
}

// Remote-mode replacement for opening gateway-local file paths with file://.
// The file lives on the gateway, so ask the Electron main process to fetch the
// bytes through the authenticated backend connection and save them locally. This
// avoids browser/OS downloads losing OAuth cookies and avoids the data-URL cap
// used by preview endpoints.
export async function downloadGatewayMediaFile(
  path: string,
  origin?: GatewayMediaOrigin
): Promise<{ canceled?: boolean; path?: string; saved: boolean }> {
  // URI conversion belongs to the gateway OS, not the renderer's URL parser.
  const file = path
  const conn = $connection.get()

  if (!window.hermesDesktop?.saveGatewayFile) {
    throw new Error('Desktop file download bridge is unavailable')
  }

  return window.hermesDesktop.saveGatewayFile({
    connectionId: origin?.connectionId ?? conn?.connectionId,
    path: file,
    profile: origin?.profile ?? conn?.profile,
    ...(origin?.targetProfile ? { targetProfile: origin.targetProfile } : {}),
    ...(origin?.sessionId ? { sessionId: origin.sessionId } : {}),
    suggestedName: mediaName(file).replace(/(?:%[0-9a-f]{2})+/gi, encoded => {
      try {
        return decodeURIComponent(encoded)
      } catch {
        return encoded
      }
    })
  })
}

export function mediaDisplayLabel(path: string): string {
  const escaped = mediaName(path).replace(/[[\]\\]/g, '\\$&')
  const kind = mediaKind(path)

  return `${capitalize(kind)}: ${escaped}`
}
