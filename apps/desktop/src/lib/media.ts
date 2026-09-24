import { LOCAL_CONNECTION_ID } from '@hermes/shared'

import { capabilityScoped, hermesApi, type OwnerScope } from '@/api/client'
import type { HermesConnection } from '@/global'
import { translateNow } from '@/i18n'
import { desktopFsCacheKey, readDesktopFileDataUrl } from '@/lib/desktop-fs'
import { LruCache } from '@/lib/lru-cache'
import { capitalize } from '@/lib/text'
import { notify, notifyError } from '@/store/notifications'
import { $connection } from '@/store/session'

export type MediaKind = 'audio' | 'image' | 'video' | 'file'

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
  // `C:\Users\…` parses as a URL with scheme `c:`; a drive letter is a path, not a scheme.
  if (!/^[A-Za-z]:[\\/]/.test(path)) {
    try {
      return new URL(path).pathname.split('/').filter(Boolean).pop() || path
    } catch {
      // not a URL — fall through to the path split
    }
  }

  return path.split(/[\\/]/).filter(Boolean).pop() || path
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

export async function resolveMediaDisplaySrc(path: string, owner?: OwnerScope): Promise<string> {
  if (isInlineMediaSrc(path) || !isFileMediaPath(path)) {
    return path
  }

  // An explicit local owner is this device, even with a remote foreground.
  // Keep the native reader and its configured size cap; the backend preview
  // endpoint has a separate fixed limit.
  if (owner?.connectionId === LOCAL_CONNECTION_ID && window.hermesDesktop?.readFileDataUrl) {
    return window.hermesDesktop.readFileDataUrl(filePathFromMediaPath(path))
  }

  // A tile can belong to a different gateway than the foreground. Pin both
  // halves at read admission rather than resolving them when the read settles.
  if (window.hermesDesktop && (owner?.connectionId || owner?.profile)) {
    const result = await hermesApi<string | { dataUrl?: string }>({
      path: `/api/fs/read-data-url?path=${encodeURIComponent(filePathFromMediaPath(path))}`,
      ...(owner.connectionId ? { connectionId: owner.connectionId } : {}),
      ...(owner.profile ? { profile: owner.profile } : {})
    })

    return typeof result === 'string' ? result : result.dataUrl || ''
  }

  if (window.hermesDesktop && isRemoteGateway()) {
    return gatewayMediaDataUrl(path)
  }

  if (!window.hermesDesktop?.readFileDataUrl) {
    return mediaExternalUrl(path)
  }

  return window.hermesDesktop.readFileDataUrl(filePathFromMediaPath(path))
}

export interface MediaImageDimensions {
  width: number
  height: number
}

// Decoded geometry only: never retain image bytes/data URLs for every visited
// transcript. A miss costs a letterboxed first display, not a collapsed row.
// 'broken' remembers a failed load so a remount starts as a text line instead
// of reserving a frame that collapses again.
const imageDimensions = new LruCache<string, 'broken' | MediaImageDimensions>(512)

export function mediaImageKey(path: string, connection: HermesConnection | null, owner?: OwnerScope): string {
  // File reads ignore URL query/fragment, but callers can use them to identify
  // a new revision. Keep them in the geometry key while joining proven aliases.
  const revision = /^file:/i.test(path) ? (path.match(/[?#].*$/)?.[0] ?? '') : ''

  return JSON.stringify([
    owner?.connectionId ||
      connection?.connectionId ||
      desktopFsCacheKey(connection && { ...connection, profile: owner?.profile ?? connection.profile }),
    owner?.profile ?? connection?.profile ?? '',
    mediaSourceIdentity(path),
    revision
  ])
}

// Inline sources are the bytes themselves (often megabytes) and this key is
// rebuilt every render. Length plus head (format header, usually the size)
// and tail keeps it small and cacheable; a collision can only borrow another
// image's frame shape, never its pixels.
function mediaSourceIdentity(path: string): string {
  return /^data:/i.test(path) && path.length > 512
    ? `data#${path.length}:${path.slice(0, 160)}:${path.slice(-96)}`
    : filePathFromMediaPath(path)
}

export function validImageDimensions(width: unknown, height: unknown): MediaImageDimensions | undefined {
  const w = Number(width)
  const h = Number(height)

  return Number.isFinite(w) && w > 0 && Number.isFinite(h) && h > 0 ? { width: w, height: h } : undefined
}

export function getMediaImageDimensions(key: string): MediaImageDimensions | undefined {
  const entry = imageDimensions.get(key)

  return entry === 'broken' ? undefined : entry
}

export function isKnownBrokenMediaImage(key: string): boolean {
  return imageDimensions.get(key) === 'broken'
}

// A pathological URL can still make a huge key; keep this metadata cache from
// becoming a second copy of it.
function rememberMediaImage(key: string, entry: 'broken' | MediaImageDimensions): void {
  if (key.length <= 4096) {
    imageDimensions.set(key, entry)
  }
}

export function rememberMediaImageDimensions(key: string, width: number, height: number): void {
  const dimensions = validImageDimensions(width, height)

  if (dimensions) {
    rememberMediaImage(key, dimensions)
  }
}

export function rememberMediaImageFailure(key: string): void {
  rememberMediaImage(key, 'broken')
}

// Audio/video need a seekable source instead of a whole-file data URL. Keep
// remote URLs untouched and route filesystem paths through the Electron media
// protocol. Its main-process handler reads local files directly or proxies a
// remote gateway with the connection's bearer/cookie/token authentication.
export async function resolveMediaPlaybackSrc(path: string): Promise<string> {
  if (isInlineMediaSrc(path)) {
    return path
  }

  if (window.hermesDesktop && ['audio', 'video'].includes(mediaKind(path))) {
    return isRemoteGateway() ? mediaGatewayStreamUrl(path) : mediaStreamUrl(path)
  }

  return resolveMediaDisplaySrc(path)
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
// out of renderer URLs while forwarding Range requests to /api/files/stream.
export function mediaGatewayStreamUrl(path: string): string {
  const conn = $connection.get()

  if (isRemoteGateway()) {
    const file = encodeURIComponent(filePathFromMediaPath(path))

    const scope = [
      conn?.connectionId ? `connectionId=${encodeURIComponent(conn.connectionId)}` : '',
      conn?.profile ? `profile=${encodeURIComponent(conn.profile)}` : ''
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
export async function gatewayMediaDataUrl(path: string): Promise<string> {
  return readDesktopFileDataUrl(filePathFromMediaPath(path))
}

export interface GatewayFileSaveResult {
  canceled?: boolean
  path?: string
  saved: boolean
}

export interface GatewayFileOrigin {
  /** Backend that owns `path`, fixed when the path was read (see
   *  `captureGatewayFileDownload`). It replaces the window's active connection
   *  outright: an absent `connectionId` keeps legacy profile-pool routing, and
   *  an explicit `local` stays pinned when the registry primary is remote. */
  owner?: OwnerScope
  /** Profile of the session that produced the file (artifacts). */
  profile?: string
  sessionId?: string
  suggestedName?: string
}

// Replacement for opening gateway-local file paths with file://. The file lives
// on the gateway, so ask the Electron main process to fetch the bytes through
// the authenticated backend connection and save them locally. This avoids
// browser/OS downloads losing OAuth cookies and avoids the data-URL cap used by
// preview endpoints. Every renderer gateway-file save resolves its backend here.
export async function downloadGatewayMediaFile(
  path: string,
  origin: GatewayFileOrigin = {}
): Promise<GatewayFileSaveResult> {
  // URI conversion belongs to the gateway OS, not the renderer's URL parser.
  if (!path?.trim()) {
    throw new Error('Missing gateway file path')
  }

  if (!window.hermesDesktop?.saveGatewayFile) {
    throw new Error('Desktop file download bridge is unavailable')
  }

  const conn = $connection.get()
  const owner = origin.owner ?? { connectionId: conn?.connectionId, profile: origin.profile ?? conn?.profile }

  return window.hermesDesktop.saveGatewayFile({
    ...(owner.connectionId ? { connectionId: owner.connectionId } : {}),
    path,
    ...(owner.profile ? { profile: owner.profile } : {}),
    ...(origin.sessionId ? { sessionId: origin.sessionId } : {}),
    suggestedName:
      origin.suggestedName ||
      mediaName(path).replace(/(?:%[0-9a-f]{2})+/gi, encoded => {
        try {
          return decodeURIComponent(encoded)
        } catch {
          return encoded
        }
      })
  })
}

/** A user-initiated gateway file save with the shared feedback: a brief
 *  "Saved" toast, a "Download failed" toast carrying the reason, and silence
 *  when the user cancels the save dialog. Never rejects. */
export async function downloadGatewayFileWithFeedback(path: string, origin?: GatewayFileOrigin): Promise<void> {
  try {
    const result = await downloadGatewayMediaFile(path, origin)

    if (result.saved) {
      notify({ durationMs: 1500, kind: 'info', message: translateNow('fileMenu.downloadSaved') })
    }
  } catch (error) {
    notifyError(error, translateNow('fileMenu.downloadFailed'))
  }
}

/** Capture a gateway file download alongside a REST read, never at click time:
 *  a cached resource belongs to the connection/profile that returned it, not
 *  to whichever backend is focused when the user clicks. The path stays on the
 *  gateway; only Electron's authenticated save bridge sees it. */
export function captureGatewayFileDownload() {
  const { connectionId, profile } = capabilityScoped()

  return (path: string, suggestedName: string) =>
    downloadGatewayFileWithFeedback(path, { owner: { connectionId, profile }, suggestedName })
}

export function mediaDisplayLabel(path: string): string {
  const escaped = mediaName(path).replace(/[[\]\\]/g, '\\$&')
  const kind = mediaKind(path)

  return `${capitalize(kind)}: ${escaped}`
}
