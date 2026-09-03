import { readDesktopFileDataUrl } from '@/lib/desktop-fs'
import type { MediaDeliverableMeta } from '@/lib/media-store'
import { capitalize } from '@/lib/text'
import { $connection } from '@/store/session'

export type MediaKind = 'audio' | 'image' | 'video' | 'file'

export type MediaFailureReason =
  /** Fetch cancelled before completion. */
  | 'cancelled'
  /** Gateway policy (media.roots) or filesystem permissions denied the read. */
  | 'denied'
  /** Path missing, or no longer a regular file on the gateway. */
  | 'enotdir'
  /** Reader/renderer could not decode or fetch the file. */
  | 'error'
  /** Gateway answered with a non-mapped HTTP error. */
  | 'http'
  /** File exceeds the inline preview size cap. */
  | 'too-large'
  /** No inline rendering exists for this file class (the `file` kind). */
  | 'unsupported'

/**
 * Why a media ref could not be rendered. Thrown (not returned) by the media
 * resolvers so the renderer's single catch path maps it onto the fallback
 * card; `statusCode` rides along when the gateway answered with an HTTP error.
 */
export type MediaFailure = {
  reason: MediaFailureReason
  statusCode?: number
}

/**
 * Shape of the Electron bridge's fetch errors: the main process attaches the
 * HTTP status when a gateway media endpoint answers 4xx/5xx (see
 * finalizeGatewayDownload). Renderer failure mapping keys on it.
 */
interface MediaBridgeError extends Error {
  statusCode?: number
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

/**
 * Kind resolution with event metadata: the structured payload is authoritative
 * (it knows the true class — e.g. `.oga` audio) where the extension table can
 * only guess. Falls back to the extension table for refs without an event.
 */
export function mediaKindWithMeta(path: string, meta?: MediaDeliverableMeta | null): MediaKind {
  return meta?.kind ?? mediaKind(path)
}

/** Mime resolution with event metadata — same precedence as the kind. */
export function mediaMimeWithMeta(path: string, meta?: MediaDeliverableMeta | null): string {
  return meta?.mime ?? mediaMime(path)
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

export function isFileMediaPath(path: string): boolean {
  return /^(?:file:|\/|~\/|[a-z]:[\\/]|\\\\)/i.test(path)
}

export async function resolveMediaDisplaySrc(path: string): Promise<string> {
  if (isInlineMediaSrc(path) || !isFileMediaPath(path)) {
    return path
  }

  if (window.hermesDesktop && isRemoteGateway()) {
    return readDesktopFileDataUrlChecked(filePathFromMediaPath(path))
  }

  if (!window.hermesDesktop?.readFileDataUrl) {
    return mediaExternalUrl(path)
  }

  return readDesktopFileDataUrlChecked(filePathFromMediaPath(path))
}

// Audio/video need a seekable source instead of a whole-file data URL. Keep
// remote URLs untouched and route filesystem paths through the Electron media
// protocol. Its main-process handler reads local files directly or proxies a
// remote gateway with the connection's bearer/cookie/token authentication.
// MediaFailure-tagged like the display path; the stream URL itself is built
// optimistically (the <audio>/<video> element reports load failures via
// onError, which the card maps onto the same fallback card).
export async function resolveMediaPlaybackSrc(path: string): Promise<string> {
  if (isInlineMediaSrc(path)) {
    return path
  }

  if (window.hermesDesktop && ['audio', 'video'].includes(mediaKind(path))) {
    return isRemoteGateway() ? mediaGatewayStreamUrl(path) : mediaStreamUrl(path)
  }

  return resolveMediaDisplaySrc(path)
}

// ── Never-silent helpers ────────────────────────────────────────────────────
//
// A deliverable's metadata rides with the ref, so a card can render (or fall
// back) even when the file itself is unreachable. Zero-silent means: the label
// carries the size when the gateway reported one, and every resolve failure
// rejects with a *tagged* MediaFailure the renderer maps onto the fallback
// card — never a bare string, never an empty catch.

const SIZE_UNITS = ['B', 'KB', 'MB', 'GB', 'TB'] as const

/** Human byte size, one decimal below 10 (media-card convention). */
export function formatMediaSize(size: number | undefined): null | string {
  if (size === undefined || !Number.isFinite(size) || size < 0) {
    return null
  }

  let value = size
  let unit = 0

  while (value >= 1000 && unit < SIZE_UNITS.length - 1) {
    value /= 1000
    unit += 1
  }

  const rounded = unit === 0 ? Math.round(value).toString() : value >= 10 ? value.toFixed(0) : value.toFixed(1)

  return `${rounded} ${SIZE_UNITS[unit]}`
}

/** Display label with event metadata: `Image · 1.2 MB: name.png`. */
export function mediaDisplayLabel(path: string, meta?: MediaDeliverableMeta | null): string {
  const escaped = mediaName(path).replace(/[[\]\\]/g, '\\$&')
  const kind = mediaKindWithMeta(path, meta)
  const size = formatMediaSize(meta?.size)
  const kindLabel = capitalize(kind)

  return size ? `${kindLabel} · ${size}: ${escaped}` : `${kindLabel}: ${escaped}`
}

/** HTTP status → MediaFailure for the /api/fs/* remote fetch paths. */
function remoteDataUrlFailure(status: number): MediaFailure {
  if (status === 403 || status === 401) {
    return { reason: 'denied', statusCode: status }
  }

  if (status === 404) {
    return { reason: 'enotdir', statusCode: status }
  }

  if (status === 413) {
    return { reason: 'too-large', statusCode: status }
  }

  return { reason: 'http', statusCode: status }
}

function bridgeFailure(error: unknown): MediaFailure {
  const message = error instanceof Error ? error.message : String(error)

  if (/cancel/i.test(message)) {
    return { reason: 'cancelled' }
  }

  if (/\b(413|too large)\b/i.test(message)) {
    return { reason: 'too-large' }
  }

  if (/\b(403|401|permission|denied)\b/i.test(message)) {
    return { reason: 'denied' }
  }

  if (/\b(404|ENOENT|no such file|not a directory)\b/i.test(message)) {
    return { reason: 'enotdir' }
  }

  return { reason: 'error' }
}

async function readDesktopFileDataUrlChecked(path: string): Promise<string> {
  try {
    return await readDesktopFileDataUrl(path)
  } catch (error) {
    const statusCode = error instanceof Error ? (error as MediaBridgeError).statusCode : undefined

    if (typeof statusCode === 'number') {
      throw remoteDataUrlFailure(statusCode)
    }

    throw bridgeFailure(error)
  }
}

// Resolve a media path to a URL the shell can open. Remote mode rewrites
// gateway-local paths to an authenticated /api/files/download URL (the file
// lives on the gateway, not this disk); local mode keeps the file:// form.
// MediaFailure-tagged: the renderer renders a fallback card from the reason.
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

// Remote-mode replacement for opening gateway-local file paths with file://.
// The file lives on the gateway, so ask the Electron main process to fetch the
// bytes through the authenticated backend connection and save them locally. This
// avoids browser/OS downloads losing OAuth cookies and avoids the data-URL cap
// used by preview endpoints.
export async function downloadGatewayMediaFile(
  path: string
): Promise<{ canceled?: boolean; path?: string; saved: boolean }> {
  const file = filePathFromMediaPath(path)
  const conn = $connection.get()

  if (!window.hermesDesktop?.saveGatewayFile) {
    throw new Error('Desktop file download bridge is unavailable')
  }

  return window.hermesDesktop.saveGatewayFile({
    connectionId: conn?.connectionId,
    path: file,
    profile: conn?.profile,
    suggestedName: mediaName(file)
  })
}

// ── Media href size query (M4) ──────────────────────────────────────────────
//
// The capture-time media link carries the gateway-reported byte size in the
// href query (`#media:<enc>?~=<n>`), so a fallback card shows name + size even
// in a reopened transcript with no event row in memory. `~=` is chosen to be
// invisible in rendered URLs and collision-free with real query params on
// remote refs. Encoded without encodeURIComponent (markdown parentheses only
// need `(` `)` escaped; digits and `~` are already safe).

export function mediaHrefWithSize(path: string, size: number | undefined): string {
  const href = mediaMarkdownHref(path)

  if (typeof size !== 'number' || !Number.isFinite(size) || size < 0) {
    return href
  }

  return `${href}?~=${Math.round(size)}`
}

/** Read the `?~=` size query off a media href. Returns undefined when absent. */
export function mediaPathAndSizeFromMarkdownHref(href?: string): { path: string; size?: number } | null {
  const mediaPath = mediaPathFromMarkdownHref(href)

  if (mediaPath === null) {
    return null
  }

  const match = /\?~=(\d{1,15})$/.exec(mediaPath)
  const size = match ? Number(match[1]) : undefined

  return { path: match ? mediaPath.slice(0, match.index) : mediaPath, size }
}