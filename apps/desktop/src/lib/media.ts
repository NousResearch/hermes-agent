import { readDesktopFileDataUrl } from '@/lib/desktop-fs'
import { capitalize } from '@/lib/text'
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
  const info = mediaInfo(path)

  if (info) {
    return info.kind
  }

  // No recognizable extension. Remote (http/https) URLs are almost always
  // images even without a file extension (e.g. Unsplash
  // `images.unsplash.com/photo-…`, `picsum.photos/200`), and the extension
  // strip above drops any query string. Treat them as images so they render
  // inline instead of being misclassified as a generic `file` and blocked.
  if (/^https?:/i.test(path)) {
    return 'image'
  }

  return 'file'
}

export function mediaMime(path: string): string {
  const info = mediaInfo(path)

  if (info) {
    return info.mime
  }

  // Mirror mediaKind: extensionless remote URLs are treated as images.
  if (/^https?:/i.test(path)) {
    return 'image/'
  }

  return 'application/octet-stream'
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
// The file lives on the gateway, so fetch it over the authenticated fs bridge
// and hand the bytes to the local browser shell as a download.
// Resolve a media path to a URL the renderer can display. Local files are read
// through the desktop FS bridge (window.hermesDesktop.readFileDataUrl); remote
// gateway files are fetched over the authenticated API; http(s)/data URLs pass
// through; audio/video may stream via the custom protocol to bypass the data
// URL size cap.
export async function mediaSrc(path: string): Promise<string> {
  if (/^(?:https?|data):/i.test(path)) {
    return path
  }

  if (window.hermesDesktop && ['audio', 'video'].includes(mediaKind(path))) {
    return mediaStreamUrl(path)
  }

  if (window.hermesDesktop && isRemoteGateway()) {
    return gatewayMediaDataUrl(path)
  }

  if (!window.hermesDesktop?.readFileDataUrl) {
    return mediaExternalUrl(path)
  }

  return window.hermesDesktop.readFileDataUrl(filePathFromMediaPath(path))
}

export async function downloadGatewayMediaFile(path: string): Promise<void> {
  const dataUrl = await readDesktopFileDataUrl(filePathFromMediaPath(path))

  if (!dataUrl) {
    throw new Error('Gateway returned no file data')
  }

  const response = await fetch(dataUrl)
  const blobUrl = URL.createObjectURL(await response.blob())
  const anchor = document.createElement('a')
  anchor.href = blobUrl
  anchor.download = mediaName(path)
  anchor.rel = 'noopener noreferrer'
  document.body.appendChild(anchor)
  anchor.click()
  anchor.remove()
  window.setTimeout(() => URL.revokeObjectURL(blobUrl), 30_000)
}

export function mediaDisplayLabel(path: string): string {
  const escaped = mediaName(path).replace(/[[\\]\\\\]/g, '\\\\$&')
  const kind = mediaKind(path)

  return `${capitalize(kind)}: ${escaped}`
}

// ─── Image gallery (MEDIA-GALLERY) ────────────────────────────────────────
// A gallery lets the agent emit a sequence of screenshots as a single
// auto-playing carousel in the chat (Replit-agent style progress reel). It is
// encoded as a `#gallery:` markdown link (parallel to the `#media:` links used
// for single-file attachments) so it rides the existing Streamdown → React
// link pipeline and needs no new assistant-ui message-part type.

export interface MediaGalleryImage {
  src: string
  title?: string
}

export interface MediaGalleryPayload {
  title?: string
  intervalMs?: number
  images: MediaGalleryImage[]
}

const GALLERY_HREF_PREFIX = '#gallery:'

export function galleryMarkdownHref(payload: MediaGalleryPayload): string {
  return `${GALLERY_HREF_PREFIX}${encodeURIComponent(JSON.stringify(payload))}`
}

// Decode a `#gallery:` href back into a payload. Returns null for anything
// that isn't a valid gallery link, or that has fewer than two images (a
// single/zero image "gallery" is meaningless and should fall back to plain
// MEDIA rendering).
export function galleryPayloadFromHref(href?: string): MediaGalleryPayload | null {
  if (!href?.startsWith(GALLERY_HREF_PREFIX)) {
    return null
  }

  try {
    const raw = decodeURIComponent(href.slice(GALLERY_HREF_PREFIX.length))
    const parsed = JSON.parse(raw) as Partial<MediaGalleryPayload>

    if (!Array.isArray(parsed.images)) {
      return null
    }

    const images = parsed.images
      .filter((img): img is MediaGalleryImage => !!img && typeof img.src === 'string' && img.src.length > 0)
      .map(img => ({
        src: img.src,
        title: typeof img.title === 'string' && img.title ? img.title : undefined
      }))

    if (images.length < 2) {
      return null
    }

    const intervalMs =
      typeof parsed.intervalMs === 'number' && Number.isFinite(parsed.intervalMs) ? parsed.intervalMs : undefined
    const title = typeof parsed.title === 'string' && parsed.title ? parsed.title : undefined

    return { images, intervalMs, title }
  } catch {
    return null
  }
}
