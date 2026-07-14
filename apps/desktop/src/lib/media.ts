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

// Custom Electron scheme (registered in electron/main.cjs) that streams a local
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

// True when the inline media reader must fetch via the gateway API rather
// than the Electron IPC (which reads THIS machine's disk).
//
// Reasons to route via gateway (#63669):
//   1. Remote gateway mode: image lives on the gateway machine.
//   2. Windows client + WSL POSIX path + local mode: connection.mode is
//      'local' but the gateway runs on the WSL Linux side, so the file
//      is on the gateway's filesystem, not Windows'. Electron IPC
//      cannot read WSL paths from a Windows process.
//
// Returns false for plain Windows paths in local mode (Electron IPC reads
// the Windows disk directly) and for POSIX paths on macOS/Linux (same
// filesystem). Connection state is required to know the mode; if
// $connection is null (cold start) we don't have enough signal to make
// a Windows+WSL call and fall back to the existing readFileDataUrl
// path — same as today.
export function shouldRouteMediaViaGateway(path: string): boolean {
  if (isRemoteGateway()) return true
  // Local mode + connection hasn't hydrated yet: no signal.
  if (!$connection.get()) return false
  if (typeof navigator === 'undefined') return false
  if (!/^Win/i.test(navigator.platform || '')) return false
  // Windows client. Route through gateway for any POSIX absolute path —
  // these are WSL filesystem paths that Electron IPC can't read.
  const file = filePathFromMediaPath(path)
  return /^(\/home\/|\/tmp\/|\/var\/|\/opt\/|\/srv\/|\/mnt\/|\/root\/)/.test(file)
}

// Fetch a gateway-local image as a data URL via the authenticated REST bridge.
// Used in remote mode where readFileDataUrl (which reads THIS machine's disk)
// can't see files the agent wrote on the gateway. Requires the gateway to
// expose GET /api/media (hermes_cli/web_server.py).
export async function gatewayMediaDataUrl(path: string): Promise<string> {
  const file = filePathFromMediaPath(path)

  const result = await window.hermesDesktop!.api<{ data_url: string }>({
    path: `/api/media?path=${encodeURIComponent(file)}`
  })

  return result.data_url
}

export function mediaDisplayLabel(path: string): string {
  const escaped = mediaName(path).replace(/[[\]\\]/g, '\\$&')
  const kind = mediaKind(path)

  return `${kind[0].toUpperCase()}${kind.slice(1)}: ${escaped}`
}
