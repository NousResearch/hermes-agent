import path from 'node:path'

/**
 * Shared MIME tables for the Electron main process.
 *
 * `mimeTypeForPath` labels files for data-URL reads (`readFileDataUrl`) and
 * attachment-upload IPC; the composer's upload pipeline forwards that MIME
 * verbatim and the backend's `_decode_chat_image_upload` rejects anything
 * that is not `image/*` BEFORE inspecting bytes — so a missing entry here
 * fails the upload even when the backend would accept the format.
 *
 * `extensionForMimeType` names persisted blobs (clipboard/drop saves); an
 * unknown image MIME falls back to `.png`, which would store HEIC/HEIF bytes
 * under a lying extension.
 */
export const MEDIA_MIME_TYPES: Record<string, string> = {
  '.avi': 'video/x-msvideo',
  '.avif': 'image/avif',
  '.bmp': 'image/bmp',
  '.flac': 'audio/flac',
  '.gif': 'image/gif',
  '.heic': 'image/heic',
  '.heif': 'image/heif',
  '.jpeg': 'image/jpeg',
  '.jpg': 'image/jpeg',
  '.m4a': 'audio/mp4',
  '.mkv': 'video/x-matroska',
  '.mov': 'video/quicktime',
  '.mp3': 'audio/mpeg',
  '.mp4': 'video/mp4',
  '.ogg': 'audio/ogg',
  '.opus': 'audio/ogg; codecs=opus',
  '.pdf': 'application/pdf',
  '.png': 'image/png',
  '.svg': 'image/svg+xml',
  '.wav': 'audio/wav',
  '.webm': 'video/webm',
  '.webp': 'image/webp'
}

export function mimeTypeForPath(filePath: string): string {
  const ext = path.extname(filePath || '').toLowerCase()

  return MEDIA_MIME_TYPES[ext] || 'application/octet-stream'
}

const EXTENSION_BY_MIME: Record<string, string> = {
  'image/avif': '.avif',
  'image/bmp': '.bmp',
  'image/gif': '.gif',
  'image/heic': '.heic',
  'image/heif': '.heif',
  'image/jpeg': '.jpg',
  'image/png': '.png',
  'image/svg+xml': '.svg',
  'image/webp': '.webp'
}

export function extensionForMimeType(mimeType: string): string {
  const type = String(mimeType || '')
    .split(';')[0]
    .trim()
    .toLowerCase()

  return EXTENSION_BY_MIME[type] || ''
}
