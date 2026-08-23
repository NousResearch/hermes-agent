import { createReadStream, promises as fsp } from 'node:fs'
import path from 'node:path'
import { Readable } from 'node:stream'

/**
 * Range-aware responses for local audio/video served through `hermes-media://stream/…`.
 *
 * The media protocol used to delegate local files to `net.fetch(file://…)` and forward the
 * renderer's `Range` header, expecting a `206 Partial Content` back. In practice Electron's
 * file:// loader ignores `Range` and always answers `200` with the whole body and no
 * `Accept-Ranges`, so Chromium reports `video.seekable` as `[0, 0]`: the progress bar cannot be
 * dragged and any seek snaps back to 0. Short clips hide this (they buffer fully within a
 * second and seeking inside buffered data works), long recordings expose it immediately.
 *
 * This module answers the request itself: it parses `Range`, slices the file with
 * `createReadStream`, and returns `206` with `Content-Range` / `Accept-Ranges` (plus `416` for
 * unsatisfiable ranges, header-only `HEAD` responses, and `404` when the file is gone).
 */

const MEDIA_MIME: Record<string, string> = {
  '.aac': 'audio/aac',
  '.avi': 'video/x-msvideo',
  '.flac': 'audio/flac',
  '.m4a': 'audio/mp4',
  '.m4v': 'video/mp4',
  '.mkv': 'video/x-matroska',
  '.mov': 'video/quicktime',
  '.mp3': 'audio/mpeg',
  '.mp4': 'video/mp4',
  '.oga': 'audio/ogg',
  '.ogg': 'audio/ogg',
  '.ogv': 'video/ogg',
  '.opus': 'audio/ogg',
  '.wav': 'audio/wav',
  '.webm': 'video/webm'
}

export function mediaMimeFor(filePath: string): string {
  return MEDIA_MIME[path.extname(filePath).toLowerCase()] || 'application/octet-stream'
}

export interface ByteRange {
  start: number
  /** Inclusive. */
  end: number
}

/**
 * Parse `Range: bytes=a-b` / `bytes=a-` / `bytes=-n`.
 *
 * Multi-range requests (any comma in the header) are deliberately ignored as a whole — RFC 7233
 * lets a server answer `200` with the full representation instead of `multipart/byteranges`,
 * and Chromium's media stack only ever sends a single range. Returns `null` when there is no
 * usable Range header (serve the whole file) and `'unsatisfiable'` when the range starts beyond
 * the end of the file (answer 416).
 */
export function parseByteRange(header: string | null | undefined, size: number): ByteRange | null | 'unsatisfiable' {
  if (!header || header.includes(',')) {
    return null
  }

  const m = /^\s*bytes\s*=\s*(\d*)\s*-\s*(\d*)\s*$/i.exec(header)

  if (!m || (m[1] === '' && m[2] === '')) {
    return null
  }

  if (size <= 0) {
    return 'unsatisfiable'
  }

  let start: number
  let end: number

  if (m[1] === '') {
    // Suffix range: the last n bytes.
    const suffix = Math.min(Number(m[2]), size)

    if (suffix <= 0) {
      return 'unsatisfiable'
    }

    start = size - suffix
    end = size - 1
  } else {
    start = Number(m[1])
    end = m[2] === '' ? size - 1 : Math.min(Number(m[2]), size - 1)
  }

  if (!Number.isFinite(start) || !Number.isFinite(end) || start >= size || start > end) {
    return 'unsatisfiable'
  }

  return { start, end }
}

export interface LocalMediaResponseInit {
  method?: string
  rangeHeader?: string | null
}

/** Build the 200 / 206 / 404 / 416 response for a local media file (HEAD gets headers only). */
export async function buildLocalMediaResponse(resolvedPath: string, init: LocalMediaResponseInit = {}): Promise<Response> {
  let handle: fsp.FileHandle

  try {
    handle = await fsp.open(resolvedPath, 'r')
  } catch (error) {
    const code = (error as NodeJS.ErrnoException)?.code

    if (code === 'ENOENT' || code === 'ENOTDIR') {
      return new Response('Media not found', { status: 404 })
    }

    throw error
  }

  // fstat the handle we are about to read from, so `Content-Length` and the streamed bytes come
  // from the same open file even if the path is truncated or replaced meanwhile.
  const stat = await handle.stat()
  const size = stat.size
  const mime = mediaMimeFor(resolvedPath)
  const isHead = (init.method || 'GET').toUpperCase() === 'HEAD'
  const range = parseByteRange(init.rangeHeader, size)

  const baseHeaders: Record<string, string> = {
    'Accept-Ranges': 'bytes',
    'Cache-Control': 'no-store',
    'Content-Type': mime,
    'Last-Modified': stat.mtime.toUTCString()
  }

  const stream = (opts: { end?: number; start?: number }) =>
    // Hand the FileHandle itself (not the raw fd) to the stream: it owns and closes it on end, so
    // the handle is never closed twice (EBADF on garbage collection).
    Readable.toWeb(createReadStream(resolvedPath, { ...opts, autoClose: true, fd: handle })) as unknown as ReadableStream

  if (range === 'unsatisfiable') {
    await handle.close()

    return new Response(null, { headers: { ...baseHeaders, 'Content-Range': `bytes */${size}` }, status: 416 })
  }

  if (range === null) {
    const headers = { ...baseHeaders, 'Content-Length': String(size) }

    if (isHead || size === 0) {
      await handle.close()

      return new Response(null, { headers, status: 200 })
    }

    return new Response(stream({}), { headers, status: 200 })
  }

  const length = range.end - range.start + 1
  const headers = {
    ...baseHeaders,
    'Content-Length': String(length),
    'Content-Range': `bytes ${range.start}-${range.end}/${size}`
  }

  if (isHead) {
    await handle.close()

    return new Response(null, { headers, status: 206 })
  }

  return new Response(stream({ end: range.end, start: range.start }), { headers, status: 206 })
}
