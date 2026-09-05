// Helpers for saving a gateway-hosted file to the local disk from the Electron
// main process. Extracted from main.ts so the streaming, data-URL decoding, and
// filename derivation are unit-testable without spinning up Electron.
//
// The transport wrappers (token / OAuth) live in main.ts because they need
// main-process singletons (https/http, electronNet, the OAuth session). They
// delegate the byte-moving to `pumpStreamToFile` here, which streams the
// response to a user-selected destination with backpressure and cleans up a
// partial file on error — so a large download never has to be buffered whole in
// the native process.

import path from 'node:path'

// Minimal shape of the response objects we consume. Both Node's
// http.IncomingMessage and Electron net's IncomingMessage satisfy it.
export interface ReadableLike {
  on(event: 'data', listener: (chunk: Buffer | Uint8Array | string) => void): unknown
  on(event: 'end', listener: () => void): unknown
  on(event: 'error', listener: (err: Error) => void): unknown
  pause?: () => void
  resume?: () => void
  destroy?: (err?: Error) => void
}

export interface WriteStreamLike {
  write(chunk: Buffer): boolean
  end(cb: () => void): void
  destroy(err?: Error): void
  on(event: 'error', listener: (err: Error) => void): unknown
  once(event: 'drain', listener: () => void): unknown
}

export interface PumpDeps {
  createWriteStream: (destPath: string) => WriteStreamLike
  unlink: (destPath: string) => Promise<unknown>
}

// Stream `res` into `destPath`, honoring backpressure. On any read/write error
// the write stream is torn down and the (partial) destination file is removed
// before the returned promise rejects, so a failed download never leaves a
// truncated file behind.
export function pumpStreamToFile(res: ReadableLike, destPath: string, deps: PumpDeps): Promise<void> {
  return new Promise((resolve, reject) => {
    const ws = deps.createWriteStream(destPath)
    let failed = false

    const fail = (err: Error) => {
      if (failed) {
        return
      }

      failed = true

      try {
        res.destroy?.(err)
      } catch {
        // best effort — the socket may already be closed
      }

      try {
        ws.destroy()
      } catch {
        // best effort
      }

      Promise.resolve(deps.unlink(destPath))
        .catch(() => {})
        .then(() => reject(err))
    }

    ws.on('error', fail)
    res.on('error', fail)

    res.on('data', chunk => {
      if (failed) {
        return
      }

      const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk as Uint8Array)
      const ok = ws.write(buffer)

      // Backpressure: pause the source until the file stream drains so we never
      // accumulate the whole payload in memory.
      if (!ok && typeof res.pause === 'function') {
        res.pause()
        ws.once('drain', () => {
          if (!failed) {
            res.resume?.()
          }
        })
      }
    })

    res.on('end', () => {
      if (failed) {
        return
      }

      ws.end(() => resolve())
    })
  })
}

// Decode a `data:[<mime>][;base64],<payload>` URL into a Buffer. Used by the
// compatibility fallback that reads through the capped `/api/fs/read-data-url`
// route when the gateway predates `/api/fs/download`.
export function parseDataUrlToBuffer(dataUrl: string): Buffer {
  const match = /^data:([^,]*),([\s\S]*)$/.exec(String(dataUrl || ''))

  if (!match) {
    throw new Error('Malformed data URL')
  }

  const meta = match[1] || ''
  const payload = match[2] || ''

  if (/;base64/i.test(meta)) {
    return Buffer.from(payload, 'base64')
  }

  return Buffer.from(decodeURIComponent(payload), 'utf8')
}

// Extract a filename from a Content-Disposition header, preferring the RFC 5987
// `filename*` form. Returns '' when none is present. Always reduced to a
// basename so a malicious header can't redirect the save outside the picked dir.
export function filenameFromContentDisposition(value: unknown): string {
  const text = String(value || '')
  const encoded = text.match(/filename\*=(?:UTF-8'')?([^;]+)/i)?.[1]
  const plain = text.match(/filename="?([^";]+)"?/i)?.[1]
  const raw = encoded || plain || ''

  if (!raw) {
    return ''
  }

  try {
    return path.basename(decodeURIComponent(raw.trim()))
  } catch {
    return path.basename(raw.trim())
  }
}

// Normalize a gateway file path that may arrive as a bare path or a file:// URL.
export function gatewayFilePath(rawPath: unknown): string {
  const value = String(rawPath || '').trim()

  if (!value) {
    return ''
  }

  if (!/^file:/i.test(value)) {
    return value
  }

  try {
    return decodeURIComponent(new URL(value).pathname)
  } catch {
    return value.replace(/^file:\/\//i, '')
  }
}

// True when an error thrown by a transport wrapper represents an HTTP 404, used
// to trigger the data-URL compatibility fallback (and nothing else).
export function isNotFoundError(error: unknown): boolean {
  return Boolean(error) && (error as { statusCode?: number }).statusCode === 404
}
