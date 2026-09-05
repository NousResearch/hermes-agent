// Helpers for saving a gateway-hosted file to the local disk from the Electron
// main process. Extracted from main.ts so the streaming, data-URL decoding, and
// filename derivation are unit-testable without spinning up Electron.
//
// The transport wrappers (token / OAuth) live in main.ts because they need
// main-process singletons (https/http, electronNet, the OAuth session). They
// delegate the byte-moving to `pumpStreamToFile` here, which streams the
// response into a sibling temp file with backpressure and renames it onto the
// user-selected destination only once the body has landed in full — so a large
// download never has to be buffered whole in the native process, and a failed
// one never touches a file that was already at the destination.

import crypto from 'node:crypto'
import fs from 'node:fs'
import path from 'node:path'
import { Readable } from 'node:stream'

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
  // fs.WriteStream's close() ends the stream and calls back only after the
  // descriptor is released. end()'s callback fires on 'finish', while the fd can
  // still be open — and Windows refuses to rename a file with an open handle.
  close?(cb: (err?: Error | null) => void): void
  destroy(err?: Error): void
  on(event: 'error', listener: (err: Error) => void): unknown
  // 'open' is the ownership signal: only after it fires did THIS pump create
  // the temp file, and only then may cleanup unlink it.
  once(event: 'close' | 'drain' | 'open', listener: () => void): unknown
}

export interface PumpDeps {
  // Must open the temp path exclusively (`flags: 'wx'`): the pump relies on
  // creating a brand-new file, never on truncating or following something that
  // already sits at that name.
  createWriteStream: (tempPath: string) => WriteStreamLike
  rename: (fromPath: string, toPath: string) => Promise<unknown>
  unlink: (tempPath: string) => Promise<unknown>
  // Test seam: pick the temp path deterministically so a regression can seed
  // it and prove a pre-open collision leaves the seeded file untouched.
  tempPathFor?: (destPath: string) => string
}

// Production deps: exclusive create on the real filesystem. Shared by the
// streaming save and the data-URL fallback in main.ts, and exercised directly
// by the real-filesystem tests so the guarantees are proven against node:fs,
// not only against fakes.
export function fsPumpDeps(): PumpDeps {
  return {
    createWriteStream: tempPath => fs.createWriteStream(tempPath, { flags: 'wx' }),
    rename: (fromPath, toPath) => fs.promises.rename(fromPath, toPath),
    unlink: tempPath => fs.promises.unlink(tempPath)
  }
}

// How long to wait for a destroyed write stream to emit 'close' before giving
// up and unlinking anyway. fs.WriteStream always emits it; the grace period only
// protects against a stream shape that never does.
const CLOSE_GRACE_MS = 2000

// Resolve once `ws` has released its descriptor. destroy() closes the fd
// asynchronously, and Windows rejects unlink/rename on a path whose handle is
// still open, so cleanup must not run until 'close' has fired.
function awaitClosed(ws: WriteStreamLike): Promise<void> {
  return new Promise(resolve => {
    const timer = setTimeout(resolve, CLOSE_GRACE_MS)

    ws.once('close', () => {
      clearTimeout(timer)
      resolve()
    })
  })
}

export interface GatewayFileBackendDeps<T> {
  ensureLegacy: (profile: null | string) => Promise<T>
  ensureRegistry: (connectionId: string, profile: null | string) => Promise<T>
}

export interface GatewayFileBackendRoute<T> {
  connection: T
  connectionId: null | string
  profile: null | string
}
export interface GatewayFileRequestPaths {
  dataUrl: string
  download: string
}

export function gatewayFileRequestPaths(
  filePath: string,
  scopePath: (requestPath: string) => string
): GatewayFileRequestPaths {
  const encodedPath = encodeURIComponent(filePath)

  return {
    dataUrl: scopePath(`/api/fs/read-data-url?path=${encodedPath}`),
    download: scopePath(`/api/fs/download?path=${encodedPath}`)
  }
}

/**
 * Resolve the backend that owns a renderer-requested gateway file. Registered
 * connections must never fall through to the legacy profile pool: that pool
 * can point at another machine with another authentication credential.
 */
export async function resolveGatewayFileBackend<T>(
  payload: { connectionId?: unknown; profile?: unknown },
  deps: GatewayFileBackendDeps<T>
): Promise<GatewayFileBackendRoute<T>> {
  const connectionId = String(payload.connectionId ?? '').trim() || null
  const profile = String(payload.profile ?? '').trim() || null

  const connection = connectionId ? await deps.ensureRegistry(connectionId, profile) : await deps.ensureLegacy(profile)

  return { connection, connectionId, profile }
}

// Sibling temp name for an in-flight download. It lives in the destination's own
// directory so the final step is a same-volume rename (and stays inside whatever
// directory the save dialog approved). The name is short and fixed rather than
// derived from the destination's basename so a long user-chosen filename cannot
// push the temp name past the filesystem limit, and the random suffix keeps two
// concurrent saves into the same directory from sharing a temp file. The leading
// dot hides the in-flight file in Finder/ls while it exists.
export function downloadTempPath(destPath: string): string {
  return path.join(path.dirname(destPath), `.hermes-download-${crypto.randomBytes(4).toString('hex')}.part`)
}

// Stream `res` to `destPath`, honoring backpressure. Bytes land in a sibling
// temp file first and are renamed onto `destPath` only after the whole body has
// been written and the descriptor released. The destination itself is never
// opened before that point, so a download that fails part-way leaves any file
// already at `destPath` exactly as it was — only the temp file is removed before
// the returned promise rejects. (Opening `destPath` directly truncated it on the
// spot and the error path then unlinked it, destroying a pre-existing file the
// user had chosen to overwrite; #96597.)
export function pumpStreamToFile(res: ReadableLike, destPath: string, deps: PumpDeps): Promise<void> {
  return new Promise((resolve, reject) => {
    const tempPath = (deps.tempPathFor ?? downloadTempPath)(destPath)
    const ws = deps.createWriteStream(tempPath)
    let failed = false

    // Ownership gate. An exclusive open can fail BEFORE this pump has created
    // anything at `tempPath` (EEXIST on a collision, EACCES, a missing parent);
    // in that case the path belongs to someone else and cleanup must not touch
    // it. fs.WriteStream emits 'open' exactly when the create succeeded.
    let owned = false

    ws.once('open', () => {
      owned = true
    })

    // `.then(() => dep())` rather than `Promise.resolve(dep())` so a dep that
    // throws synchronously still lands on the rejection path instead of escaping
    // the stream callback it was invoked from.
    const discardTemp = (): Promise<void> => {
      if (!owned) {
        return Promise.resolve()
      }

      return Promise.resolve()
        .then(() => deps.unlink(tempPath))
        .then(
          () => {},
          () => {} // best effort
        )
    }

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

      // Register the 'close' listener BEFORE destroy(): on a stream that is
      // already tearing down after its own 'error', 'close' can follow on the
      // next tick.
      const closed = awaitClosed(ws)

      try {
        ws.destroy()
      } catch {
        // best effort
      }

      closed.then(discardTemp).then(() => reject(err))
    }

    // Flush and release the temp file, then move it into place. A rename failure
    // (destination locked, permissions) must not leave the temp file behind.
    const finish = () => {
      const onClosed = (err?: Error | null) => {
        if (failed) {
          return
        }

        if (err) {
          fail(err)

          return
        }

        Promise.resolve()
          .then(() => deps.rename(tempPath, destPath))
          .then(
            () => {
              // A failure that raced the rename has already taken the reject
              // path; never report success on top of it.
              if (!failed) {
                resolve()
              }
            },
            (renameErr: Error) => {
              if (failed) {
                return
              }

              failed = true
              discardTemp().then(() => reject(renameErr))
            }
          )
      }

      if (typeof ws.close === 'function') {
        ws.close(onClosed)
      } else {
        ws.end(() => onClosed())
      }
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

      finish()
    })
  })
}

// Write an in-memory body to `destPath` with the same failure-atomic contract as
// `pumpStreamToFile` (temp file, exclusive create, close, rename). Used by the
// data-URL compatibility fallback, which has the whole body up front; a plain
// `fs.promises.writeFile(destPath, buffer)` would truncate an existing file
// before the write completes and so could destroy it on a mid-write failure.
export function writeBufferToFile(buffer: Buffer, destPath: string, deps: PumpDeps): Promise<void> {
  return pumpStreamToFile(Readable.from([buffer]), destPath, deps)
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
