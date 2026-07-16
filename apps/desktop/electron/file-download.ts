import crypto from 'node:crypto'
import fs from 'node:fs'
import path from 'node:path'
import { pipeline } from 'node:stream/promises'

const DOWNLOAD_INACTIVITY_TIMEOUT_MS = 60_000
const ERROR_BODY_MAX_BYTES = 64 * 1024

interface DownloadResponse extends NodeJS.ReadableStream {
  statusCode?: number
  statusMessage?: string
}

interface DownloadRequest {
  abort?: () => void
  end: () => void
  on: (event: 'error' | 'response', listener: (...args: any[]) => void) => DownloadRequest
}

interface DownloadOptions {
  fs?: typeof fs
  inactivityTimeoutMs?: number
  tempPath?: string
}

function downloadFilename(filePath: string): string {
  const stripped = String(filePath || '')
    .trim()
    .replace(/[\\/]+$/, '')

  const name = stripped.split(/[\\/]/).pop()?.trim()

  if (!name || name === '.' || name === '..') {
    return 'download'
  }

  // Native save dialogs accept a basename, never a path. Strip control
  // characters so a remote filename cannot distort the dialog title/value.
  return Array.from(name, character => {
    const codePoint = character.codePointAt(0) || 0

    return codePoint <= 31 || codePoint === 127 ? '_' : character
  }).join('')
}

function temporaryDownloadPath(destination: string): string {
  const nonce = crypto.randomBytes(6).toString('hex')

  return path.join(path.dirname(destination), `.${path.basename(destination)}.hermes-download-${process.pid}-${nonce}`)
}

async function replaceWithTemporaryFile(fsImpl: typeof fs, temporary: string, destination: string): Promise<void> {
  try {
    await fsImpl.promises.rename(temporary, destination)
  } catch (error: any) {
    // Windows can reject replacing an existing file with EPERM/EEXIST even
    // after the native save dialog confirmed overwrite. The complete payload
    // is already durable in the sibling temp file, so remove only at commit.
    if (error?.code !== 'EEXIST' && error?.code !== 'EPERM') {
      throw error
    }

    await fsImpl.promises.rm(destination, { force: true })
    await fsImpl.promises.rename(temporary, destination)
  }
}

async function copyFileAtomically(source: string, destination: string, options: DownloadOptions = {}): Promise<void> {
  const fsImpl = options.fs || fs
  const temporary = options.tempPath || temporaryDownloadPath(destination)

  try {
    await fsImpl.promises.copyFile(source, temporary)
    await replaceWithTemporaryFile(fsImpl, temporary, destination)
  } catch (error) {
    await fsImpl.promises.rm(temporary, { force: true }).catch(() => undefined)
    throw error
  }
}

function responseError(statusCode: number, statusMessage: string | undefined, body: string): Error {
  let detail = body.trim()

  try {
    const parsed = JSON.parse(detail)

    if (typeof parsed?.detail === 'string') {
      detail = parsed.detail
    }
  } catch {
    // Plain-text gateway errors are already useful as-is.
  }

  return new Error(`${statusCode}: ${detail || statusMessage || 'File download failed'}`)
}

async function readErrorBody(response: DownloadResponse): Promise<string> {
  const chunks: Buffer[] = []
  let total = 0

  for await (const chunk of response as any) {
    if (total >= ERROR_BODY_MAX_BYTES) {
      continue
    }

    const buffer = Buffer.from(chunk)
    const remaining = ERROR_BODY_MAX_BYTES - total
    const next = buffer.subarray(0, remaining)

    chunks.push(next)
    total += next.length
  }

  return Buffer.concat(chunks).toString('utf8')
}

function streamDownloadRequest(
  request: DownloadRequest,
  destination: string,
  options: DownloadOptions = {}
): Promise<void> {
  const fsImpl = options.fs || fs
  const temporary = options.tempPath || temporaryDownloadPath(destination)
  const inactivityTimeoutMs = Math.max(1_000, options.inactivityTimeoutMs || DOWNLOAD_INACTIVITY_TIMEOUT_MS)

  return new Promise((resolve, reject) => {
    let settled = false
    let timer: ReturnType<typeof setTimeout> | null = null

    const clearTimer = () => {
      if (timer) {
        clearTimeout(timer)
        timer = null
      }
    }

    const cleanupAndReject = (error: unknown) => {
      if (settled) {
        return
      }

      settled = true
      clearTimer()

      try {
        request.abort?.()
      } catch {
        // Request already ended.
      }

      void fsImpl.promises
        .rm(temporary, { force: true })
        .catch(() => undefined)
        .finally(() => reject(error instanceof Error ? error : new Error(String(error))))
    }

    const resetTimer = () => {
      clearTimer()
      timer = setTimeout(
        () => cleanupAndReject(new Error(`File download stalled for ${inactivityTimeoutMs}ms`)),
        inactivityTimeoutMs
      )
    }

    request.on('error', cleanupAndReject)
    request.on('response', (response: DownloadResponse) => {
      void (async () => {
        try {
          const statusCode = response.statusCode || 500

          if (statusCode < 200 || statusCode >= 300) {
            throw responseError(statusCode, response.statusMessage, await readErrorBody(response))
          }

          response.on('data', resetTimer)
          await pipeline(response as any, fsImpl.createWriteStream(temporary, { flags: 'wx' }))

          if (settled) {
            return
          }

          await replaceWithTemporaryFile(fsImpl, temporary, destination)

          if (!settled) {
            settled = true
            clearTimer()
            resolve()
          }
        } catch (error) {
          cleanupAndReject(error)
        }
      })()
    })

    resetTimer()
    request.end()
  })
}

export {
  copyFileAtomically,
  DOWNLOAD_INACTIVITY_TIMEOUT_MS,
  downloadFilename,
  responseError,
  streamDownloadRequest,
  temporaryDownloadPath
}
