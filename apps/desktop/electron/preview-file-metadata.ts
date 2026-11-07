import fs from 'node:fs'
import path from 'node:path'

const PREVIEW_FILE_TIMEOUT_MS = 1000

export type PreviewFileStatResult = { kind: 'found'; stat: fs.Stats } | { kind: 'missing' | 'unavailable' }

interface PreviewFileMetadataOptions {
  maxTextBytes?: number
  stat?: Pick<fs.Stats, 'size'>
  timeoutMs?: number
}

export function previewTargetNeedsBase(filePath: string, platform: NodeJS.Platform = process.platform): boolean {
  if (/^file:/i.test(filePath)) {
    return false
  }

  const paths = platform === 'win32' ? path.win32 : path.posix

  // A Windows root-only path still needs the cwd drive; drive-absolute and UNC paths do not.
  return !paths.isAbsolute(filePath) || (platform === 'win32' && paths.parse(filePath).root.length === 1)
}

export function looksBinary(buffer: Uint8Array): boolean {
  if (!buffer.length) {
    return false
  }

  let suspicious = 0

  for (const byte of buffer) {
    if (byte === 0) {
      return true
    }

    // Allow common whitespace controls: tab, LF, CR.
    if (byte < 32 && byte !== 9 && byte !== 10 && byte !== 13) {
      suspicious += 1
    }
  }

  return suspicious / buffer.length > 0.12
}

/** Classification is best-effort; a cloud provider must never hold its IPC open forever. */
export async function withPreviewFileDeadline<T>(
  operation: Promise<T>,
  timeoutMs = PREVIEW_FILE_TIMEOUT_MS,
  onTimeout?: () => void
): Promise<T | null> {
  let timer: ReturnType<typeof setTimeout> | undefined

  try {
    return await Promise.race([
      operation,
      new Promise<null>(resolve => {
        timer = setTimeout(() => {
          onTimeout?.()
          resolve(null)
        }, timeoutMs)
      })
    ])
  } catch {
    // Explicit preview/read handlers surface errors when the user opens the file.
    return null
  } finally {
    clearTimeout(timer)
  }
}

export async function previewFileStat(
  filePath: string,
  timeoutMs = PREVIEW_FILE_TIMEOUT_MS
): Promise<PreviewFileStatResult> {
  const operation = fs.promises.stat(filePath).then<PreviewFileStatResult, PreviewFileStatResult>(
    stat => ({ kind: 'found', stat }),
    (error: NodeJS.ErrnoException) => ({
      kind: error?.code === 'ENOENT' || error?.code === 'ENOTDIR' ? 'missing' : 'unavailable'
    })
  )

  return (await withPreviewFileDeadline(operation, timeoutMs)) ?? { kind: 'unavailable' }
}

export async function previewFileMetadata(
  filePath: string,
  mimeType: string,
  options: PreviewFileMetadataOptions = {}
): Promise<{ binary: boolean; byteSize: number; large: boolean }> {
  const result = options.stat ? null : await previewFileStat(filePath, options.timeoutMs)
  const stat = options.stat ?? (result?.kind === 'found' ? result.stat : null)
  const byteSize = stat?.size ?? 0
  const metadata = { binary: false, byteSize, large: byteSize > (options.maxTextBytes ?? 512 * 1024) }
  const ext = path.extname(filePath).toLowerCase()

  // Known preview kinds need no content sampling, which can recall an evicted cloud file.
  if (
    !byteSize ||
    mimeType.startsWith('image/') ||
    mimeType === 'application/pdf' ||
    mimeType === 'text/html' ||
    ext === '.pdf' ||
    ext === '.html' ||
    ext === '.htm'
  ) {
    return metadata
  }

  let expired = false

  const sample = (async () => {
    const handle = await fs.promises.open(filePath, 'r')

    try {
      if (expired) {
        return false
      }

      const buffer = Buffer.alloc(Math.min(byteSize, 4096))
      const { bytesRead } = await handle.read(buffer, 0, buffer.length, 0)

      return looksBinary(buffer.subarray(0, bytesRead))
    } finally {
      // A deadline does not cancel OS I/O; the straggler still owns and closes its handle.
      await handle.close()
    }
  })()

  const binary = await withPreviewFileDeadline(sample, options.timeoutMs, () => {
    expired = true
  })

  return { ...metadata, binary: binary ?? false }
}
