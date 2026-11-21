import { createHash } from 'node:crypto'
import { createReadStream } from 'node:fs'
import { lstat, open, readFile, rename, unlink, writeFile } from 'node:fs/promises'
import type { FileHandle } from 'node:fs/promises'
import path from 'node:path'

export interface PinnedArtifact {
  url: string
  sha256: string
  size: number
  format: 'zip' | 'msix' | 'msixbundle'
}

/** How long a connection may go without delivering a byte, and how many
 *  connections in a row may end without adding one before the download fails. */
export interface PinnedDownloadPolicy {
  stallMs: number
  attemptsWithoutProgress: number
}

// A total cap failed every slow link on an artifact of hundreds of MB; only a
// transfer that stops moving fails, and a dropped one resumes where it stopped.
export const PINNED_DOWNLOAD_POLICY: PinnedDownloadPolicy = { stallMs: 120_000, attemptsWithoutProgress: 3 }

/** Bytes on disk so far and the pinned total, after every received chunk. */
export type PinnedDownloadProgress = (received: number, total: number) => void

export class PinnedDownloadStalledError extends Error {
  readonly code = 'stalled'
}

/** The server or the bytes said no: resuming cannot change the answer. */
class PinnedDownloadRefusedError extends Error {}

export async function verifyPinnedArtifact(file: string, artifact: PinnedArtifact): Promise<void> {
  const info = await lstat(file)

  if (!info.isFile() || info.isSymbolicLink() || info.size !== artifact.size) {
    throw new Error('Pinned artifact size or type mismatch')
  }

  const hash = createHash('sha256')

  for await (const chunk of createReadStream(file)) {
    hash.update(chunk)
  }

  if (hash.digest('hex') !== artifact.sha256) {
    throw new Error('Pinned artifact digest mismatch')
  }
}

function ignoreMissing(error: NodeJS.ErrnoException): void {
  if (error.code !== 'ENOENT') {
    throw error
  }
}

async function partialSize(file: string): Promise<number> {
  const info = await lstat(file).catch((error: NodeJS.ErrnoException): null => {
    ignoreMissing(error)

    return null
  })

  if (info && (!info.isFile() || info.isSymbolicLink())) {
    throw new PinnedDownloadRefusedError('Pinned artifact partial is not a regular file')
  }

  return info?.size ?? 0
}

interface Partial {
  file: string
  /** ETag (or Last-Modified) of the entity the partial bytes came from. */
  validatorFile: string
}

/** One connection. Resumes from `offset` when the server honours the range for
 *  the same entity, else restarts from zero. Returns the bytes now on disk. */
async function transfer(
  artifact: PinnedArtifact,
  partial: Partial,
  offset: number,
  policy: PinnedDownloadPolicy,
  onBytes?: PinnedDownloadProgress
): Promise<number> {
  const controller = new AbortController()
  let stalled = false
  let timer: NodeJS.Timeout | undefined

  const rearm = (): void => {
    clearTimeout(timer)
    timer = setTimeout((): void => {
      stalled = true
      controller.abort()
    }, policy.stallMs)
  }

  const validator = offset > 0 ? (await readFile(partial.validatorFile, 'utf8').catch((): string => '')).trim() : ''
  // If-Range: a changed object answers 200 with its whole body instead of
  // splicing new bytes onto old ones. The digest still decides at the end.
  const headers: Record<string, string> = validator ? { Range: `bytes=${offset}-`, 'If-Range': validator } : {}
  let size = 0
  let handle: FileHandle | null = null

  rearm()

  try {
    const response: Response = await fetch(artifact.url, { redirect: 'error', signal: controller.signal, headers })
    const resumed = response.status === 206 && validator !== ''

    if (!(response.status === 200 || resumed) || !response.body) {
      throw new PinnedDownloadRefusedError(`Pinned artifact download failed (${response.status})`)
    }

    if (resumed && !response.headers.get('content-range')?.startsWith(`bytes ${offset}-`)) {
      throw new PinnedDownloadRefusedError('Pinned artifact resume answered a different range')
    }

    if (resumed) {
      size = offset
      handle = await open(partial.file, 'a', 0o600)
    } else {
      await unlink(partial.file).catch(ignoreMissing)
      handle = await open(partial.file, 'wx', 0o600)
      await writeFile(
        partial.validatorFile,
        response.headers.get('etag') ?? response.headers.get('last-modified') ?? '',
        {
          mode: 0o600
        }
      )
    }

    for await (const chunk of response.body) {
      rearm()
      size += chunk.byteLength

      if (size > artifact.size) {
        throw new PinnedDownloadRefusedError('Pinned artifact download exceeds pinned size')
      }

      await handle.writeFile(chunk)
      onBytes?.(size, artifact.size)
    }

    await handle.sync()

    return size
  } catch (error) {
    if (stalled) {
      throw new PinnedDownloadStalledError(`Pinned artifact download received nothing for ${policy.stallMs} ms`)
    }

    throw error
  } finally {
    clearTimeout(timer)
    // An attempt that failed before reading its body (refused range, partial
    // not openable) must not leave that response downloading unobserved.
    controller.abort()
    await handle?.close()
  }
}

/** Caller owns private storage and its lifetime. Redirects cannot change authority.
 *  `<file>.partial` outlives a failed call, so the next call into the same
 *  directory resumes it; only the final sha256 makes the bytes usable. */
export async function downloadPinnedArtifact(
  directory: string,
  artifact: PinnedArtifact,
  policy: PinnedDownloadPolicy = PINNED_DOWNLOAD_POLICY,
  onBytes?: PinnedDownloadProgress
): Promise<string> {
  const file: string = path.join(directory, `destination.${artifact.format}`)

  try {
    await verifyPinnedArtifact(file, artifact)

    return file
  } catch (error) {
    if (!(error instanceof Error) || !('code' in error) || error.code !== 'ENOENT') {
      throw error
    }
  }

  const partial: Partial = { file: `${file}.partial`, validatorFile: `${file}.partial-validator` }
  let size: number = await partialSize(partial.file)
  // Progress is a new high-water mark: a server that ignores ranges restarts
  // from zero, and re-downloading bytes already reached once is not progress.
  let furthest: number = size
  let idleAttempts = 0

  while (size !== artifact.size) {
    try {
      size = await transfer(artifact, partial, size, policy, onBytes)

      if (size !== artifact.size) {
        throw new Error('Pinned artifact download ended early')
      }
    } catch (error) {
      size = await partialSize(partial.file)
      idleAttempts = size > furthest ? 0 : idleAttempts + 1
      furthest = Math.max(furthest, size)

      if (error instanceof PinnedDownloadRefusedError || idleAttempts >= policy.attemptsWithoutProgress) {
        throw error
      }
    }
  }

  try {
    await verifyPinnedArtifact(partial.file, artifact)
  } catch (error) {
    // Complete but wrong bytes never become right by resuming them.
    await unlink(partial.file).catch(ignoreMissing)

    throw error
  }

  await rename(partial.file, file)
  await unlink(partial.validatorFile).catch(ignoreMissing)

  return file
}
