import { createHash } from 'node:crypto'
import { createReadStream } from 'node:fs'
import { lstat, open, rename, unlink } from 'node:fs/promises'
import type { FileHandle } from 'node:fs/promises'
import path from 'node:path'

export interface PinnedArtifact {
  url: string
  sha256: string
  size: number
  format: 'zip' | 'msix' | 'msixbundle'
}

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

/** Caller owns private storage and its lifetime. Redirects cannot change authority. */
export async function downloadPinnedArtifact(directory: string, artifact: PinnedArtifact): Promise<string> {
  const file: string = path.join(directory, `destination.${artifact.format}`)

  try {
    await verifyPinnedArtifact(file, artifact)

    return file
  } catch (error) {
    if (!(error instanceof Error) || !('code' in error) || error.code !== 'ENOENT') {
      throw error
    }
  }

  const temporary: string = `${file}.partial`
  await unlink(temporary).catch((error: NodeJS.ErrnoException): void => {
    if (error.code !== 'ENOENT') {
      throw error
    }
  })
  const response: Response = await fetch(artifact.url, { redirect: 'error', signal: AbortSignal.timeout(600_000) })

  if (!response.ok || !response.body) {
    throw new Error(`Pinned artifact download failed (${response.status})`)
  }

  const handle: FileHandle = await open(temporary, 'wx', 0o600)
  let size: number = 0

  try {
    for await (const chunk of response.body) {
      size += chunk.byteLength

      if (size > artifact.size) {
        throw new Error('Pinned artifact download exceeds pinned size')
      }

      await handle.writeFile(chunk)
    }

    await handle.sync()
  } finally {
    await handle.close()
  }

  await verifyPinnedArtifact(temporary, artifact)
  await rename(temporary, file)

  return file
}
