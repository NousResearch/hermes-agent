import crypto from 'node:crypto'
import fs from 'node:fs'
import path from 'node:path'

const writesByPath = new Map<string, Promise<void>>()

export function sha256Bytes(value: Buffer | string): string {
  return crypto.createHash('sha256').update(value).digest('hex')
}

async function readCurrent(pathname: string): Promise<Buffer> {
  try {
    const stat = await fs.promises.stat(pathname)

    if (!stat.isFile()) {
      throw new Error('Only regular files can be written')
    }

    return await fs.promises.readFile(pathname)
  } catch (error) {
    if ((error as NodeJS.ErrnoException)?.code === 'ENOENT') {
      return Buffer.alloc(0)
    }

    throw error
  }
}

async function writeLocked(
  pathname: string,
  content: string,
  expectedHash?: string
): Promise<{ contentHash: string; path: string }> {
  const assertUnchanged = async () => {
    if (expectedHash && sha256Bytes(await readCurrent(pathname)) !== expectedHash) {
      throw new Error('FILE_CHANGED')
    }
  }

  await assertUnchanged()

  const contentHash = sha256Bytes(content)

  const temporary = path.join(
    path.dirname(pathname),
    `.${path.basename(pathname)}.hermes-tmp-${process.pid}-${crypto.randomUUID()}`
  )

  await fs.promises.writeFile(temporary, content, 'utf8')

  try {
    // Re-check after staging. Together with the per-path queue this closes the
    // lost-update window between concurrent Desktop saves and catches an
    // ordinary external edit that landed while the temp file was written.
    await assertUnchanged()
    await fs.promises.rename(temporary, pathname)
  } catch (error) {
    await fs.promises.rm(temporary, { force: true }).catch(() => undefined)
    throw error
  }

  return { contentHash, path: pathname }
}

/**
 * Optimistic compare-and-replace for the Desktop text editor.
 *
 * Calls targeting the same path are serialized in-process. The replacement
 * itself is atomic, and a caller that supplies the hash returned by read-text
 * receives FILE_CHANGED instead of silently overwriting a newer revision.
 */
export async function writeTextFileCas(
  pathname: string,
  content: string,
  expectedHash?: string
): Promise<{ contentHash: string; path: string }> {
  const previous = writesByPath.get(pathname) ?? Promise.resolve()
  let release!: () => void

  const current = new Promise<void>(resolve => {
    release = resolve
  })

  const chain = previous.catch(() => undefined).then(() => current)

  writesByPath.set(pathname, chain)
  await previous.catch(() => undefined)

  try {
    return await writeLocked(pathname, content, expectedHash)
  } finally {
    release()

    if (writesByPath.get(pathname) === chain) {
      writesByPath.delete(pathname)
    }
  }
}
