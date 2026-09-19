import * as fs from 'node:fs/promises'
import * as path from 'node:path'

/** Download the descriptor before teardown. Windows verifies the referenced package. */
export async function stageAppInstallerFile(
  url: string,
  directory: string,
  fetchFile: typeof fetch = fetch
): Promise<string> {
  const response = await fetchFile(url, { signal: AbortSignal.timeout(30_000) })

  if (!response.ok || !response.body) {
    throw new Error(`App Installer descriptor download failed: HTTP ${response.status}`)
  }

  const chunks: Uint8Array[] = []
  let size = 0
  const reader = response.body.getReader()

  try {
    for (;;) {
      const { value, done } = await reader.read()

      if (done) { break }
      size += value.byteLength

      if (size > 1024 * 1024) {
        throw new Error('App Installer descriptor exceeds 1 MiB')
      }

      chunks.push(value)
    }
  } finally {
    await reader.cancel()
  }

  if (size === 0) { throw new Error('App Installer descriptor is empty') }
  await fs.mkdir(directory, { recursive: true })
  const target = path.join(directory, 'update.appinstaller')
  const temporary = `${target}.tmp`

  try {
    await fs.writeFile(temporary, Buffer.concat(chunks))
    await fs.rename(temporary, target)
  } finally {
    await fs.rm(temporary, { force: true })
  }

  return target
}
