import { execFile } from 'node:child_process'
import fs from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { promisify } from 'node:util'

const execFileAsync = promisify(execFile)
const heicBrands = new Set(['heic', 'heix', 'hevc', 'hevx', 'heim', 'heis', 'hevm', 'hevs'])

function isHeic(data: Buffer): boolean {
  if (data.length < 16 || data.toString('ascii', 4, 8) !== 'ftyp') {
    return false
  }

  const end = Math.min(data.readUInt32BE(0), data.length)

  for (let offset = 8; offset + 4 <= end; offset += 4) {
    if (offset !== 12 && heicBrands.has(data.toString('ascii', offset, offset + 4))) {
      return true
    }
  }

  return false
}

/** Convert only HEIC previews on macOS, whose browser decoder lacks HEIC support.
 * Work from already-authorized bytes, never reopen a user path after validation.
 * Uploads do not use this path; no source file or persisted attachment is changed.
 */
export async function imagePreviewDataUrl(data: Buffer, mimeType: string): Promise<string> {
  if (process.platform !== 'darwin' || !isHeic(data)) {
    return `data:${mimeType};base64,${data.toString('base64')}`
  }

  const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'hermes-image-preview-'))

  try {
    const input = path.join(dir, 'source.heic')
    const output = path.join(dir, 'preview.jpg')
    await fs.writeFile(input, data, { mode: 0o600 })
    // sips uses ImageIO (orientation-aware), preserves aspect ratio, and avoids
    // Quick Look's square/padded thumbnails. No shell or user-controlled flags.
    await execFileAsync(
      '/usr/bin/sips',
      ['-s', 'format', 'jpeg', '-s', 'formatOptions', '85', '-Z', '2048', input, '--out', output],
      {
        timeout: 30_000,
        maxBuffer: 64 * 1024
      }
    )
    const jpeg = await fs.readFile(output)

    if (jpeg.length < 3 || jpeg[0] !== 0xff || jpeg[1] !== 0xd8 || jpeg[2] !== 0xff) {
      throw new Error('HEIC preview conversion did not produce a JPEG')
    }

    return `data:image/jpeg;base64,${jpeg.toString('base64')}`
  } finally {
    await fs.rm(dir, { recursive: true, force: true })
  }
}
