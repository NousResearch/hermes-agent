import crypto from 'node:crypto'
import fs from 'node:fs'
import path from 'node:path'

/**
 * Persist a large plain-text paste as a `.txt` file the composer can attach
 * as a chip instead of flooding the input. The renderer never chooses the
 * path: the file lands in a Desktop-managed directory with a generated name,
 * mirroring how `writeComposerImage` handles pasted images.
 */
export async function writeComposerPaste(userDataDir: string, text: string): Promise<string> {
  const dir = path.join(userDataDir, 'composer-pastes')
  await fs.promises.mkdir(dir, { recursive: true })
  const stamp = new Date().toISOString().replace(/[:.]/g, '-').replace('T', '_').replace('Z', '')
  const random = crypto.randomBytes(3).toString('hex')
  const filePath = path.join(dir, `pasted_content_${stamp}_${random}.txt`)
  await fs.promises.writeFile(filePath, text, 'utf8')

  return filePath
}
