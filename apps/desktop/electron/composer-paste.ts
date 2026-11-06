import crypto from 'node:crypto'
import fs from 'node:fs'
import path from 'node:path'

export const COMPOSER_PASTES_DIRNAME = 'composer-pastes'

/**
 * Persist a large plain-text paste as a `.txt` file the composer can attach
 * as a chip instead of flooding the input. The renderer never chooses the
 * path: the file lands in a Desktop-managed directory with a generated name,
 * mirroring how `writeComposerImage` handles pasted images.
 *
 * `hermesHome` must be the HERMES_HOME root, not Electron's userData dir: the
 * backend admits `@file:` attachments outside the chat's cwd only from
 * `<HERMES_HOME>/composer-pastes` (agent/context_references.py::_resolve_path),
 * and on Linux/macOS userData is a different tree (#117149).
 */
export async function writeComposerPaste(hermesHome: string, text: string): Promise<string> {
  const dir = path.join(hermesHome, COMPOSER_PASTES_DIRNAME)
  await fs.promises.mkdir(dir, { recursive: true })
  const stamp = new Date().toISOString().replace(/[:.]/g, '-').replace('T', '_').replace('Z', '')
  const random = crypto.randomBytes(3).toString('hex')
  const filePath = path.join(dir, `pasted_content_${stamp}_${random}.txt`)
  await fs.promises.writeFile(filePath, text, 'utf8')

  return filePath
}
