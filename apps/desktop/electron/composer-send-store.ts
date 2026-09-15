// Persistence for the composer's send behaviour (Settings → Keyboards).
//
// Deliberately electron-free so it can be exercised against a temp directory:
// the IPC wrapper in composer-send-ipc.ts supplies the userData path.
import fs from 'node:fs'
import path from 'node:path'

import { type ComposerSendPrefs, normalizeComposerSendPrefs } from '../../shared/src/composer-send'

export const COMPOSER_SEND_CONFIG_FILENAME = 'composer-send.json'

/** Never throws: a missing, truncated, or hand-mangled file falls back to the
 *  defaults, because a bad preference must not take the composer with it. */
export function readComposerSendPrefs(filePath: string): ComposerSendPrefs {
  try {
    return normalizeComposerSendPrefs(JSON.parse(fs.readFileSync(filePath, 'utf8')))
  } catch {
    return normalizeComposerSendPrefs(null)
  }
}

/** Clamps before writing, so a hand-edit or a buggy caller can never persist a
 *  value the settings control would refuse. Returns what actually landed. */
export function writeComposerSendPrefs(prefs: unknown, filePath: string): ComposerSendPrefs {
  const next = normalizeComposerSendPrefs(prefs)

  try {
    fs.mkdirSync(path.dirname(filePath), { recursive: true })
    fs.writeFileSync(filePath, `${JSON.stringify(next, null, 2)}\n`, 'utf8')
  } catch (error) {
    // The in-memory value still applies for this session; only persistence is
    // lost, so carry on rather than failing the settings write.
    console.warn(`[composer-send] write failed: ${(error as Error).message}`)
  }

  return next
}
