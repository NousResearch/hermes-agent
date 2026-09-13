import { createHash } from 'node:crypto'
import fs from 'node:fs'
import path from 'node:path'

import { app, ipcMain } from 'electron'

import { writeSecretFileAtomic } from './hardening'

// Match localStorage's origin isolation; destination keys additionally carry
// connection/profile/session authority. No renderer-provided filesystem paths.
export function preparedJournal(userData: string, origin: string) {
  const file = path.join(userData, `prepared-submissions-${createHash('sha256').update(origin).digest('hex')}.json`)

  const read = (): Record<string, unknown> => {
    try {
      return JSON.parse(fs.readFileSync(file, 'utf8'))
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === 'ENOENT') {return {}}
      throw error
    }
  }

  return {
    read,
    update(key: string, entry: unknown | null) {
      const journal = read()

      if (entry === null) {delete journal[key]}
      else {Object.defineProperty(journal, key, { value: entry, enumerable: true, configurable: true })}

      fs.mkdirSync(userData, { recursive: true })
      // Same private atomic replacement used for native connection settings.
      // Return only after write+rename: process termination cannot lose an ACKed
      // entry to Chromium's deferred localStorage commit. Not a power-loss promise.
      writeSecretFileAtomic(file, JSON.stringify(journal), { encoding: 'utf8' })
    }
  }
}

export function registerPreparedSubmissions() {
  const store = (event: Electron.IpcMainInvokeEvent) =>
    preparedJournal(app.getPath('userData'), new URL(event.senderFrame!.url).origin)

  ipcMain.handle('hermes:prepared-submissions:read', event => JSON.stringify(store(event).read()))
  ipcMain.handle('hermes:prepared-submissions:update', (event, key: string, entry: string | null) => {
    if (typeof key !== 'string' || (entry !== null && typeof entry !== 'string')) {
      throw new Error('Invalid prepared submission')
    }

    store(event).update(key, entry === null ? null : JSON.parse(entry))
  })
}
