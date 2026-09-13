import { createHash, randomUUID } from 'node:crypto'
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
      const value: unknown = JSON.parse(fs.readFileSync(file, 'utf8'))

      if (!value || typeof value !== 'object' || Array.isArray(value)) {throw new Error('Invalid prepared submission journal')}

      return value as Record<string, unknown>
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === 'ENOENT') {return {}}
      throw error
    }
  }

  const write = (journal: Record<string, unknown>) => {
    fs.mkdirSync(userData, { recursive: true })
    // Private atomic replacement, not a power-loss durability promise.
    writeSecretFileAtomic(file, JSON.stringify(journal), { encoding: 'utf8' })
  }

  return {
    read,
    update(key: string, entry: unknown | null) {
      const journal = read()

      if (entry === null) {delete journal[key]}
      else {Object.defineProperty(journal, key, { value: entry, enumerable: true, configurable: true })}

      write(journal)
    },
    compareAndSet(key: string, expected: unknown | null, entry: unknown | null): boolean {
      const journal = read()
      const current = Object.hasOwn(journal, key) ? journal[key] : null

      if (JSON.stringify(current) !== JSON.stringify(expected)) {return false}

      if (entry === null) {delete journal[key]}
      else {Object.defineProperty(journal, key, { value: entry, enumerable: true, configurable: true })}

      write(journal)

      return true
    }
  }
}

export function registerPreparedSubmissions() {
  // Stable across reloads of one window, never inherited by another window.
  const owners = new WeakMap<Electron.WebContents, string>()
  ipcMain.handle('hermes:prepared-submissions:owner', event => {
    let owner = owners.get(event.sender)

    if (!owner) {owner = randomUUID(); owners.set(event.sender, owner)}

    return owner
  })

  const store = (event: Electron.IpcMainInvokeEvent) =>
    preparedJournal(app.getPath('userData'), new URL(event.senderFrame!.url).origin)

  ipcMain.handle('hermes:prepared-submissions:read', event => JSON.stringify(store(event).read()))
  ipcMain.handle('hermes:prepared-submissions:update', (event, key: string, entry: string | null) => {
    if (typeof key !== 'string' || (entry !== null && typeof entry !== 'string')) {
      throw new Error('Invalid prepared submission')
    }

    store(event).update(key, entry === null ? null : JSON.parse(entry))
  })
  ipcMain.handle('hermes:prepared-submissions:compare-and-set', (event, key: string, expected: string | null, entry: string | null) => {
    if (typeof key !== 'string' || key.length > 2048 || [expected, entry].some(value => value !== null && (typeof value !== 'string' || value.length > 1024 * 1024))) {
      throw new Error('Invalid prepared submission comparison')
    }

    return store(event).compareAndSet(key, expected === null ? null : JSON.parse(expected), entry === null ? null : JSON.parse(entry))
  })
  // Ordinary Sends retain the previous update API's large frozen payload and
  // legacy-key support without weakening Group creation's bounded CAS API.
  ipcMain.handle('hermes:prepared-submissions:compare-send', (event, key: string, expected: string | null, entry: string | null) => {
    if (typeof key !== 'string' || [expected, entry].some(value => value !== null && typeof value !== 'string')) {
      throw new Error('Invalid prepared submission comparison')
    }

    return store(event).compareAndSet(key, expected === null ? null : JSON.parse(expected), entry === null ? null : JSON.parse(entry))
  })
}
