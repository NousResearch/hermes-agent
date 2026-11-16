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
    let raw: string

    try {
      raw = fs.readFileSync(file, 'utf8')
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === 'ENOENT') {return {}}
      throw error
    }

    try {
      const parsed: unknown = JSON.parse(raw)

      if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {return parsed as Record<string, unknown>}
    } catch { /* quarantined below */ }

    // A torn/garbled journal (crash mid-write on a filesystem without atomic rename, disk
    // corruption) is unrecoverable as JSON. Throwing here blocked every later send of the origin,
    // since each send must journal first. Move it aside with its bytes intact for diagnosis; its
    // uncertain entries are lost to automatic retry, which only ever was an explicit user action.
    fs.renameSync(file, `${file}.corrupt-${Date.now()}`)
    console.warn(`[prepared-submissions] quarantined unreadable journal ${path.basename(file)}`)

    return {}
  }

  const write = (journal: Record<string, unknown>, key: string, entry: unknown | null) => {
    if (entry === null) {delete journal[key]}
    else {Object.defineProperty(journal, key, { value: entry, enumerable: true, configurable: true })}

    fs.mkdirSync(userData, { recursive: true })
    // Same private atomic replacement used for native connection settings.
    // Return only after write+rename: process termination cannot lose an ACKed
    // entry to Chromium's deferred localStorage commit. Not a power-loss promise.
    writeSecretFileAtomic(file, JSON.stringify(journal), { encoding: 'utf8' })
  }

  return {
    read,
    update(key: string, entry: unknown | null) {
      write(read(), key, entry)
    },
    /** Replace `key` only while it still holds exactly `expected` (null = absent): create-if-absent
     *  and compare-and-delete in one synchronous main-process step, so windows sharing the origin's
     *  journal cannot interleave between the check and the write. Returns the record now stored. */
    compareAndSet(key: string, expected: unknown | null, entry: unknown | null): { applied: boolean; current: unknown | null } {
      const journal = read()
      const current = Object.hasOwn(journal, key) ? journal[key] : null

      if (JSON.stringify(current) !== JSON.stringify(expected)) {return { applied: false, current }}
      write(journal, key, entry)

      return { applied: true, current: entry }
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
  ipcMain.handle('hermes:prepared-submissions:compare-and-set', (event, key: string, expected: string | null, entry: string | null) => {
    if (typeof key !== 'string' || [expected, entry].some(value => value !== null && typeof value !== 'string')) {
      throw new Error('Invalid prepared submission comparison')
    }

    const parse = (value: string | null) => value === null ? null : JSON.parse(value)
    const { applied, current } = store(event).compareAndSet(key, parse(expected), parse(entry))

    return { applied, current: current === null ? null : JSON.stringify(current) }
  })
}
