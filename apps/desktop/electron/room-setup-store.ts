/** Private, per-obligation custody under the existing native storage policy. An unreadable record is never an empty journal. */
import fs from 'node:fs/promises'
import path from 'node:path'

import { writeSecretFileAtomic } from './hardening'
import type { SetupRoute } from './room-setup-types'

export class RoomSetupError extends Error {
  constructor(readonly reason: string) {super(reason)}
}
export interface SetupRecord {
  id: string; setupId: string; kind: 'home' | 'peer' | 'custody'; route: SetupRoute; installationId: string
  roomId: string; committed?: boolean; creation?: Record<string, unknown>; invitation?: Record<string, unknown>; grant?: string
  /** A backup computer's obligation also names the host that may hold its grant. */
  home?: SetupRoute; homeInstallationId?: string
}
const ID = /^[0-9a-f-]{36}$/

export function roomSetupStore(options: {
  directory: string
  encrypt: (text: string) => string; decrypt: (sealed: string) => string
}) {
  const file = (id: string) => {
    if (!ID.test(id)) {throw new RoomSetupError('invalid_setup_record')}
    return path.join(options.directory, `${id}.json`)
  }
  const syncDirectory = async (directory: string) => {
    // Node cannot open a directory handle on Windows. Match the existing
    // desktop-boot-preference writer; the file itself is flushed on every OS.
    if (process.platform === 'win32') {return}
    const handle = await fs.open(directory, 'r')
    try {await handle.sync()} finally {await handle.close()}
  }
  const privateDirectory = async (create = false) => {
    let created = false
    if (create) {
      try {await fs.mkdir(options.directory, { mode: 0o700 }); created = true} catch (error) {
        if ((error as NodeJS.ErrnoException).code !== 'EEXIST') {throw error}
      }
    }
    const stat = await fs.lstat(options.directory)
    if (!stat.isDirectory() || stat.isSymbolicLink() ||
        (process.getuid && (stat.uid !== process.getuid() || (stat.mode & 0o077) !== 0))) {
      throw new RoomSetupError('setup_journal_unreadable')
    }
    // The new directory entry must survive too, not just its first file.
    if (created) {await syncDirectory(path.dirname(options.directory))}
  }
  const decode = (plaintext: string, id: string): SetupRecord => {
    const result = JSON.parse(plaintext)
    if (result?.id !== id || !ID.test(result.setupId) || !['home', 'peer', 'custody'].includes(result.kind) ||
        !result.route?.connectionId || !result.route.profile || !result.installationId || !result.roomId ||
        (result.kind === 'custody' && (!result.home?.connectionId || !result.home.profile || !result.homeInstallationId))) {throw new Error()}
    return result
  }
  const get = async (id: string): Promise<SetupRecord> => {
    try {
      await privateDirectory()
      const stat = await fs.lstat(file(id))
      if (!stat.isFile() || stat.isSymbolicLink() || stat.size > 65536 ||
          (process.getuid && (stat.uid !== process.getuid() || (stat.mode & 0o077) !== 0))) {throw new Error()}
      return decode(options.decrypt(await fs.readFile(file(id), 'utf8')), id)
    } catch {throw new RoomSetupError('setup_journal_unreadable')}
  }

  return {
    get,
    async list() {
      let names: string[]
      try {await privateDirectory(); names = await fs.readdir(options.directory)} catch (error) {
        if ((error as NodeJS.ErrnoException).code === 'ENOENT') {return { records: [], unreadable: [] as string[] }}
        throw new RoomSetupError('setup_journal_unreadable')
      }
      const records: SetupRecord[] = [], unreadable: string[] = []
      for (const name of names.filter(name => name.endsWith('.json'))) {
        const id = name.slice(0, -5)
        try {records.push(await get(id))} catch {unreadable.push(id)}
      }
      return { records, unreadable }
    },
    async put(record: SetupRecord) {
      const destination = file(record.id)
      try {await privateDirectory(true)} catch {throw new RoomSetupError('setup_journal_unreadable')}
      const serialized = JSON.stringify(record)
      const sealed = options.encrypt(serialized)
      if (Buffer.byteLength(sealed) > 65536) {throw new RoomSetupError('setup_journal_full')}
      try {
        writeSecretFileAtomic(destination, sealed, { encoding: 'utf8', durable: {
          verify: bytes => {
            // Read and decrypt the actual staged bytes BEFORE replacing a valid
            // issuance intent. Wrong seals and readback faults cannot erase it.
            const plaintext = options.decrypt(bytes.toString('utf8'))
            if (plaintext !== serialized) {throw new Error()}
            decode(plaintext, record.id)
          }
        } })
      } catch {throw new RoomSetupError('setup_journal_write_failed')}
    },
    async remove(id: string) {
      await privateDirectory()
      const destination = file(id)
      // Only after an owner receipt: remove abandoned stages before the live
      // obligation, so a crash never leaves a credential without its journal.
      for (const name of await fs.readdir(options.directory)) {
        if (name.startsWith(`${id}.json.`) && name.endsWith('.tmp')) {await fs.unlink(path.join(options.directory, name))}
      }
      await fs.unlink(destination)
      await syncDirectory(options.directory)
    }
  }
}
