/** Private, per-obligation custody under the existing native storage policy. An unreadable record is never an empty journal. */
import fs from 'node:fs/promises'
import path from 'node:path'

export class RoomSetupError extends Error {
  constructor(readonly reason: string) {super(reason)}
}
export interface SetupRoute { connectionId: string; profile: string }
export interface SetupRecord {
  id: string; setupId: string; kind: 'home' | 'peer'; route: SetupRoute; installationId: string
  roomId: string; committed?: boolean; creation?: Record<string, unknown>; invitation?: Record<string, unknown>; grant?: string
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
  const get = async (id: string): Promise<SetupRecord> => {
    try {
      const stat = await fs.lstat(file(id))
      if (!stat.isFile() || stat.isSymbolicLink() || stat.size > 65536 ||
          (process.getuid && (stat.uid !== process.getuid() || (stat.mode & 0o077) !== 0))) {throw new Error()}
      const result = JSON.parse(options.decrypt(await fs.readFile(file(id), 'utf8')))
      if (result?.id !== id || !ID.test(result.setupId) || !['home', 'peer'].includes(result.kind) ||
          !result.route?.connectionId || !result.route.profile || !result.installationId || !result.roomId) {throw new Error()}
      return result
    } catch {throw new RoomSetupError('setup_journal_unreadable')}
  }

  return {
    get,
    async list() {
      let names: string[]
      try {names = await fs.readdir(options.directory)} catch (error) {
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
      const serialized = JSON.stringify(record)
      const sealed = options.encrypt(serialized)
      if (Buffer.byteLength(sealed) > 65536) {throw new RoomSetupError('setup_journal_full')}
      await fs.mkdir(options.directory, { recursive: true, mode: 0o700 })
      const temporary = `${destination}.${crypto.randomUUID()}.tmp`
      try {
        await fs.writeFile(temporary, sealed, { mode: 0o600, flag: 'wx' })
        await fs.rename(temporary, destination)
        if (JSON.stringify(await get(record.id)) !== serialized) {throw new Error()}
      } catch {throw new RoomSetupError('setup_journal_write_failed')}
      finally {await fs.rm(temporary, { force: true }).catch(() => undefined)}
    },
    async remove(id: string) {
      const destination = file(id)
      // Only after an owner receipt: remove abandoned stages before the live
      // obligation, so a crash never leaves a credential without its journal.
      for (const name of await fs.readdir(options.directory)) {
        if (name.startsWith(`${id}.json.`) && name.endsWith('.tmp')) {await fs.unlink(path.join(options.directory, name))}
      }
      await fs.unlink(destination)
    }
  }
}
