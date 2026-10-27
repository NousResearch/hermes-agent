// Machine-level UI preferences (#145213): one JSON document beside the ROOT
// Hermes home, shared by native Desktop (this module, over the typed preload
// capability) and every browser that opens this machine's Webapp (the Python
// twin `hermes_cli/ui_preferences.py`, over `/api/ui-preferences`). Not per
// profile, not per gateway connection: a Desktop connected to a remote gateway
// still reads and writes ITS OWN machine's file.
//
// On-disk contract, kept in lockstep with the Python writer:
//   { "version": 1, "generation": <str>, "revision": <int>, "values": { <key>: <string> } }
// A write merges per key under the `<store>.lock` directory lock (mkdir is
// atomic on every OS and in Python), bumps `revision`, and lands by temp file +
// rename. A null value deletes. A `seed` write (a client's first contact with a
// generation) only adds the keys the store lacks. A write that finds no
// readable store (missing, or corrupt and kept aside as `<store>.corrupt-<ms>`)
// starts a new generation, which every window rejoins. Only a missing file
// reads as no store: any other read failure refuses the write.
import { randomBytes } from 'node:crypto'
import fs from 'node:fs'
import path from 'node:path'

export const UI_PREFERENCES_FILE = 'ui-preferences.json'
const KEY_PREFIXES = ['hermes.desktop.', 'hermes-desktop-']
const MAX_KEY_CHARS = 256
const MAX_VALUE_CHARS = 256 * 1024
const MAX_KEYS = 512
const MAX_TOTAL_CHARS = 2 * 1024 * 1024
const LOCK_WAIT_MS = 3000
const LOCK_STALE_MS = 10_000
const LOCK_POLL_MS = 20
const WATCH_INTERVAL_MS = 1500

export interface UiPreferencesSnapshot {
  exists: boolean
  /** One store's lifetime; a replaced (deleted or corrupt) store gets a new one. */
  generation?: string
  revision: number
  values: Record<string, string>
}

export interface UiPreferencesWrite {
  seed?: boolean
  values: Record<string, null | string>
}

export const uiPreferencesPath = (hermesHome: string) => path.join(hermesHome, UI_PREFERENCES_FILE)

/** The store's directory for main's root home, through symlinks, as the Python
 *  writer's `store_path()` resolves it: a symlinked root or profile directory
 *  must not split the machine into two stores. */
export function uiPreferencesRoot(hermesHome: string): string {
  try {
    return fs.realpathSync.native(hermesHome)
  } catch {
    return path.resolve(hermesHome)
  }
}

const sanitize = (values: unknown): Record<string, string> =>
  values && typeof values === 'object' && !Array.isArray(values)
    ? Object.fromEntries(
        Object.entries(values).filter((entry): entry is [string, string] => typeof entry[1] === 'string')
      )
    : {}

interface StoreRead {
  corrupt: boolean
  snapshot: UiPreferencesSnapshot
}

function readStore(file: string): StoreRead {
  let raw: string

  try {
    raw = fs.readFileSync(file, 'utf8')
  } catch (error) {
    // Only a missing file is no store. EACCES, EBUSY, EMFILE… say nothing about
    // the content, and a write merging over them would drop every key.
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') {
      return { corrupt: false, snapshot: { exists: false, revision: 0, values: {} } }
    }

    throw new Error(`preferences store is unreadable: ${(error as Error).message}`)
  }

  let doc: unknown = null

  try {
    doc = JSON.parse(raw.replace(/^\uFEFF/, '')) as unknown
  } catch {
    // Corrupt: no store, so clients re-import into a fresh generation.
  }

  if (!doc || typeof doc !== 'object' || Array.isArray(doc)) {
    return { corrupt: true, snapshot: { exists: false, revision: 0, values: {} } }
  }

  const { generation, revision, values } = doc as { generation?: unknown; revision?: unknown; values?: unknown }

  return {
    corrupt: false,
    snapshot: {
      exists: true,
      generation: typeof generation === 'string' ? generation : '',
      revision: Number.isInteger(revision) ? (revision as number) : 0,
      values: sanitize(values)
    }
  }
}

/** Synchronous so the preload can hand the first paint a hydrated snapshot.
 *  Throws when the file exists but cannot be read. */
export function readUiPreferences(file: string): UiPreferencesSnapshot {
  return readStore(file).snapshot
}

/** For the paths that must never throw (the preload's first paint, the
 *  watcher): an unreadable store is simply not known yet. */
function tryReadUiPreferences(file: string): null | UiPreferencesSnapshot {
  try {
    return readUiPreferences(file)
  } catch {
    return null
  }
}

function mergeValues(current: Record<string, string>, write: UiPreferencesWrite): Record<string, string> {
  const values = { ...current }

  for (const [key, value] of Object.entries(write.values)) {
    if (write.seed) {
      // A client's first contact: fill what the store lacks, never overwrite or delete.
      if (value !== null && !(key in values)) {
        values[key] = value
      }
    } else if (value === null) {
      delete values[key]
    } else {
      values[key] = value
    }
  }

  return values
}

/** The IPC payload, validated: anything else from a renderer is refused whole. */
export function parseUiPreferencesWrite(payload: unknown): UiPreferencesWrite {
  const { seed, values } = (payload && typeof payload === 'object' ? payload : {}) as {
    seed?: unknown
    values?: unknown
  }

  if (!values || typeof values !== 'object' || Array.isArray(values)) {
    throw new Error('ui preferences write needs a values object')
  }

  for (const [key, value] of Object.entries(values)) {
    if (!KEY_PREFIXES.some(prefix => key.startsWith(prefix)) || key.length > MAX_KEY_CHARS) {
      throw new Error(`unsupported preference key: ${key}`)
    }

    if (value !== null && (typeof value !== 'string' || value.length > MAX_VALUE_CHARS)) {
      throw new Error(`unsupported value for ${key}`)
    }
  }

  return { seed: seed === true, values: values as Record<string, null | string> }
}

const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms))

const ownerFile = (lockDir: string) => path.join(lockDir, 'owner')

async function owns(lockDir: string, token: string): Promise<boolean> {
  return fs.promises.readFile(ownerFile(lockDir), 'utf8').then(
    owner => owner === token,
    () => false
  )
}

async function breakIfStale(lockDir: string): Promise<void> {
  const mtime = await fs.promises.stat(lockDir).then(
    stat => stat.mtimeMs,
    () => null
  )

  // A writer that died holding the lock must not wedge every later write.
  // Writes hold it for milliseconds; the data file is only ever replaced whole.
  if (mtime !== null && Date.now() - mtime > LOCK_STALE_MS) {
    await fs.promises.rm(lockDir, { force: true, recursive: true }).catch(() => undefined)
  }
}

/** Cross-process lock shared with the Python writer: the `<store>.lock`
 *  directory, whose `owner` file names the holder. `run` gets a check that
 *  the lock is still ours, for right before the file is replaced. */
async function withStoreLock<T>(file: string, run: (stillOurs: () => Promise<boolean>) => Promise<T>): Promise<T> {
  const lockDir = `${file}.lock`
  const token = `${process.pid}-${randomBytes(8).toString('hex')}`
  const deadline = Date.now() + LOCK_WAIT_MS

  for (;;) {
    try {
      await fs.promises.mkdir(lockDir)

      break
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== 'EEXIST') {
        throw error
      }

      await breakIfStale(lockDir)

      if (Date.now() > deadline) {
        throw new Error('preferences store is busy')
      }

      await sleep(LOCK_POLL_MS)
    }
  }

  try {
    await fs.promises.writeFile(ownerFile(lockDir), token, 'utf8')

    return await run(() => owns(lockDir, token))
  } finally {
    // Only while still ours: a lock broken as stale and re-taken by another
    // writer must not be removed by the writer that lost it.
    if (await owns(lockDir, token)) {
      await fs.promises.rm(lockDir, { force: true, recursive: true }).catch(() => undefined)
    }
  }
}

async function replaceAtomically(file: string, body: string): Promise<void> {
  const temp = path.join(path.dirname(file), `.${path.basename(file)}.${process.pid}.${Date.now()}.tmp`)

  try {
    await fs.promises.writeFile(temp, body, { encoding: 'utf8', mode: 0o600 })
    await fs.promises.rename(temp, file)
  } catch (error) {
    await fs.promises.rm(temp, { force: true }).catch(() => undefined)

    throw error
  }
}

// In-process writes queue behind each other; the directory lock covers other processes.
let writeChain: Promise<unknown> = Promise.resolve()

export function writeUiPreferences(file: string, write: UiPreferencesWrite): Promise<UiPreferencesSnapshot> {
  const next = writeChain.then(async () => {
    await fs.promises.mkdir(path.dirname(file), { recursive: true })

    return withStoreLock(file, async stillOurs => {
      const { corrupt, snapshot: current } = readStore(file)
      const values = mergeValues(current.values, write)

      const entries = Object.entries(values)

      if (
        entries.length > MAX_KEYS ||
        entries.reduce((sum, [k, v]) => sum + k.length + v.length, 0) > MAX_TOTAL_CHARS
      ) {
        throw new Error('preferences store is full')
      }

      if (JSON.stringify(values) === JSON.stringify(current.values) && (current.exists || entries.length === 0)) {
        return current
      }

      const revision = current.revision + 1
      const generation = current.generation || randomBytes(8).toString('hex')

      if (!(await stillOurs())) {
        // Held past the stale limit and broken by another writer: replacing the
        // file now could drop the keys that writer merged meanwhile.
        throw new Error('preferences store is busy')
      }

      if (corrupt) {
        // Kept aside, never silently replaced: it may hold preferences worth recovering.
        await fs.promises.copyFile(file, `${file}.corrupt-${Date.now()}`)
      }

      await replaceAtomically(file, `${JSON.stringify({ version: 1, generation, revision, values }, null, 2)}\n`)

      return { exists: true, generation, revision, values }
    })
  })

  writeChain = next.catch(() => undefined)

  return next
}

interface IpcMainLike {
  handle(channel: string, listener: (event: unknown, ...args: unknown[]) => unknown): void
  on(channel: string, listener: (event: { returnValue: unknown }, ...args: unknown[]) => void): void
}

export interface UiPreferencesIpcDeps {
  ipcMain: IpcMainLike
  hermesHome: string
  /** Every app window's webContents; changes are pushed to all of them. */
  broadcast: (snapshot: UiPreferencesSnapshot) => void
}

/**
 * `hermes:ui-preferences:*` — the renderer's only door to the store: a
 * synchronous first-paint snapshot, a read, and a validated per-key write. A
 * write made here, or by the Webapp's Python process on this machine (seen by
 * a cheap stat poll), is pushed to every window so open ones follow live.
 */
export function registerUiPreferencesIpc({ ipcMain, hermesHome, broadcast }: UiPreferencesIpcDeps): () => void {
  const file = uiPreferencesPath(uiPreferencesRoot(hermesHome))

  const stamp = (snapshot: null | UiPreferencesSnapshot) =>
    snapshot ? `${snapshot.generation ?? ''}:${snapshot.revision}:${snapshot.exists}` : null

  let last = stamp(tryReadUiPreferences(file))

  const publish = (snapshot: null | UiPreferencesSnapshot) => {
    const next = stamp(snapshot)

    if (snapshot && next !== last) {
      last = next
      broadcast(snapshot)
    }
  }

  ipcMain.on('hermes:ui-preferences:snapshot', event => {
    event.returnValue = tryReadUiPreferences(file)
  })
  ipcMain.handle('hermes:ui-preferences:read', () => readUiPreferences(file))
  ipcMain.handle('hermes:ui-preferences:write', async (_event, payload) => {
    const snapshot = await writeUiPreferences(file, parseUiPreferencesWrite(payload))

    publish(snapshot)

    return snapshot
  })

  const onStat = () => publish(tryReadUiPreferences(file))

  fs.watchFile(file, { interval: WATCH_INTERVAL_MS, persistent: false }, onStat)

  return () => fs.unwatchFile(file, onStat)
}
