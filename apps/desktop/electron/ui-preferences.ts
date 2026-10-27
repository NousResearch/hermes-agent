// Machine-level UI preferences (#145213): one JSON document beside the ROOT
// Hermes home, shared by native Desktop (this module, over the typed preload
// capability) and every browser that opens this machine's Webapp (the Python
// twin `hermes_cli/ui_preferences.py`, over `/api/ui-preferences`). Not per
// profile, not per gateway connection: a Desktop connected to a remote gateway
// still reads and writes ITS OWN machine's file.
//
// On-disk contract, kept in lockstep with the Python writer:
//   { "version": 1, "revision": <int>, "values": { <key>: <string> } }
// A write merges per key under the `<store>.lock` directory lock (mkdir is
// atomic on every OS and in Python), bumps `revision`, and lands by temp file +
// rename. A null value deletes; a `seed` write applies only with no store yet.
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
  revision: number
  values: Record<string, string>
}

export interface UiPreferencesWrite {
  seed?: boolean
  values: Record<string, null | string>
}

export const uiPreferencesPath = (hermesHome: string) => path.join(hermesHome, UI_PREFERENCES_FILE)

const sanitize = (values: unknown): Record<string, string> =>
  values && typeof values === 'object' && !Array.isArray(values)
    ? Object.fromEntries(
        Object.entries(values).filter((entry): entry is [string, string] => typeof entry[1] === 'string')
      )
    : {}

/** Synchronous so the preload can hand the first paint a hydrated snapshot. */
export function readUiPreferences(file: string): UiPreferencesSnapshot {
  let raw: string

  try {
    raw = fs.readFileSync(file, 'utf8')
  } catch {
    return { exists: false, revision: 0, values: {} }
  }

  let doc: { revision?: unknown; values?: unknown } = {}

  try {
    const parsed = JSON.parse(raw.replace(/^\uFEFF/, '')) as unknown

    doc = parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? parsed : {}
  } catch {
    // A corrupt file reads as an empty existing store; the next write replaces it whole.
  }

  return {
    exists: true,
    revision: Number.isInteger(doc.revision) ? (doc.revision as number) : 0,
    values: sanitize(doc.values)
  }
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
      const current = readUiPreferences(file)

      if (write.seed && current.exists) {
        return current
      }

      const values = { ...current.values }

      for (const [key, value] of Object.entries(write.values)) {
        if (value === null) {
          delete values[key]
        } else {
          values[key] = value
        }
      }

      const entries = Object.entries(values)

      if (
        entries.length > MAX_KEYS ||
        entries.reduce((sum, [k, v]) => sum + k.length + v.length, 0) > MAX_TOTAL_CHARS
      ) {
        throw new Error('preferences store is full')
      }

      if (current.exists && JSON.stringify(values) === JSON.stringify(current.values)) {
        return current
      }

      const revision = current.revision + 1

      if (!(await stillOurs())) {
        // Held past the stale limit and broken by another writer: replacing the
        // file now could drop the keys that writer merged meanwhile.
        throw new Error('preferences store is busy')
      }

      await replaceAtomically(file, `${JSON.stringify({ version: 1, revision, values }, null, 2)}\n`)

      return { exists: true, revision, values }
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
  const file = uiPreferencesPath(hermesHome)
  let lastRevision = readUiPreferences(file).revision

  const publish = (snapshot: UiPreferencesSnapshot) => {
    if (snapshot.revision !== lastRevision) {
      lastRevision = snapshot.revision
      broadcast(snapshot)
    }
  }

  ipcMain.on('hermes:ui-preferences:snapshot', event => {
    event.returnValue = readUiPreferences(file)
  })
  ipcMain.handle('hermes:ui-preferences:read', () => readUiPreferences(file))
  ipcMain.handle('hermes:ui-preferences:write', async (_event, payload) => {
    const snapshot = await writeUiPreferences(file, parseUiPreferencesWrite(payload))

    publish(snapshot)

    return snapshot
  })

  const onStat = () => publish(readUiPreferences(file))

  fs.watchFile(file, { interval: WATCH_INTERVAL_MS, persistent: false }, onStat)

  return () => fs.unwatchFile(file, onStat)
}
