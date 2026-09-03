import { execFileSync } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'

export const STATE_DB_EMERGENCY_BACKUP_LIMIT = 2
// A physical copy must not leave the updater itself starved for working space.
// macOS `cp -c` can silently make one when cloning is unavailable, so it is
// guarded too; forced reflinks on other platforms remain safe to try first.
export const STATE_DB_BACKUP_FREE_RESERVE_BYTES = 10 * 1024 * 1024 * 1024

const BACKUP_PREFIX = 'state.db.pre-update-emergency-'
const BACKUP_SUFFIX = '.bak'
const MIN_SQLITE_DB_BYTES = 100
const SQLITE_HEADER = Buffer.from('SQLite format 3\0')

type Log = (message: string) => void

type StateDbPreflightDeps = {
  closeSync?: (fd: number) => void
  cloneFile?: ((source: string, destination: string) => void) | null
  copyFileSync?: (source: string, destination: string) => void
  existsSync?: (file: string) => boolean
  now?: () => Date
  openSync?: (file: string, flags: string) => number
  readSync?: (fd: number, buffer: Buffer, offset: number, length: number, position: number) => number
  readdirSync?: (directory: string) => string[]
  statfsSync?: (directory: string) => { bavail: bigint | number; bsize: bigint | number }
  statSync?: (file: string) => { size: number }
  unlinkSync?: (file: string) => void
}

type StateDbPreflightResult =
  | { status: 'not-found' | 'too-small'; path: string }
  | { status: 'created'; path: string; method: 'clone' | 'clone-or-physical' | 'physical' }
  | { status: 'failed' | 'skipped-insufficient-space' | 'skipped-unverified-space'; path: string }

function defaultCloneFile() {
  if (process.platform === 'darwin') {
    return (source: string, destination: string) => {
      execFileSync('/bin/cp', ['-c', source, destination], {
        stdio: 'ignore',
        timeout: 30_000
      })
    }
  }

  if (process.platform === 'win32') {
    return null
  }

  return (source: string, destination: string) => {
    fs.copyFileSync(source, destination, fs.constants.COPYFILE_FICLONE_FORCE)
  }
}

function defaultDeps(): Required<StateDbPreflightDeps> {
  return {
    closeSync: fs.closeSync,
    cloneFile: defaultCloneFile(),
    copyFileSync: fs.copyFileSync,
    existsSync: fs.existsSync,
    now: () => new Date(),
    openSync: fs.openSync,
    readSync: fs.readSync,
    readdirSync: directory => fs.readdirSync(directory, { encoding: 'utf8' }),
    statfsSync: fs.statfsSync,
    statSync: fs.statSync,
    unlinkSync: fs.unlinkSync
  }
}

function errorMessage(error: unknown) {
  return error instanceof Error ? error.message : String(error)
}

function availableBytes(stat: { bavail: bigint | number; bsize: bigint | number }) {
  const value = BigInt(stat.bavail) * BigInt(stat.bsize)

  return value > BigInt(Number.MAX_SAFE_INTEGER) ? Number.MAX_SAFE_INTEGER : Number(value)
}

function removeIfPresent(file: string, io: Required<StateDbPreflightDeps>, rememberLog: Log) {
  if (!io.existsSync(file)) {
    return
  }

  try {
    io.unlinkSync(file)
  } catch (error) {
    rememberLog(`[updates] could not remove partial emergency state.db backup: ${errorMessage(error)}`)
  }
}

function validateCompletedSnapshot(file: string, io: Required<StateDbPreflightDeps>) {
  const size = io.statSync(file).size

  if (size <= MIN_SQLITE_DB_BYTES) {
    throw new Error(`snapshot too small: ${size} bytes`)
  }

  const fd = io.openSync(file, 'r')
  const header = Buffer.alloc(SQLITE_HEADER.length)

  try {
    io.readSync(fd, header, 0, header.length, 0)
  } finally {
    io.closeSync(fd)
  }

  if (!header.equals(SQLITE_HEADER)) {
    throw new Error(`snapshot has invalid SQLite header: ${header.toString('hex')}`)
  }

  return size
}

function guardPhysicalAllocation(
  hermesHome: string,
  stateDbPath: string,
  emergencyPath: string,
  io: Required<StateDbPreflightDeps>,
  rememberLog: Log
): StateDbPreflightResult | null {
  let freeBytes: number

  try {
    freeBytes = availableBytes(io.statfsSync(hermesHome))
  } catch (error) {
    rememberLog(
      `[updates] emergency state.db physical backup skipped: free space could not be verified (${errorMessage(error)})`
    )

    return { status: 'skipped-unverified-space', path: emergencyPath }
  }

  let sourceSize: number

  try {
    sourceSize = io.statSync(stateDbPath).size
  } catch (error) {
    rememberLog(`[updates] could not refresh state.db size before physical backup: ${errorMessage(error)}`)

    return { status: 'failed', path: emergencyPath }
  }

  const requiredBytes = sourceSize + STATE_DB_BACKUP_FREE_RESERVE_BYTES

  if (freeBytes < requiredBytes) {
    rememberLog(
      `[updates] emergency state.db physical backup skipped: ` +
        `free=${freeBytes}, required=${requiredBytes}, reserve=${STATE_DB_BACKUP_FREE_RESERVE_BYTES}`
    )

    return { status: 'skipped-insufficient-space', path: emergencyPath }
  }

  return null
}

function pruneBeforeBackup(hermesHome: string, io: Required<StateDbPreflightDeps>, rememberLog: Log) {
  try {
    const backups = io
      .readdirSync(hermesHome)
      .filter(file => file.startsWith(BACKUP_PREFIX) && file.endsWith(BACKUP_SUFFIX))
      .sort()
      .reverse()

    for (const old of backups.slice(STATE_DB_EMERGENCY_BACKUP_LIMIT - 1)) {
      try {
        io.unlinkSync(path.join(hermesHome, old))
      } catch (error) {
        rememberLog(`[updates] could not prune stale emergency state.db backup: ${errorMessage(error)}`)
      }
    }
  } catch (error) {
    rememberLog(`[updates] could not list emergency state.db backups: ${errorMessage(error)}`)
  }
}

export function preflightStateDb(
  hermesHome: string,
  rememberLog: Log,
  overrides: StateDbPreflightDeps = {}
): StateDbPreflightResult {
  const io = { ...defaultDeps(), ...overrides }
  const stateDbPath = path.join(hermesHome, 'state.db')

  if (!io.existsSync(stateDbPath)) {
    rememberLog('[updates] state.db pre-flight: not found (fresh install?)')

    return { status: 'not-found', path: stateDbPath }
  }

  let sourceSize: number

  try {
    sourceSize = io.statSync(stateDbPath).size
  } catch (error) {
    rememberLog(`[updates] could not stat state.db before update: ${errorMessage(error)}`)

    return { status: 'failed', path: stateDbPath }
  }

  if (sourceSize <= MIN_SQLITE_DB_BYTES) {
    rememberLog(`[updates] state.db too small (${sourceSize} bytes) for a valid SQLite database`)

    return { status: 'too-small', path: stateDbPath }
  }

  try {
    const fd = io.openSync(stateDbPath, 'r')
    const header = Buffer.alloc(SQLITE_HEADER.length)

    try {
      io.readSync(fd, header, 0, header.length, 0)
    } finally {
      io.closeSync(fd)
    }

    const headerOk = header.equals(SQLITE_HEADER)

    rememberLog(
      `[updates] state.db pre-flight: size=${sourceSize}, ` +
        `headerOk=${headerOk}, headerHex=${header.toString('hex')}`
    )

    if (!headerOk) {
      rememberLog(
        '[updates] state.db header is INVALID before update — ' +
          'this indicates pre-existing corruption or a concurrent write issue'
      )
    }
  } catch (error) {
    rememberLog(`[updates] could not read state.db before update: ${errorMessage(error)}`)
  }

  const ts = io.now().toISOString().replace(/[:.]/g, '-')
  const emergencyPath = path.join(hermesHome, `${BACKUP_PREFIX}${ts}${BACKUP_SUFFIX}`)

  pruneBeforeBackup(hermesHome, io, rememberLog)

  if (io.cloneFile !== null) {
    if (process.platform === 'darwin') {
      const guardResult = guardPhysicalAllocation(hermesHome, stateDbPath, emergencyPath, io, rememberLog)

      if (guardResult !== null) {
        return guardResult
      }
    }

    try {
      io.cloneFile(stateDbPath, emergencyPath)
      const backupSize = validateCompletedSnapshot(emergencyPath, io)
      const method = process.platform === 'darwin' ? 'clone-or-physical' : 'clone'

      rememberLog(`[updates] emergency state.db backup: ${emergencyPath} (${backupSize} bytes, ${method})`)

      return { status: 'created', path: emergencyPath, method }
    } catch (error) {
      removeIfPresent(emergencyPath, io, rememberLog)
      rememberLog(`[updates] emergency state.db clone unavailable: ${errorMessage(error)}`)
    }
  }

  const guardResult = guardPhysicalAllocation(hermesHome, stateDbPath, emergencyPath, io, rememberLog)

  if (guardResult !== null) {
    return guardResult
  }

  try {
    io.copyFileSync(stateDbPath, emergencyPath)
    const backupSize = validateCompletedSnapshot(emergencyPath, io)

    rememberLog(`[updates] emergency state.db backup: ${emergencyPath} (${backupSize} bytes, physical)`)

    return { status: 'created', path: emergencyPath, method: 'physical' }
  } catch (error) {
    removeIfPresent(emergencyPath, io, rememberLog)
    rememberLog(`[updates] emergency state.db backup failed: ${errorMessage(error)}`)

    return { status: 'failed', path: emergencyPath }
  }
}
