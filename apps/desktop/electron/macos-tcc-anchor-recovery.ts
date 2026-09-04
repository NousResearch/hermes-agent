import { randomUUID } from 'node:crypto'
import fs from 'node:fs'
import path from 'node:path'

type RecoveryResult =
  | { status: 'not-applicable'; reason: 'platform' | 'marker-absent' | 'python-not-anchored' }
  | {
      status: 'failed'
      reason:
        | 'invalid-marker'
        | 'unsafe-source'
        | 'source-probe-failed'
        | 'venv-probe-failed'
        | 'filesystem-error'
    }
  | { status: 'recovered' }

interface RecoveryOptions {
  platform?: NodeJS.Platform | string
  venvRoot: string
  probePython: (pythonPath: string) => boolean
}

interface Replacement {
  destination: string
  target: string
  staged: string
  backup: string
  backupCreated: boolean
}

const MARKER_NAME = '.tcc-anchor-source'
const MAX_MARKER_BYTES = 4096
const PYTHON_ALIAS = /^python3(?:\.\d+)?$/
const MACH_O_MAGICS = new Set(['feedface', 'feedfacf', 'cefaedfe', 'cffaedfe', 'cafebabe', 'bebafeca', 'cafebabf', 'bfbafeca'])

function isInside(parent: string, candidate: string): boolean {
  const relative = path.relative(parent, candidate)

  return relative === '' || (!relative.startsWith(`..${path.sep}`) && relative !== '..' && !path.isAbsolute(relative))
}

function probeSafely(probePython: (pythonPath: string) => boolean, pythonPath: string): boolean {
  try {
    return probePython(pythonPath) === true
  } catch {
    return false
  }
}

function hasMachOHeader(filePath: string): boolean {
  const file = fs.openSync(filePath, 'r')

  try {
    const header = Buffer.alloc(4)

    return fs.readSync(file, header, 0, header.length, 0) === header.length && MACH_O_MAGICS.has(header.toString('hex'))
  } finally {
    fs.closeSync(file)
  }
}

function cleanupReplacementFiles(replacements: Replacement[], removeBackups = false): void {
  for (const replacement of replacements) {
    const candidates = [replacement.staged]

    if (removeBackups || !replacement.backupCreated) {
      candidates.push(replacement.backup)
    }

    for (const candidate of candidates) {
      try {
        fs.unlinkSync(candidate)
      } catch {
        // These names are unique to this recovery attempt. Leaving one
        // behind is safer than disturbing a working interpreter.
      }
    }
  }
}

function rollback(replacements: Replacement[]): void {
  for (const replacement of [...replacements].reverse()) {
    try {
      if (replacement.backupCreated) {
        fs.renameSync(replacement.backup, replacement.destination)
        replacement.backupCreated = false
      }
    } catch {
      // Keep an unrestored backup in place for manual recovery. Never throw
      // from the early boot path or delete the only remaining original.
    }
  }

  cleanupReplacementFiles(replacements)
}

/**
 * Repair venvs damaged by the reverted macOS TCC interpreter anchor before
 * Desktop asks Python to import anything. The marker and real-file interpreter
 * are both legacy artifacts from #95131/#95478; ordinary venvs are untouched.
 *
 * Every replacement is staged in venv/bin and can be rolled back until the
 * restored interpreter passes the caller's real Hermes import probe. The
 * marker is removed only after that verification succeeds.
 */
function recoverLegacyMacosTccAnchor(options: RecoveryOptions): RecoveryResult {
  const platform = options.platform ?? process.platform

  if (platform !== 'darwin') {
    return { status: 'not-applicable', reason: 'platform' }
  }

  const venvRoot = options.venvRoot
  const bin = path.join(venvRoot, 'bin')
  const marker = path.join(bin, MARKER_NAME)
  const python = path.join(bin, 'python')

  let markerStat: fs.Stats

  try {
    markerStat = fs.lstatSync(marker)
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') {
      return { status: 'not-applicable', reason: 'marker-absent' }
    }

    return { status: 'failed', reason: 'filesystem-error' }
  }

  if (!path.isAbsolute(venvRoot) || !markerStat.isFile() || markerStat.size <= 0 || markerStat.size > MAX_MARKER_BYTES) {
    return { status: 'failed', reason: 'invalid-marker' }
  }

  let pythonStat: fs.Stats

  try {
    pythonStat = fs.lstatSync(python)
  } catch {
    return { status: 'failed', reason: 'filesystem-error' }
  }

  if (!pythonStat.isSymbolicLink() && !pythonStat.isFile()) {
    return { status: 'not-applicable', reason: 'python-not-anchored' }
  }

  let source: string

  try {
    source = fs.readFileSync(marker, 'utf8')
  } catch {
    return { status: 'failed', reason: 'filesystem-error' }
  }

  if (source !== source.trim() || source.includes('\0') || source.includes('\n') || source.includes('\r') || !path.isAbsolute(source)) {
    return { status: 'failed', reason: 'invalid-marker' }
  }

  try {
    const sourceStat = fs.statSync(source)
    const realVenv = fs.realpathSync(venvRoot)
    const realSource = fs.realpathSync(source)

    if (!sourceStat.isFile() || isInside(realVenv, realSource) || !hasMachOHeader(source)) {
      return { status: 'failed', reason: 'unsafe-source' }
    }

    fs.accessSync(source, fs.constants.X_OK)
    source = realSource
  } catch {
    return { status: 'failed', reason: 'unsafe-source' }
  }

  if (!probeSafely(options.probePython, source)) {
    return { status: 'failed', reason: 'source-probe-failed' }
  }

  let pythonAlreadyRestored = false

  if (pythonStat.isSymbolicLink()) {
    try {
      pythonAlreadyRestored = fs.realpathSync(python) === source
    } catch {
      return { status: 'not-applicable', reason: 'python-not-anchored' }
    }

    if (!pythonAlreadyRestored) {
      return { status: 'not-applicable', reason: 'python-not-anchored' }
    }
  }

  const attempt = `${process.pid}-${randomUUID()}`
  let aliases: string[]

  try {
    aliases = fs
      .readdirSync(bin)
      .filter(name => PYTHON_ALIAS.test(name))
      .map(name => path.join(bin, name))
      .filter(alias => {
        const stat = fs.lstatSync(alias)

        return stat.isFile() || stat.isSymbolicLink()
      })
  } catch {
    return { status: 'failed', reason: 'filesystem-error' }
  }

  const replacements: Replacement[] = [
    ...(pythonAlreadyRestored
      ? []
      : [{ destination: python, target: source, staged: '', backup: '', backupCreated: false }]),
    ...aliases.map(alias => ({
      destination: alias,
      target: 'python',
      staged: '',
      backup: '',
      backupCreated: false
    }))
  ]

  for (const replacement of replacements) {
    const name = path.basename(replacement.destination)
    replacement.staged = path.join(bin, `.${name}.tcc-recovery-${attempt}.new`)
    replacement.backup = path.join(bin, `.${name}.tcc-recovery-${attempt}.old`)
  }

  try {
    for (const replacement of replacements) {
      fs.symlinkSync(replacement.target, replacement.staged)
    }

    for (const replacement of replacements) {
      const destinationStat = fs.lstatSync(replacement.destination)

      if (destinationStat.isSymbolicLink()) {
        fs.symlinkSync(fs.readlinkSync(replacement.destination), replacement.backup)
      } else {
        fs.linkSync(replacement.destination, replacement.backup)
      }

      replacement.backupCreated = true
      fs.renameSync(replacement.staged, replacement.destination)
    }

    if (!probeSafely(options.probePython, python)) {
      rollback(replacements)

      return { status: 'failed', reason: 'venv-probe-failed' }
    }

    fs.unlinkSync(marker)
  } catch {
    rollback(replacements)

    return { status: 'failed', reason: 'filesystem-error' }
  }

  cleanupReplacementFiles(replacements, true)

  return { status: 'recovered' }
}

export { recoverLegacyMacosTccAnchor, type RecoveryOptions, type RecoveryResult }
