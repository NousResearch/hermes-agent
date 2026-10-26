import crypto from 'node:crypto'
import fs from 'node:fs'
import path from 'node:path'

import { formatCreateTime, lockHolderIsLiveSync, ownCreateTimeSync, processCreateTimeSync } from './update-marker'

const INSTALLATION_ID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i

function parseInstallationId(raw) {
  try {
    const value = JSON.parse(String(raw || ''))?.installationId

    return INSTALLATION_ID_RE.test(value) ? value.toLowerCase() : ''
  } catch {
    return ''
  }
}

function readInstallationId(filePath) {
  try {
    const stat = fs.lstatSync(filePath)

    if (!stat.isFile() || stat.isSymbolicLink()) {
      return ''
    }

    if (typeof process.getuid === 'function' && stat.uid !== process.getuid()) {
      return ''
    }

    if (process.platform !== 'win32' && (stat.mode & 0o777) !== 0o600) {
      fs.chmodSync(filePath, 0o600)
    }

    return parseInstallationId(fs.readFileSync(filePath, 'utf8'))
  } catch {
    return ''
  }
}

function waitForRepair() {
  const buffer = new SharedArrayBuffer(4)
  Atomics.wait(new Int32Array(buffer), 0, 0, 25)
}

/**
 * The repair lock names its holder (`<pid>\nct:<creation time>\n`), so a lock
 * left by a crashed launch is reclaimed by owner liveness instead of making
 * every later launch throw at module init (desktop V20). An EMPTY lock is a
 * pre-fix holder (or one between create and write): reclaimed only after it
 * stayed empty for many polls, since it can name no owner to probe.
 */
const EMPTY_LOCK_RECLAIM_POLLS = 20

type RepairLockHolderJudge = (pid: number, recordedCt: number | null, writtenAtS: number) => boolean

/**
 * The repair lock holder's liveness for ONE wait: liveness is re-read every
 * 25 ms poll, but each pid's creation time is probed once — on macOS a `ps`,
 * on Windows a PowerShell of up to 15 s, both blocking the main thread at
 * module init. Scoped to the wait like `cachedCreateTimeProbe`.
 */
function repairLockHolderJudge(): RepairLockHolderJudge {
  const createTimes = new Map<number, number | null>()

  const createTime = (pid: number) => {
    if (!createTimes.has(pid)) {
      createTimes.set(pid, processCreateTimeSync(pid))
    }

    return createTimes.get(pid) ?? null
  }

  return (pid, recordedCt, writtenAtS) => lockHolderIsLiveSync(pid, recordedCt, writtenAtS, createTime)
}

function repairLockBody(): string {
  const ct = ownCreateTimeSync()

  return `${process.pid}\n${ct === null ? '' : `ct:${formatCreateTime(ct)}\n`}`
}

function reclaimDeadRepairLock(repairPath: string, emptyPolls: number, holderIsLive: RepairLockHolderJudge): boolean {
  let raw: Buffer
  let writtenAtS: number

  try {
    raw = fs.readFileSync(repairPath)
    writtenAtS = fs.statSync(repairPath).mtimeMs / 1000
  } catch {
    return false
  }

  const lines = raw.toString('utf8').split('\n')
  const pid = /^\d+$/.test(lines[0] || '') ? Number(lines[0]) : null
  const ctMatch = /^ct:(\d+(?:\.\d+)?)$/.exec(lines[1] || '')

  const dead =
    pid === null
      ? emptyPolls >= EMPTY_LOCK_RECLAIM_POLLS
      : !holderIsLive(pid, ctMatch ? Number(ctMatch[1]) : null, writtenAtS)

  if (!dead) {
    return false
  }

  // Rename-then-verify: a read/compare/unlink by name could delete the fresh
  // lock a peer created after reclaiming the same dead one. The rename is the
  // atomic claim (one reclaimer wins it); the moved file is unlinked only if it
  // still holds the bytes judged dead, else it goes back under the name.
  const claim = `${repairPath}.${process.pid}.${crypto.randomBytes(4).toString('hex')}.reclaim`

  try {
    fs.renameSync(repairPath, claim)
  } catch {
    return false
  }

  let moved: Buffer | null = null

  try {
    moved = fs.readFileSync(claim)
  } catch {
    moved = null
  }

  if (moved?.equals(raw)) {
    dropReclaimClaim(claim)

    return true
  }

  try {
    fs.linkSync(claim, repairPath) // never over a lock created meanwhile
  } catch (error: any) {
    if (error?.code !== 'EEXIST') {
      // No hard links on this filesystem: move it back (EEXIST = a newer lock
      // already holds the name, and its holder proceeds).
      try {
        fs.renameSync(claim, repairPath)
      } catch {
        void 0
      }
    }
  }

  dropReclaimClaim(claim)

  return false
}

function dropReclaimClaim(claim: string) {
  try {
    fs.rmSync(claim, { force: true })
  } catch {
    void 0 // a leftover `.reclaim` sibling is inert: nothing reads it
  }
}

/** Count consecutive empty-lock polls, then reclaim a dead lock or back off. */
function waitOnContendedRepairLock(
  repairPath: string,
  emptyPolls: number,
  holderIsLive: RepairLockHolderJudge
): number {
  let polls: number

  try {
    polls = fs.statSync(repairPath).size === 0 ? emptyPolls + 1 : 0
  } catch {
    polls = 0
  }

  if (!reclaimDeadRepairLock(repairPath, polls, holderIsLive)) {
    waitForRepair()
  }

  return polls
}

/** Replace a corrupt/unreadable ID file (never following another user's file) with a fresh ID. */
function replaceInstallationIdFile(filePath: string, installationId: string): void {
  try {
    const stat = fs.lstatSync(filePath)

    if (!stat.isFile() && !stat.isSymbolicLink()) {
      throw new Error('Desktop installation ID path is not a regular file.')
    }

    if (!stat.isSymbolicLink() && typeof process.getuid === 'function' && stat.uid !== process.getuid()) {
      throw new Error('Desktop installation ID is owned by another user.')
    }

    fs.unlinkSync(filePath)
  } catch (error: any) {
    if (error?.code !== 'ENOENT') {
      throw error
    }
  }

  fs.writeFileSync(filePath, JSON.stringify({ installationId }), { encoding: 'utf8', flag: 'wx', mode: 0o600 })
}

function releaseRepairLock(repairFd: number | undefined, repairPath: string, ownedBody: Buffer | null): void {
  if (repairFd !== undefined) {
    fs.closeSync(repairFd)
  }

  try {
    // Compare-and-delete: never unlink a lock a reclaimer handed to someone else.
    if (ownedBody && fs.readFileSync(repairPath).equals(ownedBody)) {
      fs.unlinkSync(repairPath)
    }
  } catch {
    void 0
  }
}

function loadOrCreateInstallationId(filePath, randomUUID = crypto.randomUUID) {
  const existing = readInstallationId(filePath)

  if (existing) {
    return existing
  }

  fs.mkdirSync(path.dirname(filePath), { recursive: true })
  const installationId = randomUUID().toLowerCase()

  if (!INSTALLATION_ID_RE.test(installationId)) {
    throw new Error('Could not generate a valid desktop installation ID.')
  }

  const repairPath = `${filePath}.repair.lock`

  const holderIsLive = repairLockHolderJudge()
  let emptyPolls = 0
  let ownedBody: Buffer | null = null

  for (let attempt = 0; attempt < 40; attempt++) {
    let repairFd

    try {
      repairFd = fs.openSync(repairPath, 'wx', 0o600)
    } catch (error: any) {
      if (error?.code !== 'EEXIST') {
        throw error
      }

      const winner = readInstallationId(filePath)

      if (winner) {
        return winner
      }

      emptyPolls = waitOnContendedRepairLock(repairPath, emptyPolls, holderIsLive)

      continue
    }

    try {
      ownedBody = Buffer.from(repairLockBody(), 'utf8')
      fs.writeSync(repairFd, ownedBody)

      const winner = readInstallationId(filePath)

      if (winner) {
        return winner
      }

      replaceInstallationIdFile(filePath, installationId)

      return installationId
    } finally {
      releaseRepairLock(repairFd, repairPath, ownedBody)
    }
  }

  throw new Error('Could not repair the desktop installation ID.')
}

function sshOwnershipId(installationId, scope) {
  if (!INSTALLATION_ID_RE.test(String(installationId || ''))) {
    throw new Error('Desktop installation ID is invalid.')
  }

  return crypto
    .createHash('sha256')
    .update(`${installationId}\0${String(scope || '')}`)
    .digest('hex')
    .slice(0, 32)
}

export { INSTALLATION_ID_RE, loadOrCreateInstallationId, parseInstallationId, readInstallationId, sshOwnershipId }
