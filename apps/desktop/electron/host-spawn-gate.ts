import crypto from 'node:crypto'
import fs from 'node:fs'

interface HostSpawnGateClaim {
  claim?: string
  isOwnerAlive?: (pid: number) => boolean
  now?: number
  pid?: number
  staleAfterMs?: number
  startedAt?: number
}

interface HostSpawnGateRecord {
  claim: string
  pid: number
  startedAt: number
}

function ownerIsAlive(pid: number): boolean {
  try {
    process.kill(pid, 0)

    return true
  } catch {
    return false
  }
}

function parseRecord(contents: string): HostSpawnGateRecord | null {
  try {
    const record = JSON.parse(contents)

    if (
      typeof record?.claim !== 'string' ||
      !record.claim ||
      !Number.isInteger(record.pid) ||
      record.pid <= 0 ||
      !Number.isFinite(record.startedAt)
    ) {
      return null
    }

    return record
  } catch {
    return null
  }
}

function createClaim(gatePath: string, record: HostSpawnGateRecord): boolean {
  try {
    fs.writeFileSync(gatePath, JSON.stringify(record), {
      flag: 'wx',
      mode: 0o600
    })

    return true
  } catch {
    return false
  }
}

/**
 * Atomically claim the host-level backend spawn gate.
 *
 * Exclusive creation makes one contender win when several Desktop processes
 * start together. The release checks its opaque claim before unlinking, so a
 * delayed cleanup cannot remove a later process's gate.
 */
export function claimHostSpawnGate(
  gatePath: string,
  {
    pid = process.pid,
    startedAt = Date.now(),
    claim = crypto.randomUUID(),
    now = Date.now(),
    staleAfterMs,
    isOwnerAlive = ownerIsAlive
  }: HostSpawnGateClaim = {}
): (() => void) | null {
  const record = { claim, pid, startedAt }

  if (!createClaim(gatePath, record)) {
    if (staleAfterMs === undefined) {
      return null
    }

    let staleContents: string

    try {
      staleContents = fs.readFileSync(gatePath, 'utf8')
    } catch {
      return null
    }

    const staleRecord = parseRecord(staleContents)

    const reclaimable =
      staleRecord === null || !isOwnerAlive(staleRecord.pid) || now - staleRecord.startedAt >= staleAfterMs

    if (!reclaimable) {
      return null
    }

    try {
      // Re-read immediately before unlinking so a replacement claim is not
      // removed using a decision made about the old owner.
      if (fs.readFileSync(gatePath, 'utf8') !== staleContents) {
        return null
      }

      fs.unlinkSync(gatePath)
    } catch {
      return null
    }

    if (!createClaim(gatePath, record)) {
      return null
    }
  }

  return () => {
    try {
      const record = JSON.parse(fs.readFileSync(gatePath, 'utf8'))

      if (record?.claim === claim) {
        fs.unlinkSync(gatePath)
      }
    } catch {
      // Already gone, unreadable, or replaced.
    }
  }
}
