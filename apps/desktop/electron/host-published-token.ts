// Session token published by a Hermes backend for same-user attach.
//
// `GET /` withholds `window.__HERMES_SESSION_TOKEN__` when the dashboard is
// auth-gated. The backend still writes the live token next to its host
// rendezvous record (`gateway/host_rendezvous.py`): `host-serve.token` for the
// machine owner, `host-desktop-serve.token` for a Desktop-spawned child. The
// post-update relaunch has to adopt that token instead of logging "did not
// publish a session token" and spawning a second backend.

import { createHash } from 'node:crypto'
import path from 'node:path'

import type { HostBackendRecord } from './backend-discovery'

const HOST_PROTOCOL_VERSION = 1
const PUBLISHED_ROLES = ['serve', 'desktop-serve'] as const

export interface PublishedHostToken {
  recordText: string
  role: string
  token: string
}

export interface HostTokenStat {
  isDirectory: () => boolean
  isFile: () => boolean
  isSymbolicLink: () => boolean
  mode: number
  uid: number
}

export interface HostTokenIo {
  lstat: (target: string) => HostTokenStat
  readFile: (target: string) => string
  uid: number | null
}

export interface HostRendezvousEnv {
  home: string
  lockDir?: string
  platform: string
  stateHome?: string
}

export function hostRendezvousDirectory(env: HostRendezvousEnv): string {
  const override = String(env.lockDir || '').trim()

  if (override) {
    return path.resolve(override)
  }

  const stateHomeEnv = String(env.stateHome || '').trim()

  const stateHome =
    stateHomeEnv && path.isAbsolute(stateHomeEnv) ? stateHomeEnv : path.join(env.home, '.local', 'state')

  return path.join(stateHome, 'hermes', 'gateway-locks')
}

function tokenFingerprint(token: string): string {
  return createHash('sha256').update(token, 'utf8').digest('hex').slice(0, 16)
}

function isPrivateEntry(stat: HostTokenStat, platform: string, uid: number | null): boolean {
  if (stat.isSymbolicLink()) {
    return false
  }

  if (platform === 'win32') {
    return true
  }

  if ((stat.mode & 0o077) !== 0) {
    return false
  }

  return uid === null || stat.uid === uid
}

function tokenForPublication(record: HostBackendRecord, publication: PublishedHostToken): string | null {
  if (!PUBLISHED_ROLES.includes(publication.role as (typeof PUBLISHED_ROLES)[number])) {
    return null
  }

  let parsed: unknown

  try {
    parsed = JSON.parse(publication.recordText)
  } catch {
    return null
  }

  if (!parsed || typeof parsed !== 'object') {
    return null
  }

  const payload = parsed as Record<string, unknown>
  const token = publication.token.trim()
  const fingerprint = String(payload.tokenFingerprint ?? '')

  if (
    payload.role !== publication.role ||
    payload.protocolVersion !== HOST_PROTOCOL_VERSION ||
    payload.pid !== record.pid ||
    payload.port !== record.port ||
    !token ||
    !/^[0-9a-f]{16}$/.test(fingerprint) ||
    tokenFingerprint(token) !== fingerprint
  ) {
    return null
  }

  return token
}

/** The published token that belongs to this ledger record, or null. */
export function publishedTokenForRecord(record: HostBackendRecord, publications: PublishedHostToken[]): string | null {
  for (const publication of publications) {
    const token = tokenForPublication(record, publication)

    if (token) {
      return token
    }
  }

  return null
}

/** Read owner-only host-serve and host-desktop-serve token pairs. Never throws. */
export function readPublishedHostTokens(directory: string, io: HostTokenIo, platform: string): PublishedHostToken[] {
  try {
    const directoryStat = io.lstat(directory)

    if (!directoryStat.isDirectory() || !isPrivateEntry(directoryStat, platform, io.uid)) {
      return []
    }
  } catch {
    return []
  }

  const publications: PublishedHostToken[] = []

  for (const role of PUBLISHED_ROLES) {
    const recordPath = path.join(directory, `host-${role}.json`)
    const tokenPath = path.join(directory, `host-${role}.token`)

    try {
      const recordStat = io.lstat(recordPath)
      const tokenStat = io.lstat(tokenPath)

      if (
        !recordStat.isFile() ||
        !tokenStat.isFile() ||
        !isPrivateEntry(recordStat, platform, io.uid) ||
        !isPrivateEntry(tokenStat, platform, io.uid)
      ) {
        continue
      }

      publications.push({
        recordText: io.readFile(recordPath),
        role,
        token: io.readFile(tokenPath)
      })
    } catch {
      continue
    }
  }

  return publications
}

export function lookupPublishedSessionToken(
  record: HostBackendRecord,
  env: HostRendezvousEnv,
  io: HostTokenIo
): string | null {
  return publishedTokenForRecord(record, readPublishedHostTokens(hostRendezvousDirectory(env), io, env.platform))
}
