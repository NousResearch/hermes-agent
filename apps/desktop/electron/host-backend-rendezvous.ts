import { createHash } from 'node:crypto'

import type { HostBackendRecord } from './backend-discovery'
import type { HostBackendRendezvous } from './host-backend-attach'

const LOOPBACK_DIALABLE = new Set(['', '0.0.0.0', '127.0.0.1', '::', '::1', 'localhost'])
const HOST_PROTOCOL_VERSION = 1

export interface HostBackendIdentityProbe {
  ok: boolean
  reason?: string
}

function tokenFingerprint(token: string): string {
  return createHash('sha256').update(token, 'utf8').digest('hex').slice(0, 16)
}

export function validateHostBackendIdentity(identity: unknown, record: HostBackendRecord): HostBackendIdentityProbe {
  if (!identity || typeof identity !== 'object') {
    return { ok: false, reason: 'the endpoint returned an invalid identity payload' }
  }

  const payload = identity as Record<string, unknown>

  if (
    payload.ok !== true ||
    payload.protocolVersion !== HOST_PROTOCOL_VERSION ||
    payload.role !== 'serve' ||
    payload.pid !== record.pid ||
    payload.servesSpa !== true
  ) {
    return { ok: false, reason: 'the endpoint did not prove the recorded serve owner' }
  }

  if (record.createTime !== null) {
    const createTime = payload.createTime

    if (typeof createTime !== 'number' || !Number.isFinite(createTime)) {
      return { ok: false, reason: 'the endpoint did not report a valid process creation time' }
    }

    if (Math.abs(createTime - record.createTime) > 0.001) {
      return { ok: false, reason: 'the endpoint reported a different process creation time' }
    }
  }

  return { ok: true }
}

export function rendezvousPortFromRecord(recordContents: unknown): number | null {
  let parsed: unknown

  try {
    parsed = JSON.parse(String(recordContents ?? ''))
  } catch {
    return null
  }

  if (!parsed || typeof parsed !== 'object') {
    return null
  }

  const record = parsed as Record<string, unknown>
  const host = String(record.host ?? '')
  const port = record.port

  if (
    record.role !== 'serve' ||
    record.protocolVersion !== HOST_PROTOCOL_VERSION ||
    !Number.isInteger(port) ||
    Number(port) <= 0 ||
    Number(port) > 65535 ||
    !LOOPBACK_DIALABLE.has(host.toLowerCase())
  ) {
    return null
  }

  return Number(port)
}

/**
 * Parse the private same-user host rendezvous files written by Hermes serve.
 *
 * The record nominates a process and the token file supplies its credential.
 * Neither file is trusted by itself: callers still have to prove readiness and
 * websocket authentication before adopting the backend.
 */
export function parseHostBackendRendezvous(
  recordContents: unknown,
  tokenContents: unknown
): HostBackendRendezvous | null {
  let parsed: unknown

  try {
    parsed = JSON.parse(String(recordContents ?? ''))
  } catch {
    return null
  }

  if (!parsed || typeof parsed !== 'object') {
    return null
  }

  const record = parsed as Record<string, unknown>
  const pid = record.pid
  const port = record.port
  const host = String(record.host ?? '')
  const token = String(tokenContents ?? '').trim()
  const fingerprint = String(record.tokenFingerprint ?? '')
  const rawCreateTime = record.createTime

  if (
    record.role !== 'serve' ||
    record.protocolVersion !== HOST_PROTOCOL_VERSION ||
    !Number.isInteger(pid) ||
    Number(pid) <= 0 ||
    !Number.isInteger(port) ||
    Number(port) <= 0 ||
    Number(port) > 65535 ||
    !LOOPBACK_DIALABLE.has(host.toLowerCase()) ||
    !token ||
    !/^[0-9a-f]{16}$/.test(fingerprint) ||
    tokenFingerprint(token) !== fingerprint ||
    typeof rawCreateTime !== 'number' ||
    !Number.isFinite(rawCreateTime) ||
    rawCreateTime <= 0
  ) {
    return null
  }

  const updatedAt = Date.parse(String(record.updatedAt ?? ''))
  const profiles = Array.isArray(record.profiles) ? record.profiles : []
  const profile = profiles.find(value => typeof value === 'string') || ''

  const normalized: HostBackendRecord = {
    createTime: rawCreateTime,
    host,
    pid: Number(pid),
    port: Number(port),
    profile,
    purpose: 'serve',
    registeredAt: Number.isFinite(updatedAt) ? updatedAt : 0
  }

  return { record: normalized, token }
}
