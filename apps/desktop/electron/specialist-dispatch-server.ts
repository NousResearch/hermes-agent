import crypto from 'node:crypto'
import fs from 'node:fs'
import net from 'node:net'
import path from 'node:path'

import type { SpecialistDispatchAdmissionRequest, SpecialistDispatchAdmissionResult } from './specialist-dispatch-admission'

export const SPECIALIST_DISPATCH_ENVELOPE_SCHEMA = 'AresDesktopSpecialistDispatchEnvelopeV1'
export const SPECIALIST_DISPATCH_ENDPOINT_SCHEMA = 'AresDesktopSpecialistDispatchEndpointV1'
const MAX_FRAME_BYTES = 64 * 1024
const RUN_ID = /^specialist-run-[a-z0-9][a-z0-9-]{7,63}$/
const DIGEST = /^sha256:[0-9a-f]{64}$/
const PROFILE = /^[a-z0-9][a-z0-9_-]{0,63}$/
const QUIESCE_LEASE_ID = /^specialist-quiesce-[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/

type TerminalState = 'running' | 'released' | 'runner_failed' | 'cleanup_failed' | 'unknown'

export interface SpecialistDispatchAdmission {
  admit: (request: SpecialistDispatchAdmissionRequest) => Promise<SpecialistDispatchAdmissionResult>
  hasActive: () => boolean
  poolKey: (runId: string) => string
  release: (runId: string, terminal: Exclude<TerminalState, 'running' | 'unknown'>) => void
  status: (runId: string) => TerminalState
}

export interface SpecialistDispatchServerDeps {
  admission: SpecialistDispatchAdmission
  cancel: (runId: string) => Promise<void>
  quiesce: {
    acquire: (profileIds: string[]) => Promise<{ leaseId: string; profileIds: string[] }>
    release: (leaseId: string) =>
      | { outcome: 'released'; profileIds: string[] }
      | { outcome: 'rejected'; reasonCode: 'UNKNOWN_QUIESCE_LEASE' }
  }
  root: string
}

export interface SpecialistDispatchServer {
  close: () => Promise<void>
  endpointPath: string
  port: number
}

function privateStatePath(root: string): string {
  return path.join(root, 'specialist-dispatch.json')
}

function writePrivateEndpoint(root: string, token: string, port: number): string {
  fs.mkdirSync(root, { mode: 0o700, recursive: true })
  const target = privateStatePath(root)
  const temporary = `${target}.${crypto.randomUUID()}.tmp`
  const value = JSON.stringify({ schema: SPECIALIST_DISPATCH_ENDPOINT_SCHEMA, host: '127.0.0.1', port, token }) + '\n'
  fs.writeFileSync(temporary, value, { encoding: 'utf8', mode: 0o600 })
  fs.chmodSync(temporary, 0o600)
  fs.renameSync(temporary, target)
  return target
}

function sameToken(actual: unknown, expected: string): boolean {
  if (typeof actual !== 'string') {
    return false
  }
  const left = Buffer.from(actual)
  const right = Buffer.from(expected)
  return left.length === right.length && crypto.timingSafeEqual(left, right)
}

function response(socket: net.Socket, value: Record<string, unknown>): void {
  socket.end(`${JSON.stringify(value)}\n`)
}

function invalidEnvelope(): Record<string, unknown> {
  return { outcome: 'rejected', reasonCode: 'INVALID_ENVELOPE' }
}

function isObject(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value)
}

/**
 * Parse bounded untrusted JSON without allowing JSON.parse() to silently
 * collapse duplicate object keys. The transport accepts only one request at a
 * time, so a small recursive scanner is preferable to a permissive parser
 * dependency. JSON.parse remains the grammar oracle after this scan.
 */
function parseStrictJson(raw: Buffer): unknown | null {
  let text: string
  try {
    text = new TextDecoder('utf-8', { fatal: true }).decode(raw)
  } catch {
    return null
  }

  let index = 0
  const whitespace = () => {
    while (/\s/.test(text[index] || '')) {
      index += 1
    }
  }
  const string = (): string => {
    const start = index
    if (text[index] !== '"') {
      throw new Error('expected JSON string')
    }
    index += 1
    while (index < text.length) {
      const char = text[index]
      if (char === '"') {
        index += 1
        return JSON.parse(text.slice(start, index)) as string
      }
      if (char === '\\') {
        index += 1
        if (text[index] === 'u') {
          index += 4
        }
      } else if (char < ' ') {
        throw new Error('unescaped control character')
      }
      index += 1
    }
    throw new Error('unterminated JSON string')
  }
  const value = (): void => {
    whitespace()
    if (text[index] === '{') {
      index += 1
      whitespace()
      const keys = new Set<string>()
      if (text[index] === '}') {
        index += 1
        return
      }
      while (true) {
        whitespace()
        const key = string()
        if (keys.has(key)) {
          throw new Error('duplicate JSON key')
        }
        keys.add(key)
        whitespace()
        if (text[index] !== ':') {
          throw new Error('missing object separator')
        }
        index += 1
        value()
        whitespace()
        if (text[index] === '}') {
          index += 1
          return
        }
        if (text[index] !== ',') {
          throw new Error('missing object delimiter')
        }
        index += 1
      }
    }
    if (text[index] === '[') {
      index += 1
      whitespace()
      if (text[index] === ']') {
        index += 1
        return
      }
      while (true) {
        value()
        whitespace()
        if (text[index] === ']') {
          index += 1
          return
        }
        if (text[index] !== ',') {
          throw new Error('missing array delimiter')
        }
        index += 1
      }
    }
    if (text[index] === '"') {
      string()
      return
    }
    const start = index
    while (index < text.length && !/[\s,}\]]/.test(text[index])) {
      index += 1
    }
    if (start === index) {
      throw new Error('missing JSON value')
    }
  }

  try {
    value()
    whitespace()
    return index === text.length ? JSON.parse(text) : null
  } catch {
    return null
  }
}

function parseSubmit(value: Record<string, unknown>): SpecialistDispatchAdmissionRequest | null {
  if (new Set(Object.keys(value)).size !== 4 || !('schema' in value && 'operation' in value && 'token' in value && 'request' in value)) {
    return null
  }
  const request = value.request
  if (!isObject(request)) {
    return null
  }
  const runId = request.run_id
  const requestDigest = request.request_digest
  const profileIds = request.profile_ids
  if (
    typeof runId !== 'string' || !RUN_ID.test(runId) ||
    typeof requestDigest !== 'string' || !DIGEST.test(requestDigest) ||
    !Array.isArray(profileIds) || profileIds.length < 1 || profileIds.length > 4 ||
    profileIds.some(profile => typeof profile !== 'string' || !PROFILE.test(profile)) ||
    profileIds.some((profile, index) => index > 0 && profileIds[index - 1] >= profile)
  ) {
    return null
  }
  const runnerInput = JSON.stringify(request)
  if (Buffer.byteLength(runnerInput, 'utf8') > MAX_FRAME_BYTES) {
    return null
  }
  return { runId, requestDigest, profileIds: profileIds as string[], runnerInput }
}

function parseRunIdOperation(value: Record<string, unknown>): string | null {
  if (new Set(Object.keys(value)).size !== 4 || !('schema' in value && 'operation' in value && 'token' in value && 'run_id' in value)) {
    return null
  }
  return typeof value.run_id === 'string' && RUN_ID.test(value.run_id) ? value.run_id : null
}

function parseProfileIdsOperation(value: Record<string, unknown>): string[] | null {
  if (new Set(Object.keys(value)).size !== 4 || !('schema' in value && 'operation' in value && 'token' in value && 'profile_ids' in value)) {
    return null
  }
  const profileIds = value.profile_ids

  return Array.isArray(profileIds) &&
    profileIds.length >= 1 &&
    profileIds.length <= 4 &&
    profileIds.every(profileId => typeof profileId === 'string' && PROFILE.test(profileId)) &&
    profileIds.every((profileId, index) => index === 0 || profileIds[index - 1] < profileId)
    ? profileIds as string[]
    : null
}

function parseQuiesceLeaseOperation(value: Record<string, unknown>): string | null {
  if (new Set(Object.keys(value)).size !== 4 || !('schema' in value && 'operation' in value && 'token' in value && 'lease_id' in value)) {
    return null
  }

  return typeof value.lease_id === 'string' && QUIESCE_LEASE_ID.test(value.lease_id) ? value.lease_id : null
}

async function handle(
  raw: Buffer,
  token: string,
  deps: SpecialistDispatchServerDeps
): Promise<Record<string, unknown>> {
  if (raw.length > MAX_FRAME_BYTES) {
    return invalidEnvelope()
  }
  const value = parseStrictJson(raw)
  if (value === null) {
    return invalidEnvelope()
  }
  if (!isObject(value) || value.schema !== SPECIALIST_DISPATCH_ENVELOPE_SCHEMA || !sameToken(value.token, token)) {
    return value && isObject(value) && value.schema === SPECIALIST_DISPATCH_ENVELOPE_SCHEMA
      ? { outcome: 'rejected', reasonCode: 'AUTHENTICATION_FAILED' }
      : invalidEnvelope()
  }
  if (value.operation === 'submit') {
    const request = parseSubmit(value)
    if (!request) {
      return invalidEnvelope()
    }
    try {
      const result = await deps.admission.admit(request)
      return { ...result }
    } catch {
      return { outcome: 'rejected', reasonCode: 'RUNNER_START_FAILED', runId: request.runId }
    }
  }
  if (value.operation === 'status') {
    const runId = parseRunIdOperation(value)
    return runId ? { outcome: 'status', runId, terminalState: deps.admission.status(runId) } : invalidEnvelope()
  }
  if (value.operation === 'cancel') {
    const runId = parseRunIdOperation(value)
    if (!runId) {
      return invalidEnvelope()
    }
    try {
      await deps.cancel(runId)
      return { outcome: 'released', runId }
    } catch {
      return { outcome: 'rejected', reasonCode: 'CANCEL_FAILED', runId }
    }
  }
  if (value.operation === 'quiesce') {
    const profileIds = parseProfileIdsOperation(value)

    if (!profileIds) {
      return invalidEnvelope()
    }
    try {
      const lease = await deps.quiesce.acquire(profileIds)

      return { outcome: 'quiesced', leaseId: lease.leaseId, profileIds: lease.profileIds }
    } catch (error: unknown) {
      const code = error && typeof error === 'object' && 'code' in error ? (error as { code?: unknown }).code : undefined

      return {
        outcome: 'rejected',
        reasonCode: code === 'INVALID_QUIESCE_PROFILE_SET' || code === 'QUIESCE_CONFLICT' || code === 'QUIESCE_STOP_FAILED'
          ? code
          : 'QUIESCE_FAILED'
      }
    }
  }
  if (value.operation === 'unquiesce') {
    const leaseId = parseQuiesceLeaseOperation(value)

    if (!leaseId) {
      return invalidEnvelope()
    }
    const result = deps.quiesce.release(leaseId)

    return result.outcome === 'released'
      ? { outcome: 'unquiesced', leaseId, profileIds: result.profileIds }
      : { outcome: 'rejected', reasonCode: result.reasonCode }
  }
  return invalidEnvelope()
}

/** Start the local-only fixed-command transport; no listener is exposed remotely. */
export async function startSpecialistDispatchServer(deps: SpecialistDispatchServerDeps): Promise<SpecialistDispatchServer> {
  const token = crypto.randomBytes(32).toString('base64url')
  const server = net.createServer(socket => {
    let received = Buffer.alloc(0)
    socket.setTimeout(10_000, () => socket.destroy())
    socket.on('data', chunk => {
      received = Buffer.concat([received, chunk])
      const newline = received.indexOf(0x0a)
      if (received.length > MAX_FRAME_BYTES || newline < 0) {
        if (received.length > MAX_FRAME_BYTES) {
          response(socket, invalidEnvelope())
        }
        return
      }
      const frame = received.subarray(0, newline)
      socket.pause()
      void handle(frame, token, deps).then(value => response(socket, value))
    })
  })
  await new Promise<void>((resolve, reject) => {
    server.once('error', reject)
    server.listen(0, '127.0.0.1', () => {
      server.off('error', reject)
      resolve()
    })
  })
  const address = server.address()
  if (!address || typeof address === 'string') {
    server.close()
    throw new Error('specialist dispatch did not bind a loopback TCP port')
  }
  const endpointPath = writePrivateEndpoint(deps.root, token, address.port)
  return {
    endpointPath,
    port: address.port,
    close: async () => {
      await new Promise<void>(resolve => server.close(() => resolve()))
      try {
        const current = JSON.parse(fs.readFileSync(endpointPath, 'utf8'))
        if (sameToken(current?.token, token)) {
          fs.rmSync(endpointPath, { force: true })
        }
      } catch {
        // Another Desktop generation may already own the endpoint projection.
      }
    }
  }
}
