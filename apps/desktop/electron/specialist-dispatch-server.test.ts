import crypto from 'node:crypto'
import fs from 'node:fs'
import net from 'node:net'
import os from 'node:os'
import path from 'node:path'

import { afterEach, expect, test, vi } from 'vitest'

import { type SpecialistDispatchAdmission, startSpecialistDispatchServer } from './specialist-dispatch-server'

const roots: string[] = []
const unusedQuiesce = {
  acquire: async (profileIds: string[]) => ({ leaseId: 'specialist-quiesce-00000000-0000-4000-8000-000000000000', profileIds }),
  release: () => ({ outcome: 'rejected' as const, reasonCode: 'UNKNOWN_QUIESCE_LEASE' as const })
}

afterEach(() => {
  for (const root of roots.splice(0)) {
    fs.rmSync(root, { force: true, recursive: true })
  }
})

function request(port: number, payload: object): Promise<Record<string, unknown>> {
  return new Promise((resolve, reject) => {
    const socket = net.createConnection({ host: '127.0.0.1', port })
    let response = ''
    socket.once('error', reject)
    socket.on('data', chunk => { response += chunk.toString() })
    socket.once('end', () => resolve(JSON.parse(response)))
    socket.once('connect', () => socket.write(`${JSON.stringify(payload)}\n`))
  })
}

function rawRequest(port: number, payload: string): Promise<Record<string, unknown>> {
  return new Promise((resolve, reject) => {
    const socket = net.createConnection({ host: '127.0.0.1', port })
    let response = ''
    socket.once('error', reject)
    socket.on('data', chunk => { response += chunk.toString() })
    socket.once('end', () => resolve(JSON.parse(response)))
    socket.once('connect', () => socket.write(`${payload}\n`))
  })
}

test('loopback transport requires its private capability and never exposes it', async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'ares-specialist-dispatch-'))
  roots.push(root)
  const admitted: unknown[] = []

  const admit: SpecialistDispatchAdmission['admit'] = async request => {
    admitted.push(request)

    return { outcome: 'admitted', reasonCode: 'ADMITTED', runId: request.runId, reservedCapacity: 1 }
  }

  const server = await startSpecialistDispatchServer({
    admission: { admit, hasActive: () => false, poolKey: runId => `specialist-run:${runId}`, release: vi.fn(), status: () => 'running' },
    cancel: async () => undefined,
    quiesce: unusedQuiesce,
    root
  })

  const endpoint = JSON.parse(fs.readFileSync(path.join(root, 'specialist-dispatch.json'), 'utf8'))

  const payload = {
    schema: 'AresDesktopSpecialistDispatchEnvelopeV1',
    operation: 'submit',
    token: endpoint.token,
    request: {
      run_id: 'specialist-run-00000001',
      request_digest: `sha256:${'a'.repeat(64)}`,
      profile_ids: ['explorer']
    }
  }

  const accepted = await request(endpoint.port, payload)
  const rejected = await request(endpoint.port, { ...payload, token: crypto.randomBytes(32).toString('hex') })

  expect(accepted).toMatchObject({ outcome: 'admitted', runId: 'specialist-run-00000001' })
  expect(JSON.stringify(accepted)).not.toContain(endpoint.token)
  expect(rejected).toMatchObject({ outcome: 'rejected', reasonCode: 'AUTHENTICATION_FAILED' })
  expect(admitted).toHaveLength(1)
  expect(fs.statSync(path.join(root, 'specialist-dispatch.json')).mode & 0o777).toBe(0o600)
  await server.close()
})

test('status and cancel are explicit operations; unknown payload fields are rejected', async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'ares-specialist-dispatch-'))
  roots.push(root)
  const cancel = vi.fn(async () => undefined)

  const server = await startSpecialistDispatchServer({
    admission: { admit: vi.fn(), hasActive: () => false, poolKey: runId => `specialist-run:${runId}`, release: vi.fn(), status: () => 'released' },
    cancel,
    quiesce: unusedQuiesce,
    root
  })

  const endpoint = JSON.parse(fs.readFileSync(path.join(root, 'specialist-dispatch.json'), 'utf8'))
  const base = { schema: 'AresDesktopSpecialistDispatchEnvelopeV1', token: endpoint.token, run_id: 'specialist-run-00000001' }

  const status = await request(endpoint.port, { ...base, operation: 'status' })
  const cancelled = await request(endpoint.port, { ...base, operation: 'cancel' })
  const malformed = await request(endpoint.port, { ...base, operation: 'status', unexpected: true })

  expect(status).toEqual({ outcome: 'status', runId: 'specialist-run-00000001', terminalState: 'released' })
  expect(cancelled).toEqual({ outcome: 'released', runId: 'specialist-run-00000001' })
  expect(malformed).toMatchObject({ outcome: 'rejected', reasonCode: 'INVALID_ENVELOPE' })
  expect(cancel).toHaveBeenCalledWith('specialist-run-00000001')
  await server.close()
})

test('authenticated quiesce leases are exact, releasable, and never expose the endpoint capability', async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'ares-specialist-dispatch-'))
  roots.push(root)
  const acquire = vi.fn(async (profileIds: string[]) => ({ leaseId: 'specialist-quiesce-00000000-0000-4000-8000-000000000000', profileIds }))
  const release = vi.fn((leaseId: string) =>
    leaseId === 'specialist-quiesce-00000000-0000-4000-8000-000000000000'
      ? { outcome: 'released' as const, profileIds: ['explorer', 'public'] }
      : { outcome: 'rejected' as const, reasonCode: 'UNKNOWN_QUIESCE_LEASE' as const }
  )
  const server = await startSpecialistDispatchServer({
    admission: { admit: vi.fn(), hasActive: () => false, poolKey: (runId: string) => `specialist-run:${runId}`, release: vi.fn(), status: () => 'unknown' },
    cancel: async () => undefined,
    quiesce: { acquire, release },
    root
  } as any)
  const endpoint = JSON.parse(fs.readFileSync(path.join(root, 'specialist-dispatch.json'), 'utf8'))
  const base = { schema: 'AresDesktopSpecialistDispatchEnvelopeV1', token: endpoint.token }

  const acquired = await request(endpoint.port, { ...base, operation: 'quiesce', profile_ids: ['explorer', 'public'] })
  const released = await request(endpoint.port, {
    ...base,
    operation: 'unquiesce',
    lease_id: 'specialist-quiesce-00000000-0000-4000-8000-000000000000'
  })
  const malformed = await request(endpoint.port, { ...base, operation: 'quiesce', profile_ids: ['public', 'explorer'] })

  expect(acquired).toEqual({
    outcome: 'quiesced',
    leaseId: 'specialist-quiesce-00000000-0000-4000-8000-000000000000',
    profileIds: ['explorer', 'public']
  })
  expect(released).toEqual({
    outcome: 'unquiesced',
    leaseId: 'specialist-quiesce-00000000-0000-4000-8000-000000000000',
    profileIds: ['explorer', 'public']
  })
  expect(malformed).toEqual({ outcome: 'rejected', reasonCode: 'INVALID_ENVELOPE' })
  expect(acquire).toHaveBeenCalledWith(['explorer', 'public'])
  expect(release).toHaveBeenCalledWith('specialist-quiesce-00000000-0000-4000-8000-000000000000')
  expect(JSON.stringify(acquired)).not.toContain(endpoint.token)
  await server.close()
})

test('duplicate keys anywhere in the untrusted envelope reject before admission', async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'ares-specialist-dispatch-'))
  roots.push(root)
  const admit = vi.fn()
  const server = await startSpecialistDispatchServer({
    admission: { admit, hasActive: () => false, poolKey: runId => `specialist-run:${runId}`, release: vi.fn(), status: () => 'unknown' },
    cancel: async () => undefined,
    quiesce: unusedQuiesce,
    root
  })
  const endpoint = JSON.parse(fs.readFileSync(path.join(root, 'specialist-dispatch.json'), 'utf8'))
  const duplicateRequest = `{"schema":"AresDesktopSpecialistDispatchEnvelopeV1","operation":"submit","token":"${endpoint.token}","request":{"run_id":"specialist-run-00000001","run_id":"specialist-run-00000001","request_digest":"sha256:${'a'.repeat(64)}","profile_ids":["explorer"]}}`

  await expect(rawRequest(endpoint.port, duplicateRequest)).resolves.toEqual({ outcome: 'rejected', reasonCode: 'INVALID_ENVELOPE' })
  expect(admit).not.toHaveBeenCalled()
  await server.close()
})
