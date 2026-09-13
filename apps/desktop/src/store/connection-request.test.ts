import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  $connectionRequests,
  clearConnectionRequest,
  type ConnectionRequest,
  hasConnectionRequest,
  normalizeConnectionRequest,
  respondToConnectionRequest,
  setConnectionRequest,
  skipConnectionRequest
} from './connection-request'
import { $gateway } from './gateway'

const WIRE = {
  deadline_at: 1_800_000_000,
  op_id: 'op-1',
  reason: 'tickets',
  request_id: 'req-1',
  targets: [
    { action: 'install', kind: 'mcp', name: 'linear' },
    { action: 'install', kind: 'mcp', name: 'figma' }
  ]
}

type Gateway = NonNullable<ReturnType<typeof $gateway.get>>

function fakeGateway(rpc: Gateway['request']): Gateway {
  // SAFETY: the store calls only `request`; the rest of the client is never touched in these tests.
  return { request: rpc } as Gateway
}

function request(sessionId: string | null, requestId = 'req-1'): ConnectionRequest {
  return normalizeConnectionRequest({ ...WIRE, request_id: requestId }, sessionId)!
}

describe('connection-request store', () => {
  beforeEach(() => {
    $connectionRequests.set({})
  })

  afterEach(() => {
    $connectionRequests.set({})
    $gateway.set(null)
  })

  it('normalizes the wire payload and keeps the server-owned deadline verbatim', () => {
    const parsed = normalizeConnectionRequest(WIRE, 's1')

    expect(parsed?.deadlineAt).toBe(WIRE.deadline_at)
    expect(parsed?.opId).toBe('op-1')
    expect(parsed?.targets.map(t => t.name)).toEqual(['linear', 'figma'])
  })

  it('rejects a payload with no targets, no op id or no deadline', () => {
    expect(normalizeConnectionRequest({ ...WIRE, targets: [] }, 's1')).toBeNull()
    expect(normalizeConnectionRequest({ ...WIRE, op_id: undefined }, 's1')).toBeNull()
    expect(normalizeConnectionRequest({ ...WIRE, deadline_at: 0 }, 's1')).toBeNull()
    expect(normalizeConnectionRequest(null, 's1')).toBeNull()
  })

  it('keeps requests from concurrent sessions independent', () => {
    setConnectionRequest(request('a', 'req-a'))
    setConnectionRequest(request('b', 'req-b'))

    expect(hasConnectionRequest('a')).toBe(true)
    clearConnectionRequest('req-a', 'a')
    expect(hasConnectionRequest('a')).toBe(false)
    expect(hasConnectionRequest('b')).toBe(true)
  })

  it('a stale request id never clears a newer card', () => {
    setConnectionRequest(request('a', 'req-new'))
    clearConnectionRequest('req-old', 'a')

    expect($connectionRequests.get().a?.requestId).toBe('req-new')
  })

  it('respond clears the entry before the RPC and refuses a second answer', async () => {
    const rpc = vi.fn().mockResolvedValue({ status: 'ok' })
    $gateway.set(fakeGateway(rpc))
    const req = request('a')
    setConnectionRequest(req)

    const first = await respondToConnectionRequest(req, { targets: [{ name: 'linear', status: 'installed' }] })
    const second = await respondToConnectionRequest(req, { targets: [{ name: 'linear', status: 'declined' }] })

    expect(first).toBe(true)
    expect(second).toBe(false)
    expect(rpc).toHaveBeenCalledTimes(1)
    expect(rpc.mock.calls[0][0]).toBe('connection.respond')
    expect(JSON.parse(rpc.mock.calls[0][1].result).targets[0].status).toBe('installed')
  })

  it('skip declines every target of the pending operation', async () => {
    const rpc = vi.fn().mockResolvedValue({ status: 'ok' })
    $gateway.set(fakeGateway(rpc))
    setConnectionRequest(request('a'))

    expect(await skipConnectionRequest('a')).toBe(true)
    expect(await skipConnectionRequest('a')).toBe(false)
    const sent = JSON.parse(rpc.mock.calls[0][1].result)

    expect(sent.targets.map((t: { name: string; status: string }) => [t.name, t.status])).toEqual([
      ['linear', 'declined'],
      ['figma', 'declined']
    ])
  })
})
