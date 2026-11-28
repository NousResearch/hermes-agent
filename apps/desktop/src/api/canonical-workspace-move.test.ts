import { JsonRpcGatewayClient } from '@hermes/shared'
import { afterEach, expect, test, vi } from 'vitest'

import { HermesGateway } from './client'

afterEach(() => { vi.restoreAllMocks() })

// Move to project names its sidebar row as `session_key` (no `session_id`). A row the user never
// opened on this socket must attach first so the CAS revision is known — the branch-stored path
// already does this for `parent_session_id`.
test.each([
  ['no profile', undefined],
  ['default profile', 'default'],
  ['named profile', 'work']
])('Move to project on a never-opened row attaches it first and presents its revision (%s)', async (_label, profile) => {
  vi.spyOn(JsonRpcGatewayClient.prototype, 'connect').mockResolvedValue()
  const calls: Array<{ method: string; params: Record<string, unknown> }> = []
  vi.spyOn(JsonRpcGatewayClient.prototype, 'request').mockImplementation(async (method, params) => {
    calls.push({ method, params: params as Record<string, unknown> })

    if (method === 'session.resume') {
      return { session_id: 's', revision: 7, execution_generation: 3, messages: [],
        info: profile && profile !== 'default' ? { profile_name: profile } : {} } as never
    }

    return { session_id: 's', revision: 8, operation: 'workspace', cwd: '/tmp/p' } as never
  })
  const client = new HermesGateway()
  await client.connect('ws://localhost/api/ws?native_dial=fixture&ticket=fixture')

  try {
    await client.request('session.workspace.move', { cwd: '/tmp/p', session_key: 's', ...(profile ? { profile } : {}) })
    expect(calls.map(c => c.method)).toEqual(['session.resume', 'session.mutate'])
    expect(calls[1].params).toMatchObject({ session_id: 's', operation: 'workspace', expected_revision: 7 })
  } finally { client.close() }
})

// A compressed local conversation is listed under its tip segment's id, but `session.resume`
// attaches (and caches the revision under) the conversation's owner id. The move must name that
// owner — its policy is the one a workspace edit rewrites — instead of failing with "Session
// revision unavailable" before any request is sent.
test.each([
  ['no profile', undefined],
  ['named profile', 'work']
])('Move to project on a compressed row moves the conversation owner the resume attached (%s)', async (_label, profile) => {
  vi.spyOn(JsonRpcGatewayClient.prototype, 'connect').mockResolvedValue()
  const calls: Array<{ method: string; params: Record<string, unknown> }> = []
  vi.spyOn(JsonRpcGatewayClient.prototype, 'request').mockImplementation(async (method, params) => {
    calls.push({ method, params: params as Record<string, unknown> })

    if (method === 'session.resume') {
      return { session_id: 'local-owner', stored_session_id: 'local-owner', revision: 7, execution_generation: 3,
        messages: [], info: profile ? { profile_name: profile } : {} } as never
    }

    return { session_id: 'local-owner', revision: 8, operation: 'workspace', cwd: '/tmp/p' } as never
  })
  const client = new HermesGateway()
  await client.connect('ws://localhost/api/ws?native_dial=fixture&ticket=fixture')

  try {
    await client.request('session.workspace.move', { cwd: '/tmp/p', session_key: '20261109_090650_tip', ...(profile ? { profile } : {}) })
    expect(calls.map(c => c.method)).toEqual(['session.resume', 'session.mutate'])
    expect(calls[0].params).toMatchObject({ session_id: '20261109_090650_tip' })
    expect(calls[1].params).toMatchObject({ session_id: 'local-owner', operation: 'workspace', expected_revision: 7 })
  } finally { client.close() }
})
