import { JsonRpcGatewayClient } from '@hermes/shared'
import { afterEach, expect, test, vi } from 'vitest'

import { CanonicalDesktopProtocol } from './canonical-protocol'
import { HermesGateway } from './client'

afterEach(() => vi.restoreAllMocks())

test('retained prompts keep their own profile when imported profiles share a stored session ID', async () => {
  vi.spyOn(JsonRpcGatewayClient.prototype, 'connect').mockResolvedValue()
  const replies: Record<string, unknown>[] = []
  vi.spyOn(JsonRpcGatewayClient.prototype, 'request').mockImplementation(async (method, params) => {
    if (method === 'session.resume') {return {
      session_id: 'imported-same-id', revision: 1, execution_generation: 2,
      prompts: [{ kind: 'approval', prompt_id: `prompt-${params?.profile}`, execution_generation: 2,
        command: 'build', choices: ['once'] }]
    } as never}

    if (method === 'approval.respond') {replies.push(params!)}

    return { status: 'resolved' } as never
  })
  const client = new HermesGateway()
  const shown: any[] = []
  client.onRequest(request => { shown.push(request) })
  await client.connect('ws://localhost/api/ws?native_dial=fixture&ticket=first')
  await client.request('session.resume', { session_id: 'imported-same-id', profile: 'alpha' })
  await client.request('session.resume', { session_id: 'imported-same-id', profile: 'beta' })
  shown[0].respond({ choice: 'once' })
  await vi.waitFor(() => expect(replies).toHaveLength(1))
  expect(replies[0].profile).toBe('alpha')
  client.close()
})

test('equal stored IDs keep independent revisions, generations, mutation retries and unknown admissions', () => {
  const protocol = new CanonicalDesktopProtocol()

  for (const [profile, revision, generation] of [['alpha', 4, 2], ['beta', 20, 9]] as const) {
    protocol.result('session.resume', { session_id: 'same', profile }, {
      session_id: 'same', revision, execution_generation: generation,
      prompts: [{ kind: 'approval', prompt_id: 'same-prompt-id', execution_generation: generation }],
      pending: [{ admission_id: 'same-admission-id', status: 'unknown', execution_generation: generation - 1 }]
    })
  }

  const alpha = protocol.prepare('session.title', { session_id: 'same', profile: 'alpha', title: 'next' })
  const beta = protocol.prepare('session.title', { session_id: 'same', profile: 'beta', title: 'next' })
  expect(alpha.expected_revision).toBe(4)
  expect(beta.expected_revision).toBe(20)
  expect(alpha.request_id).not.toBe(beta.request_id)
  protocol.event({ type: 'session.info', session_id: 'same', profile: 'beta', payload: {
    pending: [], revision: 21, execution_generation: 10
  } })
  expect(protocol.prepare('session.title', { session_id: 'same', profile: 'alpha', title: 'next' })).toEqual(alpha)
  expect(protocol.prepare('session.interrupt', { session_id: 'same', profile: 'alpha' }).execution_generation).toBe(2)
  expect(protocol.prepare('session.interrupt', { session_id: 'same', profile: 'beta' }).execution_generation).toBe(10)
  expect(protocol.prepare('approval.respond', { session_id: 'same', profile: 'alpha', prompt_id: 'same-prompt-id', choice: 'once' }).execution_generation).toBe(2)
  expect(protocol.prepare('prompt.resolve_unknown', { session_id: 'same', profile: 'alpha', admission_id: 'same-admission-id' }).execution_generation).toBe(1)
  expect(() => protocol.prepare('prompt.resolve_unknown', { session_id: 'same', profile: 'beta', admission_id: 'same-admission-id' })).toThrow('unknown')
})

test('implicit named-profile creation and explicit follow-ups share one owner and its mutation retry', async () => {
  vi.spyOn(JsonRpcGatewayClient.prototype, 'connect').mockResolvedValue()
  const calls: Array<[string, Record<string, unknown> | undefined]> = []
  let failFirstMutation = true
  vi.spyOn(JsonRpcGatewayClient.prototype, 'request').mockImplementation(async (method, params) => {
    calls.push([method, params])

    if (method === 'session.create' || method === 'session.resume') {
      return { session_id: 'named-session', revision: 4, execution_generation: 2,
        replay_epoch: 'same-owner', info: { profile_name: 'alpha', profile_id: '/profiles/alpha' }, prompts: [] } as never
    }

    if (method === 'prompt.submit') {
      return { admission_id: 'a', status: 'queued', ref: { session_id: 'named-session', profile_id: '/profiles/alpha' } } as never
    }

    if (method === 'session.mutate' && failFirstMutation) {
      failFirstMutation = false
      throw new Error('lost acknowledgement')
    }

    return { session_id: 'named-session', revision: 5, operation: 'rename' } as never
  })
  const client = new HermesGateway()
  await client.connect('ws://localhost/api/ws?native_dial=fixture&ticket=first')
  await client.request('session.create', {})
  await client.request('prompt.submit', { session_id: 'named-session', profile: 'alpha', submission_id: 'input', text: 'next' })
  expect(calls.filter(([method]) => method === 'session.resume')).toHaveLength(0)
  await expect(client.request('session.title', { session_id: 'named-session', title: 'retained' })).rejects.toThrow('lost acknowledgement')
  const first = calls.find(([method]) => method === 'session.mutate')![1]!
  await client.connect('ws://localhost/api/ws?native_dial=fixture&ticket=second')
  await client.request('session.resume', { session_id: 'named-session' })
  await client.request('session.title', { session_id: 'named-session', profile: '/profiles/alpha', title: 'retained' })
  const second = calls.filter(([method]) => method === 'session.mutate').at(-1)![1]!
  expect(second.request_id).toBe(first.request_id)
  expect(second.profile).toBe('alpha')
  expect(second.expected_revision).toBe(4)
  client.close()
})
