import { JsonRpcGatewayClient } from '@hermes/shared'
import { afterEach, expect, test, vi } from 'vitest'

import { HermesGateway } from './client'

afterEach(() => vi.restoreAllMocks())

// Equal stored session IDs under two profiles: a frame with an epoch neither snapshot named
// cannot be routed until an authoritative owner probe answers.
async function equalIdClient(resume: (profile: string) => Promise<unknown>) {
  vi.spyOn(JsonRpcGatewayClient.prototype, 'connect').mockResolvedValue()
  const calls: string[] = []

  vi.spyOn(JsonRpcGatewayClient.prototype, 'request').mockImplementation(async (method, params) => {
    calls.push(`${method}:${params?.profile}`)

    return (method === 'session.resume' ? resume(String(params?.profile)) : { status: 'resolved' }) as never
  })
  const client = new HermesGateway()
  await client.connect('ws://localhost/api/ws?native_dial=fixture&ticket=first')

  return { calls, client, emit: (event: Record<string, unknown>) => (client as any).handleEvent(event) }
}

const snapshot = (profile: string, epoch: string) => ({ session_id: 'same', revision: 1, execution_generation: 2, replay_epoch: epoch, prompts: [], info: { profile_name: profile } })

test('a failed owner reconciliation is probed again by the next frame', async () => {
  let down = false

  const { calls, client, emit } = await equalIdClient(async profile => {
    if (down) { throw new Error('owner unavailable') }

    return snapshot(profile, `epoch-${profile}`)
  })

  await client.request('session.resume', { session_id: 'same', profile: 'alpha' })
  await client.request('session.resume', { session_id: 'same', profile: 'beta' })
  client.onAny(() => undefined)
  down = true
  const probes = () => calls.filter(call => call.startsWith('session.resume')).length - 2
  emit({ type: 'message.delta', session_id: 'same', replay_epoch: 'restarted', payload: { text: 'a' } })
  await vi.waitFor(() => expect(probes()).toBe(2))
  await new Promise(resolve => setTimeout(resolve, 0))
  emit({ type: 'message.delta', session_id: 'same', replay_epoch: 'restarted', payload: { text: 'b' } })
  await vi.waitFor(() => expect(probes()).toBe(4))
  client.close()
})

test('the frame that triggered a successful reconciliation is delivered to its owner', async () => {
  let restarted = false

  const { client, emit } = await equalIdClient(async profile =>
    snapshot(profile, restarted && profile === 'beta' ? 'beta-restarted' : `epoch-${profile}`))

  await client.request('session.resume', { session_id: 'same', profile: 'alpha' })
  await client.request('session.resume', { session_id: 'same', profile: 'beta' })
  const named: any[] = []
  const wildcard: any[] = []
  client.on('message.complete', event => named.push(event))
  client.onAny(event => { if (event.type === 'message.complete') { wildcard.push(event) } })
  restarted = true
  emit({ type: 'message.complete', session_id: 'same', replay_epoch: 'beta-restarted', payload: { text: 'done' } })
  await vi.waitFor(() => expect(named).toHaveLength(1))
  expect(named[0].profile).toBe('beta')
  expect(wildcard.map(event => event.profile)).toEqual(['beta'])
  client.close()
})

test('a prompt declined locally is shown again while the authority still lists it', async () => {
  const prompt = { kind: 'approval', prompt_id: 'pending', execution_generation: 2, command: 'build', choices: ['once'] }
  const { calls, client } = await equalIdClient(async () => ({ session_id: 's', revision: 1, execution_generation: 2, prompts: [prompt] }))
  const shown: any[] = []

  client.onRequest(request => {
    shown.push(request)

    if (shown.length === 1) { request.fail(-32603, 'session interrupted') }
  })
  await client.request('session.resume', { session_id: 's' })
  await client.request('session.resume', { session_id: 's' })
  expect(shown).toHaveLength(2)
  expect(calls.filter(call => call.startsWith('approval.respond'))).toHaveLength(0)
  client.close()
})
