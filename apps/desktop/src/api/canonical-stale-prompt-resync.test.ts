import { JsonRpcGatewayClient, type ServerRequest } from '@hermes/shared'
import { afterEach, expect, test, vi } from 'vitest'

import type { CanonicalDesktopProtocol } from './canonical-protocol'
import { HermesGateway } from './client'

// A phone's backgrounded tab: the socket sleeps through a prompt's settle, a newer request or the
// turn's generation change, and the card it still shows was refused as stale on every press until
// a full reload. The owner's attach snapshot is the recovery.

afterEach(() => vi.restoreAllMocks())

const approval = (promptId: string, generation: number) =>
  ({ kind: 'approval', prompt_id: promptId, execution_generation: generation, command: 'build', choices: ['once', 'deny'] })

interface Owner { generation: number; prompts: Array<Record<string, unknown>> }

async function attachedViewer(owner: Owner) {
  vi.spyOn(JsonRpcGatewayClient.prototype, 'connect').mockResolvedValue()

  const wire = vi.spyOn(JsonRpcGatewayClient.prototype, 'request').mockImplementation(async (method, params) => {
    if (method === 'session.resume') {
      return { session_id: 's', revision: 1, running: true, replay_epoch: 'epoch', execution_generation: owner.generation,
        prompts: owner.prompts } as never
    }

    return { status: 'resolved', prompt_id: params?.prompt_id } as never
  })

  const client = new HermesGateway()
  const cards: ServerRequest[] = []
  const settled: string[] = []
  client.onRequest(request => { cards.push(request) })
  client.on('approval.settled', event => { settled.push(String(event.payload?.prompt_id)) })
  await client.connect('ws://localhost/api/ws?native_dial=fixture&ticket=first')
  await client.request('session.resume', { session_id: 's' })
  const sent = (method: string) => wire.mock.calls.filter(([name]) => name === method).map(([, params]) => params)

  return { cards, client, sent, settled }
}

// The one frame the sleeping socket did carry: the next turn's start.
const nextTurn = (client: HermesGateway, generation: number) =>
  (client as unknown as { protocol: CanonicalDesktopProtocol }).protocol.event(
    { type: 'message.start', session_id: 's', profile: 'default', payload: { execution_generation: generation } })

test('an answer refused as stale re-reads the owner\'s prompts and replaces the dead card with the open one', async () => {
  const owner = { generation: 2, prompts: [approval('p1', 2)] }
  const { cards, client, sent, settled } = await attachedViewer(owner)
  expect(cards.map(card => card.id)).toEqual(['p1'])

  // Asleep, the viewer missed p1's settle and p2's request; the owner's next turn holds p2.
  Object.assign(owner, { generation: 3, prompts: [approval('p2', 3)] })
  nextTurn(client, 3)

  await expect(client.request('approval.respond', { session_id: 's', request_id: 'p1', choice: 'once' }))
    .resolves.toEqual({ status: 'already_resolved', prompt_id: 'p1' })
  expect(sent('approval.respond')).toEqual([])
  expect(sent('session.resume').at(-1)).toMatchObject({ session_id: 's', omit_messages: true })
  expect(settled).toEqual(['p1'])
  expect(cards.map(card => [card.id, card.replayed])).toEqual([['p1', true], ['p2', true]])

  // The fresh card answers at the owner's generation.
  cards[1].respond({ choice: 'deny' })
  await vi.waitFor(() => expect(sent('approval.respond')).toEqual([
    { session_id: 's', profile: 'default', execution_generation: 3, prompt_id: 'p2', choice: 'deny' }
  ]))
  client.close()
})

test('a stale answer is sent once more when the owner still holds that prompt at its current generation', async () => {
  const owner: Owner = { generation: 2, prompts: [] }
  const { client, sent } = await attachedViewer(owner)

  // The request frame was lost; the card came back from `approval.pending`, which names no generation.
  Object.assign(owner, { generation: 3, prompts: [approval('p2', 3)] })
  nextTurn(client, 3)

  await expect(client.request('approval.respond', { session_id: 's', request_id: 'p2', choice: 'once' }))
    .resolves.toEqual({ status: 'resolved', prompt_id: 'p2' })
  expect(sent('approval.respond')).toEqual([{ session_id: 's', execution_generation: 3, prompt_id: 'p2', choice: 'once' }])
  expect(sent('session.resume')).toHaveLength(2)
  client.close()
})

test('a stale answer is not reported resolved when the re-read snapshot carries no prompt list', async () => {
  const owner: Owner = { generation: 2, prompts: [approval('p1', 2)] }
  const { client, sent, settled } = await attachedViewer(owner)

  // The owner's next turn fenced p1 out of the cache; its resume says nothing about open prompts.
  Object.assign(owner, { generation: 3, prompts: undefined })
  nextTurn(client, 3)

  await expect(client.request('approval.respond', { session_id: 's', request_id: 'p1', choice: 'once' }))
    .rejects.toThrow('Prompt is stale or unavailable')
  expect(sent('approval.respond')).toEqual([])
  expect(sent('session.resume')).toHaveLength(2)
  expect(settled).toEqual([])
  client.close()
})

test('a stale answer is not reported resolved while the snapshot still lists that prompt behind a fence', async () => {
  const owner: Owner = { generation: 2, prompts: [approval('p1', 2)] }
  const { client, sent, settled } = await attachedViewer(owner)

  // The owner still lists p1, but only at the generation the session has left: listed, not answerable.
  Object.assign(owner, { generation: 3 })
  nextTurn(client, 3)

  await expect(client.request('approval.respond', { session_id: 's', request_id: 'p1', choice: 'once' }))
    .rejects.toThrow('Prompt is stale or unavailable')
  expect(sent('approval.respond')).toEqual([])
  expect(sent('session.resume')).toHaveLength(2)
  expect(settled).toEqual([])
  client.close()
})

test('a page coming back re-reads its open prompts and takes down a card the owner settled meanwhile', async () => {
  const owner = { generation: 2, prompts: [approval('p1', 2)] }
  const { client, sent, settled } = await attachedViewer(owner)
  owner.prompts = []

  const visibility = vi.spyOn(document, 'visibilityState', 'get').mockReturnValue('hidden')
  document.dispatchEvent(new Event('visibilitychange'))
  await Promise.resolve()
  expect(sent('session.resume')).toHaveLength(1)

  visibility.mockReturnValue('visible')
  document.dispatchEvent(new Event('visibilitychange'))
  await vi.waitFor(() => expect(settled).toEqual(['p1']))
  expect(sent('session.resume').at(-1)).toMatchObject({ session_id: 's', omit_messages: true })

  // Nothing is cached any more, and a closed client no longer listens.
  document.dispatchEvent(new Event('visibilitychange'))
  client.close()
  document.dispatchEvent(new Event('visibilitychange'))
  await Promise.resolve()
  expect(sent('session.resume')).toHaveLength(2)
})

test('a snapshot that carries no prompt list takes no card down', async () => {
  const owner: Owner = { generation: 2, prompts: [approval('p1', 2)] }
  const { client, sent, settled } = await attachedViewer(owner)
  Object.assign(owner, { prompts: undefined })

  await client.resyncPrompts()
  expect(sent('session.resume')).toHaveLength(2)
  expect(settled).toEqual([])
  client.close()
})

test('a redial re-reads the open prompts its cache holds and delivers the owner\'s newer ones', async () => {
  const owner = { generation: 2, prompts: [approval('p1', 2)] }
  const { cards, client, sent, settled } = await attachedViewer(owner)
  Object.assign(owner, { generation: 3, prompts: [approval('p2', 3)] })

  await client.connect('ws://localhost/api/ws?native_dial=fixture&ticket=second')
  await vi.waitFor(() => expect(cards.map(card => card.id)).toEqual(['p1', 'p2']))
  expect(settled).toEqual(['p1'])
  expect(sent('session.resume')).toHaveLength(2)
  client.close()
})
