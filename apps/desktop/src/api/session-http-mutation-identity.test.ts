import { expect, test } from 'vitest'

import { createSessionMutationClient } from '../../../shared/src/session-http-mutations'

const key = (field: string, value: unknown) => JSON.stringify([{ connectionId: null, profile: 'p' }, 's', 'PATCH', { [field]: value }])

// N101: A:X (lost ACK) -> B:Y (acknowledged) -> A:X must read a fresh precondition; the retained
// A identity is superseded by B's committed revision. B2-22: definitive refusals drop the identity.
test.each([
  ['an acknowledged write on the same session', async (mutate: ReturnType<typeof createSessionMutationClient>, read: () => Promise<any>) => {
    await mutate(key('hidden', true), read, async () => 'ok')
  }],
  ...['unknown_execution', 'not_found', 'mutation_identity_required'].map(reason => [`a definitive ${reason} refusal`,
    async (mutate: ReturnType<typeof createSessionMutationClient>, read: () => Promise<any>) => {
      await expect(mutate(key('title', 'A'), read, async () => { throw new Error(`409 ${reason}`) })).rejects.toThrow(reason)
    }] as const)
] as const)('%s retires the retained ambiguous identity', async (_label, intervene) => {
  const mutate = createSessionMutationClient()
  let revision = 7
  const read = async () => ({ session_id: 's', exists: true, runtime_revision: revision, runtime_generation: 1 })
  const sent: number[] = []
  await expect(mutate(key('title', 'A'), read, async id => { sent.push(id.expected_revision); throw new Error('timeout') })).rejects.toThrow('timeout')
  revision = 8
  await intervene(mutate, read)
  revision = 9
  await mutate(key('title', 'A'), read, async id => { sent.push(id.expected_revision); return 'ok' })
  expect(sent).toEqual([7, 9])
})

test('a retained identity on another session survives an acknowledged write', async () => {
  const mutate = createSessionMutationClient()
  let revision = 3
  const read = async () => ({ session_id: 'x', exists: true, runtime_revision: revision, runtime_generation: 1 })
  const other = JSON.stringify([{ connectionId: null, profile: 'p' }, 'other', 'PATCH', { title: 'A' }])
  const sent: number[] = []
  await expect(mutate(other, read, async id => { sent.push(id.expected_revision); throw new Error('timeout') })).rejects.toThrow('timeout')
  revision = 4
  await mutate(key('hidden', true), read, async () => 'ok')
  await mutate(other, read, async id => { sent.push(id.expected_revision); return 'ok' })
  expect(sent).toEqual([3, 3])
})
