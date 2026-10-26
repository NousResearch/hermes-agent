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

// A DELETE raced by another surface's delete (or rolled back after its own optimistic removal
// was resurrected by a later event) reads exists: false. With `absent` that is committed state:
// the caller's synthetic result wins, nothing is sent, and the session's retained identities
// retire exactly as an acknowledged write would.
test('an absent-session DELETE resolves through absent() without sending', async () => {
  const mutate = createSessionMutationClient()
  const read = async () => ({ session_id: 's', exists: false, runtime_revision: 0, runtime_generation: null })
  const deleteKey = JSON.stringify([{ connectionId: null, profile: 'p' }, 's', 'DELETE', {}])
  let sent = 0
  const result = await mutate<{ ok: boolean, already_absent?: boolean }>(deleteKey, read,
    async () => { sent += 1; return { ok: true } }, false,
    () => ({ ok: true, already_absent: true }))
  expect(result).toEqual({ ok: true, already_absent: true })
  expect(sent).toBe(0)
})

test('an absent snapshot without absent() still refuses the write', async () => {
  const mutate = createSessionMutationClient()
  const read = async () => ({ session_id: 's', exists: false, runtime_revision: 0, runtime_generation: null })
  const patchKey = JSON.stringify([{ connectionId: null, profile: 'p' }, 's', 'PATCH', { title: 'x' }])
  await expect(mutate(patchKey, read, async () => 'ok')).rejects.toThrow('Session mutation snapshot unavailable')
})

// The ghost loop's second half: a retained ambiguous identity (a timed-out rename on the same
// session) must not outlive the session's deletion. The absent DELETE retires the session's
// retained identities exactly like an acknowledged write, so a later PATCH re-reads its
// precondition instead of replaying a stale receipt against a deleted row.
test('an absent delete retires the session retained identities like an acknowledged write', async () => {
  const mutate = createSessionMutationClient()
  let exists = true
  let revision = 5
  const read = async () => ({ session_id: 's', exists, runtime_revision: revision, runtime_generation: 1 })
  const patchKey = JSON.stringify([{ connectionId: null, profile: 'p' }, 's', 'PATCH', { title: 'x' }])
  const deleteKey = JSON.stringify([{ connectionId: null, profile: 'p' }, 's', 'DELETE', {}])
  const sent: number[] = []
  // A rename times out after reading its precondition (ambiguous write: identity retained).
  await expect(mutate(patchKey, read, async id => { sent.push(id.expected_revision); throw new Error('timeout') }))
    .rejects.toThrow('timeout')
  expect(sent).toEqual([5])
  // The session is deleted on the backend while the row still shows in the sidebar.
  exists = false
  const result = await mutate<{ ok: boolean }>(deleteKey, read, async () => { sent.push(-1); return { ok: true } }, false,
    () => ({ ok: true, already_absent: true }))
  expect(result).toEqual({ ok: true, already_absent: true })
  expect(sent).toEqual([5])
  // The deletion retired the retained rename identity: a later write on this session
  // re-reads its precondition instead of replaying revision 5.
  exists = true
  revision = 6
  await mutate(patchKey, read, async id => { sent.push(id.expected_revision); return 'ok' })
  expect(sent).toEqual([5, 6])
})
