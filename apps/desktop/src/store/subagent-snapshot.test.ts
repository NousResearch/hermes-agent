import { afterEach, expect, it } from 'vitest'

import { $subagentsBySession, reconcileSubagentSnapshot, type SubagentPayload, upsertSubagent } from './subagents'
import { subagentEvent, subagentRosterRow } from './subagents.test-util'

afterEach(() => $subagentsBySession.set({}))
it('hydrates last tool activity without claiming it is active and preserves stream history on refresh', () => {
  const child = subagentRosterRow({
    subagent_id: 'worker',
    goal: 'Actual worker',
    status: 'running',
    started_at: 1001,
    last_tool: 'read_file'
  })

  reconcileSubagentSnapshot('owner', [child])
  const first = $subagentsBySession.get().owner
  expect(first[0].stream).toMatchObject([{ kind: 'tool', text: 'Read File' }])
  expect(first[0].currentTool).toBeUndefined()
  expect(first[0].startedAt).toBe(child.started_at * 1000)
  reconcileSubagentSnapshot('owner', [child])
  expect($subagentsBySession.get().owner).toBe(first)
  upsertSubagent('other', subagentEvent({ subagent_id: 'worker', goal: 'Other owner' }))
  const other = $subagentsBySession.get().other
  upsertSubagent('owner', subagentEvent({ subagent_id: 'worker', text: 'New progress' }), false, 'subagent.progress')
  const stream = $subagentsBySession.get().owner[0].stream
  reconcileSubagentSnapshot('owner', [child])
  expect($subagentsBySession.get().owner[0].stream).toBe(stream)
  reconcileSubagentSnapshot('owner', [])
  expect($subagentsBySession.get().owner).toEqual([])
  expect($subagentsBySession.get().other).toBe(other)
})

it('fails closed on an unknown terminal status from a newer emitter (off-contract probe)', () => {
  upsertSubagent('owner', subagentEvent({ subagent_id: 'worker', goal: 'Future worker', status: 'running' }))
  upsertSubagent(
    'owner',
    subagentEvent({
      subagent_id: 'worker',
      // SAFETY: deliberately off-contract — a future backend may add a terminal status this build
      // does not know; the store must coerce it to 'failed' rather than leave the row spinning.
      status: 'some_future_terminal_status' as unknown as SubagentPayload['status']
    }),
    false,
    'subagent.complete'
  )
  expect($subagentsBySession.get().owner[0].status).toBe('failed')
})
