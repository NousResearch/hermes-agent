import { afterEach, expect, it } from 'vitest'

import { $subagentsBySession, reconcileSubagentSnapshot } from './subagents'

afterEach(() => $subagentsBySession.set({}))
it('projects unresolved running delegations once, replaces them with real workers, and retires completed units', () => {
  const delegation = {
    delegation_id: 'batch',
    goal: 'Queued unit',
    status: 'running',
    dispatched_at: 1000,
    subagent_ids: []
  }
  reconcileSubagentSnapshot('owner', [], [delegation])
  expect($subagentsBySession.get().owner).toMatchObject([
    { id: 'delegation:batch', goal: 'Queued unit', startedAt: 1000000 }
  ])
  const child = {
    subagent_id: 'worker',
    delegation_id: 'batch',
    goal: 'Actual worker',
    status: 'running',
    started_at: 1001
  }
  reconcileSubagentSnapshot('owner', [child], [delegation])
  expect($subagentsBySession.get().owner.map(row => row.id)).toEqual(['worker'])
  const before = $subagentsBySession.get().owner
  reconcileSubagentSnapshot('owner', [child], [delegation])
  expect($subagentsBySession.get().owner).toBe(before)
  reconcileSubagentSnapshot('owner', [], [{ ...delegation, status: 'completed' }])
  expect($subagentsBySession.get().owner).toEqual([])
})
