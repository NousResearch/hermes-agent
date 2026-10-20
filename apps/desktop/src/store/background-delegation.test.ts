import { afterEach, expect, it, vi } from 'vitest'

import { sessionBackgroundResume } from './background-delegation'
import { $activeSessionId, $busy } from './session'
import { $subagentsBySession, upsertSubagent } from './subagents'

afterEach(() => {
  vi.restoreAllMocks()
  $busy.set(false)
  $activeSessionId.set(null)
  $subagentsBySession.set({})
})

it('tracks only the owning runtime session, independent of foreground busy state and child stream text', () => {
  const owner = sessionBackgroundResume('owner')
  const other = sessionBackgroundResume('other')
  const unbound = sessionBackgroundResume(null)
  upsertSubagent('owner', { subagent_id: 'working', status: 'running' }, true, 'subagent.start')
  upsertSubagent('owner', { subagent_id: 'waiting', status: 'queued' }, true, 'subagent.spawn_requested')
  upsertSubagent('other', { subagent_id: 'working', status: 'running' }, true, 'subagent.start')
  $activeSessionId.set('other')
  $busy.set(true)
  upsertSubagent('owner', { subagent_id: 'working', text: '(°□°) pondering...' }, false, 'subagent.thinking')
  expect(owner.get()).toMatchObject({ count: 2 })
  expect(other.get()).toMatchObject({ count: 1 })
  expect(unbound.get()).toBeNull()

  upsertSubagent('owner', { subagent_id: 'working', status: 'completed' }, false, 'subagent.complete')
  expect(owner.get()).toMatchObject({ count: 1 })
  upsertSubagent('owner', { subagent_id: 'waiting', status: 'timeout' }, false, 'subagent.complete')
  $activeSessionId.set(null)
  $busy.set(false)
  expect(owner.get()).toBeNull()
  expect(other.get()).toMatchObject({ count: 1 })
})

it('times the wait from the oldest child still running, and moves on when it settles', () => {
  const now = vi.spyOn(Date, 'now')
  const owner = sessionBackgroundResume('owner')
  const seen: unknown[] = []
  const unbind = owner.listen(value => seen.push(value))

  now.mockReturnValue(1_000)
  upsertSubagent('owner', { subagent_id: 'first', status: 'running' }, true, 'subagent.start')
  now.mockReturnValue(5_000)
  upsertSubagent('owner', { subagent_id: 'second', status: 'running' }, true, 'subagent.start')
  expect(owner.get()).toEqual({ count: 2, since: 1_000 })

  // A child's relayed activity is not a new wait: nothing downstream repaints.
  const before = seen.length
  now.mockReturnValue(9_000)
  upsertSubagent('owner', { subagent_id: 'first', text: 'reading files' }, false, 'subagent.progress')
  expect(seen.length).toBe(before)

  upsertSubagent('owner', { subagent_id: 'first', status: 'completed' }, false, 'subagent.complete')
  expect(owner.get()).toEqual({ count: 1, since: 5_000 })
  unbind()
})
