import { afterEach, expect, it } from 'vitest'

import { sessionBackgroundResume } from './background-delegation'
import { $activeSessionId, $busy } from './session'
import { $subagentsBySession, upsertSubagent } from './subagents'

afterEach(() => {
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
  expect(owner.get()).toEqual({ count: 2 })
  expect(other.get()).toEqual({ count: 1 })
  expect(unbound.get()).toBeNull()

  upsertSubagent('owner', { subagent_id: 'working', status: 'completed' }, false, 'subagent.complete')
  expect(owner.get()).toEqual({ count: 1 })
  upsertSubagent('owner', { subagent_id: 'waiting', status: 'timeout' }, false, 'subagent.complete')
  $activeSessionId.set(null)
  $busy.set(false)
  expect(owner.get()).toBeNull()
  expect(other.get()).toEqual({ count: 1 })
})
