import { beforeEach, describe, expect, it } from 'vitest'

import { $subagentsBySession, reconcileSubagentRoster, upsertSubagent } from './subagents'

const listFor = (sid: string) => $subagentsBySession.get()[sid] ?? []

describe('subagent roster reconciliation', () => {
  beforeEach(() => $subagentsBySession.set({}))

  it('keeps live delegate-tool fallback rows outside native roster reconciliation', () => {
    upsertSubagent('s1', {
      goal: 'fallback',
      status: 'running',
      subagent_id: 'delegate-tool:active-call:0',
      task_index: 0
    })

    reconcileSubagentRoster('s1', [])
    expect(listFor('s1').map(item => item.id)).toEqual(['delegate-tool:active-call:0'])

    upsertSubagent('s1', {
      goal: 'fallback',
      status: 'completed',
      subagent_id: 'delegate-tool:active-call:0',
      task_index: 0
    })
    expect(listFor('s1')[0]?.status).toBe('completed')
  })
})
