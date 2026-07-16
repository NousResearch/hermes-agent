import { beforeEach, describe, expect, it, vi } from 'vitest'

import { reconcileActiveSubagents, SUBAGENT_LIVENESS_GRACE_MS, SUBAGENT_ORPHAN_GRACE_MS } from './subagent-liveness'
import { $subagentsBySession, type SubagentStatus, upsertSubagent } from './subagents'

const idsFor = (sid: string) => ($subagentsBySession.get()[sid] ?? []).map(item => item.id)
const snapshot = (activeIds: string[] = [], profile = 'default') => ({ activeIds, profile })

const spawn = (sid: string, id: string, status: SubagentStatus = 'running', profile = 'default', parentId?: string) =>
  upsertSubagent(
    sid,
    {
      goal: id,
      parent_id: parentId,
      status,
      subagent_id: id,
      task_index: 0
    },
    true,
    undefined,
    profile
  )

describe('subagent liveness reconciliation', () => {
  beforeEach(() => $subagentsBySession.set({}))

  it('removes stale running rows missing from an authoritative profile snapshot', () => {
    const nowSpy = vi.spyOn(Date, 'now').mockReturnValue(1_000)

    spawn('s1', 'stale')
    spawn('s2', 'orphan')
    nowSpy.mockReturnValue(2_000)
    spawn('s1', 'live')
    nowSpy.mockReturnValue(3_000)
    spawn('s1', 'done', 'completed')

    reconcileActiveSubagents([snapshot(['live'])], 1_000 + SUBAGENT_LIVENESS_GRACE_MS + 1)
    nowSpy.mockRestore()

    expect(idsFor('s1')).toEqual(['live', 'done'])
    expect($subagentsBySession.get().s2).toBeUndefined()
  })

  it('keeps recent rows while gateway events and the registry race', () => {
    const nowSpy = vi.spyOn(Date, 'now').mockReturnValue(1_000)

    spawn('s1', 'starting')
    reconcileActiveSubagents([snapshot()], 1_000 + SUBAGENT_LIVENESS_GRACE_MS - 1)
    expect(idsFor('s1')).toEqual(['starting'])

    reconcileActiveSubagents([snapshot()], 1_000 + SUBAGENT_LIVENESS_GRACE_MS + 1)
    nowSpy.mockRestore()
    expect($subagentsBySession.get().s1).toBeUndefined()
  })

  it('gives queued and fallback rows a longer grace without keeping them forever', () => {
    const nowSpy = vi.spyOn(Date, 'now').mockReturnValue(1_000)

    spawn('s1', 'queued', 'queued')
    spawn('s1', 'delegate-tool:call:0')
    reconcileActiveSubagents([snapshot()], 1_000 + SUBAGENT_LIVENESS_GRACE_MS + 1)
    expect(idsFor('s1')).toEqual(['queued', 'delegate-tool:call:0'])

    reconcileActiveSubagents([snapshot()], 1_000 + SUBAGENT_ORPHAN_GRACE_MS + 1)
    nowSpy.mockRestore()
    expect($subagentsBySession.get().s1).toBeUndefined()
  })

  it('keeps a missing parent while an authoritative child is active', () => {
    const nowSpy = vi.spyOn(Date, 'now').mockReturnValue(1_000)

    spawn('s1', 'parent')
    spawn('s1', 'child', 'running', 'default', 'parent')
    reconcileActiveSubagents([snapshot(['child'])], 1_000 + SUBAGENT_LIVENESS_GRACE_MS + 1)
    nowSpy.mockRestore()

    expect(idsFor('s1')).toEqual(['parent', 'child'])
  })

  it('reconciles healthy profiles without trusting an unavailable profile', () => {
    const nowSpy = vi.spyOn(Date, 'now').mockReturnValue(1_000)

    spawn('local', 'local', 'running', 'default')
    spawn('remote', 'remote', 'running', 'remote')
    reconcileActiveSubagents([snapshot()], 1_000 + SUBAGENT_LIVENESS_GRACE_MS + 1, false)
    nowSpy.mockRestore()

    expect($subagentsBySession.get().local).toBeUndefined()
    expect(idsFor('remote')).toEqual(['remote'])
    reconcileActiveSubagents([snapshot()], 1_000 + SUBAGENT_ORPHAN_GRACE_MS * 2, false)
    expect(idsFor('remote')).toEqual(['remote'])
  })

  it('does not let an active id in one profile retain a stale row in another', () => {
    const nowSpy = vi.spyOn(Date, 'now').mockReturnValue(1_000)

    spawn('local', 'shared-id', 'running', 'default')
    spawn('remote', 'shared-id', 'running', 'remote')
    reconcileActiveSubagents(
      [snapshot([], 'default'), snapshot(['shared-id'], 'remote')],
      1_000 + SUBAGENT_LIVENESS_GRACE_MS + 1
    )
    nowSpy.mockRestore()

    expect($subagentsBySession.get().local).toBeUndefined()
    expect(idsFor('remote')).toEqual(['shared-id'])
  })

  it('keeps unprofiled legacy ancestors without consulting another explicit profile', () => {
    const nowSpy = vi.spyOn(Date, 'now').mockReturnValue(1_000)

    upsertSubagent('legacy', { goal: 'grandparent', status: 'running', subagent_id: 'grandparent', task_index: 0 })
    upsertSubagent('legacy', {
      goal: 'parent',
      parent_id: 'grandparent',
      status: 'running',
      subagent_id: 'parent',
      task_index: 0
    })
    upsertSubagent('legacy', {
      goal: 'child',
      parent_id: 'parent',
      status: 'running',
      subagent_id: 'child',
      task_index: 0
    })
    spawn('other', 'parent', 'running', 'other')
    reconcileActiveSubagents(
      [snapshot(['child'], 'remote'), snapshot([], 'other')],
      1_000 + SUBAGENT_LIVENESS_GRACE_MS + 1
    )
    nowSpy.mockRestore()

    expect(idsFor('legacy')).toEqual(['grandparent', 'parent', 'child'])
    expect($subagentsBySession.get().other).toBeUndefined()
  })
})
