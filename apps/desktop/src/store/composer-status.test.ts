import { beforeEach, describe, expect, it } from 'vitest'

import {
  $backgroundStatusBySession,
  $goalStatusBySession,
  dismissBackgroundProcess,
  groupStatusItems,
  reconcileBackgroundProcesses,
  setGoalStatusFromText
} from './composer-status'

const SID = 'sess-1'

const running = (id: string, command = `cmd ${id}`) => ({ command, session_id: id, status: 'running' })

const exited = (id: string, exit_code = 0, command = `cmd ${id}`) => ({
  command,
  exit_code,
  session_id: id,
  status: 'exited'
})

const items = () => $backgroundStatusBySession.get()[SID] ?? []

describe('reconcileBackgroundProcesses', () => {
  beforeEach(() => {
    $backgroundStatusBySession.set({})
    $goalStatusBySession.set({})
  })

  it('maps registry entries to status items', () => {
    reconcileBackgroundProcesses(SID, [running('a'), exited('b', 0), exited('c', 1)])

    expect(items().map(i => [i.id, i.state])).toEqual([
      ['a', 'running'],
      ['b', 'done'],
      ['c', 'failed']
    ])
    expect(items()[2]!.exitCode).toBe(1)
  })

  it('keeps row order stable when a process flips state or the snapshot reorders', () => {
    reconcileBackgroundProcesses(SID, [running('a'), running('b')])
    // Snapshot arrives reordered AND `a` has exited — rows must not move.
    reconcileBackgroundProcesses(SID, [running('b'), exited('a', 0)])

    expect(items().map(i => [i.id, i.state])).toEqual([
      ['a', 'done'],
      ['b', 'running']
    ])
  })

  it('appends new processes after existing rows', () => {
    reconcileBackgroundProcesses(SID, [running('a')])
    reconcileBackgroundProcesses(SID, [running('b'), running('a')])

    expect(items().map(i => i.id)).toEqual(['a', 'b'])
  })

  it('preserves object identity for unchanged rows (memo stability)', () => {
    reconcileBackgroundProcesses(SID, [running('a'), running('b')])
    const [a1] = items()

    reconcileBackgroundProcesses(SID, [running('a'), exited('b', 0)])
    const [a2, b2] = items()

    expect(a2).toBe(a1)
    expect(b2!.state).toBe('done')
  })

  it('is a no-op store write when nothing changed', () => {
    reconcileBackgroundProcesses(SID, [running('a')])
    const before = $backgroundStatusBySession.get()

    reconcileBackgroundProcesses(SID, [running('a')])

    expect($backgroundStatusBySession.get()).toBe(before)
  })

  it('never resurrects a dismissed process while the registry still reports it', () => {
    reconcileBackgroundProcesses(SID, [exited('a', 0), running('b')])
    dismissBackgroundProcess(SID, 'a')

    reconcileBackgroundProcesses(SID, [exited('a', 0), running('b')])

    expect(items().map(i => i.id)).toEqual(['b'])
  })

  it('forgets a dismissal once the registry prunes the process', () => {
    reconcileBackgroundProcesses(SID, [exited('a', 0)])
    dismissBackgroundProcess(SID, 'a')

    // Registry pruned it…
    reconcileBackgroundProcesses(SID, [])
    // …so a future process reusing the id (new spawn) shows again.
    reconcileBackgroundProcesses(SID, [running('a')])

    expect(items().map(i => i.id)).toEqual(['a'])
  })

  it('drops the session key entirely when the last row goes away', () => {
    reconcileBackgroundProcesses(SID, [running('a')])
    reconcileBackgroundProcesses(SID, [])

    expect($backgroundStatusBySession.get()).toEqual({})
  })
})

describe('goal status items', () => {
  beforeEach(() => {
    $goalStatusBySession.set({})
  })

  it('parses goal set notices into a status row', () => {
    setGoalStatusFromText(SID, '⊙ Goal set (20-turn budget): fix the desktop slash bug')

    expect($goalStatusBySession.get()[SID]).toMatchObject({
      goalStatus: 'active',
      maxTurns: 20,
      state: 'running',
      title: 'fix the desktop slash bug',
      type: 'goal'
    })
  })

  it('updates judge turn metadata and clears on stop output', () => {
    setGoalStatusFromText(SID, '⊙ Goal set (20-turn budget): fix the desktop slash bug')
    setGoalStatusFromText(SID, '↻ Continuing toward goal (3/20): still working')

    expect($goalStatusBySession.get()[SID]).toMatchObject({
      goalVerdict: 'continue',
      reason: 'still working',
      turnsUsed: 3
    })

    setGoalStatusFromText(SID, 'Goal cleared')

    expect($goalStatusBySession.get()[SID]).toBeUndefined()
  })

  it('clears stale goal rows when backend reports no active goal', () => {
    setGoalStatusFromText(SID, '⊙ Goal set (20-turn budget): fix the desktop slash bug')
    setGoalStatusFromText(SID, 'No active goal. Set one with /goal <text>.')

    expect($goalStatusBySession.get()[SID]).toBeUndefined()
  })

  it('parses backend goal status lines into status rows', () => {
    setGoalStatusFromText(SID, '⊙ Goal (active, 2/20 turns): keep this thread alive')

    expect($goalStatusBySession.get()[SID]).toMatchObject({
      goalStatus: 'active',
      maxTurns: 20,
      state: 'running',
      title: 'keep this thread alive',
      turnsUsed: 2
    })

    setGoalStatusFromText(SID, '✓ Goal done (4/20 turns): keep this thread alive')

    expect($goalStatusBySession.get()[SID]).toMatchObject({
      goalStatus: 'done',
      state: 'done',
      title: 'keep this thread alive',
      turnsUsed: 4
    })
  })

  it('groups goal rows before todos and background rows', () => {
    const groups = groupStatusItems([
      { id: 'bg', state: 'running', title: 'bg', type: 'background' },
      { id: 'todo:a', state: 'running', title: 'todo', todoStatus: 'in_progress', type: 'todo' },
      { goalStatus: 'active', id: 'goal', state: 'running', title: 'goal', type: 'goal' }
    ])

    expect(groups.map(group => group.type)).toEqual(['goal', 'todo', 'background'])
  })
})
