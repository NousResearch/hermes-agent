import { beforeEach, describe, expect, it } from 'vitest'

import {
  $backgroundStatusBySession,
  clearFinishedBackgroundProcesses,
  dismissBackgroundProcess,
  reconcileBackgroundProcesses
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

  it('maps negative exit code to stopped state', () => {
    reconcileBackgroundProcesses(SID, [exited('a', -15)])

    expect(items()[0]!.state).toBe('stopped')
    expect(items()[0]!.exitCode).toBe(-15)
  })

  it('maps exit code 1 to failed state', () => {
    reconcileBackgroundProcesses(SID, [exited('a', 1)])

    expect(items()[0]!.state).toBe('failed')
  })

  it('maps exit code 0 to done state', () => {
    reconcileBackgroundProcesses(SID, [exited('a', 0)])

    expect(items()[0]!.state).toBe('done')
  })
})

describe('clearFinishedBackgroundProcesses', () => {
  beforeEach(() => {
    $backgroundStatusBySession.set({})
  })

  it('keeps running rows and drops finished rows', () => {
    reconcileBackgroundProcesses(SID, [running('a'), exited('b', 0), exited('c', 1)])

    clearFinishedBackgroundProcesses(SID)

    expect(items().map(i => i.id)).toEqual(['a'])
  })

  it('does nothing when session id is empty', () => {
    reconcileBackgroundProcesses(SID, [exited('a', 0)])

    clearFinishedBackgroundProcesses('')

    expect(items().length).toBeGreaterThan(0)
  })
})
