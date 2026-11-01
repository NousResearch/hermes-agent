import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  $backgroundRunningSessionIds,
  $backgroundStatusBySession,
  dismissBackgroundProcess,
  reconcileBackgroundProcesses
} from './composer-status'
import { $liveWorkRoster, setLiveWorkRosterSource } from './live-work-roster'
import { $delegatingSessionIds } from './session-dot-state'
import { $subagentsBySession, type SubagentProgress } from './subagents'

const child = (status: SubagentProgress['status'], updatedAt: number) =>
  ({ id: 'c1', index: 0, status, updatedAt }) as unknown as SubagentProgress

afterEach(() => {
  vi.clearAllTimers()
  vi.useRealTimers()
  $liveWorkRoster.set({})
  $subagentsBySession.set({})
  $backgroundStatusBySession.set({})
})

describe('backend roster feeds rows no mounted tile speaks for', () => {
  it('lights a conversation the roster reports with running children', () => {
    setLiveWorkRosterSource('ambient', [{ session_id: 'off-screen', subagents: 2, background: 0 }])

    expect($delegatingSessionIds.get()).toContain('off-screen')
  })

  it('keeps an event-fed entry newer than the roster read authoritative', () => {
    setLiveWorkRosterSource('ambient', [{ session_id: 'open-pane', subagents: 1, background: 0 }], 1_000)
    $subagentsBySession.set({ 'open-pane': [child('completed', 2_000)] })

    expect($delegatingSessionIds.get()).not.toContain('open-pane')
  })

  it('never lets retained child history hide work a newer roster read saw', () => {
    $subagentsBySession.set({ 'old-pane': [child('completed', 1_000)] })
    setLiveWorkRosterSource('ambient', [{ session_id: 'old-pane', subagents: 1, background: 0 }], 2_000)

    expect($delegatingSessionIds.get()).toContain('old-pane')
  })

  it('never lets a retained background list hide a process a newer roster read saw', () => {
    $backgroundStatusBySession.set({ 'old-pane': [{ id: 'p1', state: 'done', title: 'sleep', type: 'background' }] })
    setLiveWorkRosterSource('ambient', [{ session_id: 'old-pane', subagents: 0, background: 1 }])

    expect($backgroundRunningSessionIds.get()).toContain('old-pane')
  })

  it('never stamps a local dismissal as a fresher backend snapshot than the roster read', () => {
    vi.useFakeTimers()
    vi.setSystemTime(1_000)
    reconcileBackgroundProcesses('pane', [
      { command: 'a', exit_code: 0, session_id: 'p1', status: 'exited' },
      { command: 'b', exit_code: 0, session_id: 'p2', status: 'exited' }
    ])
    setLiveWorkRosterSource('ambient', [{ session_id: 'pane', subagents: 0, background: 1 }], 2_000)

    vi.setSystemTime(3_000)
    dismissBackgroundProcess('pane', 'p1')

    expect($backgroundStatusBySession.get().pane?.map(item => item.id)).toEqual(['p2'])
    expect($backgroundRunningSessionIds.get()).toContain('pane')
  })
})
