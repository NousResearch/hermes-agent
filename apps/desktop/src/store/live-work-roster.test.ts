import { afterEach, describe, expect, it } from 'vitest'

import { $liveWorkRoster, setLiveWorkRosterSource } from './live-work-roster'
import { $delegatingSessionIds } from './session-dot-state'
import { $subagentsBySession, type SubagentProgress } from './subagents'

const child = (status: SubagentProgress['status']) => ({ id: 'c1', index: 0, status }) as unknown as SubagentProgress

afterEach(() => {
  $liveWorkRoster.set({})
  $subagentsBySession.set({})
})

describe('backend roster feeds rows no mounted tile speaks for', () => {
  it('lights a conversation the roster reports with running children', () => {
    setLiveWorkRosterSource('ambient', [{ session_id: 'off-screen', subagents: 2, background: 0 }])

    expect($delegatingSessionIds.get()).toContain('off-screen')
  })

  it('keeps a live event-fed entry authoritative over a stale roster count', () => {
    setLiveWorkRosterSource('ambient', [{ session_id: 'open-pane', subagents: 1, background: 0 }])
    $subagentsBySession.set({ 'open-pane': [child('completed')] })

    expect($delegatingSessionIds.get()).not.toContain('open-pane')
  })
})
