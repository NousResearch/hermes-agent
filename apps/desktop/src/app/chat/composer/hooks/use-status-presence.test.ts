import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { $composerActionsBySession } from '@/store/composer-actions'
import { $previewStatusBySession } from '@/store/preview-status'
import { $sessionControlBySession, type SessionControlEntry } from '@/store/session-control'
import { $todosBySession } from '@/store/todos'

import { useSessionStatusPresence } from './use-status-presence'

const SID = 'presence-session-1'

const mockEntry = (overrides?: Partial<SessionControlEntry>): SessionControlEntry => ({
  capability: 'supported',
  error: null,
  loading: false,
  pendingAction: null,
  snapshot: {
    goal: null,
    heartbeat: null,
    loop: null,
    revision: 'rev-1',
    updated_at: 1000
  },
  ...overrides
})

describe('useSessionStatusPresence', () => {
  beforeEach(() => {
    $todosBySession.set({})
    $composerActionsBySession.set({})
    $previewStatusBySession.set({})
    $sessionControlBySession.set({})
  })

  afterEach(() => {
    cleanup()
    $todosBySession.set({})
    $composerActionsBySession.set({})
    $previewStatusBySession.set({})
    $sessionControlBySession.set({})
  })

  it('returns false when session is null or empty', () => {
    const { result } = renderHook(() => useSessionStatusPresence(null))
    expect(result.current).toBe(false)

    const { result: emptyResult } = renderHook(() => useSessionStatusPresence(SID))
    expect(emptyResult.current).toBe(false)
  })

  it('returns true when legacy status items exist', () => {
    const { result } = renderHook(() => useSessionStatusPresence(SID))
    expect(result.current).toBe(false)

    act(() => {
      $todosBySession.set({
        [SID]: [{ content: 'task 1', id: '1', status: 'in_progress' }]
      })
    })

    expect(result.current).toBe(true)
  })

  it('returns true when structured goal exists in session control', () => {
    const { result } = renderHook(() => useSessionStatusPresence(SID))
    expect(result.current).toBe(false)

    act(() => {
      $sessionControlBySession.set({
        [SID]: mockEntry({
          snapshot: {
            goal: {
              contract: {
                boundaries: '',
                constraints: '',
                outcome: 'test outcome',
                stop_when: '',
                verification: ''
              },
              gates: [],
              max_turns: 10,
              status: 'active',
              subgoals: [],
              title: 'Structured Goal',
              turns_used: 1
            },
            heartbeat: null,
            loop: null,
            revision: 'rev-2',
            updated_at: 2000
          }
        })
      })
    })

    expect(result.current).toBe(true)
  })

  it('returns true when structured loop exists in session control', () => {
    const { result } = renderHook(() => useSessionStatusPresence(SID))
    expect(result.current).toBe(false)

    act(() => {
      $sessionControlBySession.set({
        [SID]: mockEntry({
          snapshot: {
            goal: null,
            heartbeat: null,
            loop: {
              awaiting_response: false,
              created_at: 1000,
              current_delay: 60,
              deferred_by_goal: false,
              interval_seconds: 60,
              last_fired_at: 1000,
              max_ticks: 10,
              mode: 'interval',
              next_due_at: 2000,
              prompt: 'Run loop',
              status: 'active',
              ticks_fired: 0,
              times: 5,
              until: ''
            },
            revision: 'rev-3',
            updated_at: 2000
          }
        })
      })
    })

    expect(result.current).toBe(true)
  })

  it('returns true when structured heartbeat exists in session control', () => {
    const { result } = renderHook(() => useSessionStatusPresence(SID))
    expect(result.current).toBe(false)

    act(() => {
      $sessionControlBySession.set({
        [SID]: mockEntry({
          snapshot: {
            goal: null,
            heartbeat: {
              created_at: 1000,
              fire_count: 1,
              interval_seconds: 300,
              last_fired_at: 1000,
              prompt: 'Heartbeat check',
              status: 'active'
            },
            loop: null,
            revision: 'rev-4',
            updated_at: 2000
          }
        })
      })
    })

    expect(result.current).toBe(true)
  })

  it('returns false when session control entry has empty snapshot (null goal, loop, heartbeat)', () => {
    const { result } = renderHook(() => useSessionStatusPresence(SID))
    expect(result.current).toBe(false)

    act(() => {
      $sessionControlBySession.set({
        [SID]: mockEntry({
          snapshot: {
            goal: null,
            heartbeat: null,
            loop: null,
            revision: 'rev-empty',
            updated_at: 2000
          }
        })
      })
    })

    expect(result.current).toBe(false)
  })

  it('returns true when session control entry has only an error', () => {
    const { result } = renderHook(() => useSessionStatusPresence(SID))
    expect(result.current).toBe(false)

    act(() => {
      $sessionControlBySession.set({
        [SID]: mockEntry({
          error: 'Gateway connection failed',
          snapshot: null
        })
      })
    })

    expect(result.current).toBe(true)
  })
})
