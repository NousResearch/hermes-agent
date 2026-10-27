import { afterEach, describe, expect, it } from 'vitest'

import { $clarifyRequests } from './clarify'
import { clearApprovalRequest, setApprovalRequest } from './prompts'
import { $rosterSessionSignals } from './roster-session-signals'
import { $cronSessions, $sessions } from './session'
import { $sessionStates } from './session-states'

afterEach(() => {
  $clarifyRequests.set({})
  clearApprovalRequest()
  $sessionStates.set({})
  $cronSessions.set([])
  $sessions.set([])
})

describe('$rosterSessionSignals', () => {
  it('quotes a blocking prompt under the STORED id its runtime belongs to', () => {
    $sessionStates.set({ 'rt-1': { storedSessionId: 'stored-1', busy: false, messages: [] } } as never)
    $clarifyRequests.set({
      'rt-1': {
        requestId: 'r1',
        sessionId: 'rt-1',
        questions: [{ qid: 'q', question: 'Which region?', choices: null, multiSelect: false }]
      }
    })

    expect($rosterSessionSignals.get().prompts['stored-1']).toEqual({ kind: 'clarify', text: 'Which region?' })

    setApprovalRequest({ sessionId: 'rt-1', command: 'terraform apply', description: 'dangerous' })
    expect($rosterSessionSignals.get().prompts['stored-1']).toEqual({ kind: 'approval', text: 'terraform apply' })

    $clarifyRequests.set({})
    clearApprovalRequest()
    expect($rosterSessionSignals.get().prompts['stored-1']).toBeUndefined()
  })

  it('quotes the question a partly answered batch still waits on', () => {
    $clarifyRequests.set({
      'rt-1': {
        requestId: 'r1',
        sessionId: 'rt-1',
        questions: [
          { qid: 'q0', question: 'Which region?', choices: null, multiSelect: false },
          { qid: 'q1', question: 'Which instance size?', choices: null, multiSelect: false }
        ],
        lockedAnswers: { q0: 'us-east-1' }
      }
    })

    expect($rosterSessionSignals.get().prompts['rt-1']).toEqual({ kind: 'clarify', text: 'Which instance size?' })
  })

  it('keeps a live prompt map by reference across an unrelated recompute', () => {
    setApprovalRequest({ sessionId: 'rt-1', command: 'terraform apply', description: 'dangerous' })
    const before = $rosterSessionSignals.get().prompts

    $cronSessions.set([{ id: 'c9', profile: 'alpha', scheduler_owned: false, is_active: true }] as never)

    expect($rosterSessionSignals.get().prompts).toBe(before)
  })

  it('lists only cron runs the scheduler still owns, scoped by connection and profile', () => {
    $cronSessions.set([
      { id: 'c1', profile: 'alpha', scheduler_owned: true, is_active: true },
      { id: 'c2', profile: 'beta', connection_id: 'box', scheduler_owned: true, is_active: true },
      { id: 'c3', profile: 'gamma', scheduler_owned: false, is_active: true }
    ] as never)

    expect($rosterSessionSignals.get().routines).toEqual(['::alpha', 'box::beta'])
  })

  it('lights a listed in-flight run the list never stamped with scheduler_owned', () => {
    // The sessions LIST row of a run the scheduler is executing right now:
    // open, inside the activity window, no ownership stamp.
    $cronSessions.set([
      { id: 'c1', profile: 'alpha', ended_at: null, is_active: true },
      { id: 'c2', profile: 'beta', ended_at: 1_700_000_000, is_active: true },
      { id: 'c3', profile: 'gamma', ended_at: null, is_active: false }
    ] as never)

    expect($rosterSessionSignals.get().routines).toEqual(['::alpha'])
  })

  it('keeps its reference while nothing it reports changed', () => {
    const first = $rosterSessionSignals.get()
    // A finished cron run changes the input list but nothing this reports.
    $cronSessions.set([{ id: 'c9', profile: 'alpha', scheduler_owned: false, is_active: true }] as never)

    expect($rosterSessionSignals.get()).toBe(first)
  })
})
