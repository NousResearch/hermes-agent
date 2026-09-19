import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import {
  $connectionRequests,
  applyConnectionUpdate,
  type ConnectionRequest,
  normalizeConnectionRequest,
  setConnectionRequest
} from '@/store/connection-request'

import { restorePendingConnectionFromSnapshot } from './restore-pending-connection'

const SESSION_ID = 'session-1'

const SNAPSHOT = {
  deadline_at: 1_800_000_000,
  op_id: 'op-1',
  seq: 3,
  timeout_seconds: 300,
  targets: [{ action: 'connect' as const, kind: 'connector' as const, name: 'gmail', state: 'pending' as const }],
  tool_call_id: 'call-1'
}

const cached = (overrides: Partial<ConnectionRequest> = {}): ConnectionRequest => ({
  ...normalizeConnectionRequest(SNAPSHOT, SESSION_ID)!,
  ...overrides
})

beforeEach(() => {
  $connectionRequests.set({})
})

afterEach(() => {
  $connectionRequests.set({})
})

describe('restoring a pending connection from a resume snapshot', () => {
  it('never revives a card the operation already settled', () => {
    const settled = cached({ settled: true, settledBy: 'continue' })
    setConnectionRequest(settled)

    const state = restorePendingConnectionFromSnapshot({ pending_connection: SNAPSHOT }, SESSION_ID, Date.now() / 1000)

    expect($connectionRequests.get()[SESSION_ID]).toBe(settled)
    // A refused snapshot is no pending card: the caller must not flag the session as needing
    // input behind a summary that has no controls.
    expect(state.request).toBeNull()
  })

  it('a settled card for one operation does not hide the next operation the session opened', () => {
    setConnectionRequest(cached({ opId: 'op-0', settled: true, settledBy: 'all_resolved' }))

    const state = restorePendingConnectionFromSnapshot({ pending_connection: SNAPSHOT }, SESSION_ID, Date.now() / 1000)

    expect(state.request?.opId).toBe('op-1')
    expect($connectionRequests.get()[SESSION_ID].opId).toBe('op-1')
    expect($connectionRequests.get()[SESSION_ID].settled).toBe(false)
  })

  it('never regresses a row a newer frame already moved', () => {
    const live = applyConnectionUpdate(cached(), {
      deadline_at: SNAPSHOT.deadline_at,
      op_id: 'op-1',
      owner: { session_id: SESSION_ID, type: 'session' },
      seq: 7,
      settled: false,
      targets: [{ action: 'connect', kind: 'connector', name: 'gmail', state: 'connected' }]
    })

    setConnectionRequest(live)

    const state = restorePendingConnectionFromSnapshot({ pending_connection: SNAPSHOT }, SESSION_ID, Date.now() / 1000)

    expect($connectionRequests.get()[SESSION_ID]).toBe(live)
    expect($connectionRequests.get()[SESSION_ID].targets[0].state).toBe('connected')
    // The live card is still the pending card: the session keeps waiting on it.
    expect(state.request).toBe(live)
  })

  it('takes the snapshot when it is the newer word on the operation', () => {
    setConnectionRequest(cached({ seq: 1 }))

    const newer = { ...SNAPSHOT, seq: 4 }
    const state = restorePendingConnectionFromSnapshot({ pending_connection: newer }, SESSION_ID, Date.now() / 1000)

    expect(state.request?.seq).toBe(4)
    expect($connectionRequests.get()[SESSION_ID].seq).toBe(4)
  })
})
