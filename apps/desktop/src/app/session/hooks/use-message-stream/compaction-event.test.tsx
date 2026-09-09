import { act, cleanup } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { createClientSessionState } from '@/lib/chat-runtime'
import { $compactingSessions, setSessionCompacting } from '@/store/compaction'
import type { RpcEvent } from '@/types/hermes'

import { type MessageStreamHarness, renderMessageStream } from './test-harness'

const SID = 'session-1'
const OTHER_SID = 'session-2'
let stream: MessageStreamHarness

function mountStream() {
  stream = renderMessageStream(SID)
}

function emit(type: RpcEvent['type'], payload: RpcEvent['payload'] = {}) {
  act(() => stream.handleEvent({ payload, session_id: SID, type }))
}

describe('useMessageStream compaction lifecycle', () => {
  beforeEach(() => {
    $compactingSessions.set({})
  })

  afterEach(() => {
    cleanup()
    $compactingSessions.set({})
    vi.restoreAllMocks()
  })

  it.each([
    ['message.delta', { text: 'resumed' }],
    ['thinking.delta', { text: 'still working' }],
    ['reasoning.delta', { text: 'thinking again' }],
    ['tool.start', { name: 'terminal', tool_id: 'tool-1' }]
  ] as const)('clears the stale compaction phase when %s resumes the turn', (type, payload) => {
    mountStream()
    setSessionCompacting(OTHER_SID, true)

    emit('status.update', { kind: 'compacting' })
    expect($compactingSessions.get()).toEqual({ [OTHER_SID]: true, [SID]: true })

    emit(type, payload)

    expect($compactingSessions.get()).toEqual({ [OTHER_SID]: true })
  })

  it('clears the compaction phase on the structured completion edge', () => {
    mountStream()
    setSessionCompacting(OTHER_SID, true)

    emit('status.update', { kind: 'compacting' })
    emit('status.update', { kind: 'compacted' })

    expect($compactingSessions.get()).toEqual({ [OTHER_SID]: true })
  })

  // #97948: a manual /compress whose RPC answered `pending` (the compute host
  // outlived the gateway's wait) has no turn-end hydrate — the `compacted`
  // edge is the only signal the transcript changed.
  it('rehydrates the idle active session on the compacted edge', () => {
    const hydrateFromStoredSession = vi.fn(async () => undefined)
    const states = new Map([[SID, { ...createClientSessionState(), storedSessionId: 'stored-1' }]])

    stream = renderMessageStream(SID, { hydrateFromStoredSession, states })

    emit('status.update', { kind: 'compacted' })

    expect(hydrateFromStoredSession).toHaveBeenCalledWith(3, 'stored-1', SID)
  })

  it('leaves the transcript to the turn settle path when compaction ends mid-turn', () => {
    const hydrateFromStoredSession = vi.fn(async () => undefined)
    const states = new Map([[SID, { ...createClientSessionState(), busy: true, storedSessionId: 'stored-1' }]])

    stream = renderMessageStream(SID, { hydrateFromStoredSession, states })

    emit('status.update', { kind: 'compacted' })

    expect(hydrateFromStoredSession).not.toHaveBeenCalled()
  })

  it('reconciles a reconnecting compaction only from trusted terminal server state', () => {
    mountStream()
    emit('status.update', { kind: 'compacting' })

    // A running heartbeat is not terminal evidence and must not hide real work.
    emit('session.info', { running: true })
    expect($compactingSessions.get()).toEqual({ [SID]: true })

    // A server-reported terminal turn is trusted reconnect evidence.
    emit('session.info', { running: false })
    expect($compactingSessions.get()).toEqual({})
  })
})
