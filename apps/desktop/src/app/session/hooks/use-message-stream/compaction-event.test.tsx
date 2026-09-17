import type { GatewayEvent } from '@hermes/shared'
import { act, cleanup } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { createClientSessionState } from '@/lib/chat-runtime'
import { $compactingSessions, setSessionCompacting } from '@/store/compaction'

import { type MessageStreamHarness, renderMessageStream } from './test-harness'

const SID = 'session-1'
const OTHER_SID = 'session-2'
let stream: MessageStreamHarness

function mountStream() {
  stream = renderMessageStream(SID)
}

function emit(type: GatewayEvent['type'], payload: GatewayEvent['payload'] = {}) {
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

  // Manual /compress takes a different route through the gateway than
  // auto-compaction and tags its status `compressing`, not `compacting`:
  // methods_session.py pins the kind itself, while server.py::_status_update
  // only re-tags the auto path's generic "lifecycle" status. The desktop
  // listened for `compacting` alone, so the whole manual run — routinely two
  // minutes on a large session — showed no indicator at all.
  it('shows the compaction phase for a manual compress', () => {
    mountStream()

    emit('status.update', { kind: 'compressing', text: '⠋ compressing 328 messages (~85,980 tok)…' })

    expect($compactingSessions.get()).toEqual({ [SID]: true })
  })

  it('ends a manual compress on the same compacted edge as auto-compaction', () => {
    mountStream()

    emit('status.update', { kind: 'compressing', text: '⠋ compressing 12 messages (~4,000 tok)…' })
    emit('status.update', { kind: 'compacted' })

    expect($compactingSessions.get()).toEqual({})
  })

  it('clears a manual compress phase when the turn resumes', () => {
    mountStream()
    setSessionCompacting(OTHER_SID, true)

    emit('status.update', { kind: 'compressing', text: '⠋ compressing 40 messages…' })
    expect($compactingSessions.get()).toEqual({ [OTHER_SID]: true, [SID]: true })

    emit('message.delta', { text: 'back to work' })

    expect($compactingSessions.get()).toEqual({ [OTHER_SID]: true })
  })
})
