import type { GatewayEvent } from '@hermes/shared'
import { act, cleanup } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { createClientSessionState } from '@/lib/chat-runtime'
import { $compactingSessions, markCompressDeferred, setSessionCompacting, takeCompressDeferred } from '@/store/compaction'
import { $notifications, clearNotifications } from '@/store/notifications'

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
    clearNotifications()
    takeCompressDeferred(SID)
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

  // Manual /compress pins `compressing` (methods_session._compress_live) and
  // always clears it with `ready` from that function's `finally`. The desktop
  // matched only the auto-compaction spelling, so /compress showed no phase at
  // all — the TUI has handled both since createGatewayEventHandler.ts:904.
  it('drives the compaction phase from the manual /compress spelling', () => {
    mountStream()
    setSessionCompacting(OTHER_SID, true)

    emit('status.update', { kind: 'compressing', text: '\u280b compressing 42 messages (~120,000 tok)\u2026' })
    expect($compactingSessions.get()).toEqual({ [OTHER_SID]: true, [SID]: true })

    emit('status.update', { kind: 'ready' })

    expect($compactingSessions.get()).toEqual({ [OTHER_SID]: true })
  })

  it('retires the manual /compress phase even when the compress aborted', () => {
    mountStream()

    emit('status.update', { kind: 'compressing', text: 'compressing\u2026' })
    expect($compactingSessions.get()).toEqual({ [SID]: true })

    // CompressionLockHeld / raise both land on the same `ready` edge.
    emit('status.update', { kind: 'ready' })

    expect($compactingSessions.get()).toEqual({})
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

  // A deferred /compress (#97948) answers `pending` and returns, so the
  // handler's summary + success toast never run. The terminal edge is the only
  // completion the client sees — before this it merely cleared a spinner and
  // the compression finished in total silence.
  it('announces a deferred /compress completion on the terminal edge', async () => {
    const hydrateFromStoredSession = vi.fn(async () => undefined)
    const states = new Map([[SID, { ...createClientSessionState(), storedSessionId: 'stored-1' }]])

    stream = renderMessageStream(SID, { hydrateFromStoredSession, states })
    markCompressDeferred(SID)

    await act(async () => {
      stream.handleEvent({
        payload: { kind: 'compacted', text: '\u2713 Context compaction complete' },
        session_id: SID,
        type: 'status.update'
      })
    })

    // Same notice id the compress handler used, so the pending toast is
    // replaced in place rather than stacked.
    expect($notifications.get().find(entry => entry.id === `session-compress:${SID}`)).toMatchObject({
      kind: 'success',
      message: '\u2713 Context compaction complete'
    })

    // Appended after the hydrate resolves, or the refresh would drop it.
    expect(hydrateFromStoredSession).toHaveBeenCalledWith(3, 'stored-1', SID)
    expect(stream.state(SID).messages.at(-1)).toMatchObject({ role: 'system' })
    expect(stream.text(SID)).toBe('\u2713 Context compaction complete')
  })

  it('stays silent on an auto-compaction the user never asked for', async () => {
    const states = new Map([[SID, { ...createClientSessionState(), storedSessionId: 'stored-1' }]])

    stream = renderMessageStream(SID, { states })

    await act(async () => {
      stream.handleEvent({ payload: { kind: 'compacting' }, session_id: SID, type: 'status.update' })
      stream.handleEvent({ payload: { kind: 'compacted' }, session_id: SID, type: 'status.update' })
    })

    expect($notifications.get()).toEqual([])
    expect(stream.state(SID).messages).toEqual([])
  })

  it('announces a deferred compress once, not on every later compaction', async () => {
    const states = new Map([[SID, { ...createClientSessionState(), storedSessionId: 'stored-1' }]])

    stream = renderMessageStream(SID, { states })
    markCompressDeferred(SID)

    await act(async () => {
      stream.handleEvent({ payload: { kind: 'ready' }, session_id: SID, type: 'status.update' })
    })
    expect(stream.state(SID).messages).toHaveLength(1)

    clearNotifications()
    await act(async () => {
      stream.handleEvent({ payload: { kind: 'compacted' }, session_id: SID, type: 'status.update' })
    })

    expect($notifications.get()).toEqual([])
    expect(stream.state(SID).messages).toHaveLength(1)
  })

  it('falls back to translated copy when the edge carries no text', async () => {
    const states = new Map([[SID, { ...createClientSessionState(), storedSessionId: 'stored-1' }]])

    stream = renderMessageStream(SID, { states })
    markCompressDeferred(SID)

    await act(async () => {
      stream.handleEvent({ payload: { kind: 'ready' }, session_id: SID, type: 'status.update' })
    })

    expect(stream.text(SID)).toBe('Context compression finished')
  })
})
