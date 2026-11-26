import type { GatewayEvent } from '@hermes/shared'
// A reply its interrupter will continue (a host plugin stopped the turn to apply
// a grant: session.interrupt with `resume`) ends with message.complete
// {status: interrupted, paused: true}. Until the continuation's turn completes,
// the chat is paused: its reply reads "Continuing…" until that turn starts, and a
// send waits in the queue instead of barging into the continuation
// (store/paused-replies.ts). A sandbox restart in between gives the chat a new
// runtime session, so the pause belongs to the stored chat.
import { act, cleanup } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { createClientSessionState } from '@/lib/chat-runtime'
import { endReplyPause, isReplyAwaitingContinuation, isReplyPaused, PAUSE_CAP_MS } from '@/store/paused-replies'
import { $sessions, resolveComposerSessionKey } from '@/store/session'
import type { SessionInfo } from '@/types/hermes'

import { type MessageStreamHarness, renderMessageStream } from './test-harness'

const CHAT = 'paused-reply-chat'
const BEFORE = 'paused-reply-runtime-before-restart'
const AFTER = 'paused-reply-runtime-after-restart'

let stream: MessageStreamHarness

const emit = (sessionId: string, event: Omit<GatewayEvent, 'session_id'>) =>
  act(() => stream.handleEvent({ ...event, session_id: sessionId }))

const chatState = (storedSessionId = CHAT): ClientSessionState => ({ ...createClientSessionState(), storedSessionId })

function mountRunningTurn() {
  vi.useFakeTimers()
  stream = renderMessageStream(BEFORE, {
    states: new Map([
      [BEFORE, chatState()],
      [AFTER, chatState()]
    ])
  })
  emit(BEFORE, { payload: {}, type: 'message.start' })
  emit(BEFORE, {
    payload: { args: { path: 'DELIVERY-PLAN.md' }, name: 'read_file', tool_id: 'call-1' },
    type: 'tool.start'
  })
}

describe('a reply paused for an automatic continuation', () => {
  afterEach(() => {
    cleanup()
    endReplyPause(CHAT)
    $sessions.set([])
    vi.useRealTimers()
  })

  it('stays paused from its paused completion until the continuation completes, across a backend restart', () => {
    mountRunningTurn()

    emit(BEFORE, { payload: { paused: true, status: 'interrupted', text: '' }, type: 'message.complete' })
    expect(isReplyPaused(CHAT)).toBe(true)
    expect(isReplyAwaitingContinuation(CHAT)).toBe(true)

    // The sandbox restarted: the chat is back on a new runtime session, and the
    // plugin's hidden continuation starts its turn there. The reply goes on (no
    // "Continuing…" row any more), but a send still waits for the turn to end.
    emit(AFTER, { payload: {}, type: 'message.start' })
    expect(isReplyAwaitingContinuation(CHAT)).toBe(false)
    expect(isReplyPaused(CHAT)).toBe(true)

    emit(AFTER, { payload: { status: 'complete', text: 'Read it.' }, type: 'message.complete' })
    expect(isReplyPaused(CHAT)).toBe(false)
  })

  // A compressed chat has several ids: the runtime shows its newest (the compression tip), while the composer
  // and the queue are keyed by its lineage root (resolveComposerSessionKey). The pause is the conversation's.
  it('holds a compressed chat under every id of its lineage, to its continuation on another of them', () => {
    const ROOT = 'paused-reply-lineage-root'
    const TIP = 'paused-reply-compression-tip'
    $sessions.set([{ _lineage_root_id: ROOT, id: TIP } as SessionInfo])
    vi.useFakeTimers()
    stream = renderMessageStream(BEFORE, {
      states: new Map([
        [BEFORE, chatState(TIP)],
        [AFTER, chatState(ROOT)]
      ])
    })
    emit(BEFORE, { payload: {}, type: 'message.start' })

    emit(BEFORE, { payload: { paused: true, status: 'interrupted', text: '' }, type: 'message.complete' })
    expect(isReplyPaused(resolveComposerSessionKey(TIP, $sessions.get()))).toBe(true)
    expect(isReplyAwaitingContinuation(TIP)).toBe(true)
    expect(isReplyPaused(CHAT)).toBe(false)

    emit(AFTER, { payload: {}, type: 'message.start' })
    expect(isReplyAwaitingContinuation(TIP)).toBe(false)
    expect(isReplyPaused(ROOT)).toBe(true)

    emit(AFTER, { payload: { status: 'complete', text: 'Read it.' }, type: 'message.complete' })
    expect(isReplyPaused(ROOT, TIP)).toBe(false)
  })

  it('is paused again when the continuation itself is paused (a second grant)', () => {
    mountRunningTurn()
    emit(BEFORE, { payload: { paused: true, status: 'interrupted', text: '' }, type: 'message.complete' })
    emit(AFTER, { payload: {}, type: 'message.start' })

    emit(AFTER, { payload: { paused: true, status: 'interrupted', text: '' }, type: 'message.complete' })

    expect(isReplyAwaitingContinuation(CHAT)).toBe(true)
  })

  it('gives a continuation that started the cap again, then lets sends through', () => {
    mountRunningTurn()
    emit(BEFORE, { payload: { paused: true, status: 'interrupted', text: '' }, type: 'message.complete' })

    act(() => {
      vi.advanceTimersByTime(PAUSE_CAP_MS - 1_000)
    })
    emit(AFTER, { payload: {}, type: 'message.start' })
    act(() => {
      vi.advanceTimersByTime(PAUSE_CAP_MS - 1_000)
    })
    expect(isReplyPaused(CHAT)).toBe(true)

    act(() => {
      vi.advanceTimersByTime(1_000)
    })
    expect(isReplyPaused(CHAT)).toBe(false)
  })

  it('is not paused by a plain interrupt (a Stop here or in another window)', () => {
    mountRunningTurn()

    emit(BEFORE, { payload: { status: 'interrupted', text: '' }, type: 'message.complete' })

    expect(isReplyPaused(CHAT)).toBe(false)
  })

  it('reads as stopped again when nothing continues it', () => {
    mountRunningTurn()
    emit(BEFORE, { payload: { paused: true, status: 'interrupted', text: '' }, type: 'message.complete' })

    act(() => {
      vi.advanceTimersByTime(PAUSE_CAP_MS)
    })

    expect(isReplyPaused(CHAT)).toBe(false)
  })
})
