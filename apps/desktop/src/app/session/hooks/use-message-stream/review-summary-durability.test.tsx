import { act, cleanup } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { graftRefreshedTailOntoBackfill } from '@/app/chat/transcript-backfill'
import { chatMessageText, preserveLocalAssistantErrors, toChatMessages } from '@/lib/chat-messages'
import type { SessionMessage } from '@/types/hermes'

import { renderMessageStream } from './test-harness'

const SID = 'review-session'
const TEXT = "💾 Self-improvement review: Skill 'canary' patched"

const base: SessionMessage[] = [
  { id: 1, role: 'user', content: 'Inspect it', timestamp: 100 },
  { id: 2, role: 'assistant', content: 'Done', timestamp: 101 }
]

const stored = (id: number, reviewId: string): SessionMessage => ({
  id,
  role: 'system',
  content: TEXT,
  timestamp: 100 + id,
  display_kind: 'review_summary',
  display_metadata: { review_id: reviewId }
})

const refresh = (rows: SessionMessage[], previous: ReturnType<typeof toChatMessages>, beforeRead = previous) =>
  preserveLocalAssistantErrors(graftRefreshedTailOntoBackfill(toChatMessages(rows), previous), previous, beforeRead)

afterEach(cleanup)

describe('durable review confirmations', () => {
  it('uses one identity for live delivery, replay, REST hydration and cold reopening', () => {
    const stream = renderMessageStream(SID)
    const payload = { text: TEXT, review_id: 'review-a', row_id: 3, timestamp: 103, stored_session_id: SID }
    act(() => {
      stream.handleEvent({ type: 'review.summary', session_id: SID, payload })
      stream.handleEvent({ type: 'review.summary', session_id: SID, payload })
    })
    expect(stream.state(SID).messages).toHaveLength(1)
    const live = stream.state(SID).messages[0]
    const cold = toChatMessages([stored(3, 'review-a')])[0]
    expect(cold.id).toBe(live.id)
    expect(chatMessageText(cold)).toBe(chatMessageText(live))
    expect(cold.timestamp).toBe(103)
    const refreshed = refresh([...base, stored(3, 'review-a')], [...toChatMessages(base), live])
    expect(refreshed.filter(message => message.role === 'system')).toHaveLength(1)
    expect(refreshed.at(-1)?.id).toBe(live.id)
  })

  it('keeps a receipt newer than an in-flight snapshot without merging distinct reviews by text', () => {
    const stream = renderMessageStream(SID)
    act(() =>
      stream.handleEvent({
        type: 'review.summary',
        session_id: SID,
        payload: { text: TEXT, review_id: 'review-a', row_id: 3, timestamp: 103 }
      })
    )
    const previous = [...toChatMessages(base), ...stream.state(SID).messages]
    const stale = refresh(base, previous, toChatMessages(base))
    expect(stale.filter(message => message.role === 'system')).toHaveLength(1)
    const current = refresh([...base, stored(3, 'review-a'), stored(4, 'review-b')], stale)
    const reviews = current.filter(message => message.role === 'system')
    expect(reviews).toHaveLength(2)
    expect(reviews[0].id).not.toBe(reviews[1].id)
    // An authoritative rewind started after delivery must not resurrect its removed receipt.
    expect(refresh(base, current).filter(message => message.role === 'system')).toHaveLength(0)
    // An authoritative empty/reset transcript must not resurrect old receipts.
    expect(refresh([], current)).toHaveLength(0)
  })
})

// The same stored id in a focused A chat must not swallow a hidden B receipt.
// Exercise the real unread writer as well: a visible A row is not B's owner.
describe('review receipt owner scope', () => {
  it.each([false, true])('marks hidden profile B, not focused A (A row loaded: %s)', async loaded => {
    const { $activeGatewayProfile } = await import('@/store/profile')

    const { $activeSessionId, $selectedStoredSessionId, $sessions, $unreadFinishedSessionIds } =
      await import('@/store/session')

    const { $unreadFinishedMarkers } = await import('@/store/session-unread')
    const { makeSessionInfo } = await import('@/test/session-info')
    const { clearAllSessionStates } = await import('@/store/session-states')
    clearAllSessionStates()
    act(() => {
      $activeGatewayProfile.set('A')
      $activeSessionId.set('runtime-A')
      $selectedStoredSessionId.set('same-id')
      $sessions.set(loaded ? [makeSessionInfo({ id: 'same-id', profile: 'A' })] : [])
      $unreadFinishedMarkers.set({})
      $unreadFinishedSessionIds.set([])
    })
    const stream = renderMessageStream('runtime-A')
    act(() =>
      stream.handleEvent({
        type: 'review.summary',
        session_id: 'runtime-B',
        profile: 'B',
        payload: { text: TEXT, review_id: 'scope-b', timestamp: 103, stored_session_id: 'same-id' }
      })
    )
    expect($unreadFinishedMarkers.get().B).toEqual(['same-id'])
    expect($unreadFinishedMarkers.get().A).toBeUndefined()
    // The actual focused A receipt stays read; replay of B stays idempotent.
    act(() =>
      stream.handleEvent({
        type: 'review.summary',
        session_id: 'runtime-A',
        profile: 'A',
        payload: { text: TEXT, review_id: 'scope-a', timestamp: 104, stored_session_id: 'same-id' }
      })
    )
    expect($unreadFinishedMarkers.get().A).toBeUndefined()
    act(() => {
      clearAllSessionStates()
      $sessions.set([])
      $selectedStoredSessionId.set(null)
      $activeSessionId.set(null)
      $activeGatewayProfile.set('default')
      $unreadFinishedMarkers.set({})
      $unreadFinishedSessionIds.set([])
    })
  })
})
