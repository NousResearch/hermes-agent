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
