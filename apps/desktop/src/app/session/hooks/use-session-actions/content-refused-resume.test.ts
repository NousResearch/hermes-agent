import { describe, expect, it } from 'vitest'

import { type ChatMessage, chatMessageText, textPart, toChatMessages } from '@/lib/chat-messages'
import type { SessionMessage } from '@/types/hermes'

import { reconcileDurableHistory } from './utils'

// A safety-refused turn is retained for display as ONE stored row the backend
// projects `display_kind: 'content_refused'` (agent/turn_failure_copy.py).
// Hydration renders it through the greyed `system` path (NOTICE_DISPLAY_KINDS),
// NOT as a `user` bubble — so warm-resume matching can no longer pair it with
// the optimistic `user-*` prompt by role. Without treating the content_refused
// row as that prompt's stored counterpart, the resume merge appends the old
// prompt a second time beside the greyed block (the reported duplicate).
const PROMPT = 'the request the filter blocked'
const NOTICE = "This message was refused by the model's safety filter — try a different query."

const storedRefusedRow = (): ChatMessage => {
  const row: SessionMessage = {
    role: 'user',
    content: PROMPT,
    display_kind: 'content_refused',
    display_metadata: { content_refused: { notice: NOTICE } } as SessionMessage['display_metadata'],
    timestamp: 1,
    row_id: 10
  } as SessionMessage

  const [hydrated] = toChatMessages([row])

  return hydrated
}

// The optimistic prompt the composer painted on submit. A canonical submit can
// leave it without a rowId (its receipt names no user-row address), so the
// merge cannot dedupe it by row id either.
const optimisticPrompt = (): ChatMessage => ({ id: 'user-123', role: 'user', parts: [textPart(PROMPT)] })

describe('warm-resume merge of a content-refused turn', () => {
  it('does not duplicate the optimistic prompt beside the stored greyed block', () => {
    const stored = storedRefusedRow()
    // Sanity: hydration really did route it through the greyed system path.
    expect(stored.role).toBe('system')
    expect(stored.contentRefused).toEqual({ notice: NOTICE })

    const merged = reconcileDurableHistory([stored], [optimisticPrompt()])

    // Exactly the stored greyed row survives — the optimistic prompt is its
    // local counterpart, not a second unsent turn.
    expect(merged).toHaveLength(1)
    expect(merged[0].role).toBe('system')
    expect(merged[0].contentRefused).toEqual({ notice: NOTICE })

    // No leftover blue user bubble carrying the same prompt text.
    const promptBubbles = merged.filter(message => message.role === 'user' && chatMessageText(message) === PROMPT)

    expect(promptBubbles).toHaveLength(0)
  })
})
