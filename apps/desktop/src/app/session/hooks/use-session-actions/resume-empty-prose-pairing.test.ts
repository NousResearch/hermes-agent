import { describe, expect, it } from 'vitest'

import type { ChatMessage } from '@/lib/chat-messages'

import { reconcileResumeMessages } from './utils'

const user = (id: string, text: string): ChatMessage => ({
  id,
  parts: [{ type: 'text', text }],
  role: 'user'
})

/**
 * Empty-prose pairing (#114543): the reconciler pairs cached and hydrated rows
 * by role ordinal. Prose that carries no identity must not establish that two
 * rows are the same turn — otherwise the later cached structure is grafted
 * onto an unrelated earlier row and duplicated on its own row as well.
 */
describe('reconcileResumeMessages — empty-prose pairing', () => {
  it('does not graft a later cached structure onto an earlier tool-only row with empty prose', () => {
    const cachedLive: ChatMessage = {
      id: 'assistant-stream-late',
      pending: true,
      parts: [
        { type: 'reasoning', text: 'Later planning', timestamp: 5 },
        { type: 'tool-call', toolCallId: 'late', toolName: 'terminal', timestamp: 9 }
      ],
      role: 'assistant'
    }

    const authoritative: ChatMessage[] = [
      {
        id: 'stored-early',
        parts: [{ type: 'tool-call', toolCallId: 'early', toolName: 'skill_view', timestamp: 1 }],
        role: 'assistant'
      },
      { ...cachedLive, id: 'stored-late', pending: false }
    ]

    const reconciled = reconcileResumeMessages(authoritative, [cachedLive])

    // The earlier row keeps only its own tool call: [1], not [5, 9, 1].
    expect(reconciled[0].parts.map(part => part.timestamp)).toEqual([1])

    // The later row keeps its own structure instead of inheriting the early
    // tool call on the next ordinal.
    expect(reconciled[1].parts.map(part => part.timestamp)).toEqual([5, 9])
  })

  it('does not overwrite a settled text-only target when a structured cached row shares its ordinal', () => {
    // Cached: a live structured row at assistant ordinal 0 (mid-turn switch).
    const cachedLive: ChatMessage = {
      id: 'assistant-stream-1',
      pending: true,
      parts: [
        { type: 'reasoning', text: 'current thinking', timestamp: 50 },
        { type: 'tool-call', toolCallId: 'current', toolName: 'terminal', timestamp: 60 },
        { type: 'text', text: 'interim narration', timestamp: 55 }
      ],
      role: 'assistant'
    }

    // Hydrated: an unrelated settled text-only assistant at the same ordinal
    // (compression rewrote history and shifted ordinals).
    const authoritative: ChatMessage[] = [
      user('u1', 'unrelated earlier prompt'),
      { id: 'stored-0', parts: [{ type: 'text', text: 'a settled unrelated answer' }], role: 'assistant' }
    ]

    const reconciled = reconcileResumeMessages(authoritative, [cachedLive])

    expect(reconciled[1].parts).toEqual([{ type: 'text', text: 'a settled unrelated answer' }])
  })
})
