import { describe, expect, it } from 'vitest'

import { type ChatMessage, toChatMessages } from '@/lib/chat-messages'
import type { SessionMessage } from '@/types/hermes'

import { graftRefreshedTailOntoBackfill } from './transcript-backfill'

/**
 * The refresh merge's logical-row key must not collapse rows the backend
 * keeps distinct.
 *
 * The backend's display dedupe key (`_display_dedupe_key` in
 * `hermes_state_messages.py`) also carries `tool_call_id`, `tool_calls` and
 * `tool_name`: a tool-bearing stored assistant and a plain stored assistant
 * with the same visible text and timestamp are distinct durable rows. Text-key
 * matching after compaction re-ids rows is only safe for rows whose parts are
 * all text; otherwise the plain page copy supersedes the tool-bearing
 * original and a whole turn disappears.
 */

const stored = (id: number, role: 'assistant' | 'user', content: string, timestamp: number): SessionMessage => ({
  content,
  id,
  role,
  timestamp
})

describe('refresh merge logical-row identity', () => {
  it('keeps a tool-bearing reply distinct from a plain reply with the same text and timestamp', () => {
    const oldToolReply: ChatMessage = {
      id: 'old-tool-reply',
      parts: [
        { text: 'answer', type: 'text' },
        { args: {}, toolCallId: 'call-3', toolName: 'read_file', type: 'tool-call' }
      ],
      role: 'assistant',
      rowId: 3,
      timestamp: 20
    }

    const previous = [...toChatMessages([stored(2, 'assistant', 'anchor', 2)]), oldToolReply]

    const refreshed = toChatMessages([
      stored(1, 'user', 'before', 1),
      stored(2, 'assistant', 'anchor', 2),
      stored(4, 'assistant', 'answer', 20)
    ])

    expect(graftRefreshedTailOntoBackfill(refreshed, previous).map(message => message.rowId)).toEqual([1, 2, 3, 4])
  })
})
