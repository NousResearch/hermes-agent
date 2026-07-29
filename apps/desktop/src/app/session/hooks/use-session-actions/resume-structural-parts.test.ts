import { describe, expect, it } from 'vitest'

import type { ChatMessage } from '@/lib/chat-messages'

import { reconcileResumeMessages } from './utils'

const user = (id: string, text: string): ChatMessage => ({
  id,
  parts: [{ type: 'text', text }],
  role: 'user'
})

/**
 * Switching away from a running session and back re-hydrates it from
 * `session.activate`, whose transcript is TEXT-ONLY for the live turn (the
 * gateway's `inflight` projection carries `user`/`assistant` strings, nothing
 * structural). The renderer's cached state is the only carrier of the turn's
 * tool calls and reasoning, so reconcile must not drop them.
 */
describe('reconcileResumeMessages — structural parts on a mid-turn switch', () => {
  it('keeps tool-call parts the authoritative text-only row cannot carry', () => {
    const cached: ChatMessage[] = [
      user('u1', 'read the config'),
      {
        id: 'a1',
        parts: [
          { type: 'reasoning', text: 'I should read the file first.' },
          { type: 'tool-call', toolCallId: 'call-1', toolName: 'read_file', result: 'contents' },
          { type: 'text', text: 'Reading it now' }
        ],
        role: 'assistant'
      }
    ]

    // What activate returns mid-turn: the same rows, but flattened to text and
    // one delta further along, so the text no longer matches the cached copy.
    const authoritative: ChatMessage[] = [
      user('u1', 'read the config'),
      { id: 'a1', parts: [{ type: 'text', text: 'Reading it now — found the key' }], role: 'assistant' }
    ]

    const [, assistant] = reconcileResumeMessages(authoritative, cached)

    expect(assistant.parts.filter(p => p.type === 'tool-call')).toHaveLength(1)
    expect(assistant.parts.filter(p => p.type === 'reasoning')).toHaveLength(1)
    // The newer authoritative text still wins.
    expect(assistant.parts.filter(p => p.type === 'text').at(-1)).toMatchObject({
      text: 'Reading it now — found the key'
    })
  })

  it('does not duplicate tool calls the authoritative row already carries', () => {
    const cached: ChatMessage[] = [
      {
        id: 'a1',
        parts: [{ type: 'tool-call', toolCallId: 'call-1', toolName: 'read_file', result: 'contents' }],
        role: 'assistant'
      }
    ]

    const authoritative: ChatMessage[] = [
      {
        id: 'a1',
        parts: [
          { type: 'tool-call', toolCallId: 'call-1', toolName: 'read_file', result: 'contents' },
          { type: 'text', text: 'done' }
        ],
        role: 'assistant'
      }
    ]

    const [assistant] = reconcileResumeMessages(authoritative, cached)

    expect(assistant.parts.filter(p => p.type === 'tool-call')).toHaveLength(1)
  })
})
