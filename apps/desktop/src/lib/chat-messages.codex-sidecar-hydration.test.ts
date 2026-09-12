import { describe, expect, it } from 'vitest'

import type { SessionMessage } from '@/types/hermes'

import { chatMessageText, toChatMessages } from './chat-messages'

// #68321 / GregKM 2026-09-02 v0.21.0 DB-level evidence: an assistant row
// whose persisted `content` is empty but whose user-visible response text
// lives only in `reasoning` / `reasoning_content` is dropped by hydration
// after a session/profile switch-back — the live stream rendered it, the
// rehydrated transcript loses it. This file pins that no reasoning-carrying
// assistant row may vanish.

const reasoningOnlyRow: SessionMessage = {
  id: 71,
  role: 'assistant',
  content: '',
  reasoning: 'Here is the plan: first inspect the repo, then propose a fix, then run the tests.',
  timestamp: 2
}

describe('#68321 assistant rows whose reply persisted only in codex_message_items', () => {
  it('restores the reply text from codex_message_items when content persisted empty (#68321 GregKM repro)', () => {
    // Field-level evidence from the 2026-09-02 v0.21.0 reproduction:
    // role=assistant, content length 0, the exact user-visible response
    // exists only in reasoning / reasoning_content / codex_message_items.
    // The live stream painted it; the rehydrate dropped it.
    const row: SessionMessage = {
      id: 110,
      role: 'assistant',
      content: '',
      reasoning: null,
      reasoning_content: null,
      codex_message_items: [
        {
          type: 'message',
          id: 'msg_abc',
          role: 'assistant',
          phase: 'commentary',
          content: [{ type: 'output_text', text: 'Working through the approach...' }]
        },
        {
          type: 'message',
          id: 'msg_abd',
          role: 'assistant',
          phase: 'analysis',
          content: [{ type: 'output_text', text: 'Scratchpad thoughts.' }]
        },
        {
          type: 'message',
          id: 'msg_def',
          role: 'assistant',
          phase: 'final_answer',
          content: [{ type: 'output_text', text: 'Here is the full response you saw live.' }]
        }
      ],
      timestamp: 2
    }

    const messages = toChatMessages([
      { id: 109, role: 'user', content: 'go', timestamp: 1 },
      row,
      { id: 111, role: 'user', content: 'next', timestamp: 3 }
    ])

    expect(messages.map(m => m.role)).toEqual(['user', 'assistant', 'user'])
    // The final-answer text is painted as the bubble's reply text...
    expect(chatMessageText(messages[1])).toContain('Here is the full response you saw live.')
    // ...and commentary / analysis narration (reasoning channel on the backend) is not.
    expect(chatMessageText(messages[1])).not.toContain('Working through the approach...')
    expect(chatMessageText(messages[1])).not.toContain('Scratchpad thoughts.')
  })

  it('prefers persisted content over the codex sidecar when both exist', () => {
    const row: SessionMessage = {
      id: 120,
      role: 'assistant',
      content: 'Canonical persisted reply',
      codex_message_items: [
        {
          type: 'message',
          role: 'assistant',
          phase: 'final_answer',
          content: [{ type: 'output_text', text: 'Sidecar-only reply' }]
        }
      ],
      timestamp: 2
    }

    const [assistant] = toChatMessages([row])
    expect(chatMessageText(assistant)).toBe('Canonical persisted reply')
  })
})
