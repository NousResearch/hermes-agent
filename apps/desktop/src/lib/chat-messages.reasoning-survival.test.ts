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
  reasoning:
    'Here is the plan: first inspect the repo, then propose a fix, then run the tests.',
  timestamp: 2
}

describe('#68321 reasoning-only assistant rows survive hydration', () => {
  it('does not drop an assistant row whose text lives only in reasoning', () => {
    const messages = toChatMessages([
      { id: 70, role: 'user', content: 'go', timestamp: 1 },
      reasoningOnlyRow,
      { id: 72, role: 'user', content: 'thanks', timestamp: 3 }
    ])

    expect(messages.map((m) => m.role)).toEqual(['user', 'assistant', 'user'])
    // The reasoning text rides as a reasoning part (rendered in the
    // collapsible), keeping the row and its content paintable.
    const reasoningParts = messages[1].parts.filter((p) => p.type === 'reasoning')
    expect(reasoningParts.map((p) => ('text' in p ? p.text : '')).join('')).toContain('Here is the plan')
  })

  it('renders the reasoning part for an empty-content assistant row', () => {
    const [assistant] = toChatMessages([reasoningOnlyRow])
    expect(assistant).toBeDefined()
    expect(assistant.parts.some((p) => p.type === 'reasoning')).toBe(true)
  })

  it('keeps reasoning_content fallback rows visible', () => {
    const row: SessionMessage = {
      id: 80,
      role: 'assistant',
      content: '',
      reasoning: null,
      reasoning_content: 'Fallback reasoning text from reasoning_content',
      timestamp: 2
    }
    const [assistant] = toChatMessages([row])
    expect(assistant).toBeDefined()
    const reasoningText = assistant.parts
      .filter((p) => p.type === 'reasoning')
      .map((p) => ('text' in p ? p.text : ''))
      .join('')
    expect(reasoningText).toContain('Fallback reasoning text')
  })

  it('keeps a row with empty reasoning string AND empty content as a visible placeholder instead of dropping it silently', () => {
    // A wholly empty assistant row (e.g. a torn persist) must not silently
    // vanish from the transcript — the zero-parts drop is the reported
    // "all assistant messages gone" shape. Keep the row paintable.
    const row: SessionMessage = {
      id: 90,
      role: 'assistant',
      content: '',
      reasoning: '',
      reasoning_content: null,
      timestamp: 2
    }
    const messages = toChatMessages([
      { id: 89, role: 'user', content: 'q', timestamp: 1 },
      row,
      { id: 91, role: 'user', content: 'next', timestamp: 3 }
    ])
    // The assistant row survives with a placeholder part rather than
    // disappearing between the two user rows.
    expect(messages.map((m) => m.role)).toEqual(['user', 'assistant', 'user'])
    expect(messages[1].parts.length).toBeGreaterThan(0)
  })

  it('renders reasoning alongside tool calls for a row with both and no content', () => {
    const row: SessionMessage = {
      id: 100,
      role: 'assistant',
      content: '',
      reasoning: 'Thought about the approach first',
      tool_calls: [
        { id: 'call_1', type: 'function', function: { name: 'inspect', arguments: '{}' } }
      ],
      timestamp: 2
    }
    const [assistant] = toChatMessages([row])
    expect(assistant).toBeDefined()
    const types = assistant.parts.map((p) => p.type)
    expect(types).toContain('reasoning')
  })

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
    expect(messages.map((m) => m.role)).toEqual(['user', 'assistant', 'user'])
    // The final-answer text is painted as the bubble's reply text...
    expect(chatMessageText(messages[1])).toContain('Here is the full response you saw live.')
    // ...and the commentary narration is NOT promoted into the reply.
    expect(chatMessageText(messages[1])).not.toContain('Working through the approach...')
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

  it('paints a placeholder for a wholly empty assistant row instead of dropping it', () => {
    const row: SessionMessage = {
      id: 130,
      role: 'assistant',
      content: '',
      reasoning: '',
      reasoning_content: null,
      codex_message_items: null,
      timestamp: 2
    }
    const messages = toChatMessages([row])
    expect(messages).toHaveLength(1)
    expect(messages[0].role).toBe('assistant')
    expect(chatMessageText(messages[0]).length).toBeGreaterThan(0)
  })

  it('still drops hidden display_kind rows (interrupt scaffolding stays invisible)', () => {
    const row: SessionMessage = {
      id: 140,
      role: 'assistant',
      content: '',
      reasoning: '',
      display_kind: 'hidden',
      timestamp: 2
    }
    const messages = toChatMessages([row])
    expect(messages).toHaveLength(0)
  })
})
