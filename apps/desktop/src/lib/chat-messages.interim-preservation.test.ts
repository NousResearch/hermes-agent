import { describe, expect, it } from 'vitest'

import type { SessionMessage } from '@/types/hermes'

import {
  assistantTextPart,
  type ChatMessagePart,
  chatMessageText,
  mergeFinalAssistantText,
  reasoningPart,
  toChatMessages,
  upsertToolPart
} from './chat-messages'

const item = (phase: string, text: string) => ({
  type: 'message',
  role: 'assistant',
  phase,
  content: [{ type: 'output_text', text }]
})

const textParts = (parts: ChatMessagePart[]) => parts.flatMap(part => (part.type === 'text' ? [part.text] : []))

const assistantText = (rows: SessionMessage[]) =>
  toChatMessages(rows)
    .filter(message => message.role === 'assistant')
    .map(chatMessageText)
    .join('\n')

const withTool = (parts: ChatMessagePart[], id = 'call-1', timestamp = 2) =>
  upsertToolPart(
    parts,
    { tool_id: id, name: 'terminal', args: { command: 'pwd' }, result: 'ok' },
    'complete',
    timestamp
  )

describe('stored assistant commentary preservation', () => {
  it.each(['rpc', 'rest'] as const)('restores commentary before the canonical reply through %s history', transport => {
    const items = [
      item('commentary', 'I will inspect the files.'),
      item('analysis', 'Private analysis.'),
      item('final_answer', 'Stale sidecar final.')
    ]

    const row: SessionMessage = {
      role: 'assistant',
      content: 'Canonical final.',
      timestamp: 2,
      reasoning: 'Real reasoning.',
      codex_message_items: transport === 'rest' ? JSON.stringify(items) : items
    }

    const [message] = toChatMessages([row])
    expect(textParts(message.parts)).toEqual(['I will inspect the files.', 'Canonical final.'])
    expect(chatMessageText(message)).not.toContain('Private analysis.')
    expect(chatMessageText(message)).not.toContain('Stale sidecar final.')
    expect(message.parts.filter(part => part.type === 'reasoning')).toEqual([reasoningPart('Real reasoning.', 2)])
  })

  it.each(['rpc', 'rest'] as const)('keeps a commentary-only assistant row through %s history', transport => {
    const items = [item('commentary', 'Still checking.')]

    const rows: SessionMessage[] = [
      { role: 'user', content: 'Check it.', timestamp: 1 },
      {
        role: 'assistant',
        content: '',
        timestamp: 2,
        codex_message_items: transport === 'rest' ? JSON.stringify(items) : items
      },
      { role: 'user', content: 'Continue.', timestamp: 3 }
    ]

    expect(toChatMessages(rows).map(message => message.role)).toEqual(['user', 'assistant', 'user'])
    expect(assistantText(rows)).toBe('Still checking.')
  })

  it('keeps separate commentary items in order and joins only chunks of the same item', () => {
    const first = item('commentary', 'First ')
    first.content.push({ type: 'output_text', text: 'update.' })

    const [message] = toChatMessages([
      {
        role: 'assistant',
        content: '',
        timestamp: 2,
        codex_message_items: [first, item('commentary', 'Second update.'), item('final_answer', 'Final answer.')]
      }
    ])

    expect(textParts(message.parts)).toEqual(['First update.', 'Second update.', 'Final answer.'])
  })

  it('normalizes phase spelling without exposing analysis', () => {
    expect(
      assistantText([
        {
          role: 'assistant',
          content: '',
          timestamp: 2,
          codex_message_items: [
            item(' Commentary ', 'Visible update.'),
            item(' ANALYSIS ', 'Hidden analysis.'),
            item('final_answer', 'Answer.')
          ]
        }
      ])
    ).toBe('Visible update.Answer.')
  })

  it('does not duplicate commentary also present in canonical content', () => {
    const items = [item('commentary', 'First update.'), item('commentary', 'Second update.')]
    expect(
      assistantText([
        {
          role: 'assistant',
          content: 'First update.\n\nSecond update.',
          timestamp: 2,
          codex_message_items: items
        }
      ])
    ).toBe('First update.\n\nSecond update.')
  })

  it('keeps canonical text authoritative when it equals one commentary item', () => {
    expect(
      assistantText([
        {
          role: 'assistant',
          content: 'Same text.',
          timestamp: 2,
          codex_message_items: [item('commentary', 'Same text.'), item('final_answer', 'Stale text.')]
        }
      ])
    ).toBe('Same text.')
  })

  it('respects explicitly hidden and non-assistant rows', () => {
    const codex_message_items = [item('commentary', 'Do not surface this.')]
    expect(
      toChatMessages([{ role: 'assistant', content: '', timestamp: 2, display_kind: 'hidden', codex_message_items }])
    ).toEqual([])
    expect(assistantText([{ role: 'user', content: 'User text.', timestamp: 2, codex_message_items }])).toBe('')
  })

  it.each(['{broken', 'null', '{}', '42'])(
    'ignores malformed sidecar %s without losing canonical content',
    codex_message_items => {
      expect(assistantText([{ role: 'assistant', content: 'Keep me.', timestamp: 2, codex_message_items }])).toBe(
        'Keep me.'
      )
    }
  )

  it('does not promote a reasoning-only row into an answer', () => {
    const [message] = toChatMessages([{ role: 'assistant', content: '', timestamp: 2, reasoning: 'Only reasoning.' }])
    expect(chatMessageText(message)).toBe('')
    expect(message.parts).toEqual([reasoningPart('Only reasoning.', 2)])
  })

  it('rejects malformed and foreign-role items while keeping valid commentary', () => {
    expect(
      assistantText([
        {
          role: 'assistant',
          content: '',
          timestamp: 2,
          codex_message_items: JSON.stringify([
            null,
            1,
            [],
            { ...item('commentary', 'Foreign text.'), role: 'user' },
            { ...item('commentary', 'Wrong type.'), type: 'reasoning' },
            { ...item('commentary', 'Bad content.'), content: 'not an array' },
            { ...item('commentary', ''), content: [null, 1, [], { type: 'output_text', text: 42 }] },
            item('commentary', 'Valid update.')
          ])
        }
      ])
    ).toBe('Valid update.')
  })
})

describe('authoritative final text is scoped to the latest tool-delimited response', () => {
  it('does not erase commentary from before a tool when the interim frame is absent', () => {
    const earlier = withTool([assistantTextPart('Checking the repository.', 1)])
    const parts = [...earlier, assistantTextPart('Partial final', 3)]
    const before = structuredClone(parts)
    const result = mergeFinalAssistantText(parts, 'Final answer.', 4)
    expect(textParts(result)).toEqual(['Checking the repository.', 'Final answer.'])
    expect(result.slice(0, earlier.length)).toEqual(earlier)
    expect(parts).toEqual(before)
  })

  it('keeps all earlier tool-delimited updates, not just the latest one', () => {
    const first = withTool([assistantTextPart('First update.', 1)])
    const second = withTool([...first, assistantTextPart('Second update.', 3)], 'call-2', 4)
    const result = mergeFinalAssistantText([...second, assistantTextPart('Draft', 5)], 'Done.', 6)
    expect(textParts(result)).toEqual(['First update.', 'Second update.', 'Done.'])
    expect(result.filter(part => part.type === 'tool-call')).toHaveLength(2)
  })

  it('does not duplicate an exact cumulative final prefix while preserving the tool boundary', () => {
    const parts = [...withTool([assistantTextPart('Earlier update.', 1)]), assistantTextPart('Partial', 3)]
    const result = mergeFinalAssistantText(parts, 'Earlier update.Final answer.', 4)
    expect(textParts(result)).toEqual(['Earlier update.', 'Final answer.'])
    expect(result.findIndex(part => part.type === 'tool-call')).toBe(1)
  })

  it('keeps a longer earlier update when the final is only its short prefix', () => {
    const parts = withTool([assistantTextPart('Done. The investigation details follow.', 1)])
    expect(textParts(mergeFinalAssistantText(parts, 'Done.', 3))).toEqual([
      'Done. The investigation details follow.',
      'Done.'
    ])
  })

  it('still replaces provisional text within one response, even across reasoning parts', () => {
    const parts = [
      assistantTextPart('Wrong draft.', 1),
      reasoningPart('Reconsidering.', 2),
      assistantTextPart('Another draft.', 3)
    ]

    expect(textParts(mergeFinalAssistantText(parts, 'Correct final.', 4))).toEqual(['Correct final.'])
  })

  it('keeps a confirmed full stream and an empty terminal frame unchanged', () => {
    const parts = [...withTool([assistantTextPart('Earlier.', 1)]), assistantTextPart('Final.', 3)]
    expect(mergeFinalAssistantText(parts, 'Earlier.Final.', 4)).toBe(parts)
    expect(mergeFinalAssistantText(parts, '  ', 4)).toBe(parts)
  })
})
