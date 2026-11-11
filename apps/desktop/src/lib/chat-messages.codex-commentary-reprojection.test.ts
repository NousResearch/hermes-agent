import { describe, expect, it } from 'vitest'

import { chatMessageText, toChatMessages } from '@/lib/chat-messages'
import type { SessionMessage } from '@/types/hermes'

describe('public Codex commentary after transcript hydration', () => {
  it('keeps a tool-call preamble in assistant text, not the Thinking disclosure', () => {
    // Real row shape: the Responses adapter persists a phase=commentary item
    // alongside an empty content column and a reasoning blob containing both
    // the genuine summary and the already-delivered public preamble.
    const preamble = 'I will inspect the repository before changing code.'

    const row: SessionMessage = {
      id: 310,
      role: 'assistant',
      content: '',
      reasoning: `**Checking workflow docs**\n\n${preamble}`,
      display_reasoning: '**Checking workflow docs**',
      display_commentary: [preamble],
      tool_calls: [{ id: 'call-read', type: 'function', function: { name: 'read_file', arguments: '{}' } }],
      codex_message_items: [
        {
          type: 'message',
          role: 'assistant',
          phase: 'commentary',
          content: [{ type: 'output_text', text: preamble }]
        }
      ],
      timestamp: 2
    }

    const [message] = toChatMessages([row])
    const thoughts = message.parts.filter(part => part.type === 'reasoning')

    expect(chatMessageText(message)).toBe(preamble)
    expect(thoughts).toHaveLength(1)
    expect(thoughts[0]).toMatchObject({ text: '**Checking workflow docs**' })
  })

  it('removes only exact commentary segments, preserving surrounding and incidental reasoning', () => {
    const commentary = 'Inspect the files.\n\nThen run tests.'

    const row: SessionMessage = {
      id: 311,
      role: 'assistant',
      content: 'Final reply.',
      reasoning: `Summary before.\n\n${commentary}\n\nSummary after.`,
      display_reasoning: 'Summary before.\n\nSummary after.',
      display_commentary: [commentary],
      codex_message_items: [
        {
          type: 'message',
          role: 'assistant',
          phase: 'commentary',
          content: [{ type: 'output_text', text: commentary }]
        },
        { type: 'message', role: 'assistant', phase: 'analysis', content: [{ type: 'output_text', text: 'Private.' }] }
      ],
      timestamp: 3
    }

    const [message] = toChatMessages([row])

    expect(message.parts.filter(part => part.type === 'text').map(part => part.text)).toEqual([
      commentary,
      'Final reply.'
    ])
    expect(message.parts.filter(part => part.type === 'reasoning').map(part => part.text)).toEqual([
      'Summary before.\n\nSummary after.'
    ])

    const incidental: SessionMessage = {
      ...row,
      reasoning: `Summary mentioning ${commentary} as an example.`,
      display_reasoning: `Summary mentioning ${commentary} as an example.`
    }

    const [unchanged] = toChatMessages([incidental])

    expect(unchanged.parts.filter(part => part.type === 'reasoning').map(part => part.text)).toEqual([
      incidental.reasoning
    ])
  })

  it('honors disabled interim commentary without suppressing a separate final reply', () => {
    const commentary = 'I will inspect the files.'

    const row: SessionMessage = {
      id: 312,
      role: 'assistant',
      content: 'Here is the final result.',
      reasoning: `Private summary.\n\n${commentary}`,
      display_reasoning: 'Private summary.',
      display_commentary: [],
      codex_message_items: [
        {
          type: 'message',
          role: 'assistant',
          phase: 'commentary',
          content: [{ type: 'output_text', text: commentary }]
        }
      ],
      timestamp: 4
    }

    const [message] = toChatMessages([row])

    expect(chatMessageText(message)).toBe('Here is the final result.')
    expect(message.parts.filter(part => part.type === 'reasoning').map(part => part.text)).toEqual(['Private summary.'])
  })

  it('uses only backend-authorized display commentary, never raw replay items', () => {
    const raw = 'My key is «redacted:sk-…». <think>private</think>'

    const row: SessionMessage = {
      id: 314,
      role: 'assistant',
      content: '',
      reasoning: `Private summary.\n\n${raw}`,
      display_reasoning: 'Private summary.',
      display_commentary: ['My key is redacted.'],
      codex_message_items: [
        { type: 'message', role: 'assistant', phase: 'commentary', content: [{ type: 'output_text', text: raw }] }
      ],
      timestamp: 5
    }

    expect(chatMessageText(toChatMessages([row])[0])).toBe('My key is redacted.')
    expect(
      toChatMessages([row])[0]
        .parts.filter(part => part.type === 'reasoning')
        .map(part => part.text)
    ).toEqual(['Private summary.'])
    expect(chatMessageText(toChatMessages([{ ...row, display_commentary: undefined }])[0])).toBe('')
  })

  it('keeps a sanitized canonical answer intact when its sidecar origin is ambiguous', () => {
    const raw = 'Checking. <think>hidden</think> key=' + 'sk-' + 'demo0123456789abcdef'

    const row: SessionMessage = {
      id: 315,
      role: 'assistant',
      content: `${raw}\n\nFinal result.`,
      display_content: 'Checking. key=«redacted»\n\nFinal result.',
      display_commentary: [],
      reasoning: `Private.\n\n${raw}`,
      display_reasoning: 'Private.',
      codex_message_items: [
        { type: 'message', role: 'assistant', phase: 'commentary', content: [{ type: 'output_text', text: raw }] }
      ],
      timestamp: 6
    }

    const [shown] = toChatMessages([row])
    expect(shown.parts.filter(part => part.type === 'text').map(part => part.text)).toEqual([
      'Checking. key=«redacted»\n\nFinal result.'
    ])
    expect(shown.parts.filter(part => part.type === 'reasoning').map(part => part.text)).toEqual(['Private.'])
    expect(chatMessageText(toChatMessages([{ ...row, display_commentary: [] }])[0])).toBe(
      'Checking. key=«redacted»\n\nFinal result.'
    )
  })

  it('keeps an explicit empty display content authoritative over a stale final sidecar', () => {
    const row: SessionMessage = {
      id: 316,
      role: 'assistant',
      content: 'Working',
      display_content: '',
      display_commentary: [],
      reasoning: 'Private.\n\nWorking',
      display_reasoning: 'Private.',
      codex_message_items: [
        {
          type: 'message',
          role: 'assistant',
          phase: 'commentary',
          content: [{ type: 'output_text', text: 'Working' }]
        },
        {
          type: 'message',
          role: 'assistant',
          phase: 'final',
          content: [{ type: 'output_text', text: 'Stale sidecar final' }]
        }
      ],
      timestamp: 7
    }

    expect(chatMessageText(toChatMessages([row])[0])).toBe('')
    expect(
      toChatMessages([row])[0]
        .parts.filter(part => part.type === 'reasoning')
        .map(part => part.text)
    ).toEqual(['Private.'])
    // Older backends without a display projection still use final sidecars.
    expect(chatMessageText(toChatMessages([{ ...row, display_content: undefined }])[0])).toBe('Working')
  })

  it('keeps a final answer that starts with or equals the commentary text', () => {
    for (const final of ['Checking. The answer is 42.', 'Checking.']) {
      const row: SessionMessage = {
        id: 317,
        role: 'assistant',
        content: final,
        display_commentary: ['Checking.'],
        display_reasoning: 'Private.',
        codex_message_items: [
          {
            type: 'message',
            role: 'assistant',
            phase: 'commentary',
            content: [{ type: 'output_text', text: 'Checking.' }]
          },
          { type: 'message', role: 'assistant', phase: 'final', content: [{ type: 'output_text', text: final }] }
        ],
        timestamp: 8
      }

      const shown = toChatMessages([row])[0]
        .parts.filter(part => part.type === 'text')
        .map(part => part.text)

      expect(shown).toEqual(final === 'Checking.' ? [final] : ['Checking.', final])
      expect(chatMessageText(toChatMessages([{ ...row, display_commentary: [] }])[0])).toBe(final)
    }
  })

  it('uses each message owner’s display projection, not a foreground global setting', () => {
    const row: SessionMessage = {
      id: 313,
      role: 'assistant',
      content: '',
      reasoning: 'Checking files.',
      codex_message_items: [
        {
          type: 'message',
          role: 'assistant',
          phase: 'commentary',
          content: [{ type: 'output_text', text: 'Checking files.' }]
        }
      ],
      timestamp: 5
    }

    expect(chatMessageText(toChatMessages([{ ...row, display_commentary: [] }])[0])).toBe('')
    expect(chatMessageText(toChatMessages([{ ...row, display_commentary: ['Checking files.'] }])[0])).toBe(
      'Checking files.'
    )
  })
})
