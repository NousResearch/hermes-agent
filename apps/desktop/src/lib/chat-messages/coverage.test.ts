import { describe, expect, it } from 'vitest'

import { withoutCoveredAssistantPrefix } from './coverage'
import type { ChatMessage, ChatMessagePart } from './types'

const text = (value: string): ChatMessagePart => ({ type: 'text', text: value })
const reasoning = (value: string): ChatMessagePart => ({ type: 'reasoning', text: value })

const tool = (id: string): ChatMessagePart => ({
  type: 'tool-call',
  toolCallId: id,
  toolName: 'read_file',
  args: { path: 'same-file.ts' }
})

const assistant = (id: string, parts: ChatMessagePart[]): ChatMessage => ({ id, role: 'assistant', parts })

describe('tool-anchored assistant coverage', () => {
  it.each(['different-tool', 'different-text', 'user-boundary', 'error-boundary', 'reasoning-only'] as const)(
    'preserves the unproven occurrence after a shared tool: %s',
    scenario => {
      const stored = [
        assistant('durable', [tool('shared'), reasoning('next reasoning'), text('Continue?'), tool('durable-next')])
      ]

      const suffix: ChatMessage[] =
        scenario === 'user-boundary'
          ? [
              { id: 'new-user', role: 'user', parts: [text('Continue?')] },
              assistant('new-response', [text('Continue?'), tool('durable-next')])
            ]
          : [
              {
                ...assistant(
                  'new-response',
                  scenario === 'reasoning-only'
                    ? [reasoning('next reasoning')]
                    : [
                        reasoning('next reasoning'),
                        text(scenario === 'different-text' ? 'A different question?' : 'Continue?'),
                        tool(scenario === 'different-tool' ? 'distinct-next' : 'durable-next')
                      ]
                ),
                ...(scenario === 'error-boundary' ? { error: 'retain the failure' } : {})
              }
            ]

      const local = [assistant('covered', [tool('shared')]), ...suffix]

      const coveredTools = new Map<ChatMessagePart, ChatMessagePart>()
      expect(withoutCoveredAssistantPrefix(stored, local, coveredTools)).toEqual(suffix)
      expect([...coveredTools]).toEqual([[stored[0].parts[0], local[0].parts[0]]])
    }
  )

  it('does not use equal commentary or reasoning as an occurrence anchor', () => {
    const stored = [assistant('durable', [reasoning('durable planning'), text('Continue?'), tool('one')])]
    const local = [assistant('live', [reasoning('stream planning'), text('Continue?'), tool('another')])]

    expect(withoutCoveredAssistantPrefix(stored, local)).toBe(local)
  })
})
