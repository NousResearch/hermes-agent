import { describe, expect, it } from 'vitest'

import { type ChatMessage, chatMessageText } from '@/lib/chat-messages'

import { overlayConcurrentMessageChanges } from './utils'

const tool = { type: 'tool-call', toolCallId: 'call-1', toolName: 'terminal', args: {}, argsText: '{}', result: 'ok' }

const live = (answer: string): ChatMessage =>
  ({
    id: 'assistant-stream-1-3',
    role: 'assistant',
    pending: true,
    parts: [{ type: 'text', text: '\n\nA6i steered narration' }, tool, { type: 'text', text: `\n\n${answer}` }]
  }) as ChatMessage

const steer: ChatMessage = { id: 'user-1-steer', role: 'user', parts: [{ type: 'text', text: 'change course' }] }

// The switch-back REST page as history folds it: the turn's narration, its
// tool and its stored final reply in one bubble under the narration's row.
const committedPage = (prompt: string): ChatMessage[] => [
  { id: '18-user', role: 'user', rowId: 18, parts: [{ type: 'text', text: prompt }] } as ChatMessage,
  {
    id: '19-assistant',
    role: 'assistant',
    rowId: 19,
    parts: [
      { type: 'text', text: 'A6i steered narration', sourceRowId: 19 },
      tool,
      { type: 'text', text: 'A6 steered answer', sourceRowId: 21 }
    ]
  } as ChatMessage
]

const assistants = (messages: ChatMessage[]) =>
  messages.filter(message => message.role === 'assistant').map(message => [message.id, chatMessageText(message)])

describe('a re-attach page that commits the reply before its completion', () => {
  // Steer, switch away, switch back: the turn stores its final row while the
  // REST read is in flight, and that page resolves before message.complete
  // reaches the still-pending live row (the e2e "steer then re-attach" step).
  it('keeps the live row in the committed bubble’s place instead of beside it', () => {
    const baseline = [steer, live('A6 ')]
    const current = [steer, live('A6 steered answer')]

    const overlaid = overlayConcurrentMessageChanges(committedPage('change course'), baseline, current)

    expect(assistants(overlaid)).toEqual([['assistant-stream-1-3', chatMessageText(current[1])]])
    expect(overlaid.map(message => message.id)).toEqual(['18-user', 'assistant-stream-1-3'])
  })

  // A live row still short of the stored reply may never get its tail (a
  // completion lost to a reconnect): only the committed bubble holds it all.
  it('keeps the committed bubble while the live row is still short of it', () => {
    const baseline = [steer, live('A6 ')]
    const current = [steer, live('A6 steered')]

    const overlaid = overlayConcurrentMessageChanges(committedPage('change course'), baseline, current)

    expect(assistants(overlaid).map(([id]) => id)).toEqual(['19-assistant', 'assistant-stream-1-3'])
  })

  // A page whose newest prompt is not the live row's own is an earlier turn:
  // a new reply that happens to open with the same words is not that answer.
  it('leaves a page reply that follows another prompt alone', () => {
    const baseline = [steer, live('A6 ')]
    const current = [steer, live('A6 steered answer')]

    const overlaid = overlayConcurrentMessageChanges(committedPage('an earlier prompt'), baseline, current)

    expect(assistants(overlaid).map(([id]) => id)).toEqual(['19-assistant', 'assistant-stream-1-3'])
  })
})
