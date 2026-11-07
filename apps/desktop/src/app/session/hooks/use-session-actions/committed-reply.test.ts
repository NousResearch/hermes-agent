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
  // completion lost to a reconnect): it takes the committed bubble's place
  // AND its whole body, so the reply draws once and in full.
  it('draws one whole reply while the live row is still short of it', () => {
    const baseline = [steer, live('A6 ')]
    const current = [steer, live('A6 steered')]
    const page = committedPage('change course')

    const overlaid = overlayConcurrentMessageChanges(page, baseline, current)

    expect(assistants(overlaid)).toEqual([['assistant-stream-1-3', chatMessageText(page[1])]])
    expect(overlaid.map(message => message.id)).toEqual(['18-user', 'assistant-stream-1-3'])
    expect(overlaid[1].pending).toBe(true)
  })

  // A live row that says something the stored reply does not is not its copy.
  it('leaves a stored reply that does not extend the live row alone', () => {
    const baseline = [steer, live('A6 ')]
    const current = [steer, live('A6 rerouted')]

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

// transcript-integrity "ws drop + reconnect": the re-resume's page already holds
// the stored reply while the live row's tail (and its completion) is still on
// the wire. Release CI caught the reply drawn twice in that window.
describe('a reconnect page that stores a plain reply the live row is still short of', () => {
  const prompt: ChatMessage = { id: 'user-optimistic', role: 'user', parts: [{ type: 'text', text: 'U8 across a drop' }] }

  const stream = (text: string): ChatMessage =>
    ({ id: 'assistant-stream-9-1', role: 'assistant', pending: true, parts: [{ type: 'text', text }] }) as ChatMessage

  const page: ChatMessage[] = [
    { id: '16-user', role: 'user', rowId: 16, parts: [{ type: 'text', text: 'U8 across a drop' }] } as ChatMessage,
    {
      id: '17-assistant',
      role: 'assistant',
      rowId: 17,
      parts: [{ type: 'text', text: 'A8 survives the drop', sourceRowId: 17 }]
    } as ChatMessage
  ]

  it('draws the reply once, in full, under the live row', () => {
    const overlaid = overlayConcurrentMessageChanges(page, [prompt, stream('A8 ')], [prompt, stream('A8 survives the')])

    expect(assistants(overlaid)).toEqual([['assistant-stream-9-1', 'A8 survives the drop']])
  })

  // A shared prefix that ends inside a word is not evidence of the same reply.
  it('does not hand the live row a stored reply that only shares a word fragment with it', () => {
    const other: ChatMessage[] = [page[0], { ...page[1], parts: [{ type: 'text', text: 'A8x unrelated reply' }] }]
    const overlaid = overlayConcurrentMessageChanges(other, [prompt, stream('A')], [prompt, stream('A8')])

    expect(assistants(overlaid)).toEqual([
      ['17-assistant', 'A8x unrelated reply'],
      ['assistant-stream-9-1', 'A8']
    ])
  })
})
