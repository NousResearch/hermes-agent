import { describe, expect, it } from 'vitest'

import { type ChatMessage, type ChatMessagePart, chatMessageText, textPart, toChatMessages } from '@/lib/chat-messages'

import { preserveLocalPendingTurnMessages } from './utils'

const tool = (id: string): ChatMessagePart =>
  ({ type: 'tool-call', toolCallId: id, toolName: 'terminal', result: 'done' }) as ChatMessagePart

const message = (
  id: string,
  role: ChatMessage['role'],
  parts: ChatMessagePart[],
  extra: Partial<ChatMessage> = {}
): ChatMessage => ({ id, role, parts, ...extra }) as ChatMessage

const liveUser = message('user-live', 'user', [textPart('check the model')])

const settledStream = (parts: ChatMessagePart[], extra: Partial<ChatMessage> = {}) =>
  message('assistant-stream-live', 'assistant', parts, { interim: false, pending: false, ...extra })

const storedCall = { id: 'call-1', type: 'function', function: { name: 'terminal', arguments: '{}' } }

// #123047: the store committed the turn as a tool round with content = '' and
// no prose row after it, while the renderer streamed and settled the answer.
describe('preserveLocalPendingTurnMessages — settled reply over an empty tool shell (#123047)', () => {
  it('keeps the streamed final reply when the stored turn hydrates as a text-less tool shell', () => {
    const next = toChatMessages([
      { id: 196700, role: 'user', content: 'check the model', timestamp: 1 },
      { id: 196701, role: 'assistant', content: '', tool_calls: [storedCall], timestamp: 2 },
      { id: 196702, role: 'tool', content: 'done', tool_call_id: 'call-1', timestamp: 3 }
    ] as never)

    expect(next.map(row => chatMessageText(row))).toEqual(['check the model', ''])

    const previous = [liveUser, settledStream([textPart('I checked it.'), tool('call-1'), textPart('Model is up.')])]
    const merged = preserveLocalPendingTurnMessages(next, previous)

    expect(merged).toHaveLength(2)
    expect(merged[1]).toMatchObject({ id: 'assistant-stream-live', pending: false })
    expect(chatMessageText(merged[1])).toContain('Model is up.')
  })

  it('carries the shell row identity onto the kept reply', () => {
    const next = [
      message('user-stored', 'user', [textPart('check the model')], { rowId: 196700 }),
      message('assistant-shell', 'assistant', [tool('call-1')], { pending: false, rowId: 196701 })
    ]

    const merged = preserveLocalPendingTurnMessages(next, [liveUser, settledStream([textPart('Model is up.')])])

    expect(merged).toHaveLength(2)
    expect(merged[1]).toMatchObject({ id: 'assistant-stream-live', pending: false, rowId: 196701 })
  })

  it('yields to the committed answer once a prose row folds into the tool round (no duplicate)', () => {
    const next = toChatMessages([
      { id: 196700, role: 'user', content: 'check the model', timestamp: 1 },
      { id: 196701, role: 'assistant', content: '', tool_calls: [storedCall], timestamp: 2 },
      { id: 196702, role: 'tool', content: 'done', tool_call_id: 'call-1', timestamp: 3 },
      { id: 196703, role: 'assistant', content: 'Model is up and serving.', timestamp: 4 }
    ] as never)

    const previous = [liveUser, settledStream([textPart('I checked it.'), tool('call-1'), textPart('Model is up.')])]
    const merged = preserveLocalPendingTurnMessages(next, previous)

    expect(merged).toEqual(next)
  })

  it('does not let settled interim narration replace the shell', () => {
    const next = [
      message('user-stored', 'user', [textPart('check the model')], { rowId: 196700 }),
      message('assistant-shell', 'assistant', [tool('call-1')], { pending: false, rowId: 196701 })
    ]

    const previous = [liveUser, settledStream([textPart('Checking the logs…')], { interim: true })]

    expect(preserveLocalPendingTurnMessages(next, previous)).toEqual(next)
  })

  it('does not let a structural-only local row replace the shell', () => {
    const next = [
      message('user-stored', 'user', [textPart('check the model')], { rowId: 196700 }),
      message('assistant-shell', 'assistant', [tool('call-stored')], { pending: false, rowId: 196701 })
    ]

    expect(preserveLocalPendingTurnMessages(next, [liveUser, settledStream([tool('call-local')])])).toEqual(next)
  })

  it('never repaints a retained error row from the local partial', () => {
    const next = [
      message('user-stored', 'user', [textPart('check the model')], { rowId: 196700 }),
      message('assistant-error', 'assistant', [tool('call-1')], {
        error: 'provider failed',
        pending: false,
        rowId: 196701
      } as Partial<ChatMessage>)
    ]

    expect(preserveLocalPendingTurnMessages(next, [liveUser, settledStream([textPart('Model is up.')])])).toEqual(next)
  })
})
