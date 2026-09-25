import { expect, it } from 'vitest'

import { type ChatMessage, type ChatMessagePart, chatMessageText, textPart } from '@/lib/chat-messages'

import { preserveLocalPendingTurnMessages } from './utils'

const tool = (id: string): ChatMessagePart =>
  ({ type: 'tool-call', toolCallId: id, toolName: 'terminal', result: 'done' }) as ChatMessagePart

const message = (
  id: string,
  role: ChatMessage['role'],
  parts: ChatMessagePart[],
  extra: Partial<ChatMessage> = {}
): ChatMessage => ({ id, role, parts, ...extra }) as ChatMessage

it('keeps a settled streamed final reply when the store only has an empty tool-round shell (#123047)', () => {
  const previous = [
    message('user-live', 'user', [textPart('check the model')]),
    message(
      'assistant-stream-live',
      'assistant',
      [textPart('I checked it.'), tool('call-1'), textPart('The model is running.')],
      { interim: false, pending: false }
    )
  ]

  const next = [
    message('user-stored', 'user', [textPart('check the model')], { rowId: 196700 }),
    message('assistant-shell', 'assistant', [tool('call-1')], { pending: false, rowId: 196701 })
  ]

  const merged = preserveLocalPendingTurnMessages(next, previous)

  expect(merged).toHaveLength(2)
  expect(merged[1]).toMatchObject({ id: 'assistant-stream-live', pending: false, rowId: 196701 })
  expect(chatMessageText(merged[1])).toContain('The model is running.')
})

it('does not let settled interim narration replace the empty authoritative shell', () => {
  const previous = [
    message('user-live', 'user', [textPart('check the model')]),
    message('assistant-stream-narration', 'assistant', [textPart('I am checking the logs…')], {
      interim: true,
      pending: false
    })
  ]

  const next = [
    message('user-stored', 'user', [textPart('check the model')], { rowId: 196700 }),
    message('assistant-shell', 'assistant', [tool('call-1')], { pending: false, rowId: 196701 })
  ]

  const merged = preserveLocalPendingTurnMessages(next, previous)

  expect(merged).toEqual(next)
})
