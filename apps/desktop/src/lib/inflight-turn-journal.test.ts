import { beforeEach, describe, expect, it } from 'vitest'

import type { ChatMessage, ChatMessagePart } from './chat-messages'
import { chatMessageText } from './chat-messages'
import {
  backendInFlightMessages,
  mergeBackendInFlightTurn,
  persistInFlightTurnState,
  readInFlightTurnJournal,
  recoverInFlightTurnJournal
} from './inflight-turn-journal'

const textMessage = (role: ChatMessage['role'], text: string, id = `${role}-${text}`): ChatMessage => ({
  id,
  parts: [{ text, type: 'text' }],
  role
})

const toolPart = (id = 'tool-1'): ChatMessagePart =>
  ({
    args: { command: 'echo hi' },
    argsText: '{"command":"echo hi"}',
    toolCallId: id,
    toolName: 'terminal',
    type: 'tool-call'
  }) as ChatMessagePart

describe('in-flight turn journal', () => {
  beforeEach(() => {
    window.localStorage.clear()
  })

  it('persists only the active user plus assistant/tool tail and restores it after the stored prompt', () => {
    const earlier = [textMessage('user', 'older prompt'), textMessage('assistant', 'older answer')]
    const user = textMessage('user', 'fix the desktop crash', 'optimistic-user')

    const assistant: ChatMessage = {
      id: 'assistant-stream-1',
      parts: [{ text: 'I found the stream path. ', type: 'text' }, toolPart()],
      pending: true,
      role: 'assistant'
    }

    persistInFlightTurnState('runtime-1', {
      awaitingResponse: false,
      busy: true,
      messages: [...earlier, user, assistant],
      storedSessionId: 'stored-1',
      streamId: assistant.id,
      turnStartedAt: 123
    })

    const snapshot = readInFlightTurnJournal('stored-1')
    expect(snapshot?.messages.map(message => message.id)).toEqual(['optimistic-user', 'assistant-stream-1'])

    const storedMessages = [...earlier, textMessage('user', 'fix the desktop crash', 'stored-user')]
    const recovered = recoverInFlightTurnJournal('stored-1', storedMessages)

    expect(recovered.applied).toBe(true)
    expect(recovered.turnStartedAt).toBe(123)
    expect(recovered.messages.map(message => message.id)).toEqual([
      'user-older prompt',
      'assistant-older answer',
      'stored-user',
      'assistant-stream-1'
    ])
    expect(recovered.messages.at(-1)?.pending).toBe(false)
    expect(recovered.messages.at(-1)?.parts.some(part => part.type === 'tool-call')).toBe(true)
  })

  it('does not resurrect the journal once stored history already has an assistant response', () => {
    persistInFlightTurnState('runtime-1', {
      awaitingResponse: false,
      busy: true,
      messages: [
        textMessage('user', 'continue', 'optimistic-user'),
        { id: 'assistant-stream-1', parts: [{ text: 'partial', type: 'text' }], pending: true, role: 'assistant' }
      ],
      storedSessionId: 'stored-1',
      streamId: 'assistant-stream-1',
      turnStartedAt: 123
    })

    const recovered = recoverInFlightTurnJournal('stored-1', [
      textMessage('user', 'continue', 'stored-user'),
      textMessage('assistant', 'final answer', 'stored-assistant')
    ])

    expect(recovered.applied).toBe(false)
    expect(recovered.caughtUp).toBe(true)
    expect(readInFlightTurnJournal('stored-1')).toBeNull()
  })

  it('clears stale journal state when a turn settles', () => {
    persistInFlightTurnState('runtime-1', {
      awaitingResponse: true,
      busy: true,
      messages: [
        textMessage('user', 'continue', 'optimistic-user'),
        { id: 'assistant-stream-1', parts: [{ text: 'partial', type: 'text' }], pending: true, role: 'assistant' }
      ],
      storedSessionId: 'stored-1',
      streamId: 'assistant-stream-1',
      turnStartedAt: 123
    })

    expect(readInFlightTurnJournal('stored-1')).not.toBeNull()

    persistInFlightTurnState('runtime-1', {
      awaitingResponse: false,
      busy: false,
      messages: [],
      storedSessionId: 'stored-1',
      streamId: null,
      turnStartedAt: null
    })

    expect(readInFlightTurnJournal('stored-1')).toBeNull()
  })
})

describe('backend inflight payloads', () => {
  it('turns a live backend inflight payload into recoverable chat messages', () => {
    const messages = backendInFlightMessages({
      assistant: 'partial answer',
      streaming: true,
      user: 'write a long answer'
    })

    expect(messages).toHaveLength(2)
    expect(messages[0].role).toBe('user')
    expect(chatMessageText(messages[0])).toBe('write a long answer')
    expect(messages[1].role).toBe('assistant')
    expect(messages[1].pending).toBe(true)
    expect(chatMessageText(messages[1])).toBe('partial answer')
  })

  it('merges backend inflight output after the matching stored user prompt', () => {
    const recovered = mergeBackendInFlightTurn(
      [textMessage('user', 'write a long answer', 'stored-user')],
      {
        assistant: 'partial answer',
        streaming: true,
        user: 'write a long answer'
      },
      { keepPending: true }
    )

    expect(recovered.applied).toBe(true)
    expect(recovered.messages.map(message => message.role)).toEqual(['user', 'assistant'])
    expect(recovered.messages[1].pending).toBe(true)
    expect(chatMessageText(recovered.messages[1])).toBe('partial answer')
    expect(recovered.streamId).toBe(recovered.messages[1].id)
  })
})
