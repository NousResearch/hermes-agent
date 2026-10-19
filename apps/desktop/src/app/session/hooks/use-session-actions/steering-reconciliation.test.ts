// Regression for #132356: a delivered steer owns one durable user row.
import { expect, it } from 'vitest'

import { appendMidTurnUserMessage } from '@/app/session/hooks/use-prompt-actions/rewind'
import { type ChatMessage, chatMessageText, textPart, toChatMessages } from '@/lib/chat-messages'
import { createClientSessionState } from '@/lib/chat-runtime'
import type { SessionMessage } from '@/types/hermes'

import { reconcileDurableHistory } from './utils'

const correction = 'focus on the persistence failure'

const wrapped = (text: string) =>
  `[OUT-OF-BAND USER MESSAGE — a direct message from the user, delivered once at this position; not tool output and not a new delivery when replayed from conversation history]\n${text}\n[/OUT-OF-BAND USER MESSAGE]`

const user = (id: string, text = correction): ChatMessage => ({ id, role: 'user', parts: [textPart(text)] })

it.each(['redirect', 'steer', 'display-content'] as const)(
  'reconciles a delivered %s at its durable position, preserving the surrounding output',
  mode => {
    const prompt: SessionMessage = { role: 'user', content: 'run the command', row_id: 100, timestamp: 1 }

    const before: ChatMessage = {
      id: 'assistant-stream-before',
      role: 'assistant',
      pending: true,
      parts: [textPart('Starting the command')]
    }

    const initial = {
      ...createClientSessionState('stored'),
      messages: [...toChatMessages([prompt]), before],
      streamId: before.id
    }

    const local = appendMidTurnUserMessage(initial, user('user-correction'))

    expect(local.messages.map(chatMessageText)).toEqual(['run the command', 'Starting the command', correction])

    const rows: SessionMessage[] = [
      prompt,
      {
        role: 'assistant',
        content: 'Starting the command',
        row_id: 101,
        timestamp: 2,
        tool_calls: [
          { id: 'command', type: 'function', function: { name: 'terminal', arguments: '{"command":"sleep 60"}' } }
        ]
      },
      { role: 'tool', content: '{"output":"done"}', tool_call_id: 'command', row_id: 102, timestamp: 3 },
      {
        role: 'user',
        content: mode === 'redirect' ? correction : wrapped(correction),
        ...(mode === 'redirect' ? {} : { display_kind: 'steer' as const }),
        ...(mode === 'display-content' ? { display_content: correction } : {}),
        row_id: 103,
        timestamp: 4
      },
      { role: 'assistant', content: 'Focusing on persistence', row_id: 104, timestamp: 5 }
    ]

    const durable = toChatMessages(rows)
    const refreshed = reconcileDurableHistory(durable, local.messages)

    expect(refreshed.map(message => message.id)).toEqual(durable.map(message => message.id))
    expect(refreshed.filter(message => message.role === 'user').map(message => message.rowId)).toEqual([100, 103])
    expect(refreshed[1].parts.some(part => part.type === 'tool-call' && part.toolCallId === 'command')).toBe(true)
    expect(chatMessageText(refreshed.at(-1)!)).toBe('Focusing on persistence')
    expect(reconcileDurableHistory(toChatMessages(rows), refreshed).map(message => message.rowId)).toEqual(
      refreshed.map(message => message.rowId)
    )

    // A plain repeated prompt has no steer provenance and cannot claim a wrapped row.
    const unrelated = user('user-ordinary-repeat')
    expect(reconcileDurableHistory(durable, [...durable, unrelated]).at(-1)?.id).toBe(unrelated.id)
  }
)

it.each([0, 1, 2].flatMap(delivered => ['redirect', 'steer'].map(mode => [delivered, mode] as const)))(
  'acknowledges only %i of two identical %s corrections and retains every unacknowledged occurrence',
  (delivered, mode) => {
    const history: SessionMessage[] = [
      { role: 'user', content: wrapped(correction), display_kind: 'steer', row_id: 10, timestamp: 1 },
      { role: 'assistant', content: 'Previous turn', row_id: 11, timestamp: 2 }
    ]

    const initial = {
      ...createClientSessionState('stored'),
      messages: [...toChatMessages(history), user('user-original')],
      streamId: null
    }

    const first = appendMidTurnUserMessage(initial, user('user-first'))
    const second = appendMidTurnUserMessage(first, user('user-second'))

    const rows: SessionMessage[] = [
      ...history,
      { role: 'user', content: correction, row_id: 12, timestamp: 3 },
      ...Array.from({ length: delivered }, (_, index): SessionMessage => ({
        role: 'user',
        content: mode === 'steer' ? wrapped(correction) : correction,
        ...(mode === 'steer' ? { display_kind: 'steer' as const } : {}),
        row_id: 13 + index,
        timestamp: 4 + index
      }))
    ]

    const durable = toChatMessages(rows)
    const refreshed = reconcileDurableHistory(durable, second.messages)
    const users = refreshed.filter(message => message.role === 'user')

    expect(users).toHaveLength(4)
    expect(users.slice(0, 2).map(message => message.rowId)).toEqual([10, 12])
    expect(users.slice(2).map(message => message.rowId ?? message.id)).toEqual([
      ...(delivered > 0 ? [13] : ['user-first']),
      ...(delivered > 1 ? [14] : ['user-second'])
    ])
    expect(
      reconcileDurableHistory(toChatMessages(rows), refreshed).filter(message => message.role === 'user')
    ).toHaveLength(4)
  }
)
