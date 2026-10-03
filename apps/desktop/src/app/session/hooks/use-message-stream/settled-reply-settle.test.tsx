import type { GatewayEvent } from '@hermes/shared'
import { act, cleanup } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { type ChatMessage, chatMessageText, toChatMessages } from '@/lib/chat-messages'
import { createClientSessionState } from '@/lib/chat-runtime'
import type { SessionMessage } from '@/types/hermes'

import { renderMessageStream } from './test-harness'

/**
 * `completeAssistantMessage` chooses between settling the terminal frame onto the
 * turn's own bubble and appending a new one. A pending reply or matching durable
 * row can identify that bubble; position alone cannot. A settled background reply
 * with no matching identity must keep its text while the final appends beside it.
 */
const textPart = (text: string) => ({ type: 'text' as const, text })

const transcript = (messages: readonly { id: string; parts: unknown[]; role: string }[]) =>
  messages
    .map(message => {
      const text = message.parts
        .map(part => (part && typeof part === 'object' && 'text' in part ? String(part.text) : ''))
        .join('')

      return `${message.role}[${message.id}]: ${text}`
    })
    .join('\n')

const seed = (id: string, storedId: string, messages: ChatMessage[]) =>
  new Map([
    [
      id,
      {
        ...createClientSessionState(storedId, messages),
        awaitingResponse: true,
        busy: true
      }
    ]
  ])

describe('terminal frame vs an already-settled reply', () => {
  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it("leaves a settled reply alone and appends this turn's final beside it", async () => {
    const sid = 'settled-reply-kept'

    const stream = renderMessageStream(sid, {
      states: seed(sid, 'stored-settled', [
        { id: 'user-this-turn', parts: [textPart('start the long job')], role: 'user' },
        {
          id: 'assistant-background',
          parts: [textPart('BACKGROUND REPLY: the job finished elsewhere')],
          pending: false,
          role: 'assistant'
        }
      ])
    })

    await act(async () => {
      stream.handleEvent({
        payload: { text: 'THIS TURN: the answer you asked for' },
        session_id: sid,
        type: 'message.complete'
      } as GatewayEvent)
    })

    const after = transcript(stream.state().messages)

    expect(after, 'the settled reply keeps its own text').toContain('BACKGROUND REPLY')
    expect(after, "this turn's final is on screen too").toContain('THIS TURN')
  })

  it("settles this turn's own pending bubble instead of painting the reply twice", async () => {
    const sid = 'own-reply-settled'

    const stream = renderMessageStream(sid, {
      states: seed(sid, 'stored-own', [
        { id: 'user-own', parts: [textPart('do the thing')], role: 'user' },
        {
          id: 'assistant-live',
          parts: [textPart('partial reply that lost a delta')],
          pending: true,
          role: 'assistant'
        }
      ])
    })

    await act(async () => {
      stream.handleEvent({
        payload: { text: 'The rewritten final that shares no prefix' },
        session_id: sid,
        type: 'message.complete'
      } as GatewayEvent)
    })

    const messages = stream.state().messages
    const assistantRows = messages.filter(message => message.role === 'assistant' && !message.hidden)

    expect(assistantRows, 'one reply, one bubble').toHaveLength(1)
    expect(transcript(messages)).toContain('The rewritten final that shares no prefix')
  })

  it.each([
    { folded: false, hasNeighbour: false },
    { folded: false, hasNeighbour: true },
    { folded: true, hasNeighbour: true }
  ])('settles a reloaded reply by its persisted row ($folded, $hasNeighbour)', async ({ folded, hasNeighbour }) => {
    const sid = 'reloaded-reply'
    const history: SessionMessage[] = [{ id: 41, role: 'user', content: 'do the thing' }]

    if (folded) {
      history.push(
        {
          id: 42,
          role: 'assistant',
          content: 'Checking the file.',
          tool_calls: [{ id: 'read-1', type: 'function', function: { name: 'read_file', arguments: '{}' } }]
        },
        { id: 43, role: 'tool', content: 'file contents', tool_call_id: 'read-1' }
      )
    }

    history.push({ id: 44, role: 'assistant', content: 'The displayed draft' })
    const reloaded = toChatMessages(history)
    const reply = reloaded.at(-1)!

    const neighbour: ChatMessage = {
      id: 'background-reply',
      role: 'assistant',
      rowId: 45,
      parts: [textPart('The independent background answer')]
    }

    const before = hasNeighbour ? [...reloaded, neighbour] : reloaded
    const stream = renderMessageStream(sid, { states: seed(sid, 'stored-reloaded', before) })

    const receipt = {
      row_ids: history.map(message => message.id!),
      user_row_id: 41,
      final_assistant_row_id: 44,
      complete: true
    }

    // History has replaced the volatile streaming identity before the late
    // terminal frame arrives. A folded bubble addresses its first source row;
    // the final response's identity lives on its text part instead.
    expect(stream.state().streamId).toBeNull()
    expect(reply.pending).not.toBe(true)
    expect(reply.rowId).toBe(folded ? 42 : 44)

    await act(async () => {
      stream.handleEvent({
        payload: { text: 'The authoritative final', persisted_turn: receipt },
        session_id: sid,
        type: 'message.complete'
      })
    })

    const after = stream.state().messages
    expect(after.map(message => message.id)).toEqual(before.map(message => message.id))

    if (hasNeighbour) {
      expect(after.at(-1)).toBe(neighbour)
    }

    const settled = after.find(message => message.id === reply.id)!
    expect(settled.parts.filter(part => part.type === 'text').map(part => part.text)).toEqual(
      folded ? ['Checking the file.', 'The authoritative final'] : ['The authoritative final']
    )
    expect(settled.parts.findLast(part => part.type === 'text')?.sourceRowId).toBe(44)
    expect(settled.parts.filter(part => part.type === 'tool-call')).toEqual(
      reply.parts.filter(part => part.type === 'tool-call')
    )
    expect(settled.persistedTurn).toEqual(receipt)
  })

  it.each([
    ...[undefined, null, 0, -1, 1.5, Number.NaN, Number.MAX_SAFE_INTEGER + 1, 99].map(finalRowId => ({
      finalRowId,
      folded: false
    })),
    { finalRowId: 42, folded: true }
  ])(
    'preserves a settled neighbour without a matching final response ($finalRowId, $folded)',
    async ({ finalRowId, folded }) => {
      const sid = 'unproven-reply'
      const history: SessionMessage[] = [{ id: 41, role: 'user', content: 'do the thing' }]

      if (folded) {
        history.push(
          {
            id: 42,
            role: 'assistant',
            content: 'Earlier commentary.',
            tool_calls: [{ id: 'read-1', type: 'function', function: { name: 'read_file', arguments: '{}' } }]
          },
          { id: 43, role: 'tool', content: 'file contents', tool_call_id: 'read-1' }
        )
      }

      history.push({ id: 44, role: 'assistant', content: 'The independent background answer' })
      const reloaded = toChatMessages(history)
      const stream = renderMessageStream(sid, { states: seed(sid, 'stored-unproven', reloaded) })

      await act(async () => {
        stream.handleEvent({
          payload: {
            text: 'This turn has a different answer',
            persisted_turn: { row_ids: [], final_assistant_row_id: finalRowId, complete: false }
          },
          session_id: sid,
          type: 'message.complete'
        })
      })

      const after = stream.state().messages
      expect(after.slice(0, reloaded.length)).toEqual(reloaded)
      expect(after.filter(message => message.role === 'assistant').map(chatMessageText)).toEqual([
        chatMessageText(reloaded.at(-1)!),
        'This turn has a different answer'
      ])
    }
  )
})
