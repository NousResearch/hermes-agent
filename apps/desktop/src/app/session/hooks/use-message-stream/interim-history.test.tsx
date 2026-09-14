import { act, cleanup } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { chatMessageText, toChatMessages } from '@/lib/chat-messages'
import type { RpcEvent, SessionMessage } from '@/types/hermes'

import { renderMessageStream } from './test-harness'

const SID = 'interim-preservation'

afterEach(cleanup)

describe('intermediate assistant text survives Desktop lifecycle boundaries', () => {
  it.each(['message.complete', 'message.interim'] as const)(
    'keeps pre-tool text when %s settles the next response',
    async terminal => {
      const stream = renderMessageStream(SID)

      const emit = async (type: RpcEvent['type'], payload: RpcEvent['payload']) => {
        await act(() => stream.handleEvent({ type, payload, session_id: SID }))
      }

      await emit('message.start', {})
      await emit('message.delta', { text: 'Checking the files.', timestamp: 1 })
      // Missing/disabled interim delivery must not make this older model response
      // disposable when the later model response is finalized.
      await emit('tool.start', { tool_id: 'call-1', name: 'terminal', args: { command: 'pwd' }, timestamp: 2 })
      await emit('tool.complete', { tool_id: 'call-1', name: 'terminal', result: 'ok', timestamp: 3 })
      await emit('message.delta', { text: 'Partial answer', timestamp: 4 })
      await emit(terminal, { text: 'The result is ready.', timestamp: 5 })
      const messages = stream.state().messages
      const visible = messages.map(chatMessageText).join('\n')
      expect(visible).toContain('Checking the files.')
      expect(visible).toContain('The result is ready.')
      expect(visible).not.toContain('Partial answer')
      expect(messages.flatMap(message => message.parts).filter(part => part.type === 'tool-call')).toHaveLength(1)
      expect(stream.state().busy).toBe(terminal === 'message.interim')
    }
  )

  it.each(['rpc', 'rest'] as const)(
    'preserves the same public commentary on live → %s history projection',
    async transport => {
      const stream = renderMessageStream(SID)

      const emit = async (type: RpcEvent['type'], payload: RpcEvent['payload']) => {
        await act(() => stream.handleEvent({ type, payload, session_id: SID }))
      }

      await emit('message.start', {})
      await emit('reasoning.delta', { text: 'Real reasoning.', timestamp: 1 })
      await emit('message.interim', { text: 'I will inspect the files.', already_streamed: false, timestamp: 2 })
      await emit('tool.start', { tool_id: 'call-1', name: 'terminal', args: { command: 'pwd' }, timestamp: 3 })
      await emit('tool.complete', { tool_id: 'call-1', name: 'terminal', result: 'ok', timestamp: 4 })
      await emit('message.delta', { text: 'All checks passed.', timestamp: 5 })
      await emit('message.complete', { text: 'All checks passed.', timestamp: 6 })

      const items = [
        {
          type: 'message',
          role: 'assistant',
          phase: 'commentary',
          content: [{ type: 'output_text', text: 'I will inspect the files.' }]
        }
      ]

      const rows: SessionMessage[] = [
        {
          role: 'assistant',
          content: '',
          timestamp: 1,
          reasoning: 'Real reasoning.',
          codex_message_items: transport === 'rest' ? JSON.stringify(items) : items,
          tool_calls: [
            { id: 'call-1', type: 'function', function: { name: 'terminal', arguments: '{"command":"pwd"}' } }
          ]
        },
        { role: 'tool', content: 'ok', tool_call_id: 'call-1', name: 'terminal', timestamp: 4 },
        { role: 'assistant', content: 'All checks passed.', timestamp: 5 }
      ]

      const texts = (messages: ReturnType<typeof toChatMessages>) =>
        messages
          .flatMap(message => message.parts.flatMap(part => (part.type === 'text' ? [part.text.trim()] : [])))
          .filter(Boolean)

      const live = texts(stream.state().messages)
      const restored = texts(toChatMessages(rows))
      expect(live).toEqual(['I will inspect the files.', 'All checks passed.'])
      expect(restored).toEqual(live)
      expect(restored.join('')).not.toContain('Real reasoning.')
      expect(
        toChatMessages(rows)
          .flatMap(message => message.parts)
          .filter(part => part.type === 'tool-call')
      ).toHaveLength(1)
    }
  )
})
