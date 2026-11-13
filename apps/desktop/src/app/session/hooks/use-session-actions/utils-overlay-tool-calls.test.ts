import { describe, expect, it } from 'vitest'

import type { ChatMessage } from '@/lib/chat-messages'

import { overlayConcurrentMessageChanges } from './utils'

describe('overlayConcurrentMessageChanges tool calls', () => {
  // A re-delivered open_requests marks the cached activation row "changed";
  // the raw concat used to carry the same clarify call twice, interleaved
  // with reasoning that ran between the copies.
  it('does not duplicate a tool call the live row already carries when merging the prefix', () => {
    const authoritative = [
      {
        id: 'assistant-stream-activation',
        role: 'assistant',
        parts: [
          { type: 'text', text: 'partial A' },
          { type: 'tool-call', toolCallId: 'call-clarify', toolName: 'clarify' }
        ],
        pending: true
      }
    ] as ChatMessage[]

    const current = [
      {
        id: 'assistant-stream-runtime',
        role: 'assistant',
        parts: [
          { type: 'tool-call', toolCallId: 'call-clarify', toolName: 'clarify' },
          { type: 'reasoning', text: 'mid reasoning' },
          { type: 'text', text: ' + delta B' }
        ],
        pending: true
      }
    ] as ChatMessage[]

    const overlaid = overlayConcurrentMessageChanges(authoritative, [], current)

    expect(overlaid.map(message => message.id)).toEqual(['assistant-stream-runtime'])
    const toolCallIds = overlaid[0].parts.flatMap(part => (part.type === 'tool-call' ? [part.toolCallId] : []))
    expect(toolCallIds).toEqual(['call-clarify'])
    // The activation side still contributes what the live row lacks.
    expect(overlaid[0].parts[0]).toEqual({ type: 'text', text: 'partial A' })
  })

  // The activation page's stream row and the live store's copy of the same turn.
  const merge = (page: ChatMessage['parts'], live: ChatMessage['parts']) =>
    overlayConcurrentMessageChanges(
      [{ id: 'assistant-stream-activation', role: 'assistant', parts: page, pending: true }],
      [],
      [{ id: 'assistant-stream-runtime', role: 'assistant', parts: live, pending: true }]
    )[0].parts

  const calls = (parts: ChatMessage['parts']) =>
    parts.flatMap(part => (part.type === 'tool-call' ? [`${part.toolName}:${part.toolCallId}`] : []))

  it('keeps an answered clarify when a later call of another tool reuses its id', () => {
    // llama.cpp sends one constant id for every call.
    const parts = merge(
      [
        { type: 'text', text: 'partial A' },
        { type: 'tool-call', toolCallId: 'call_0', toolName: 'clarify', result: { answer: 'staging' } }
      ] as ChatMessage['parts'],
      [
        { type: 'tool-call', toolCallId: 'call_0', toolName: 'terminal' },
        { type: 'text', text: ' + delta B' }
      ] as ChatMessage['parts']
    )

    expect(calls(parts)).toEqual(['clarify:call_0', 'terminal:call_0'])
    expect(parts.find(part => part.type === 'tool-call' && part.toolName === 'clarify')).toMatchObject({
      result: { answer: 'staging' }
    })
  })

  it('keeps the copy that carries a result', () => {
    const args = { questions: [{ question: 'Which deployment target?' }] }

    const parts = merge(
      [
        { type: 'text', text: 'partial A' },
        { type: 'tool-call', toolCallId: 'call-clarify', toolName: 'clarify', args, result: { answer: 'staging' } }
      ] as ChatMessage['parts'],
      [
        { type: 'tool-call', toolCallId: 'call-clarify', toolName: 'clarify', args },
        { type: 'text', text: ' + delta B' }
      ] as ChatMessage['parts']
    )

    expect(calls(parts)).toEqual(['clarify:call-clarify'])
    expect(parts[1]).toMatchObject({ result: { answer: 'staging' } })
  })

  it('keeps a running call that shares a finished call of the same tool its id', () => {
    const parts = merge(
      [
        { type: 'text', text: 'partial A' },
        { type: 'tool-call', toolCallId: 'call_0', toolName: 'terminal', args: { command: 'ls' }, result: 'ok' }
      ] as ChatMessage['parts'],
      [
        { type: 'tool-call', toolCallId: 'call_0', toolName: 'terminal', args: { command: 'pwd' } },
        { type: 'text', text: ' + delta B' }
      ] as ChatMessage['parts']
    )

    expect(calls(parts)).toEqual(['terminal:call_0', 'terminal:call_0'])
    expect(parts.filter(part => part.type === 'tool-call' && part.result === undefined)).toHaveLength(1)
  })

  it('keeps an open clarify that shares an answered clarify its id', () => {
    const parts = merge(
      [
        { type: 'text', text: 'partial A' },
        {
          type: 'tool-call',
          toolCallId: 'call_0',
          toolName: 'clarify',
          args: { questions: [{ question: 'Which deployment target?' }] },
          result: { answer: 'staging' }
        }
      ] as ChatMessage['parts'],
      [
        {
          type: 'tool-call',
          toolCallId: 'call_0',
          toolName: 'clarify',
          args: { questions: [{ question: 'Which region?' }] }
        },
        { type: 'text', text: ' + delta B' }
      ] as ChatMessage['parts']
    )

    expect(calls(parts)).toEqual(['clarify:call_0', 'clarify:call_0'])
    expect(parts.filter(part => part.type === 'tool-call' && part.result === undefined)).toHaveLength(1)
  })

  it('keeps the call where the activation page had it', () => {
    const parts = merge(
      [
        { type: 'text', text: 'partial A' },
        { type: 'tool-call', toolCallId: 'call-clarify', toolName: 'clarify' },
        { type: 'reasoning', text: 'after the call' }
      ] as ChatMessage['parts'],
      [
        { type: 'tool-call', toolCallId: 'call-clarify', toolName: 'clarify' },
        { type: 'text', text: ' + delta B' }
      ] as ChatMessage['parts']
    )

    expect(parts.map(part => (part.type === 'tool-call' ? part.toolName : part.type))).toEqual([
      'text',
      'clarify',
      'reasoning',
      'text'
    ])
  })
})
