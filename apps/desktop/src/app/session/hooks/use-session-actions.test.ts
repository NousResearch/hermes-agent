import { describe, expect, it } from 'vitest'

import type { ChatMessage } from '@/lib/chat-messages'

import { chatMessageArraysEquivalent, chatMessagesEquivalent, chatPartsEquivalent } from './use-session-actions'

describe('chatPartsEquivalent', () => {
  it('returns true for identical text parts', () => {
    const partA = { type: 'text' as const, text: 'Hello world' }
    const partB = { type: 'text' as const, text: 'Hello world' }

    expect(chatPartsEquivalent(partA, partB)).toBe(true)
  })

  it('returns false for text parts with different content', () => {
    const partA = { type: 'text' as const, text: 'Hello' }
    const partB = { type: 'text' as const, text: 'World' }

    expect(chatPartsEquivalent(partA, partB)).toBe(false)
  })

  it('returns true for identical reasoning parts', () => {
    const partA = { type: 'reasoning' as const, text: 'Thinking...' }
    const partB = { type: 'reasoning' as const, text: 'Thinking...' }

    expect(chatPartsEquivalent(partA, partB)).toBe(true)
  })

  it('returns true for tool-call parts with same identity and both have no result', () => {
    const partA = { type: 'tool-call' as const, toolCallId: 'tc-1', toolName: 'read_file', args: {} as never, argsText: '{}' }
    const partB = { type: 'tool-call' as const, toolCallId: 'tc-1', toolName: 'read_file', args: {} as never, argsText: '{}' }

    expect(chatPartsEquivalent(partA, partB)).toBe(true)
  })

  it('returns true for tool-call parts with same identity and both have results', () => {
    const partA = { type: 'tool-call' as const, toolCallId: 'tc-1', toolName: 'read_file', args: {} as never, argsText: '{}', result: { content: 'file data' }, isError: false }
    const partB = { type: 'tool-call' as const, toolCallId: 'tc-1', toolName: 'read_file', args: {} as never, argsText: '{}', result: { content: 'file data' }, isError: false }

    expect(chatPartsEquivalent(partA, partB)).toBe(true)
  })

  it('returns false when only one tool-call part has a result', () => {
    const partA = { type: 'tool-call' as const, toolCallId: 'tc-1', toolName: 'read_file', args: {} as never, argsText: '{}' }
    const partB = { type: 'tool-call' as const, toolCallId: 'tc-1', toolName: 'read_file', args: {} as never, argsText: '{}', result: { content: 'file data' }, isError: false }

    expect(chatPartsEquivalent(partA, partB)).toBe(false)
  })

  it('uses reference equality fast-path for identical part objects', () => {
    const part = { type: 'text' as const, text: 'Same reference' }

    expect(chatPartsEquivalent(part, part)).toBe(true)
  })
})

describe('chatMessagesEquivalent', () => {
  it('returns true for structurally identical messages', () => {
    const messageA: ChatMessage = {
      id: 'msg-1',
      role: 'user',
      parts: [{ type: 'text', text: 'Hello' }]
    }
    const messageB: ChatMessage = {
      id: 'msg-1',
      role: 'user',
      parts: [{ type: 'text', text: 'Hello' }]
    }

    expect(chatMessagesEquivalent(messageA, messageB)).toBe(true)
  })

  it('returns false when text part content differs', () => {
    const messageA: ChatMessage = {
      id: 'msg-1',
      role: 'user',
      parts: [{ type: 'text', text: 'Hello' }]
    }
    const messageB: ChatMessage = {
      id: 'msg-1',
      role: 'user',
      parts: [{ type: 'text', text: 'World' }]
    }

    expect(chatMessagesEquivalent(messageA, messageB)).toBe(false)
  })

  it('returns false when tool result presence differs', () => {
    const messageA: ChatMessage = {
      id: 'msg-1',
      role: 'assistant',
      parts: [
        { type: 'tool-call', toolCallId: 'tc-1', toolName: 'read_file', args: {} as never, argsText: '{}' }
      ]
    }
    const messageB: ChatMessage = {
      id: 'msg-1',
      role: 'assistant',
      parts: [
        { type: 'tool-call', toolCallId: 'tc-1', toolName: 'read_file', args: {} as never, argsText: '{}', result: { content: 'data' }, isError: false }
      ]
    }

    expect(chatMessagesEquivalent(messageA, messageB)).toBe(false)
  })

  it('returns false when message IDs differ', () => {
    const messageA: ChatMessage = {
      id: 'msg-1',
      role: 'user',
      parts: [{ type: 'text', text: 'Hello' }]
    }
    const messageB: ChatMessage = {
      id: 'msg-2',
      role: 'user',
      parts: [{ type: 'text', text: 'Hello' }]
    }

    expect(chatMessagesEquivalent(messageA, messageB)).toBe(false)
  })

  it('compares large messages with embedded images structurally without JSON.stringify', () => {
    // This test asserts correctness, not timing — it verifies that two
    // structurally identical messages (that would be equal via stringify)
    // are also equal via the new cheap structural compare.
    const messageA: ChatMessage = {
      id: 'msg-1',
      role: 'assistant',
      parts: [
        { type: 'text', text: 'Here are the images:' },
        { type: 'tool-call', toolCallId: 'img-1', toolName: 'image_generate', args: { prompt: 'a cat' } as never, argsText: '{"prompt":"a cat"}', result: { image: 'data:image/png;base64,iVBORw0KG...(large base64)' }, isError: false }
      ]
    }
    const messageB: ChatMessage = {
      id: 'msg-1',
      role: 'assistant',
      parts: [
        { type: 'text', text: 'Here are the images:' },
        { type: 'tool-call', toolCallId: 'img-1', toolName: 'image_generate', args: { prompt: 'a cat' } as never, argsText: '{"prompt":"a cat"}', result: { image: 'data:image/png;base64,iVBORw0KG...(large base64)' }, isError: false }
      ]
    }

    // The new structural compare treats these as equal (both have result defined,
    // same toolCallId/toolName), without comparing the full result object.
    expect(chatMessagesEquivalent(messageA, messageB)).toBe(true)
  })
})

describe('chatMessageArraysEquivalent', () => {
  it('returns true for identical arrays via identity fast-path', () => {
    const messages: ChatMessage[] = [
      { id: 'msg-1', role: 'user', parts: [{ type: 'text', text: 'Hello' }] }
    ]

    expect(chatMessageArraysEquivalent(messages, messages)).toBe(true)
  })

  it('returns true for structurally identical arrays', () => {
    const messagesA: ChatMessage[] = [
      { id: 'msg-1', role: 'user', parts: [{ type: 'text', text: 'Hello' }] }
    ]
    const messagesB: ChatMessage[] = [
      { id: 'msg-1', role: 'user', parts: [{ type: 'text', text: 'Hello' }] }
    ]

    expect(chatMessageArraysEquivalent(messagesA, messagesB)).toBe(true)
  })

  it('returns false when a text part differs', () => {
    const messagesA: ChatMessage[] = [
      { id: 'msg-1', role: 'user', parts: [{ type: 'text', text: 'Hello' }] }
    ]
    const messagesB: ChatMessage[] = [
      { id: 'msg-1', role: 'user', parts: [{ type: 'text', text: 'World' }] }
    ]

    expect(chatMessageArraysEquivalent(messagesA, messagesB)).toBe(false)
  })

  it('returns false when tool result presence differs', () => {
    const messagesA: ChatMessage[] = [
      {
        id: 'msg-1',
        role: 'assistant',
        parts: [
          { type: 'tool-call', toolCallId: 'tc-1', toolName: 'read_file', args: {} as never, argsText: '{}' }
        ]
      }
    ]
    const messagesB: ChatMessage[] = [
      {
        id: 'msg-1',
        role: 'assistant',
        parts: [
          { type: 'tool-call', toolCallId: 'tc-1', toolName: 'read_file', args: {} as never, argsText: '{}', result: { content: 'data' }, isError: false }
        ]
      }
    ]

    expect(chatMessageArraysEquivalent(messagesA, messagesB)).toBe(false)
  })
})
