import { beforeEach, describe, expect, it } from 'vitest'

import type { ChatMessage } from '@/lib/chat-messages'
import { $approvalModes, approvalModeForProfile } from '@/store/approval-mode'
import { $activeGatewayProfile } from '@/store/profile'
import type { SessionInfo } from '@/types/hermes'

import {
  appendLiveSessionProjection,
  applyRuntimeInfo,
  chatMessageArraysEquivalent,
  chatMessagesEquivalent,
  chatPartsEquivalent,
  isSessionGoneError,
  preserveLocalPendingTurnMessages,
  reconcileResumeMessages,
  sessionMatchesStoredId,
  sessionShouldHaveTranscript,
  toBranchMessages
} from './utils'

const msg = (id: string, role: ChatMessage['role'], text: string, extra: Partial<ChatMessage> = {}): ChatMessage =>
  ({ id, role, parts: [{ type: 'text', text }], ...extra }) as ChatMessage

const session = (over: Partial<SessionInfo>): SessionInfo => over as SessionInfo

describe('applyRuntimeInfo approval mode', () => {
  beforeEach(() => {
    $approvalModes.set({})
    $activeGatewayProfile.set('work')
  })

  it('reconciles session.info against the gateway profile', () => {
    applyRuntimeInfo({ approval_mode: 'smart', desktop_contract: 3 })

    expect(approvalModeForProfile('work')).toBe('smart')
    expect(approvalModeForProfile('default')).toBe('smart')
  })
})

describe('isSessionGoneError', () => {
  it('is true for 404 / session-not-found, false otherwise', () => {
    expect(isSessionGoneError(new Error('Request failed 404'))).toBe(true)
    expect(isSessionGoneError(new Error('Session not found'))).toBe(true)
    expect(isSessionGoneError(new Error('ECONNREFUSED'))).toBe(false)
    expect(isSessionGoneError(null)).toBe(false)
  })
})

describe('sessionMatchesStoredId', () => {
  it('matches on live id or lineage root', () => {
    expect(sessionMatchesStoredId(session({ id: 'a' }), 'a')).toBe(true)
    expect(sessionMatchesStoredId(session({ id: 'live', _lineage_root_id: 'root' }), 'root')).toBe(true)
    expect(sessionMatchesStoredId(session({ id: 'a' }), 'b')).toBe(false)
  })
})

describe('sessionShouldHaveTranscript', () => {
  it('is true only when the session has messages', () => {
    expect(sessionShouldHaveTranscript(session({ message_count: 3 }))).toBe(true)
    expect(sessionShouldHaveTranscript(session({ message_count: 0 }))).toBe(false)
    expect(sessionShouldHaveTranscript(undefined)).toBe(false)
  })
})

describe('toBranchMessages', () => {
  it('keeps only user/assistant turns that carry text', () => {
    const out = toBranchMessages([
      msg('u', 'user', 'hi'),
      msg('blank', 'assistant', '   '),
      msg('sys', 'system', 'ignored'),
      msg('a', 'assistant', 'hello')
    ])

    expect(out.map(b => b.source.id)).toEqual(['u', 'a'])
    expect(out[0]).toMatchObject({ content: 'hi', role: 'user' })
  })
})

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
    const partA = {
      type: 'tool-call' as const,
      toolCallId: 'tc-1',
      toolName: 'read_file',
      args: {} as never,
      argsText: '{}'
    }

    const partB = {
      type: 'tool-call' as const,
      toolCallId: 'tc-1',
      toolName: 'read_file',
      args: {} as never,
      argsText: '{}'
    }

    expect(chatPartsEquivalent(partA, partB)).toBe(true)
  })

  it('returns true for tool-call parts with same identity and both have results', () => {
    const partA = {
      type: 'tool-call' as const,
      toolCallId: 'tc-1',
      toolName: 'read_file',
      args: {} as never,
      argsText: '{}',
      result: { content: 'file data' },
      isError: false
    }

    const partB = {
      type: 'tool-call' as const,
      toolCallId: 'tc-1',
      toolName: 'read_file',
      args: {} as never,
      argsText: '{}',
      result: { content: 'file data' },
      isError: false
    }

    expect(chatPartsEquivalent(partA, partB)).toBe(true)
  })

  it('returns false when only one tool-call part has a result', () => {
    const partA = {
      type: 'tool-call' as const,
      toolCallId: 'tc-1',
      toolName: 'read_file',
      args: {} as never,
      argsText: '{}'
    }

    const partB = {
      type: 'tool-call' as const,
      toolCallId: 'tc-1',
      toolName: 'read_file',
      args: {} as never,
      argsText: '{}',
      result: { content: 'file data' },
      isError: false
    }

    expect(chatPartsEquivalent(partA, partB)).toBe(false)
  })

  it('uses reference equality fast-path for identical part objects', () => {
    const part = { type: 'text' as const, text: 'Same reference' }

    expect(chatPartsEquivalent(part, part)).toBe(true)
  })
})

describe('chatMessagesEquivalent', () => {
  it('returns true for structurally identical messages', () => {
    expect(chatMessagesEquivalent(msg('1', 'user', 'Hello'), msg('1', 'user', 'Hello'))).toBe(true)
  })

  it('returns false when text part content differs', () => {
    expect(chatMessagesEquivalent(msg('1', 'user', 'Hello'), msg('1', 'user', 'World'))).toBe(false)
  })

  it('returns false when tool result presence differs', () => {
    const messageA: ChatMessage = {
      id: 'msg-1',
      role: 'assistant',
      parts: [{ type: 'tool-call', toolCallId: 'tc-1', toolName: 'read_file', args: {} as never, argsText: '{}' }]
    }

    const messageB: ChatMessage = {
      id: 'msg-1',
      role: 'assistant',
      parts: [
        {
          type: 'tool-call',
          toolCallId: 'tc-1',
          toolName: 'read_file',
          args: {} as never,
          argsText: '{}',
          result: { content: 'data' },
          isError: false
        }
      ]
    }

    expect(chatMessagesEquivalent(messageA, messageB)).toBe(false)
  })

  it('returns false when message IDs differ', () => {
    expect(chatMessagesEquivalent(msg('msg-1', 'user', 'Hello'), msg('msg-2', 'user', 'Hello'))).toBe(false)
  })

  it('compares large messages with embedded images structurally without JSON.stringify', () => {
    // Verifies that two structurally identical messages (that would be equal
    // via stringify) are also equal via the new cheap structural compare.
    const messageA: ChatMessage = {
      id: 'msg-1',
      role: 'assistant',
      parts: [
        { type: 'text', text: 'Here are the images:' },
        {
          type: 'tool-call',
          toolCallId: 'img-1',
          toolName: 'image_generate',
          args: { prompt: 'a cat' } as never,
          argsText: '{"prompt":"a cat"}',
          result: { image: 'data:image/png;base64,iVBORw0KG...(large base64)' },
          isError: false
        }
      ]
    }

    const messageB: ChatMessage = {
      id: 'msg-1',
      role: 'assistant',
      parts: [
        { type: 'text', text: 'Here are the images:' },
        {
          type: 'tool-call',
          toolCallId: 'img-1',
          toolName: 'image_generate',
          args: { prompt: 'a cat' } as never,
          argsText: '{"prompt":"a cat"}',
          result: { image: 'data:image/png;base64,iVBORw0KG...(large base64)' },
          isError: false
        }
      ]
    }

    // The structural compare treats these as equal (both have result defined,
    // same toolCallId/toolName), without comparing the full result object.
    expect(chatMessagesEquivalent(messageA, messageB)).toBe(true)
  })
})

describe('chatMessageArraysEquivalent', () => {
  it('returns true for identical arrays via identity fast-path', () => {
    const messages: ChatMessage[] = [msg('1', 'user', 'x')]

    expect(chatMessageArraysEquivalent(messages, messages)).toBe(true)
  })

  it('compares length and per-message equivalence', () => {
    const a = [msg('1', 'user', 'x'), msg('2', 'assistant', 'y')]
    expect(chatMessageArraysEquivalent(a, [msg('1', 'user', 'x'), msg('2', 'assistant', 'y')])).toBe(true)
    expect(chatMessageArraysEquivalent(a, [msg('1', 'user', 'x')])).toBe(false)
    expect(chatMessageArraysEquivalent(a, [msg('1', 'user', 'x'), msg('2', 'assistant', 'changed')])).toBe(false)
  })
})

describe('reconcileResumeMessages', () => {
  it('returns next untouched when there is no previous transcript', () => {
    const next = [msg('1', 'user', 'hi')]
    expect(reconcileResumeMessages(next, [])).toBe(next)
  })

  it('re-grafts reasoning parts onto a matching assistant turn', () => {
    const next = [msg('a', 'assistant', 'answer')]

    const previous = [
      msg('a', 'assistant', 'answer', {
        parts: [
          { type: 'reasoning', text: 'thinking' },
          { type: 'text', text: 'answer' }
        ]
      } as Partial<ChatMessage>)
    ]

    const [out] = reconcileResumeMessages(next, previous)
    expect(out.parts.some(p => p.type === 'reasoning')).toBe(true)
  })
})

describe('preserveLocalPendingTurnMessages', () => {
  it('keeps an optimistic user turn and pending assistant when the server projection is behind', () => {
    const next = [msg('1-user', 'user', 'first'), msg('2-assistant', 'assistant', 'first answer')]

    const previous = [
      ...next,
      msg('user-optimistic', 'user', 'new question'),
      msg('assistant-stream-1', 'assistant', 'partial answer', { pending: true })
    ]

    expect(preserveLocalPendingTurnMessages(next, previous).map(message => message.id)).toEqual([
      '1-user',
      '2-assistant',
      'user-optimistic',
      'assistant-stream-1'
    ])
  })

  it('drops the local copies once the same role ordinals are authoritative', () => {
    const previous = [
      msg('1-user', 'user', 'first'),
      msg('2-assistant', 'assistant', 'first answer'),
      msg('user-optimistic', 'user', 'new question'),
      msg('assistant-stream-1', 'assistant', 'partial answer', { pending: true })
    ]

    const next = [
      msg('1-user-stored', 'user', 'first'),
      msg('2-assistant-stored', 'assistant', 'first answer'),
      msg('3-user-stored', 'user', 'new question'),
      msg('4-assistant-stored', 'assistant', 'complete answer')
    ]

    expect(preserveLocalPendingTurnMessages(next, previous)).toBe(next)
  })
})

describe('appendLiveSessionProjection', () => {
  it('restores the running turn and accepted queued prompt after a renderer restart', () => {
    const stored = [msg('stored-user', 'user', 'earlier'), msg('stored-assistant', 'assistant', 'earlier answer')]

    const restored = appendLiveSessionProjection(stored, {
      session_id: 'runtime-1',
      inflight: {
        user: 'current prompt',
        assistant: 'partial answer',
        streaming: true
      },
      queued: { user: 'newest prompt' }
    })

    expect(restored.map(message => message.role)).toEqual(['user', 'assistant', 'user', 'assistant', 'user'])
    expect(restored.map(message => message.parts.map(part => ('text' in part ? part.text : '')).join(''))).toEqual([
      'earlier',
      'earlier answer',
      'current prompt',
      'partial answer',
      'newest prompt'
    ])
    expect(restored[3]).toMatchObject({ id: 'assistant-stream-runtime-1', pending: true })
  })

  it('preserves the original array when no live projection exists', () => {
    const stored = [msg('stored-user', 'user', 'earlier')]

    expect(appendLiveSessionProjection(stored, { session_id: 'runtime-1' })).toBe(stored)
  })
})
