import { cleanup, renderHook } from '@testing-library/react'
import type { MutableRefObject } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { requestComposerInsert } from '@/app/chat/composer/focus'
import { getSessionMessages } from '@/hermes'
import { chatMessageText, textPart } from '@/lib/chat-messages'
import type { SessionMessage } from '@/types/hermes'

import type { ClientSessionState } from '../../types'

import { usePromptActions } from './use-prompt-actions'

vi.mock('@/hermes', () => ({
  getProfiles: vi.fn(async () => ({ profiles: [] })),
  getSessionMessages: vi.fn(),
  setApiRequestProfile: vi.fn(),
  transcribeAudio: vi.fn()
}))

vi.mock('@/app/chat/composer/focus', () => ({
  requestComposerInsert: vi.fn()
}))

const ref = <T,>(value: T): MutableRefObject<T> => ({ current: value })

function baseState(messages: ClientSessionState['messages']): ClientSessionState {
  return {
    storedSessionId: 'sess-1',
    messages,
    branch: '',
    cwd: '',
    busy: false
  } as ClientSessionState
}

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('usePromptActions /undo prefill handling', () => {
  it('re-syncs the visible transcript to the server active history and prefills the composer', async () => {
    // After /undo the server returns the truncated active history (the undone
    // user+assistant exchange is gone).
    const activeMessages: SessionMessage[] = [{ role: 'user', content: 'first turn that survives' }]

    vi.mocked(getSessionMessages).mockResolvedValue({ messages: activeMessages, session_id: 'sess-1' })

    const requestGateway = vi.fn(async (method: string) => {
      if (method === 'slash.exec') {
        // /undo is a pending-input command — slash.exec rejects it so the
        // client falls through to command.dispatch.
        throw new Error('pending-input command')
      }

      if (method === 'command.dispatch') {
        return { type: 'prefill', message: 'the undone message', notice: '↶ Undid 1 turn (2 messages).' }
      }

      throw new Error(`unexpected method ${method}`)
    })

    // Stateful reducer so the sequential updateSessionState calls compose,
    // letting us assert the *final* transcript — this is what catches a
    // notice that gets clobbered by the history replace (ordering bug).
    let sessionState = baseState([
      { id: 'u1', role: 'user', parts: [textPart('first turn that survives')] },
      { id: 'a1', role: 'assistant', parts: [textPart('reply 1')] },
      { id: 'u2', role: 'user', parts: [textPart('the undone message')] },
      { id: 'a2', role: 'assistant', parts: [textPart('reply that gets undone')] }
    ])

    const updateSessionState = vi.fn((_sessionId: string, updater: (state: ClientSessionState) => ClientSessionState) => {
      sessionState = updater(sessionState)

      return sessionState
    })

    const { result } = renderHook(() =>
      usePromptActions({
        activeSessionId: 'sess-1',
        activeSessionIdRef: ref<string | null>('sess-1'),
        busyRef: ref(false),
        branchCurrentSession: vi.fn(async () => true),
        createBackendSessionForSend: vi.fn(async () => 'sess-1'),
        handleSkinCommand: vi.fn(() => ''),
        requestGateway: requestGateway as never,
        resumeStoredSession: vi.fn(async () => undefined),
        selectedStoredSessionIdRef: ref<string | null>('sess-1'),
        startFreshSessionDraft: vi.fn(),
        sttEnabled: false,
        refreshSessions: vi.fn(async () => undefined),
        updateSessionState
      })
    )

    await result.current.submitText('/undo')

    expect(getSessionMessages).toHaveBeenCalledWith('sess-1')

    // The undone exchange (u2/a2) is gone — the transcript now holds the
    // server's active history (one surviving user turn) plus the confirmation
    // notice. The notice surviving proves the replace ran *before* it.
    const texts = sessionState.messages.map(chatMessageText)

    expect(texts.some(t => t.includes('the undone message') && !t.includes('first turn'))).toBe(false)
    expect(texts.some(t => t.includes('first turn that survives'))).toBe(true)

    const last = sessionState.messages.at(-1)

    expect(last?.role).toBe('system')
    expect(chatMessageText(last!)).toContain('Undid 1 turn (2 messages).')

    expect(requestComposerInsert).toHaveBeenCalledWith('the undone message', { target: 'main' })
  })
})
