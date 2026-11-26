import { act, renderHook } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { createClientSessionState } from '@/lib/chat-runtime'
import { $confirmRequest, settleConfirm } from '@/store/confirm'
import { $activeSessionId, $sessions } from '@/store/session'
import { $sessionStates, type SessionTileDelegate, setSessionTileDelegate } from '@/store/session-states'

import { useCompactConversation } from './use-compact-conversation'

afterEach(() => {
  settleConfirm(false)
  $activeSessionId.set(null)
  $sessions.set([])
  $sessionStates.set({})
  setSessionTileDelegate(null as unknown as SessionTileDelegate)
})

async function requestAndAnswer(answer: () => void) {
  const executeSlash = vi.fn(async () => {})
  setSessionTileDelegate({ executeSlash } as unknown as SessionTileDelegate)
  $activeSessionId.set('rt-a')

  const { result } = renderHook(() => useCompactConversation({ busy: false, sessionId: 'rt-a' }))

  act(() => result.current.onRequest())
  expect($confirmRequest.get()?.title).toBe('Compress this chat into a summary?')

  await act(async () => {
    answer()
    await Promise.resolve()
  })

  return executeSlash
}

it('runs the shared /compress pipeline on the chat it was opened on once confirmed', async () => {
  const executeSlash = await requestAndAnswer(() => settleConfirm(true))

  expect(executeSlash).toHaveBeenCalledWith('/compress', 'rt-a', { typed: false })
})

it('compacts nothing when the user switched chats while the confirmation was open', async () => {
  const executeSlash = await requestAndAnswer(() => {
    $activeSessionId.set('rt-b')
    settleConfirm(true)
  })

  expect(executeSlash).not.toHaveBeenCalled()
})

it('names the chat it will compact in the confirmation when the chat has a title', () => {
  setSessionTileDelegate({ executeSlash: vi.fn(async () => {}) } as unknown as SessionTileDelegate)
  $activeSessionId.set('rt-a')
  $sessionStates.set({ 'rt-a': { ...createClientSessionState(), storedSessionId: 'stored-a' } as never })
  $sessions.set([{ id: 'stored-a', title: 'Release notes' } as never])

  const { result } = renderHook(() => useCompactConversation({ busy: false, sessionId: 'rt-a' }))

  act(() => result.current.onRequest())
  expect($confirmRequest.get()?.title).toBe('Compress “Release notes” into a summary?')
})
