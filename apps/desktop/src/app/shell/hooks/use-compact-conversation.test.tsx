import { act, renderHook } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { $confirmRequest, settleConfirm } from '@/store/confirm'
import { $activeSessionId } from '@/store/session'
import { type SessionTileDelegate, setSessionTileDelegate } from '@/store/session-states'

import { useCompactConversation } from './use-compact-conversation'

afterEach(() => {
  settleConfirm(false)
  $activeSessionId.set(null)
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
