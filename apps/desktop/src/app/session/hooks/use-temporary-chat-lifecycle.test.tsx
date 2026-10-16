import { act, cleanup, renderHook } from '@testing-library/react'
import type { ReactNode } from 'react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, expect, test, vi } from 'vitest'

import { $selectedStoredSessionId } from '@/store/session'
import { $sessionStates } from '@/store/session-states'
import {
  $temporaryChatEndRequests,
  $temporaryChats,
  endTemporaryChat,
  isTemporarySession,
  registerTemporaryChat,
  requestNewTemporaryChat
} from '@/store/temporary-chat'

import { useTemporaryChatLifecycle } from './use-temporary-chat-lifecycle'

const removeSession = vi.fn(async (_id: string) => undefined)
const startFreshSessionDraft = vi.fn()
const TTL = 60_000

beforeEach(() => {
  vi.useFakeTimers()
  $selectedStoredSessionId.set(null)
  $temporaryChats.set({})
  $temporaryChatEndRequests.set([])
  $sessionStates.set({})
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
  vi.useRealTimers()
})

const wrapper = ({ children }: { children: ReactNode }) => <MemoryRouter>{children}</MemoryRouter>

function mount() {
  return renderHook(() => useTemporaryChatLifecycle(removeSession, startFreshSessionDraft, TTL), { wrapper })
}

test('leaving a temporary chat keeps it alive (it is sticky)', () => {
  registerTemporaryChat('temp', 'rt')
  act(() => $selectedStoredSessionId.set('temp'))
  mount()

  act(() => $selectedStoredSessionId.set('saved'))
  act(() => $selectedStoredSessionId.set(null))
  expect(removeSession).not.toHaveBeenCalled()
  expect(isTemporarySession('temp')).toBe(true)
})

test('an idle temporary chat off screen is deleted after the TTL', () => {
  registerTemporaryChat('temp', 'rt')
  mount()

  act(() => void vi.advanceTimersByTime(TTL - 20_000))
  expect(removeSession).not.toHaveBeenCalled()
  act(() => void vi.advanceTimersByTime(40_000))
  expect(removeSession).toHaveBeenCalledExactlyOnceWith('temp')
  expect(isTemporarySession('temp')).toBe(false)
})

test('a running reply is never cut off by the TTL', () => {
  registerTemporaryChat('temp', 'rt')
  act(() => $sessionStates.set({ rt: { busy: true, storedSessionId: 'temp' } as never }))
  mount()

  act(() => void vi.advanceTimersByTime(TTL * 3))
  expect(removeSession).not.toHaveBeenCalled()
})

test('the chat on screen never expires', () => {
  registerTemporaryChat('temp', 'rt')
  act(() => $selectedStoredSessionId.set('temp'))
  mount()

  act(() => void vi.advanceTimersByTime(TTL * 3))
  expect(removeSession).not.toHaveBeenCalled()
})

test('End deletes it right away', () => {
  registerTemporaryChat('temp', 'rt')
  mount()

  act(() => endTemporaryChat('temp'))
  expect(removeSession).toHaveBeenCalledExactlyOnceWith('temp')
  expect(isTemporarySession('temp')).toBe(false)
})

test('a new-temporary-chat request starts a temporary draft', () => {
  mount()
  expect(startFreshSessionDraft).not.toHaveBeenCalled()

  act(() => requestNewTemporaryChat())
  expect(startFreshSessionDraft).toHaveBeenCalledExactlyOnceWith({ temporary: true })
})
