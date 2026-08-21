import { cleanup, renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { createClientSessionState } from '@/lib/chat-runtime'
import { $activeGatewayProfile } from '@/store/profile'
import { setSessions } from '@/store/session'

import { useStoredSessionHydration } from './use-stored-session-hydration'

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal()),
  getLatestSessionMessages: vi.fn()
}))

const { getLatestSessionMessages } = await import('@/hermes')

const sharedId = 'shared-session'

function transcript(answer: string) {
  return {
    messages: [
      { content: 'question', role: 'user', timestamp: 1 },
      { content: answer, role: 'assistant', timestamp: 2 }
    ],
    session_id: sharedId
  }
}

afterEach(() => {
  cleanup()
  setSessions([])
  $activeGatewayProfile.set('default')
  vi.clearAllMocks()
})

describe('useStoredSessionHydration profile ownership', () => {
  it('hydrates the selected profile when duplicate stored ids coexist', async () => {
    setSessions([
      { id: sharedId, profile: 'default', title: 'Default duplicate' } as never,
      { id: sharedId, profile: 'meta', title: 'Selected owner' } as never
    ])
    $activeGatewayProfile.set('default')

    const activeSessionIdRef = { current: 'runtime-meta' as string | null }
    const selectedStoredSessionIdRef = { current: sharedId as string | null }
    const selectedStoredSessionProfileRef = { current: 'meta' as string | null }
    const states = new Map([['runtime-meta', createClientSessionState(sharedId)]])

    const updateSessionState = vi.fn((runtimeId, updater, storedSessionId) => {
      const previous = states.get(runtimeId) ?? createClientSessionState(storedSessionId)
      const next = updater(previous)
      states.set(runtimeId, next)

      return next
    })

    vi.mocked(getLatestSessionMessages).mockImplementation(async (_storedId, profile) =>
      profile === 'meta' ? (transcript('meta answer') as never) : (transcript('wrong default answer') as never)
    )

    const { result } = renderHook(() =>
      useStoredSessionHydration({
        activeSessionIdRef,
        selectedStoredSessionIdRef,
        selectedStoredSessionProfileRef,
        updateSessionState
      })
    )

    await result.current()

    expect(getLatestSessionMessages).toHaveBeenCalledWith(sharedId, 'meta')
    expect(updateSessionState).toHaveBeenCalledWith('runtime-meta', expect.any(Function), sharedId)
    expect(states.get('runtime-meta')?.messages.at(-1)?.parts[0]).toMatchObject({ text: 'meta answer' })
  })
})
