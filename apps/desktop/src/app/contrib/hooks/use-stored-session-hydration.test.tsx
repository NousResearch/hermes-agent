import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { createClientSessionState } from '@/lib/chat-runtime'
import { $activeGatewayProfile } from '@/store/profile'
import { setSessions } from '@/store/session'
import { $todosBySession, clearSessionTodos, setSessionTodos } from '@/store/todos'

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
  clearSessionTodos('runtime-shared')
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

  it('drops a delayed same-id response after its profile owner switches', async () => {
    let resolveRequest: ((value: ReturnType<typeof transcript>) => void) | undefined

    const request = new Promise<ReturnType<typeof transcript>>(resolve => {
      resolveRequest = resolve
    })

    vi.mocked(getLatestSessionMessages).mockReturnValue(request as never)

    const activeSessionIdRef = { current: 'runtime-shared' as string | null }
    const selectedStoredSessionIdRef = { current: sharedId as string | null }
    const selectedStoredSessionProfileRef = { current: 'default' as string | null }
    const initialState = createClientSessionState(sharedId)
    initialState.profile = 'meta'
    initialState.messages = [
      {
        id: 'meta-existing',
        role: 'assistant',
        parts: [{ type: 'text', text: 'meta existing answer' }]
      }
    ]
    const states = new Map([['runtime-shared', initialState]])

    const updateSessionState = vi.fn((runtimeId, updater) => {
      const previous = states.get(runtimeId)!
      const next = updater(previous)
      states.set(runtimeId, next)

      return next
    })

    const metaTodos = [{ content: 'meta task', id: 'meta-todo', status: 'in_progress' as const }]
    setSessionTodos('runtime-shared', metaTodos)

    const { result } = renderHook(() =>
      useStoredSessionHydration({
        activeSessionIdRef,
        selectedStoredSessionIdRef,
        selectedStoredSessionProfileRef,
        updateSessionState
      })
    )

    const hydration = result.current()

    // Same durable and runtime ids, different profile owner: neither id alone
    // can distinguish the request that is now stale.
    selectedStoredSessionProfileRef.current = 'meta'
    $activeGatewayProfile.set('meta')

    await act(async () => {
      resolveRequest?.(transcript('stale default answer'))
      await hydration
    })

    expect(getLatestSessionMessages).toHaveBeenCalledWith(sharedId, 'default')
    expect(updateSessionState).not.toHaveBeenCalled()
    expect(states.get('runtime-shared')?.messages.at(-1)?.parts[0]).toMatchObject({ text: 'meta existing answer' })
    expect($todosBySession.get()['runtime-shared']).toEqual(metaTodos)
  })

  it('revalidates ownership again before publishing hydrated todos', async () => {
    vi.mocked(getLatestSessionMessages).mockResolvedValue(transcript('default answer') as never)

    const activeSessionIdRef = { current: 'runtime-shared' as string | null }
    const selectedStoredSessionIdRef = { current: sharedId as string | null }
    const selectedStoredSessionProfileRef = { current: 'default' as string | null }
    const state = createClientSessionState(sharedId)
    const metaTodos = [{ content: 'meta task', id: 'meta-todo', status: 'in_progress' as const }]

    const updateSessionState = vi.fn((_runtimeId, updater) => {
      const next = updater(state)

      // Model a synchronous profile-switch subscriber firing while the message
      // publication notifies stores. The sibling todo publication must recheck.
      selectedStoredSessionProfileRef.current = 'meta'
      $activeGatewayProfile.set('meta')
      setSessionTodos('runtime-shared', metaTodos)

      return next
    })

    const { result } = renderHook(() =>
      useStoredSessionHydration({
        activeSessionIdRef,
        selectedStoredSessionIdRef,
        selectedStoredSessionProfileRef,
        updateSessionState
      })
    )

    await act(async () => result.current())

    expect(updateSessionState).toHaveBeenCalledOnce()
    expect($todosBySession.get()['runtime-shared']).toEqual(metaTodos)
  })
})
