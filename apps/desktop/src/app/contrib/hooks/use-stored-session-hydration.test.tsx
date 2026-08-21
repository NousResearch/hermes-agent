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

type TestSessionState = ReturnType<typeof createClientSessionState>

function transcript(answer: string) {
  return {
    messages: [
      { content: 'question', role: 'user', timestamp: 1 },
      { content: answer, role: 'assistant', timestamp: 2 }
    ],
    session_id: sharedId
  }
}

function ownerTools(states: Map<string, TestSessionState>) {
  const sessionStateHasOwner = vi.fn((runtimeId: string, owner: { profile: string; storedSessionId: string }) => {
    const state = states.get(runtimeId)

    return state?.profile === owner.profile && state.storedSessionId === owner.storedSessionId
  })

  const updateOwnedSessionState = vi.fn(
    (
      runtimeId: string,
      owner: { profile: string; storedSessionId: string },
      updater: (state: TestSessionState) => TestSessionState
    ) => {
      const previous = states.get(runtimeId)

      if (previous?.profile !== owner.profile || previous.storedSessionId !== owner.storedSessionId) {
        return false
      }

      states.set(runtimeId, updater(previous))

      return sessionStateHasOwner(runtimeId, owner)
    }
  )

  return { sessionStateHasOwner, updateOwnedSessionState }
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
    const initialState = createClientSessionState(sharedId)
    initialState.profile = 'meta'
    const states = new Map([['runtime-meta', initialState]])
    const { sessionStateHasOwner, updateOwnedSessionState } = ownerTools(states)

    vi.mocked(getLatestSessionMessages).mockImplementation(async (_storedId, profile) =>
      profile === 'meta' ? (transcript('meta answer') as never) : (transcript('wrong default answer') as never)
    )

    const { result } = renderHook(() =>
      useStoredSessionHydration({
        activeSessionIdRef,
        selectedStoredSessionIdRef,
        selectedStoredSessionProfileRef,
        sessionStateHasOwner,
        updateOwnedSessionState
      })
    )

    await result.current()

    expect(getLatestSessionMessages).toHaveBeenCalledWith(sharedId, 'meta')
    expect(updateOwnedSessionState).toHaveBeenCalledWith(
      'runtime-meta',
      { profile: 'meta', storedSessionId: sharedId },
      expect.any(Function)
    )
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
    const { sessionStateHasOwner, updateOwnedSessionState } = ownerTools(states)
    const metaTodos = [{ content: 'meta task', id: 'meta-todo', status: 'in_progress' as const }]
    setSessionTodos('runtime-shared', metaTodos)

    const { result } = renderHook(() =>
      useStoredSessionHydration({
        activeSessionIdRef,
        selectedStoredSessionIdRef,
        selectedStoredSessionProfileRef,
        sessionStateHasOwner,
        updateOwnedSessionState
      })
    )

    const hydration = result.current()
    selectedStoredSessionProfileRef.current = 'meta'
    $activeGatewayProfile.set('meta')

    await act(async () => {
      resolveRequest?.(transcript('stale default answer'))
      await hydration
    })

    expect(getLatestSessionMessages).toHaveBeenCalledWith(sharedId, 'default')
    expect(updateOwnedSessionState).not.toHaveBeenCalled()
    expect(states.get('runtime-shared')?.messages.at(-1)?.parts[0]).toMatchObject({ text: 'meta existing answer' })
    expect($todosBySession.get()['runtime-shared']).toEqual(metaTodos)
  })

  it('drops a delayed ABA response when the cache runtime is reclaimed by another profile', async () => {
    let resolveRequest: ((value: ReturnType<typeof transcript>) => void) | undefined

    const request = new Promise<ReturnType<typeof transcript>>(resolve => {
      resolveRequest = resolve
    })

    vi.mocked(getLatestSessionMessages).mockReturnValue(request as never)

    const activeSessionIdRef = { current: 'runtime-shared' as string | null }
    const selectedStoredSessionIdRef = { current: sharedId as string | null }
    const selectedStoredSessionProfileRef = { current: 'default' as string | null }
    const initialState = createClientSessionState(sharedId)
    initialState.profile = 'default'
    const states = new Map([['runtime-shared', initialState]])
    const { sessionStateHasOwner, updateOwnedSessionState } = ownerTools(states)

    const { result } = renderHook(() =>
      useStoredSessionHydration({
        activeSessionIdRef,
        selectedStoredSessionIdRef,
        selectedStoredSessionProfileRef,
        sessionStateHasOwner,
        updateOwnedSessionState
      })
    )

    const hydration = result.current()
    const reclaimedState = createClientSessionState(sharedId)
    reclaimedState.profile = 'meta'
    reclaimedState.messages = [
      {
        id: 'meta-existing',
        role: 'assistant',
        parts: [{ type: 'text', text: 'meta existing answer' }]
      }
    ]
    states.set('runtime-shared', reclaimedState)
    const metaTodos = [{ content: 'meta task', id: 'meta-todo', status: 'in_progress' as const }]
    setSessionTodos('runtime-shared', metaTodos)

    // Selection refs ABA back to their captured values, but the cache slot is
    // now owned by another profile with the same runtime and stored ids.
    await act(async () => {
      resolveRequest?.(transcript('stale default answer'))
      await hydration
    })

    expect(states.get('runtime-shared')).toBe(reclaimedState)
    expect(states.get('runtime-shared')?.messages.at(-1)?.parts[0]).toMatchObject({ text: 'meta existing answer' })
    expect($todosBySession.get()['runtime-shared']).toEqual(metaTodos)
  })

  it('revalidates cache ownership again before publishing hydrated todos', async () => {
    vi.mocked(getLatestSessionMessages).mockResolvedValue(transcript('default answer') as never)

    const activeSessionIdRef = { current: 'runtime-shared' as string | null }
    const selectedStoredSessionIdRef = { current: sharedId as string | null }
    const selectedStoredSessionProfileRef = { current: 'default' as string | null }
    const state = createClientSessionState(sharedId)
    state.profile = 'default'
    const metaTodos = [{ content: 'meta task', id: 'meta-todo', status: 'in_progress' as const }]
    const sessionStateHasOwner = vi.fn(() => false)

    const updateOwnedSessionState = vi.fn((_runtimeId, _owner, updater) => {
      const next = updater(state)

      // Model a synchronous cache-reclaim subscriber firing while the message
      // publication notifies stores. The sibling todo publication must recheck.
      void next
      setSessionTodos('runtime-shared', metaTodos)

      return true
    })

    const { result } = renderHook(() =>
      useStoredSessionHydration({
        activeSessionIdRef,
        selectedStoredSessionIdRef,
        selectedStoredSessionProfileRef,
        sessionStateHasOwner,
        updateOwnedSessionState
      })
    )

    await act(async () => result.current())

    expect(updateOwnedSessionState).toHaveBeenCalledOnce()
    expect(sessionStateHasOwner).toHaveBeenCalledOnce()
    expect($todosBySession.get()['runtime-shared']).toEqual(metaTodos)
  })
})
