import { act, cleanup, render } from '@testing-library/react'
import type { MutableRefObject } from 'react'
import { afterEach, describe, expect, it } from 'vitest'

import { clearAllSessionStates, getSessionStateGeneration } from '@/store/session-states'

import { useSessionStateCache } from './use-session-state-cache'

type Cache = ReturnType<typeof useSessionStateCache>

function Harness({ onReady }: { onReady: (cache: Cache) => void }) {
  const busyRef: MutableRefObject<boolean> = { current: false }

  const cache = useSessionStateCache({
    activeSessionId: 'runtime',
    busyRef,
    selectedStoredSessionId: 'stored',
    setAwaitingResponse: () => undefined,
    setBusy: () => undefined,
    setMessages: () => undefined
  })

  onReady(cache)

  return null
}

describe('useSessionStateCache — gateway reset', () => {
  afterEach(() => {
    cleanup()
    clearAllSessionStates()
  })

  it('drops hook-owned runtime and stored-id caches when all session states are cleared', () => {
    let cache!: Cache
    render(<Harness onReady={value => (cache = value)} />)

    act(() => {
      cache.updateSessionState('runtime', state => ({ ...state, busy: true }), 'stored')
    })

    expect(cache.sessionStateByRuntimeIdRef.current.has('runtime')).toBe(true)
    expect(cache.runtimeIdByStoredSessionIdRef.current.get('stored')).toBe('runtime')

    act(() => {
      clearAllSessionStates()
    })

    expect(cache.sessionStateByRuntimeIdRef.current.size).toBe(0)
    expect(cache.runtimeIdByStoredSessionIdRef.current.size).toBe(0)
  })

  it('rejects an async update captured before the reset', () => {
    let cache!: Cache
    render(<Harness onReady={value => (cache = value)} />)
    const staleGeneration = getSessionStateGeneration()

    act(() => {
      clearAllSessionStates()
      cache.updateSessionState('runtime', state => ({ ...state, busy: true }), 'stored', staleGeneration)
    })

    expect(cache.sessionStateByRuntimeIdRef.current.size).toBe(0)
    expect(cache.runtimeIdByStoredSessionIdRef.current.size).toBe(0)
  })
})
