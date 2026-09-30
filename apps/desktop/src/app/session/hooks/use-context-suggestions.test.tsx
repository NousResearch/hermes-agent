import { act, cleanup, renderHook, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $composerContextSuggestions } from '@/store/composer-context-suggestions'
import { $contextSuggestions, setCurrentCwdTransient } from '@/store/session'

import { useContextSuggestions } from './use-context-suggestions'

afterEach(cleanup)

// The hook has no gateway in these tests — every requestGateway mock stands in
// for the `complete.path` RPC the real app sends.
function renderContextHook() {
  const requestGateway = vi.fn(
    () =>
      Promise.resolve({
        items: [{ text: '@file:src/index.ts', display: 'src/index.ts' }]
      }) as Promise<{ items?: { text: string; display: string }[] }>
  )

  const activeSessionIdRef = { current: 'session-a' }

  const hook = renderHook(() =>
    useContextSuggestions({
      activeSessionId: 'session-a',
      activeSessionIdRef,
      currentCwd: '/repo',
      gatewayState: 'open',
      requestGateway: requestGateway as unknown as <T = unknown>() => Promise<T>
    })
  )

  return { hook, requestGateway }
}

describe('useContextSuggestions', () => {
  beforeEach(() => {
    $composerContextSuggestions.set(true)
    $contextSuggestions.set([])
    setCurrentCwdTransient('/repo')
  })

  it('prefetches context suggestions while the setting is on', async () => {
    const { requestGateway } = renderContextHook()

    await waitFor(() => {
      expect($contextSuggestions.get()).toHaveLength(1)
    })

    expect(requestGateway).toHaveBeenCalledTimes(1)
  })

  it('stops prefetching and clears published suggestions when the setting is off', async () => {
    const { requestGateway } = renderContextHook()

    await waitFor(() => {
      expect($contextSuggestions.get()).toHaveLength(1)
    })

    await act(async () => {
      $composerContextSuggestions.set(false)
    })

    expect($contextSuggestions.get()).toEqual([])
    expect(requestGateway).toHaveBeenCalledTimes(1)

    // Toggling back on refetches — the flip is immediate, not next-launch.
    await act(async () => {
      $composerContextSuggestions.set(true)
    })

    await waitFor(() => {
      expect(requestGateway).toHaveBeenCalledTimes(2)
    })
  })
})
