import { act, cleanup, render, renderHook, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import type { PropsWithChildren } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { AccountUsageResponse, AccountUsageSnapshot } from '@/types/hermes'

import {
  AccountUsageMethodUnavailableError,
  CODEX_USAGE_BACKOFF_MS,
  CODEX_USAGE_REFRESH_MS,
  type CodexAccountUsageOptions,
  codexAccountUsageQueryKey,
  codexUsageRefetchInterval,
  type GatewayRequester,
  useCodexAccountUsage
} from './use-codex-account-usage'

afterEach(cleanup)

const snapshot = (usedPercent: number): AccountUsageSnapshot => ({
  available: true,
  details: [],
  fetched_at: '2026-07-16T01:02:03+00:00',
  plan: 'Plus',
  provider: 'openai-codex',
  source: 'usage_api',
  title: 'Account limits',
  unavailable_reason: null,
  windows: [{ label: 'Session', used_percent: usedPercent }]
})

function queryWrapper() {
  const client = new QueryClient({
    defaultOptions: { queries: { gcTime: Number.POSITIVE_INFINITY, retry: false } }
  })

  return function Wrapper({ children }: PropsWithChildren) {
    return <QueryClientProvider client={client}>{children}</QueryClientProvider>
  }
}

function options(requestGateway: GatewayRequester, over: Partial<CodexAccountUsageOptions> = {}) {
  return {
    connectionScope: 'local:',
    gatewayState: 'open',
    profile: 'default',
    provider: 'openai-codex',
    requestGateway,
    sessionId: 'runtime-1',
    ...over
  } satisfies CodexAccountUsageOptions
}

describe('useCodexAccountUsage', () => {
  it('does not request usage for a non-Codex provider or missing runtime session', async () => {
    const requestGateway = vi.fn() as unknown as GatewayRequester
    const { rerender } = renderHook(props => useCodexAccountUsage(props), {
      initialProps: options(requestGateway, { provider: 'anthropic' }),
      wrapper: queryWrapper()
    })

    await act(async () => undefined)
    expect(requestGateway).not.toHaveBeenCalled()

    rerender(options(requestGateway, { sessionId: null }))
    await act(async () => undefined)
    expect(requestGateway).not.toHaveBeenCalled()
  })

  it('keys data by connection, profile, runtime session, and provider without carrying stale data', async () => {
    const requestGateway = vi
      .fn<() => Promise<AccountUsageResponse>>()
      .mockResolvedValueOnce({ account_usage: snapshot(10) })
      .mockResolvedValueOnce({ account_usage: snapshot(80) })
    const first = options(requestGateway as never)
    const { result, rerender } = renderHook(props => useCodexAccountUsage(props), {
      initialProps: first,
      wrapper: queryWrapper()
    })

    await waitFor(() => expect(result.current.snapshot?.windows[0].used_percent).toBe(10))

    const second = options(requestGateway as never, {
      connectionScope: 'remote:https://backend-b',
      profile: 'work',
      sessionId: 'runtime-2'
    })
    expect(codexAccountUsageQueryKey(first)).not.toEqual(codexAccountUsageQueryKey(second))
    rerender(second)
    expect(result.current.snapshot).toBeNull()

    await waitFor(() => expect(result.current.snapshot?.windows[0].used_percent).toBe(80))
    expect(requestGateway).toHaveBeenCalledTimes(2)
  })

  it('deduplicates two consumers of the same account-usage scope', async () => {
    let resolveRequest: ((value: AccountUsageResponse) => void) | undefined
    const requestGateway = vi.fn(
      () =>
        new Promise<AccountUsageResponse>(resolve => {
          resolveRequest = resolve
        })
    ) as unknown as GatewayRequester
    const sharedOptions = options(requestGateway)

    function Pair() {
      useCodexAccountUsage(sharedOptions)
      useCodexAccountUsage(sharedOptions)
      return null
    }

    render(<Pair />, { wrapper: queryWrapper() })
    await waitFor(() => expect(requestGateway).toHaveBeenCalledTimes(1))

    await act(async () => resolveRequest?.({ account_usage: snapshot(25) }))
    expect(requestGateway).toHaveBeenCalledTimes(1)
  })

  it('recognizes an older backend and disables its polling interval', async () => {
    const error = Object.assign(new Error('Method not found: session.account_usage'), { code: -32601 })
    const requestGateway = vi.fn(async () => Promise.reject(error)) as unknown as GatewayRequester
    const { result } = renderHook(() => useCodexAccountUsage(options(requestGateway)), {
      wrapper: queryWrapper()
    })

    await waitFor(() => expect(result.current.methodUnavailable).toBe(true))
    expect(
      codexUsageRefetchInterval({
        state: { error: new AccountUsageMethodUnavailableError(), fetchFailureCount: 1 }
      })
    ).toBe(false)
  })

  it('backs polling off after repeated failures and restores the normal cadence otherwise', () => {
    expect(codexUsageRefetchInterval({ state: { error: new Error('offline'), fetchFailureCount: 2 } })).toBe(
      CODEX_USAGE_REFRESH_MS
    )
    expect(codexUsageRefetchInterval({ state: { error: new Error('offline'), fetchFailureCount: 3 } })).toBe(
      CODEX_USAGE_BACKOFF_MS
    )
    expect(codexUsageRefetchInterval({ state: { error: null, fetchFailureCount: 0 } })).toBe(CODEX_USAGE_REFRESH_MS)
  })
})
