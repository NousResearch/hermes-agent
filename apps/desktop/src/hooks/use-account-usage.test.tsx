import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, render, renderHook, waitFor } from '@testing-library/react'
import type { PropsWithChildren } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import * as sessionRequestRouter from '@/store/session-request-router'
import type { AccountUsageResponse, AccountUsageSnapshot } from '@/types/hermes'

import {
  ACCOUNT_USAGE_BACKOFF_MS,
  ACCOUNT_USAGE_REFRESH_MS,
  AccountUsageMethodUnavailableError,
  type AccountUsageOptions,
  AccountUsageUnsupportedError,
  accountUsageCacheIdentity,
  accountUsageOwnerIsResolved,
  accountUsageQueryKey,
  accountUsageRefetchInterval,
  createAccountUsageRequester,
  type GatewayRequester,
  useAccountUsage
} from './use-account-usage'

afterEach(() => {
  cleanup()
  vi.useRealTimers()
  vi.restoreAllMocks()
})

const snapshot = (usedPercent: number, over: Partial<AccountUsageSnapshot> = {}): AccountUsageSnapshot => ({
  available: true,
  details: [],
  fetched_at: '2026-07-16T01:02:03+00:00',
  plan: 'Plus',
  provider: 'openai-codex',
  source: 'usage_api',
  title: 'Account limits',
  unavailable_reason: null,
  windows: [{ label: 'Session', used_percent: usedPercent }],
  ...over
})

function queryWrapper() {
  const client = new QueryClient({
    defaultOptions: { queries: { gcTime: Number.POSITIVE_INFINITY, retry: false } }
  })

  return function Wrapper({ children }: PropsWithChildren) {
    return <QueryClientProvider client={client}>{children}</QueryClientProvider>
  }
}

function options(requestGateway: GatewayRequester, over: Partial<AccountUsageOptions> = {}) {
  return {
    connectionScope: 'local:',
    gatewayState: 'open',
    owner: { connectionId: 'local', profile: 'default' },
    profile: 'default',
    provider: 'openai-codex',
    requestGateway,
    sessionId: 'runtime-1',
    ...over
  } satisfies AccountUsageOptions
}

describe('useAccountUsage', () => {
  it('keys usage by connection, profile, provider, and session without a hardcoded provider segment', () => {
    const key = accountUsageQueryKey({
      connectionScope: 'local:',
      profile: 'default',
      provider: 'openrouter',
      sessionId: 'runtime-1'
    })

    expect(key).toEqual(['account-usage', 'local:', 'default', 'openrouter', 'runtime-1'])
    expect(key.filter(part => part === 'openai-codex')).toHaveLength(0)
    expect(
      accountUsageQueryKey({
        connectionScope: 'local:',
        profile: 'default',
        provider: 'openai-codex',
        sessionId: 'runtime-1'
      })
    ).toEqual(['account-usage', 'local:', 'default', 'openai-codex', 'runtime-1'])
  })

  it('keys owner routes by connectionId and keeps ambient scope when no owner is known', () => {
    const ambient = { connectionScope: 'local:http://127.0.0.1:8642', profile: 'default' }

    expect(
      accountUsageCacheIdentity(
        { connectionId: 'conn-ssh-1', profile: 'work', targetProfile: 'remote-work' },
        ambient
      )
    ).toEqual({ connectionScope: 'conn-ssh-1', profile: 'remote-work' })
    expect(accountUsageCacheIdentity({ connectionId: 'conn-local', profile: 'coder' }, ambient)).toEqual({
      connectionScope: 'conn-local',
      profile: 'coder'
    })
    expect(accountUsageCacheIdentity('work', ambient)).toEqual({
      connectionScope: ambient.connectionScope,
      profile: 'work'
    })
    expect(accountUsageCacheIdentity(null, ambient)).toEqual(ambient)
    expect(accountUsageCacheIdentity(undefined, ambient)).toEqual(ambient)
    expect(accountUsageOwnerIsResolved(null)).toBe(false)
    expect(accountUsageOwnerIsResolved(undefined)).toBe(false)
    expect(accountUsageOwnerIsResolved('')).toBe(false)
    expect(accountUsageOwnerIsResolved({ connectionId: '', profile: 'work' })).toBe(false)
    expect(accountUsageOwnerIsResolved('work')).toBe(true)
    expect(accountUsageOwnerIsResolved({ connectionId: 'conn-1', profile: 'work' })).toBe(true)
  })

  it('does not request usage without an open gateway, a runtime session, or a provider', async () => {
    const requestGateway = vi.fn() as unknown as GatewayRequester
    const { rerender } = renderHook(props => useAccountUsage(props), {
      initialProps: options(requestGateway, { gatewayState: 'closed' }),
      wrapper: queryWrapper()
    })

    await act(async () => undefined)
    expect(requestGateway).not.toHaveBeenCalled()

    rerender(options(requestGateway, { sessionId: null }))
    await act(async () => undefined)
    expect(requestGateway).not.toHaveBeenCalled()

    rerender(options(requestGateway, { provider: '   ' }))
    await act(async () => undefined)
    expect(requestGateway).not.toHaveBeenCalled()

    rerender(options(requestGateway, { owner: null }))
    await act(async () => undefined)
    expect(requestGateway).not.toHaveBeenCalled()

    rerender(options(requestGateway, { owner: undefined }))
    await act(async () => undefined)
    expect(requestGateway).not.toHaveBeenCalled()
  })

  it('does not route ambient when a session owner is unresolved', async () => {
    const ambient = vi.fn()
    const spy = vi.spyOn(sessionRequestRouter, 'requestForSessionProfile').mockResolvedValue({ ok: true })
    const requester = createAccountUsageRequester(null, ambient as unknown as GatewayRequester)

    await expect(requester('session.account_usage', { session_id: 'runtime-1' }, 45_000)).rejects.toThrow(
      'Account usage owner is unresolved'
    )
    expect(spy).not.toHaveBeenCalled()
    expect(ambient).not.toHaveBeenCalled()
    spy.mockRestore()
  })

  it('routes a known owner through requestForSessionProfile, never null', async () => {
    const ambient = vi.fn()
    const owner = { connectionId: 'conn-ssh-1', profile: 'work' }
    const spy = vi.spyOn(sessionRequestRouter, 'requestForSessionProfile').mockResolvedValue({ ok: true })
    const requester = createAccountUsageRequester(owner, ambient as unknown as GatewayRequester)

    await requester('session.account_usage', { session_id: 'runtime-1' }, 45_000)

    expect(spy).toHaveBeenCalledTimes(1)
    expect(spy.mock.calls[0]?.[0]).toEqual(owner)
    expect(spy.mock.calls[0]?.[0]).not.toBeNull()
    expect(ambient).not.toHaveBeenCalled()
    spy.mockRestore()
  })

  it('requests usage for any named provider, including non-Codex ones', async () => {
    const requestGateway = vi
      .fn<() => Promise<AccountUsageResponse>>()
      .mockResolvedValue({ account_usage: snapshot(10, { provider: 'anthropic' }), status: 'ok' })
    const { result } = renderHook(() => useAccountUsage(options(requestGateway as never, { provider: 'anthropic' })), {
      wrapper: queryWrapper()
    })

    await waitFor(() => expect(result.current.snapshot?.windows[0].used_percent).toBe(10))
    expect(requestGateway).toHaveBeenCalledWith(
      'session.account_usage',
      { session_id: 'runtime-1' },
      45_000,
      expect.any(AbortSignal)
    )
  })

  it('maps status ok to snapshot data', async () => {
    const requestGateway = vi
      .fn<() => Promise<AccountUsageResponse>>()
      .mockResolvedValue({ account_usage: snapshot(42), status: 'ok' })
    const { result } = renderHook(() => useAccountUsage(options(requestGateway as never)), {
      wrapper: queryWrapper()
    })

    await waitFor(() => expect(result.current.snapshot?.windows[0].used_percent).toBe(42))
    expect(result.current.error).toBe(false)
    expect(result.current.unsupported).toBe(false)
    expect(result.current.methodUnavailable).toBe(false)
  })

  it('forwards structured localization fields from the wire snapshot', async () => {
    const structured = snapshot(10, {
      details: ['Credits balance: $31.44'],
      details_structured: true,
      rows: [{ args: { currency: 'USD', value: 31.44 }, key: 'credits_balance' }],
      windows: [
        {
          label: 'API key quota',
          label_key: 'api_key_quota',
          limit: 14.25,
          limit_remaining: 4.09,
          reset_interval: 'weekly',
          used_percent: 10
        }
      ]
    })
    const requestGateway = vi
      .fn<() => Promise<AccountUsageResponse>>()
      .mockResolvedValue({ account_usage: structured, status: 'ok' })
    const { result } = renderHook(() => useAccountUsage(options(requestGateway as never, { provider: 'openrouter' })), {
      wrapper: queryWrapper()
    })

    await waitFor(() => expect(result.current.snapshot?.details_structured).toBe(true))
    expect(result.current.snapshot?.rows).toEqual([{ args: { currency: 'USD', value: 31.44 }, key: 'credits_balance' }])
    expect(result.current.snapshot?.windows[0]).toMatchObject({
      label_key: 'api_key_quota',
      limit: 14.25,
      limit_remaining: 4.09,
      reset_interval: 'weekly'
    })
  })

  it('treats unsupported as a terminal stop', async () => {
    const requestGateway = vi
      .fn<() => Promise<AccountUsageResponse>>()
      .mockResolvedValue({ account_usage: null, status: 'unsupported' })
    const { result } = renderHook(() => useAccountUsage(options(requestGateway as never, { provider: 'openai' })), {
      wrapper: queryWrapper()
    })

    await waitFor(() => expect(result.current.unsupported).toBe(true))
    expect(result.current.snapshot).toBeNull()
    expect(
      accountUsageRefetchInterval(
        { state: { error: new AccountUsageUnsupportedError(), fetchFailureCount: 1 } },
        1
      )
    ).toBe(false)
  })

  it('treats unavailable as retryable', async () => {
    const requestGateway = vi
      .fn<() => Promise<AccountUsageResponse>>()
      .mockResolvedValue({ account_usage: null, reason: 'no_live_agent', status: 'unavailable' })
    const { result } = renderHook(() => useAccountUsage(options(requestGateway as never)), {
      wrapper: queryWrapper()
    })

    await waitFor(() => expect(result.current.error).toBe(true))
    expect(result.current.unsupported).toBe(false)
    expect(result.current.methodUnavailable).toBe(false)
    expect(
      accountUsageRefetchInterval({ state: { error: new Error('unavailable'), fetchFailureCount: 1 } }, 1)
    ).toBe(ACCOUNT_USAGE_REFRESH_MS)
  })

  it('treats a missing-method backend as a terminal stop', async () => {
    const error = Object.assign(new Error('Method not found: session.account_usage'), { code: -32601 })
    const requestGateway = vi.fn(async () => Promise.reject(error)) as unknown as GatewayRequester
    const { result } = renderHook(() => useAccountUsage(options(requestGateway)), {
      wrapper: queryWrapper()
    })

    await waitFor(() => expect(result.current.methodUnavailable).toBe(true))
    expect(
      accountUsageRefetchInterval(
        { state: { error: new AccountUsageMethodUnavailableError(), fetchFailureCount: 1 } },
        1
      )
    ).toBe(false)
  })

  it('accepts a legacy response that has account_usage but no status', async () => {
    const requestGateway = vi
      .fn<() => Promise<AccountUsageResponse>>()
      .mockResolvedValue({ account_usage: snapshot(15) })
    const { result } = renderHook(() => useAccountUsage(options(requestGateway as never)), {
      wrapper: queryWrapper()
    })

    await waitFor(() => expect(result.current.snapshot?.windows[0].used_percent).toBe(15))
    expect(result.current.error).toBe(false)
  })

  it('treats a legacy empty payload as retryable unavailable', async () => {
    const requestGateway = vi.fn<() => Promise<AccountUsageResponse>>().mockResolvedValue({ account_usage: null })
    const { result } = renderHook(() => useAccountUsage(options(requestGateway as never)), {
      wrapper: queryWrapper()
    })

    await waitFor(() => expect(result.current.error).toBe(true))
    expect(result.current.methodUnavailable).toBe(false)
    expect(result.current.unsupported).toBe(false)
  })

  it('hides a never-fetched session and restores that session’s own cached snapshot', async () => {
    const requestGateway = vi
      .fn<() => Promise<AccountUsageResponse>>()
      .mockResolvedValueOnce({ account_usage: snapshot(10), status: 'ok' })
      .mockImplementationOnce(() => new Promise<AccountUsageResponse>(() => undefined))
    const first = options(requestGateway as never)
    const { result, rerender } = renderHook(props => useAccountUsage(props), {
      initialProps: first,
      wrapper: queryWrapper()
    })

    await waitFor(() => expect(result.current.snapshot?.windows[0].used_percent).toBe(10))

    const second = options(requestGateway as never, { sessionId: 'runtime-2' })
    expect(accountUsageQueryKey(first)).not.toEqual(accountUsageQueryKey(second))
    rerender(second)
    expect(result.current.snapshot).toBeNull()

    rerender(first)
    expect(result.current.snapshot?.windows[0].used_percent).toBe(10)
    expect(requestGateway).toHaveBeenCalledTimes(2)
  })

  it('does not paint the previous account while a same-provider profile switch loads', async () => {
    let resolveSecond: ((value: AccountUsageResponse) => void) | undefined
    const requestGateway = vi
      .fn<() => Promise<AccountUsageResponse>>()
      .mockResolvedValueOnce({
        account_usage: snapshot(10, { credits_balance: 50, provider: 'openrouter' }),
        status: 'ok'
      })
      .mockImplementationOnce(
        () =>
          new Promise<AccountUsageResponse>(resolve => {
            resolveSecond = resolve
          })
      )
    const first = options(requestGateway as never, { provider: 'openrouter' })
    const { result, rerender } = renderHook(props => useAccountUsage(props), {
      initialProps: first,
      wrapper: queryWrapper()
    })

    await waitFor(() => expect(result.current.snapshot?.credits_balance).toBe(50))

    const second = options(requestGateway as never, { profile: 'work', provider: 'openrouter' })
    expect(accountUsageQueryKey(first)).not.toEqual(accountUsageQueryKey(second))
    rerender(second)
    expect(result.current.snapshot).toBeNull()

    await act(async () =>
      resolveSecond?.({
        account_usage: snapshot(80, { credits_balance: 12, provider: 'openrouter' }),
        status: 'ok'
      })
    )
    await waitFor(() => expect(result.current.snapshot?.credits_balance).toBe(12))
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
      useAccountUsage(sharedOptions)
      useAccountUsage(sharedOptions)
      return null
    }

    render(<Pair />, { wrapper: queryWrapper() })
    await waitFor(() => expect(requestGateway).toHaveBeenCalledTimes(1))

    await act(async () => resolveRequest?.({ account_usage: snapshot(25), status: 'ok' }))
    expect(requestGateway).toHaveBeenCalledTimes(1)
  })

  it('backs polling off after repeated failures and restores the normal cadence otherwise', () => {
    expect(accountUsageRefetchInterval({ state: { error: new Error('offline'), fetchFailureCount: 2 } }, 2)).toBe(
      ACCOUNT_USAGE_REFRESH_MS
    )
    expect(accountUsageRefetchInterval({ state: { error: new Error('offline'), fetchFailureCount: 3 } }, 3)).toBe(
      ACCOUNT_USAGE_BACKOFF_MS
    )
    expect(accountUsageRefetchInterval({ state: { error: null, fetchFailureCount: 0 } }, 0)).toBe(
      ACCOUNT_USAGE_REFRESH_MS
    )
  })

  it('backs off after three real consecutive refetch failures', async () => {
    vi.useFakeTimers()
    const requestGateway = vi.fn(async () => {
      throw new Error('offline')
    }) as unknown as GatewayRequester
    const client = new QueryClient({
      defaultOptions: { queries: { gcTime: Number.POSITIVE_INFINITY, refetchOnWindowFocus: false, retry: false } }
    })
    const first = options(requestGateway)
    const queryKey = accountUsageQueryKey(first)
    const { result } = renderHook(() => useAccountUsage(first), {
      wrapper({ children }: PropsWithChildren) {
        return <QueryClientProvider client={client}>{children}</QueryClientProvider>
      }
    })

    await act(async () => {
      await vi.advanceTimersByTimeAsync(0)
    })
    expect(result.current.error).toBe(true)
    expect(requestGateway).toHaveBeenCalledTimes(1)

    const fetchFailureCounts: number[] = [client.getQueryState(queryKey)?.fetchFailureCount ?? -1]

    await act(async () => {
      await vi.advanceTimersByTimeAsync(ACCOUNT_USAGE_REFRESH_MS)
    })
    expect(requestGateway).toHaveBeenCalledTimes(2)
    fetchFailureCounts.push(client.getQueryState(queryKey)?.fetchFailureCount ?? -1)

    await act(async () => {
      await vi.advanceTimersByTimeAsync(ACCOUNT_USAGE_REFRESH_MS)
    })
    expect(requestGateway).toHaveBeenCalledTimes(3)
    fetchFailureCounts.push(client.getQueryState(queryKey)?.fetchFailureCount ?? -1)

    await act(async () => {
      await vi.advanceTimersByTimeAsync(ACCOUNT_USAGE_REFRESH_MS)
    })
    expect(requestGateway).toHaveBeenCalledTimes(3)
    fetchFailureCounts.push(client.getQueryState(queryKey)?.fetchFailureCount ?? -1)

    await act(async () => {
      await vi.advanceTimersByTimeAsync(ACCOUNT_USAGE_BACKOFF_MS - ACCOUNT_USAGE_REFRESH_MS)
    })
    expect(requestGateway).toHaveBeenCalledTimes(4)

    // * retry: false keeps fetchFailureCount at 1 each cycle — it does not
    // * accumulate, which is why the hook owns a consecutive-failure map.
    expect(fetchFailureCounts).toEqual([1, 1, 1, 1])
  })

  it('resets backoff to the normal cadence after a successful fetch', async () => {
    vi.useFakeTimers()
    const requestGateway = vi.fn(async (): Promise<AccountUsageResponse> => {
      throw new Error('offline')
    })
    const client = new QueryClient({
      defaultOptions: { queries: { gcTime: Number.POSITIVE_INFINITY, refetchOnWindowFocus: false, retry: false } }
    })
    const first = options(requestGateway as unknown as GatewayRequester)
    const { result } = renderHook(() => useAccountUsage(first), {
      wrapper({ children }: PropsWithChildren) {
        return <QueryClientProvider client={client}>{children}</QueryClientProvider>
      }
    })

    await act(async () => {
      await vi.advanceTimersByTimeAsync(0)
    })
    await act(async () => {
      await vi.advanceTimersByTimeAsync(ACCOUNT_USAGE_REFRESH_MS)
    })
    await act(async () => {
      await vi.advanceTimersByTimeAsync(ACCOUNT_USAGE_REFRESH_MS)
    })
    expect(requestGateway).toHaveBeenCalledTimes(3)

    await act(async () => {
      await vi.advanceTimersByTimeAsync(ACCOUNT_USAGE_REFRESH_MS)
    })
    expect(requestGateway).toHaveBeenCalledTimes(3)

    requestGateway.mockImplementation(async () => ({ account_usage: snapshot(10), status: 'ok' }))
    let refreshed: Awaited<ReturnType<typeof result.current.refresh>> | undefined
    await act(async () => {
      const pending = result.current.refresh()
      await vi.advanceTimersByTimeAsync(0)
      refreshed = await pending
    })
    expect(requestGateway).toHaveBeenCalledTimes(4)
    expect(refreshed?.data?.windows[0].used_percent).toBe(10)
    expect(result.current.snapshot?.windows[0].used_percent).toBe(10)
    expect(result.current.error).toBe(false)

    await act(async () => {
      await vi.advanceTimersByTimeAsync(ACCOUNT_USAGE_REFRESH_MS)
    })
    expect(requestGateway).toHaveBeenCalledTimes(5)
  })
})
