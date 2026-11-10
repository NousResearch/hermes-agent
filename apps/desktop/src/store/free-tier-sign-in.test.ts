import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as HermesApi from '@/hermes'
import type { FreeTierRequester } from '@/store/free-tier'
import type { AppHandoffStatus } from '@/types/hermes'

const startOAuthLogin = vi.fn()
const pollOAuthSession = vi.fn()
const cancelOAuthSession = vi.fn(async (_id: string) => ({ ok: true }))
const getNousHandoffStatus = vi.fn()

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<typeof HermesApi>()),
  cancelOAuthSession: (id: string) => cancelOAuthSession(id),
  getNousHandoffStatus: (since: null | string) => getNousHandoffStatus(since),
  listOAuthProviders: async () => ({ providers: [] }),
  pollOAuthSession: (providerId: string, sessionId: string, _profile?: unknown, wait?: number) =>
    pollOAuthSession(providerId, sessionId, wait),
  startOAuthLogin: () => startOAuthLogin()
}))

const requestGateway = (async <T>(_method: string, _params?: Record<string, unknown>): Promise<T> =>
  ({ available: true, has_guest: true }) as T) satisfies FreeTierRequester

const start = (id: string) => ({
  expires_in: 900,
  flow: 'device_code' as const,
  poll_interval: 2,
  session_id: id,
  user_code: 'ABCD-EFGH',
  verification_url: `https://portal.example/claim?code=${id}`
})

beforeEach(() => {
  vi.useFakeTimers()
  vi.spyOn(window, 'open').mockReturnValue(null)
})

afterEach(async () => {
  const { closeFreeTierSignIn } = await import('./free-tier-sign-in')
  closeFreeTierSignIn()
  vi.useRealTimers()
  vi.restoreAllMocks()
  vi.clearAllMocks()
})

describe('free-tier sign-in attempts', () => {
  it('a poll from a closed attempt never overwrites the attempt now on screen', async () => {
    const { $freeTierSignIn, beginFreeTierSignIn, closeFreeTierSignIn } = await import('./free-tier-sign-in')
    let resolveA: (value: unknown) => void = () => undefined
    startOAuthLogin.mockResolvedValueOnce(start('session-a')).mockResolvedValueOnce(start('session-b'))
    pollOAuthSession.mockImplementation((_provider: string, id: string) =>
      id === 'session-a'
        ? new Promise(resolve => (resolveA = resolve))
        : Promise.resolve({ session_id: id, status: 'pending' })
    )

    await beginFreeTierSignIn(requestGateway)
    expect($freeTierSignIn.get()).toMatchObject({ sessionId: 'session-a', status: 'code' })
    await vi.advanceTimersByTimeAsync(2000) // A's poll is now in flight

    closeFreeTierSignIn()
    await beginFreeTierSignIn(requestGateway)
    expect($freeTierSignIn.get()).toMatchObject({ sessionId: 'session-b', status: 'code' })

    resolveA({ account_email: 'old@example.com', model: 'x', session_id: 'session-a', status: 'approved' })
    await vi.advanceTimersByTimeAsync(10)

    expect($freeTierSignIn.get()).toMatchObject({ sessionId: 'session-b', status: 'code' })
    expect(cancelOAuthSession).toHaveBeenCalledWith('session-a')
  })
})

describe('free-tier sign-in failure screens', () => {
  it('maps the account service verdicts onto ruled screens', async () => {
    const { signInFailureKind } = await import('./free-tier-sign-in')

    expect(signInFailureKind('anon_rate_limited')).toBe('busy')
    expect(signInFailureKind('anon_gate_paused')).toBe('busy')
    expect(signInFailureKind('anon_unreachable')).toBe('unreachable')
    expect(signInFailureKind('anon_server_error')).toBe('unreachable')
    expect(signInFailureKind('anon_gate_closed')).toBe('unavailable')
    expect(signInFailureKind('anon_pow_required')).toBe('unavailable')
    expect(signInFailureKind('anon_account_locked')).toBe('unavailable')
    expect(signInFailureKind('timeout')).toBe('timed_out')
    expect(signInFailureKind('wat')).toBe('error')
    expect(signInFailureKind(null)).toBe('error')
  })

  it('a busy verdict keeps the wait the service named', async () => {
    const { $freeTierSignIn, beginFreeTierSignIn } = await import('./free-tier-sign-in')
    startOAuthLogin.mockResolvedValueOnce(start('session-busy'))
    pollOAuthSession.mockResolvedValue({
      error_message: 'Signing in could not finish because the service is busy.',
      reason: 'anon_rate_limited',
      retry_after: 45,
      retryable: true,
      session_id: 'session-busy',
      status: 'error'
    })

    await beginFreeTierSignIn(requestGateway)
    await vi.advanceTimersByTimeAsync(2000)

    expect($freeTierSignIn.get()).toEqual({
      kind: 'busy',
      message: 'Signing in could not finish because the service is busy.',
      retryAfter: 45,
      status: 'failed'
    })
  })
})

describe('free-tier sign-in long-polls', () => {
  it('long-polls the sign-in, and never asks again sooner than the gap when the backend answers at once', async () => {
    const { beginFreeTierSignIn } = await import('./free-tier-sign-in')
    startOAuthLogin.mockResolvedValueOnce(start('session-poll'))
    pollOAuthSession.mockResolvedValue({ session_id: 'session-poll', status: 'pending' })

    await beginFreeTierSignIn(requestGateway)
    await vi.advanceTimersByTimeAsync(10)

    expect(pollOAuthSession).toHaveBeenCalledTimes(1)
    expect(pollOAuthSession).toHaveBeenCalledWith('nous', 'session-poll', 25)

    await vi.advanceTimersByTimeAsync(2000)
    expect(pollOAuthSession).toHaveBeenCalledTimes(2)
  })

  const approvedWith = (handoff: AppHandoffStatus) => ({
    account_email: 'someone@example.com',
    handoff_status: handoff,
    model: null,
    session_id: 'session-moved',
    status: 'approved'
  })

  const retrying: AppHandoffStatus = {
    lines: ['Moving your free-tier apps to this account.'],
    state: 'retrying',
    updated_at: 't1'
  }

  it('follows a move that is still retrying until it settles', async () => {
    const { $freeTierSignIn, beginFreeTierSignIn } = await import('./free-tier-sign-in')

    const claimed: AppHandoffStatus = {
      dropped_toolkits: [],
      lines: ['Your apps moved.'],
      state: 'claimed',
      updated_at: 't2'
    }

    startOAuthLogin.mockResolvedValueOnce(start('session-moved'))
    pollOAuthSession.mockResolvedValue(approvedWith(retrying))
    getNousHandoffStatus.mockResolvedValueOnce(claimed)

    await beginFreeTierSignIn(requestGateway)
    await vi.advanceTimersByTimeAsync(10)

    expect(getNousHandoffStatus).toHaveBeenCalledWith('t1')
    expect($freeTierSignIn.get()).toMatchObject({ email: 'someone@example.com', handoff: claimed, status: 'completed' })

    await vi.advanceTimersByTimeAsync(10_000)
    expect(getNousHandoffStatus).toHaveBeenCalledTimes(1)
  })

  it('stops following the move once the dialog closes', async () => {
    const { beginFreeTierSignIn, closeFreeTierSignIn } = await import('./free-tier-sign-in')
    startOAuthLogin.mockResolvedValueOnce(start('session-moved'))
    pollOAuthSession.mockResolvedValue(approvedWith(retrying))
    getNousHandoffStatus.mockResolvedValue(retrying)

    await beginFreeTierSignIn(requestGateway)
    await vi.advanceTimersByTimeAsync(10)
    closeFreeTierSignIn()
    const asked = getNousHandoffStatus.mock.calls.length

    await vi.advanceTimersByTimeAsync(10_000)
    expect(getNousHandoffStatus.mock.calls.length).toBe(asked)
  })

  it('keeps following the move through a failed read', async () => {
    const { $freeTierSignIn, beginFreeTierSignIn } = await import('./free-tier-sign-in')
    const claimed: AppHandoffStatus = { lines: ['Your apps moved.'], state: 'claimed', updated_at: 't2' }
    startOAuthLogin.mockResolvedValueOnce(start('session-moved'))
    pollOAuthSession.mockResolvedValue(approvedWith(retrying))
    getNousHandoffStatus.mockRejectedValueOnce(new Error('connection reset')).mockResolvedValueOnce(claimed)

    await beginFreeTierSignIn(requestGateway)
    await vi.advanceTimersByTimeAsync(10_000)

    expect($freeTierSignIn.get()).toMatchObject({ handoff: claimed, status: 'completed' })
  })

  it('stops after a bounded run of failed reads, leaving the note that the move continues', async () => {
    const { $freeTierSignIn, beginFreeTierSignIn } = await import('./free-tier-sign-in')
    startOAuthLogin.mockResolvedValueOnce(start('session-moved'))
    pollOAuthSession.mockResolvedValue(approvedWith(retrying))
    getNousHandoffStatus.mockRejectedValue(new Error('backend gone'))

    await beginFreeTierSignIn(requestGateway)
    await vi.advanceTimersByTimeAsync(600_000)
    const asked = getNousHandoffStatus.mock.calls.length

    await vi.advanceTimersByTimeAsync(600_000)
    expect(getNousHandoffStatus.mock.calls.length).toBe(asked)
    expect(asked).toBeGreaterThan(1)
    expect($freeTierSignIn.get()).toMatchObject({ handoff: retrying, status: 'completed' })
  })

  it('a sign-in drops the cached connector reads, and again when a followed move settles', async () => {
    const { beginFreeTierSignIn } = await import('./free-tier-sign-in')
    const { queryClient } = await import('@/lib/query-client')
    const list = ['connectors', 'default', 'list']
    const stale = () => queryClient.getQueryState(list)?.isInvalidated
    let settle: (handoff: AppHandoffStatus) => void = () => undefined
    queryClient.setQueryData(list, [{ slug: 'linear', status: 'connected' }])
    startOAuthLogin.mockResolvedValueOnce(start('session-moved'))
    pollOAuthSession.mockResolvedValue(approvedWith(retrying))
    getNousHandoffStatus.mockReturnValueOnce(new Promise(resolve => (settle = resolve)))

    await beginFreeTierSignIn(requestGateway)
    await vi.advanceTimersByTimeAsync(10)
    expect(stale()).toBe(true)

    queryClient.setQueryData(list, [])
    expect(stale()).toBe(false)
    settle({ lines: ['Your apps moved.'], state: 'claimed', updated_at: 't2' })
    await vi.advanceTimersByTimeAsync(10)
    expect(stale()).toBe(true)
    queryClient.clear()
  })
})
