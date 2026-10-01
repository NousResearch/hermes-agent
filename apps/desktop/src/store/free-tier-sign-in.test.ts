import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as HermesApi from '@/hermes'
import type { FreeTierRequester } from '@/store/free-tier'

const startOAuthLogin = vi.fn()
const pollOAuthSession = vi.fn()
const cancelOAuthSession = vi.fn(async (_id: string) => ({ ok: true }))

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<typeof HermesApi>()),
  cancelOAuthSession: (id: string) => cancelOAuthSession(id),
  listOAuthProviders: async () => ({ providers: [] }),
  pollOAuthSession: (providerId: string, sessionId: string) => pollOAuthSession(providerId, sessionId),
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

    expect(signInFailureKind('account_busy')).toBe('busy')
    expect(signInFailureKind('anon_rate_limited')).toBe('busy')
    expect(signInFailureKind('anon_gate_paused')).toBe('busy')
    expect(signInFailureKind('anon_unreachable')).toBe('unreachable')
    expect(signInFailureKind('anon_server_error')).toBe('unreachable')
    expect(signInFailureKind('anon_gate_closed')).toBe('unavailable')
    expect(signInFailureKind('anon_pow_required')).toBe('unavailable')
    expect(signInFailureKind('anon_account_locked')).toBe('unavailable')
    expect(signInFailureKind('user_declined')).toBe('rejected')
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
