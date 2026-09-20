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

  it('retains the retry delay from a rate-limited poll', async () => {
    const { $freeTierSignIn, beginFreeTierSignIn } = await import('./free-tier-sign-in')
    startOAuthLogin.mockResolvedValueOnce(start('rate-limited-session'))
    pollOAuthSession.mockResolvedValueOnce({
      reason: 'anon_rate_limited',
      retry_after: 45,
      session_id: 'rate-limited-session',
      status: 'error'
    })

    await beginFreeTierSignIn(requestGateway)
    await vi.advanceTimersByTimeAsync(2000)

    expect($freeTierSignIn.get()).toMatchObject({ kind: 'busy', retryAfter: 45, status: 'failed' })
  })
})

it('cancels a failed polling session and ignores its other in-flight approval', async () => {
  const { $freeTierSignIn, beginFreeTierSignIn } = await import('./free-tier-sign-in')
  let approve!: (value: unknown) => void
  startOAuthLogin.mockResolvedValueOnce(start('failed-session'))
  pollOAuthSession
    .mockImplementationOnce(
      () =>
        new Promise(resolve => {
          approve = resolve
        })
    )
    .mockRejectedValueOnce(new Error('connection interrupted'))
  await beginFreeTierSignIn(requestGateway)
  await vi.advanceTimersByTimeAsync(4000)
  expect(cancelOAuthSession).toHaveBeenCalledWith('failed-session')
  expect($freeTierSignIn.get()).toMatchObject({ status: 'failed', message: 'connection interrupted' })
  approve({ status: 'approved', account_email: 'late@example.com' })
  await vi.advanceTimersByTimeAsync(1)
  expect($freeTierSignIn.get()).toMatchObject({ status: 'failed' })
})
