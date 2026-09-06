import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as HermesApi from '@/hermes'
import { $freeTierSignIn, openFreeTierSignIn } from '@/store/free-tier-sign-in'

const pollOAuthSession = vi.fn()
const requestGateway = vi.fn(async () => ({ available: true, has_guest: true }))

// Only the two calls this flow makes are replaced; everything else keeps its
// real implementation so the modules the dialog pulls in (the onboarding
// DeviceCode cell, the model picker) still resolve their imports.
vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<typeof HermesApi>()),
  pollOAuthSession: (providerId: string, sessionId: string) => pollOAuthSession(providerId, sessionId),
  startOAuthLogin: async () => ({
    expires_in: 900,
    flow: 'device_code' as const,
    poll_interval: 2,
    session_id: 'session-1',
    user_code: 'ABCD-EFGH',
    verification_url: 'https://portal.example/claim?code=ABCD-EFGH'
  })
}))

vi.mock('@/app/gateway/hooks/use-gateway-request', () => ({
  useGatewayRequest: () => ({ requestGateway })
}))

beforeEach(() => {
  vi.spyOn(window, 'open').mockReturnValue(null)
})

afterEach(() => {
  cleanup()
  $freeTierSignIn.set({ status: 'closed' })
  vi.restoreAllMocks()
  vi.clearAllMocks()
  vi.useRealTimers()
})

describe('FreeTierSignInDialog', () => {
  it('shows the transfer code, then the signed-in screen once the poll approves', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true })
    pollOAuthSession.mockResolvedValue({
      account_email: 'someone@example.com',
      model: 'Hermes-4-405B',
      reason: null,
      session_id: 'session-1',
      status: 'approved'
    })

    const { FreeTierSignInDialog } = await import('./sign-in-dialog')

    await act(async () => {
      render(
        <QueryClientProvider client={new QueryClient()}>
          <FreeTierSignInDialog />
        </QueryClientProvider>
      )
    })

    await act(async () => {
      openFreeTierSignIn()
    })

    await waitFor(() => expect(screen.getByText('Do not share this code.')).toBeTruthy())
    expect(screen.getByText('Enter this code in your browser to finish signing in.')).toBeTruthy()

    // The 2s poll tick is what carries the approval through to the last screen.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(2100)
    })

    await waitFor(() => expect(screen.getByText('Signed in as someone@example.com')).toBeTruthy())
    expect(screen.getByText('Your connectors are kept.')).toBeTruthy()
    expect(screen.getByText('Hermes-4-405B')).toBeTruthy()
  })
})
