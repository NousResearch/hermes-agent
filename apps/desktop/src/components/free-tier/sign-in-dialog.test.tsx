import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as HermesApi from '@/hermes'
import { $freeTierStatus } from '@/store/free-tier'
import { $freeTierSignIn, noteFreeTierTurnComplete, openFreeTierSignIn } from '@/store/free-tier-sign-in'
import { $onboardingGate } from '@/store/onboarding-gate'
import type { FreeTierStatus } from '@/types/hermes'

const pollOAuthSession = vi.fn()
// The two answers the dialog reads: `free_tier.status` and `free_tier.claim_nudge`.
type GatewayAnswer = { claimed: boolean } | Partial<FreeTierStatus>

const requestGateway = vi.fn(async (_method?: string): Promise<GatewayAnswer> => ({ available: true, has_guest: true }))

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
  $freeTierStatus.set(null)
  $onboardingGate.set({ guideKickoff: 'idle', guideQueued: false, phase: 'idle' })
  Reflect.deleteProperty(window, 'hermesDesktop')
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

    // The 2s poll tick is what carries the approval through to the last screen.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(2100)
    })

    await waitFor(() => expect(screen.getByText('Signed in as someone@example.com')).toBeTruthy())
    expect(screen.getByText('Hermes-4-405B')).toBeTruthy()
  })

  describe('sign-in offer', () => {
    const dueNow = { available: true, enabled: true, has_guest: true, label: '', model: '', notice_pending: false }

    async function renderDialog() {
      const { FreeTierSignInDialog } = await import('./sign-in-dialog')

      await act(async () => {
        render(
          <QueryClientProvider client={new QueryClient()}>
            <FreeTierSignInDialog />
          </QueryClientProvider>
        )
      })
    }

    beforeEach(() => {
      requestGateway.mockImplementation(async (method?: string) =>
        method === 'free_tier.claim_nudge' ? { claimed: true } : { ...dueNow, nudge_due_in: null }
      )
    })

    it('waits for guided onboarding to leave the screen, then offers sign-in', async () => {
      Object.assign(window, { hermesDesktop: { guestOnboardingEnabled: true } })
      $onboardingGate.set({ guideKickoff: 'started', guideQueued: false, phase: 'guided' })
      await renderDialog()

      await act(async () => {
        $freeTierStatus.set({ ...dueNow, nudge_due_in: 0 })
      })

      expect(requestGateway).not.toHaveBeenCalledWith('free_tier.claim_nudge')

      await act(async () => {
        $onboardingGate.set({ guideKickoff: 'started', guideQueued: false, phase: 'done' })
      })

      await waitFor(() => expect(screen.getByText('Keep going with Hermes')).toBeTruthy())
      expect(requestGateway).toHaveBeenCalledWith('free_tier.claim_nudge')
    })

    it('a finished free-tier turn re-reads the status, and a due offer opens with no user action', async () => {
      await renderDialog()
      await act(async () => {
        $freeTierStatus.set({ ...dueNow, nudge_due_in: null })
      })
      // The backend recorded the finished task before message.complete: the next read says due.
      requestGateway.mockImplementation(async (method?: string) =>
        method === 'free_tier.claim_nudge' ? { claimed: true } : { ...dueNow, nudge_due_in: 0 }
      )

      await act(async () => noteFreeTierTurnComplete())

      await waitFor(() => expect(screen.getByText('Keep going with Hermes')).toBeTruthy())
    })

    it('Sign in continues into the normal sign-in flow', async () => {
      await renderDialog()
      await act(async () => $freeTierSignIn.set({ status: 'offer' }))

      await act(async () => screen.getByRole('button', { name: 'Sign in' }).click())

      await waitFor(() => expect(screen.getByText('Do not share this code.')).toBeTruthy())
    })

    it('Not now closes the offer', async () => {
      await renderDialog()
      await act(async () => $freeTierSignIn.set({ status: 'offer' }))

      await act(async () => screen.getByRole('button', { name: 'Not now' }).click())

      expect($freeTierSignIn.get()).toEqual({ status: 'closed' })
      expect(screen.queryByText('Keep going with Hermes')).toBeNull()
    })
  })
})
