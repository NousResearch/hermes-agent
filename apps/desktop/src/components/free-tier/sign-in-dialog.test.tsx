import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as HermesApi from '@/hermes'
import { $freeTierStatus } from '@/store/free-tier'
import { $freeTierSignIn, noteFreeTierTurnComplete, openFreeTierSignIn } from '@/store/free-tier-sign-in'
import { markOnboardingDecided, setOnboardingSurfaceActive } from '@/store/onboarding-presence'
import type { AppHandoffStatus, FreeTierStatus } from '@/types/hermes'

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
  getGlobalModelOptions: async () => ({ providers: [] }),
  getNousHandoffStatus: async () => ({ lines: [], state: 'idle' }),
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
  markOnboardingDecided()
})

afterEach(() => {
  cleanup()
  window.location.hash = ''
  $freeTierSignIn.set({ status: 'closed' })
  $freeTierStatus.set(null)
  setOnboardingSurfaceActive('onboarding', false)
  vi.restoreAllMocks()
  vi.clearAllMocks()
  vi.useRealTimers()
})

describe('FreeTierSignInDialog', () => {
  it('shows the code small, then the signed-in screen once the poll approves', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true })
    pollOAuthSession.mockResolvedValueOnce({ session_id: 'session-1', status: 'pending' }).mockResolvedValue({
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

    await waitFor(() => expect(screen.getByText('Finish in your browser')).toBeTruthy())
    expect(screen.getByText('ABCD-EFGH')).toBeTruthy()
    expect(screen.getByText('Do not share it.')).toBeTruthy()

    await act(async () => {
      await vi.advanceTimersByTimeAsync(2100)
    })

    await waitFor(() => expect(screen.getByText('Signed in as')).toBeTruthy())
    expect(screen.getByText('someone@example.com')).toBeTruthy()
    expect(screen.getByText('Your account now carries inference and tools.')).toBeTruthy()
    expect(screen.getByText('Hermes-4-405B')).toBeTruthy()
  })

  describe('result screen', () => {
    async function renderCompleted(handoff: AppHandoffStatus | null) {
      const { FreeTierSignInDialog } = await import('./sign-in-dialog')

      await act(async () => {
        render(
          <QueryClientProvider client={new QueryClient()}>
            <FreeTierSignInDialog />
          </QueryClientProvider>
        )
      })
      await act(async () =>
        $freeTierSignIn.set({ email: 'someone@example.com', handoff, model: null, status: 'completed' })
      )
    }

    it('shows what moved in the backend lines, and Reconnect opens the app left behind', async () => {
      await renderCompleted({
        dropped_toolkits: ['slack'],
        lines: ['Your apps moved to someone@example.com: gmail.', 'Reconnect in someone@example.com: slack.'],
        moved_toolkits: ['gmail'],
        state: 'claimed'
      })

      expect(screen.getByText('Your apps moved to someone@example.com: gmail.')).toBeTruthy()
      expect(screen.getByText('Reconnect in someone@example.com: slack.')).toBeTruthy()
      expect(screen.queryByText('Your account now carries inference and tools.')).toBeNull()

      await act(async () => screen.getByRole('button', { name: 'Reconnect' }).click())

      expect(window.location.hash).toBe('#/capabilities?tab=connectors&connector=slack')
      expect($freeTierSignIn.get()).toEqual({ status: 'closed' })
    })

    it('a move still retrying is a waiting note with no connectors button', async () => {
      await renderCompleted({ lines: ['Moving your free-tier apps to this account.'], state: 'retrying' })

      expect(screen.getByRole('status').textContent).toContain('Moving your free-tier apps to this account.')
      expect(screen.queryByRole('button', { name: 'Reconnect' })).toBeNull()
      expect(screen.queryByRole('button', { name: 'Open Connectors' })).toBeNull()
    })

    it('apps that probably moved already offer a look at Connectors', async () => {
      await renderCompleted({
        lines: ['Your free-tier apps were probably moved already.'],
        probably_moved: true,
        state: 'gone'
      })

      expect(screen.getByRole('button', { name: 'Open Connectors' })).toBeTruthy()
    })
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
      pollOAuthSession.mockResolvedValue({ session_id: 'session-1', status: 'pending' })
      requestGateway.mockImplementation(async (method?: string) =>
        method === 'free_tier.claim_nudge' ? { claimed: true } : { ...dueNow, nudge_due_in: null }
      )
    })

    it('waits for onboarding to leave the screen, then offers sign-in', async () => {
      setOnboardingSurfaceActive('onboarding', true)
      await renderDialog()

      await act(async () => {
        $freeTierStatus.set({ ...dueNow, nudge_due_in: 0 })
      })

      expect(requestGateway).not.toHaveBeenCalledWith('free_tier.claim_nudge', undefined)

      await act(async () => {
        setOnboardingSurfaceActive('onboarding', false)
      })

      await waitFor(() => expect(screen.getByText('Keep going with Hermes')).toBeTruthy())
      expect(requestGateway).toHaveBeenCalledWith('free_tier.claim_nudge', undefined)
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
      await act(async () => $freeTierSignIn.set({ kind: 'trial', status: 'offer' }))

      await act(async () => screen.getByRole('button', { name: 'Sign in' }).click())

      await waitFor(() => expect(screen.getByText('Do not share it.')).toBeTruthy())
    })

    it('Not now closes the offer', async () => {
      await renderDialog()
      await act(async () => $freeTierSignIn.set({ kind: 'trial', status: 'offer' }))

      await act(async () => screen.getByRole('button', { name: 'Not now' }).click())

      expect($freeTierSignIn.get()).toEqual({ status: 'closed' })
      expect(screen.queryByText('Keep going with Hermes')).toBeNull()
    })

    it('an app connected without the trial: claims the connector offer and shows its copy', async () => {
      await renderDialog()
      const ownKey = { ...dueNow, available: false, connector_nudge_due_in: null }
      await act(async () => $freeTierStatus.set(ownKey))
      requestGateway.mockImplementation(async (method?: string) =>
        method === 'free_tier.claim_nudge' ? { claimed: true } : { ...ownKey, connector_nudge_due_in: 0 }
      )

      await act(async () => noteFreeTierTurnComplete())

      await waitFor(() => expect(screen.getByText('Keep your connected apps')).toBeTruthy())
      expect(requestGateway).toHaveBeenCalledWith('free_tier.claim_nudge', { kind: 'connector' })
      expect(screen.queryByText('Keep going with Hermes')).toBeNull()
    })
  })
})
