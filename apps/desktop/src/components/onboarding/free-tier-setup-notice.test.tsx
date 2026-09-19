import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { en } from '@/i18n/en'
import { $freeTierStatus, freeTierSetupFailure } from '@/store/free-tier'
import type { OnboardingContext } from '@/store/onboarding'
import type { FreeTierStatus } from '@/types/hermes'

import { FreeTierSetupNotice, setupFailureCopy } from './free-tier-setup-notice'

const NO_IDENTITY: FreeTierStatus = {
  available: false,
  enabled: true,
  has_guest: false,
  label: 'Nous · free tier',
  model: 'nous/welcome',
  notice_pending: false
}

function ctxReturning(status: FreeTierStatus, provisioned?: FreeTierStatus): OnboardingContext & { calls: string[] } {
  const calls: string[] = []
  let current = status

  return {
    calls,
    requestGateway: async <T,>(method: string): Promise<T> => {
      calls.push(method)

      if (method === 'free_tier.provision' && provisioned) {
        current = provisioned
      }

      if (method === 'free_tier.status') {
        return current as T
      }

      return { provider_configured: current.has_guest, ok: current.has_guest } as T
    }
  }
}

afterEach(() => {
  cleanup()
  $freeTierStatus.set(null)
  vi.restoreAllMocks()
})

describe('setupFailureCopy', () => {
  const copy = en.freeTier.setupFailed

  it.each([
    ['anon_gate_closed', copy.gateClosed],
    ['anon_gate_paused', copy.paused],
    ['anon_unreachable', copy.unreachable],
    ['anon_server_error', copy.serverError],
    ['anon_pow_required', copy.powRequired],
    ['anon_account_locked', copy.locked],
    ['anon_rate_limited', copy.rateLimited('about 5 minutes')]
  ])('%s has its own sentence, in the agreed voice', (code, expected) => {
    const failure = freeTierSetupFailure({ ...NO_IDENTITY, error_code: code, retry_after: 300 })
    const text = failure ? setupFailureCopy(failure, copy) : ''

    expect(text).toBe(expected)
    // Never "the free service is off" — what is unavailable is using Hermes without signing in —
    // and no jargon a first-time user would not know.
    expect(text.toLowerCase()).not.toMatch(/free (service|model|tier) is (off|switched off|unavailable|down)/)
    expect(text.toLowerCase()).not.toMatch(/anonymous|guest|credential|token|rate limit/)
  })

  it('falls back to the backend sentence for a code this build does not know', () => {
    const failure = freeTierSetupFailure({ ...NO_IDENTITY, error: 'Something new.', error_code: 'anon_newer' })

    expect(failure && setupFailureCopy(failure, copy)).toBe('Something new.')
    // Prototype names are not codes.
    expect(setupFailureCopy({ ...failure!, code: 'constructor', message: '' }, copy)).toBe(copy.generic)
  })
})

describe('FreeTierSetupNotice', () => {
  it('renders nothing when the backend reported no failure', async () => {
    const ctx = ctxReturning(NO_IDENTITY)
    render(<FreeTierSetupNotice ctx={ctx} />)

    await waitFor(() => expect(ctx.calls).toContain('free_tier.status'))
    expect(screen.queryByTestId('free-tier-setup-notice')).toBeNull()
  })

  it('shows the sentence, the sign-in door, and a retry for a retryable refusal', async () => {
    const ctx = ctxReturning({ ...NO_IDENTITY, error_code: 'anon_gate_paused', retryable: true, retry_after: 60 })
    render(<FreeTierSetupNotice ctx={ctx} />)

    await screen.findByTestId('free-tier-setup-notice')
    expect(screen.getByText(en.freeTier.setupFailed.paused)).toBeTruthy()
    expect(screen.getByText(en.freeTier.setupFailed.signInBelow)).toBeTruthy()
    expect(screen.getByRole('button', { name: en.freeTier.setupFailed.tryAgain })).toBeTruthy()
  })

  it('offers no sign-in door and no retry when the service is unreachable and the code is terminal', async () => {
    const ctx = ctxReturning({ ...NO_IDENTITY, error_code: 'anon_gate_closed', retryable: false })
    render(<FreeTierSetupNotice ctx={ctx} />)

    await screen.findByTestId('free-tier-setup-notice')
    expect(screen.queryByRole('button')).toBeNull()

    cleanup()
    $freeTierStatus.set(null)
    const unreachable = ctxReturning({
      ...NO_IDENTITY,
      error_code: 'anon_unreachable',
      retryable: true,
      retry_after: 15
    })
    render(<FreeTierSetupNotice ctx={unreachable} />)

    await screen.findByTestId('free-tier-setup-notice')
    expect(screen.queryByText(en.freeTier.setupFailed.signInBelow)).toBeNull()
    expect(screen.getByRole('button', { name: en.freeTier.setupFailed.tryAgain })).toBeTruthy()
  })

  it('the retry asks the backend once and the notice leaves when an identity appears', async () => {
    const ctx = ctxReturning(
      { ...NO_IDENTITY, error_code: 'anon_unreachable', retryable: true, retry_after: 15 },
      { ...NO_IDENTITY, available: true, has_guest: true }
    )

    render(<FreeTierSetupNotice ctx={ctx} />)

    fireEvent.click(await screen.findByRole('button', { name: en.freeTier.setupFailed.tryAgain }))

    await waitFor(() => expect(screen.queryByTestId('free-tier-setup-notice')).toBeNull())
    expect(ctx.calls.filter(method => method === 'free_tier.provision')).toHaveLength(1)
  })
})
