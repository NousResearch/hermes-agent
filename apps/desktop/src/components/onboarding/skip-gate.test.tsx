import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { $desktopOnboarding, type DesktopOnboardingState, type OnboardingContext } from '@/store/onboarding'
import { makeOAuthProvider } from '@/test/oauth-provider'

import { DesktopOnboardingOverlay } from '.'

const HEADER = "Let's get you setup with Hermes Agent"

// Never answers: the readiness effect stays in flight, so each case is observed
// on exactly the state it set up instead of racing a round to completion.
const pendingGateway = (() => new Promise(() => {})) as OnboardingContext['requestGateway']

function skippedState(overrides: Partial<DesktopOnboardingState> = {}): DesktopOnboardingState {
  return {
    configured: false,
    flow: { status: 'idle' },
    mode: 'oauth',
    providers: [makeOAuthProvider('nous', 'Nous Portal')],
    reason: null,
    requested: false,
    firstRunSkipped: true,
    manual: false,
    localEndpoint: false,
    freeTierReady: false,
    ...overrides
  }
}

afterEach(() => {
  cleanup()

  try {
    window.localStorage.clear()
  } catch {
    // jsdom localStorage should always be present; ignore if not.
  }

  $desktopOnboarding.set(skippedState({ firstRunSkipped: false, providers: null }))
})

describe('DesktopOnboardingOverlay first-run skip gate', () => {
  it('stays out of the way for a user who chose "I\'ll choose a provider later"', () => {
    $desktopOnboarding.set(skippedState())

    render(<DesktopOnboardingOverlay enabled profile="default" requestGateway={pendingGateway} />)

    expect(screen.queryByText(HEADER)).toBeNull()
  })

  it('still opens when a real credential wall asked for it', () => {
    // `requested` is only ever set by the submit-time deferred credential
    // warning or a stream that reported a provider setup error — never by a
    // passive readiness round. A durable skip must not swallow those.
    $desktopOnboarding.set(skippedState({ requested: true }))

    render(<DesktopOnboardingOverlay enabled profile="default" requestGateway={pendingGateway} />)

    expect(screen.getByText(HEADER)).toBeTruthy()
  })
})
