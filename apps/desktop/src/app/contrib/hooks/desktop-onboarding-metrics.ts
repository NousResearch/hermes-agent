/**
 * First-run funnel telemetry (hermes.desktop.onboarding). Observes the
 * onboarding stores' own transitions — the provider picker, the first-run
 * questionnaire, free-tier sign-in — and maps them onto the closed step set; nothing here
 * changes onboarding behavior. Manual (Settings "add provider") flows are not
 * first run and are ignored.
 */

import { $questionnairePhase, type QuestionnairePhase } from '@/onboarding/store'
import { closeOnboardingStep, recordDislike, recordOnboarding } from '@/store/desktop-metrics'
import { $freeTierSignIn, type FreeTierSignInState } from '@/store/free-tier-sign-in'
import { $desktopOnboarding, type DesktopOnboardingState } from '@/store/onboarding'

const OAUTH_PENDING = new Set(['awaiting_user', 'error', 'external_pending', 'polling', 'starting', 'submitting'])

function providerStepFor(state: DesktopOnboardingState): 'provider_api_key' | 'provider_local' | null {
  return state.localEndpoint ? 'provider_local' : state.mode === 'apikey' ? 'provider_api_key' : null
}

export function onboardingTransition(prev: DesktopOnboardingState, next: DesktopOnboardingState): void {
  if (prev.manual || next.manual) {
    return
  }

  if (next.configured === false && prev.configured !== false && !next.firstRunSkipped) {
    recordOnboarding('provider_setup', 'reached')
  }

  const prevStep = providerStepFor(prev)
  const nextStep = providerStepFor(next)

  if (next.configured === false && nextStep && nextStep !== prevStep) {
    recordOnboarding(nextStep, 'reached')
  }

  if (next.flow.status !== prev.flow.status) {
    if (next.flow.status === 'starting') {
      recordOnboarding('provider_oauth', 'reached')
    } else if (next.flow.status === 'success') {
      recordOnboarding('provider_oauth', 'completed')
    } else if (next.flow.status === 'confirming_model') {
      recordOnboarding('model_pick', 'reached')
    } else if (
      next.flow.status === 'idle' &&
      OAUTH_PENDING.has(prev.flow.status) &&
      next.configured !== true &&
      !next.firstRunSkipped
    ) {
      // The sign-in was backed out of (Cancel / Back), not finished.
      recordDislike('cancelled', 'provider_oauth')
      closeOnboardingStep('provider_oauth')
    }
  }

  // Only a picker the user actually saw (configured was false) completes a step: the boot readiness
  // probe flips unknown → configured on every launch of an already-set-up install.
  if (next.configured === true && prev.configured === false) {
    if (prev.flow.status === 'confirming_model') {
      recordOnboarding('model_pick', 'completed')
    }

    if (prevStep) {
      recordOnboarding(prevStep, 'completed')
    }

    recordOnboarding('provider_setup', 'completed')
  }

  if (next.firstRunSkipped && !prev.firstRunSkipped) {
    recordOnboarding('choose_later', 'completed')

    for (const step of [
      'provider_setup',
      'provider_oauth',
      'provider_api_key',
      'provider_local',
      'model_pick'
    ] as const) {
      closeOnboardingStep(step)
    }
  }

  if (next.freeTierReady !== prev.freeTierReady) {
    recordOnboarding('free_tier_ready', next.freeTierReady ? 'reached' : 'completed')
  }
}

/** D10: `guide` is reached when the questionnaire shows and completed at a successful Start. */
export function guidePhaseTransition(prev: QuestionnairePhase, next: QuestionnairePhase): void {
  if (prev === next) {
    return
  }

  if (next === 'shown') {
    recordOnboarding('guide', 'reached')
  } else if (next === 'skipped') {
    recordOnboarding('guide_skip', 'completed')
    closeOnboardingStep('guide')
  } else if (next === 'done') {
    recordOnboarding('guide', 'completed')
  } else if (next === 'failed') {
    closeOnboardingStep('guide')
  }
}

const SIGN_IN_OPEN = new Set(['code', 'failed', 'finishing', 'requested', 'setting_up'])

export function signInTransition(prev: FreeTierSignInState, next: FreeTierSignInState): void {
  if (prev.status === next.status) {
    return
  }

  // The offer reaches the step too: its Sign in goes straight to `setting_up`, never through
  // `requested`. "Not now" closes the step without counting as a cancelled sign-in.
  if (next.status === 'requested' || next.status === 'offer') {
    recordOnboarding('sign_in', 'reached')
  } else if (next.status === 'completed' || next.status === 'already_signed_in') {
    recordOnboarding('sign_in', 'completed')
  } else if (next.status === 'closed' && prev.status === 'offer') {
    closeOnboardingStep('sign_in')
  } else if (next.status === 'closed' && SIGN_IN_OPEN.has(prev.status)) {
    recordDislike('cancelled', 'free_tier_sign_in')
    closeOnboardingStep('sign_in')
  }
}

/** Subscribe the three first-run stores; returns the unsubscribe. */
export function observeOnboardingMetrics(): () => void {
  let onboarding = $desktopOnboarding.get()
  let phase = $questionnairePhase.get()
  let signIn = $freeTierSignIn.get()

  const stops = [
    $desktopOnboarding.listen(next => {
      const prev = onboarding

      onboarding = next
      onboardingTransition(prev, next)
    }),
    $questionnairePhase.listen(next => {
      const prev = phase

      phase = next
      guidePhaseTransition(prev, next)
    }),
    $freeTierSignIn.listen(next => {
      const prev = signIn

      signIn = next
      signInTransition(prev, next)
    })
  ]

  return () => stops.forEach(stop => stop())
}
