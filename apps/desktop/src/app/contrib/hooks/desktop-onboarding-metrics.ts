/**
 * First-run funnel telemetry (rabbit.desktop.onboarding). Observes the
 * onboarding stores' own transitions — the provider picker and the guided intro —
 * and maps them onto the closed step set; nothing here
 * changes onboarding behavior. Manual (Settings "add provider") flows are not
 * first run and are ignored.
 */

import { closeOnboardingStep, recordDislike, recordOnboarding } from '@/store/desktop-metrics'
import { $desktopOnboarding, type DesktopOnboardingState } from '@/store/onboarding'
import { $onboardingGate, type OnboardingPhase } from '@/store/onboarding-gate'

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

}

export function guidePhaseTransition(prev: OnboardingPhase, next: OnboardingPhase): void {
  if (prev === next) {
    return
  }

  if (next === 'guided') {
    recordOnboarding('guide', 'reached')
  } else if (next === 'skipped') {
    recordOnboarding('guide_skip', 'completed')
    closeOnboardingStep('guide')
  } else if (next === 'done') {
    recordOnboarding('guide', 'completed')
  }
}

/** Subscribe the first-run stores; returns the unsubscribe. */
export function observeOnboardingMetrics(): () => void {
  let onboarding = $desktopOnboarding.get()
  let phase = $onboardingGate.get().phase

  const stops = [
    $desktopOnboarding.listen(next => {
      const prev = onboarding

      onboarding = next
      onboardingTransition(prev, next)
    }),
    $onboardingGate.listen(next => {
      const prev = phase

      phase = next.phase
      guidePhaseTransition(prev, next.phase)
    })
  ]

  return () => stops.forEach(stop => stop())
}
