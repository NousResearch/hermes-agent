import type { OnboardingRunStateResult } from '@hermes/shared'
import { atom, computed } from 'nanostores'

import type { GatewayRequester } from '@/lib/yolo-session'

import { activeGateway, requestGatewayForAgent } from './gateway'
import { $gatewayReadyAt } from './live-sync'
import { clearFreeTierIntro, dismissFirstRunOnboarding } from './onboarding'
import { $setupGuidePhase as $onboardingPhase } from './onboarding-phase'
import { markOnboardingDecided, ONBOARDING_DECIDE_OFFER_MS, setOnboardingSurfaceActive } from './onboarding-presence'
import { captureOnboardingScope, type OnboardingScope } from './onboarding-scope'
import { $simplifiedAgentMode } from './simplified-agent-mode'

export type OnboardingRequest = <T = unknown>(
  method: string,
  params?: Parameters<GatewayRequester>[1],
  timeoutMs?: number,
  signal?: AbortSignal
) => Promise<T>
export type GuideStep = 'preferences' | 'model'
export const $guideStep = atom<GuideStep>('preferences')
export const $guideRequest = atom<OnboardingRequest | null>(null)
export const $guideScope = atom<OnboardingScope | null>(null)
export const $decideExit = atom<(() => void) | null>(null)
export const $onboardingSettleFrom = atom<number | null>(null)
let generation = 0
$simplifiedAgentMode.listen(() => {
  generation += 1
})

computed([$onboardingPhase, $simplifiedAgentMode], (phase, redesign) => redesign && phase === 'shown').subscribe(
  active => setOnboardingSurfaceActive('setup-guide', active)
)

function show(state: OnboardingRunStateResult, request: OnboardingRequest, scope: OnboardingScope): void {
  $guideRequest.set(request)
  $guideScope.set(scope)
  $guideStep.set(state.step ?? 'preferences')
  $onboardingPhase.set('shown')
}

function pinRequest(request: OnboardingRequest, scope: OnboardingScope): OnboardingRequest {
  // During initial socket boot the caller may own a connection before registry publication.
  const registered = Boolean(activeGateway())

  return (method, params, timeoutMs, signal) => {
    const optedIn = { ...params, sessionless: true }

    return registered
      ? requestGatewayForAgent(
          scope.connectionId ?? null,
          scope.profile || 'default',
          method,
          optedIn,
          timeoutMs,
          signal
        )
      : request(method, optedIn, timeoutMs, signal)
  }
}

export async function startOnboarding(request: OnboardingRequest): Promise<boolean> {
  const current = ++generation
  const scope = captureOnboardingScope()
  request = pinRequest(request, scope)
  await request('onboarding.state')

  if (current !== generation || !$simplifiedAgentMode.get()) {
    return false
  }

  const state = await request<OnboardingRunStateResult>('onboarding.set_run', {
    run: true,
    step: 'preferences',
    restart: true
  })

  if (current !== generation || !$simplifiedAgentMode.get() || !state.run) {
    return false
  }
  show(state, request, scope)
  markOnboardingDecided()

  return true
}

export async function saveGuideStep(step: GuideStep): Promise<void> {
  const request = $guideRequest.get()

  if (!request) {
    return
  }
  const current = ++generation
  const state = await request<OnboardingRunStateResult>('onboarding.set_run', { run: true, step })

  if (current !== generation || !$simplifiedAgentMode.get()) {
    return
  }

  if (state.run) {
    $guideStep.set(state.step ?? 'preferences')
  } else {
    settleGuide('done')
  }
}

function settleGuide(outcome: 'done' | 'skipped'): void {
  // Dismissing provider setup never claims readiness; a real credential wall still opens it.
  dismissFirstRunOnboarding()
  clearFreeTierIntro()
  $onboardingPhase.set(outcome)
  $onboardingSettleFrom.set(Date.now())
}

export async function finishOnboarding(outcome: 'done' | 'skipped'): Promise<boolean> {
  const request = $guideRequest.get()

  if (!request) {
    return false
  }
  const current = ++generation
  await request('onboarding.set_run', { run: false, mark_profile_offered: true })

  if (current !== generation || !$simplifiedAgentMode.get()) {
    return false
  }
  settleGuide(outcome)

  return true
}

export async function reconcileOnboarding(): Promise<void> {
  const request = $guideRequest.get()

  if (!request || !$simplifiedAgentMode.get() || $onboardingPhase.get() !== 'shown') {
    return
  }
  const current = generation
  const state = await request<OnboardingRunStateResult>('onboarding.state')

  if (current !== generation || $onboardingPhase.get() !== 'shown') {
    return
  }

  if (!state.run) {
    settleGuide('done')
  } else {
    $guideStep.set(state.step ?? 'preferences')
  }
}

let decision: Promise<void> | null = null

export function decideOnboarding(request: OnboardingRequest, revisit = false): Promise<void> {
  if (revisit) {
    decision = null
  }
  decision ??= readAndStart(request)

  return decision
}

async function readAndStart(request: OnboardingRequest): Promise<void> {
  const current = generation
  const scope = captureOnboardingScope()
  request = pinRequest(request, scope)
  let released = false
  let timer: ReturnType<typeof setTimeout> | undefined

  const unlisten = $gatewayReadyAt.subscribe(at => {
    clearTimeout(timer)

    if (at !== null) {
      timer = setTimeout(
        () =>
          $decideExit.set(() => {
            released = true
            $decideExit.set(null)
            markOnboardingDecided()
          }),
        Math.max(0, at + ONBOARDING_DECIDE_OFFER_MS - Date.now())
      )
    }
  })

  try {
    const state = await request<OnboardingRunStateResult>(
      'onboarding.state',
      window.hermesDesktop?.assumeFreshInstall ? { assume_fresh: true } : {},
      0
    )

    if (
      !released &&
      current === generation &&
      $simplifiedAgentMode.get() &&
      state.run &&
      (state.eligible || state.started)
    ) {
      // Latch before another boot step makes the install no longer look fresh.
      const saved = await request<OnboardingRunStateResult>('onboarding.set_run', {
        run: true,
        step: state.step ?? 'preferences'
      })

      if (!released && current === generation && $simplifiedAgentMode.get() && saved.run) {
        show(saved, request, scope)
      }
    }

    if (current === generation && $simplifiedAgentMode.get() && !state.run && $onboardingPhase.get() === 'shown') {
      settleGuide('done')
    }
  } catch (error) {
    console.warn('[onboarding] state could not be read', error)
  } finally {
    unlisten()
    clearTimeout(timer)
    $decideExit.set(null)
    markOnboardingDecided()
  }
}
