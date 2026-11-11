import type { OnboardingRunStateResult } from '@hermes/shared'
import { atom, computed } from 'nanostores'

import { isMissingRpcMethod, isOutOfSyncRpcParams } from '@/lib/gateway-rpc'
import type { GatewayRequester } from '@/lib/yolo-session'

import { activeGateway, requestGatewayForAgent } from './gateway'
import { $gatewayReadyAt } from './live-sync'
import { clearFreeTierIntro, dismissFirstRunOnboarding } from './onboarding'
import { $setupGuidePhase as $onboardingPhase } from './onboarding-phase'
import { markOnboardingDecided, ONBOARDING_DECIDE_OFFER_MS, setOnboardingSurfaceActive } from './onboarding-presence'
import { captureOnboardingScope, type OnboardingScope } from './onboarding-scope'
import { $setupGuideOnboarding } from './shell-policy'

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
export const $guideSuspended = atom(false)
export const $guideRecovery = atom<{
  message: string
  upgradeRequired: boolean
  scope: OnboardingScope
  retry: () => Promise<unknown>
} | null>(null)
let generation = 0
$setupGuideOnboarding.listen(() => {
  generation += 1
})

computed(
  [$onboardingPhase, $setupGuideOnboarding, $guideRecovery],
  (phase, redesign, recovery) => redesign && (phase === 'shown' || recovery !== null)
).subscribe(active => setOnboardingSurfaceActive('setup-guide', active))

function show(state: OnboardingRunStateResult, request: OnboardingRequest, scope: OnboardingScope): void {
  $guideRecovery.set(null)
  const previousScope = $guideScope.get()

  if (previousScope?.connectionId !== scope.connectionId || previousScope?.profile !== scope.profile) {
    $guideSuspended.set(false)
  }

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

function offerRecovery(error: unknown, scope: OnboardingScope, retry: () => Promise<unknown>): void {
  const message = error instanceof Error ? error.message : String(error)
  $guideRecovery.set({
    message,
    upgradeRequired:
      isMissingRpcMethod(error) ||
      isOutOfSyncRpcParams(message) ||
      (/sessionless/i.test(message) && /extra_forbidden|extra inputs are not permitted|unexpected/i.test(message)),
    scope,
    retry
  })
}

export async function startOnboarding(request: OnboardingRequest, scope = captureOnboardingScope()): Promise<boolean> {
  const current = ++generation
  const pinned = pinRequest(request, scope)

  try {
    await pinned('onboarding.state')

    if (current !== generation || !$setupGuideOnboarding.get()) {
      return false
    }

    const state = await pinned<OnboardingRunStateResult>('onboarding.set_run', {
      run: true,
      step: 'preferences',
      restart: true
    })

    if (current !== generation || !$setupGuideOnboarding.get() || !state.run) {
      return false
    }

    $guideSuspended.set(false)
    show(state, pinned, scope)
    markOnboardingDecided()

    return true
  } catch (error) {
    if (current === generation && $setupGuideOnboarding.get()) {
      offerRecovery(error, scope, () => startOnboarding(request, scope))
    }

    return false
  }
}

export function suspendOnboarding(): void {
  if ($setupGuideOnboarding.get() && $onboardingPhase.get() === 'shown') {
    $guideSuspended.set(true)
  }
}

export async function resumeOnboarding(): Promise<void> {
  await reconcileOnboarding()
  $guideSuspended.set(false)
}

export async function saveGuideStep(step: GuideStep): Promise<void> {
  const request = $guideRequest.get()

  if (!request) {
    return
  }

  const current = ++generation
  const state = await request<OnboardingRunStateResult>('onboarding.set_run', { run: true, step })

  if (current !== generation || !$setupGuideOnboarding.get()) {
    return
  }

  if (state.run) {
    $guideStep.set(state.step ?? 'preferences')
  } else {
    settleGuide('done')
  }
}

function settleGuide(outcome: 'done' | 'skipped'): void {
  $guideSuspended.set(false)
  $guideRecovery.set(null)
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

  if (current !== generation || !$setupGuideOnboarding.get()) {
    return false
  }

  settleGuide(outcome)

  return true
}

export async function reconcileOnboarding(): Promise<void> {
  const request = $guideRequest.get()

  if (!request || !$setupGuideOnboarding.get() || $onboardingPhase.get() !== 'shown') {
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

async function readAndStart(request: OnboardingRequest, scope = captureOnboardingScope()): Promise<void> {
  const current = ++generation
  const pinned = pinRequest(request, scope)
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
    const state = await pinned<OnboardingRunStateResult>(
      'onboarding.state',
      window.hermesDesktop?.assumeFreshInstall ? { assume_fresh: true } : {},
      0
    )

    if (released || current !== generation || !$setupGuideOnboarding.get()) {
      return
    }

    $guideRecovery.set(null)

    if (!state.run) {
      // Only a persisted sessionless guide owns the welcome; run=false alone may predate it.
      if (state.started || $onboardingPhase.get() === 'shown') {
        settleGuide('done')
      }

      return
    }

    if (!state.eligible && !state.started) {
      return
    }

    // Resuming is read-only: a peer may have advanced since this snapshot.
    // Only a fresh guide needs the initial durable latch.
    const saved = state.started ? state : await pinned<OnboardingRunStateResult>('onboarding.set_run', { run: true })

    if (!released && current === generation && $setupGuideOnboarding.get() && saved.run) {
      show(saved, pinned, scope)
    }
  } catch (error) {
    if (!released && current === generation && $setupGuideOnboarding.get()) {
      offerRecovery(error, scope, () => readAndStart(request, scope))
    }
  } finally {
    unlisten()
    clearTimeout(timer)
    $decideExit.set(null)
    markOnboardingDecided()
  }
}
