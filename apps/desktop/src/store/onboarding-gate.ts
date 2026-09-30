import type { OnboardingStateResult } from '@hermes/shared'
import { atom, computed } from 'nanostores'

import { isOnboardingEnabled } from '@/lib/onboarding-enabled'

import { $gateway } from './gateway'
import { DEFAULT_ANSWERS, setOnboardingAnswers } from './onboarding-answers'

export const ONBOARDING_PHASES = ['idle', 'pending', 'guided', 'skipped', 'handoff', 'done'] as const

export type OnboardingPhase = (typeof ONBOARDING_PHASES)[number]

export type GuideKickoffResult = 'started' | 'off' | 'failed'

export interface OnboardingGateState {
  phase: OnboardingPhase
  guideQueued: boolean
  guideKickoff: 'idle' | 'starting' | 'started'
}

type GuideKickoff =
  { status: 'idle' } | { status: 'starting'; promise: Promise<GuideKickoffResult> } | { status: 'started' }

export const $onboardingGate = atom<OnboardingGateState>({ phase: 'idle', guideQueued: false, guideKickoff: 'idle' })

const $onboardingStateRead = atom(false)

/** The setup profile's name, from `onboarding.state` or from the kickoff that creates it; `null` when none is known. */
export const $setupProfileName = atom<null | string>(null)

let guideKickoff: GuideKickoff = { status: 'idle' }

const guidedPhase = (phase: OnboardingPhase) => phase === 'pending' || phase === 'guided' || phase === 'handoff'

/**
 * Guided first run is behind the user: finished, skipped, or never due. The phase is only
 * trusted once the backend's `onboarding.state` (its `intro`) has been read.
 */
export const $guidedOnboardingSettled = computed(
  [$onboardingGate, $onboardingStateRead],
  (gate, read) => !isOnboardingEnabled() || (read && !guidedPhase(gate.phase))
)

export const $guideOpening = computed(
  $onboardingGate,
  gate =>
    isOnboardingEnabled() && (gate.phase === 'pending' || gate.phase === 'guided') && gate.guideKickoff !== 'started'
)

function setGuideKickoff(state: GuideKickoff): void {
  guideKickoff = state
  $onboardingGate.set({ ...$onboardingGate.get(), guideKickoff: state.status })
}

function setPhase(phase: OnboardingPhase): void {
  $onboardingGate.set({ ...$onboardingGate.get(), phase, guideQueued: false })
}

function reportOnboarding(method: 'onboarding.mark_seen' | 'onboarding.record_failed_start'): void {
  void $gateway
    .get()
    ?.request(method, {})
    .catch(error => console.warn(`[onboarding] ${method} failed`, error))
}

export function guidedOnboardingActive(): boolean {
  const { phase } = $onboardingGate.get()

  return isOnboardingEnabled() && guidedPhase(phase)
}

export function markOnboardingStateRead(): void {
  $onboardingStateRead.set(true)
}

export function afterOnboardingStateRead(run: () => void): void {
  if (!isOnboardingEnabled() || $onboardingStateRead.get()) {
    run()

    return
  }

  const stop = $onboardingStateRead.listen(() => {
    stop()
    run()
  })
}

export function beginOnboardingFlow(state: OnboardingStateResult, firstRunSkipped: boolean): void {
  if (
    !isOnboardingEnabled() ||
    !state.eligible ||
    state.intro !== 'unseen' ||
    firstRunSkipped ||
    $onboardingGate.get().phase !== 'idle'
  ) {
    return
  }

  setPhase('pending')
  $onboardingGate.set({ ...$onboardingGate.get(), guideQueued: true })
}

export function runGuideKickoff(kickoff: () => Promise<GuideKickoffResult>): Promise<GuideKickoffResult> {
  if (!isOnboardingEnabled()) {
    return Promise.resolve('off')
  }

  if (guideKickoff.status === 'starting') {
    return guideKickoff.promise
  }

  if (guideKickoff.status === 'started') {
    return Promise.resolve('started')
  }

  if (!$onboardingGate.get().guideQueued) {
    return Promise.resolve('off')
  }

  const promise = Promise.resolve()
    .then(kickoff)
    .then(
      result => {
        setGuideKickoff({ status: result === 'started' ? 'started' : 'idle' })

        if (result === 'started' && $onboardingGate.get().phase === 'pending') {
          setPhase('guided')
        }

        return result
      },
      error => {
        setGuideKickoff({ status: 'idle' })

        throw error
      }
    )

  setGuideKickoff({ status: 'starting', promise })

  return promise
}

export function beginOnboardingHandoff(): void {
  const { phase } = $onboardingGate.get()

  if (isOnboardingEnabled() && (phase === 'guided' || phase === 'skipped')) {
    setPhase('handoff')
    reportOnboarding('onboarding.mark_seen')
  }
}

export function completeOnboardingFlow(): void {
  if (isOnboardingEnabled() && $onboardingGate.get().phase === 'handoff') {
    setPhase('done')
  }
}

export function skipGuide(): void {
  const { phase } = $onboardingGate.get()

  if (isOnboardingEnabled() && (phase === 'pending' || phase === 'guided')) {
    setPhase('skipped')
    reportOnboarding('onboarding.mark_seen')
  }
}

export function abandonGuide(result: Exclude<GuideKickoffResult, 'started'>): void {
  const { phase } = $onboardingGate.get()

  if (isOnboardingEnabled() && (phase === 'pending' || phase === 'guided')) {
    setPhase('skipped')

    if (result === 'failed') {
      reportOnboarding('onboarding.record_failed_start')
    }
  }
}

export async function devResetOnboardingFlow(): Promise<void> {
  if (!import.meta.env.DEV) {
    return
  }

  await $gateway.get()?.request('onboarding.reset_setup_profile', {})
  setGuideKickoff({ status: 'idle' })
  setPhase('idle')
  setOnboardingAnswers({ ...DEFAULT_ANSWERS, connectors: [], plugins: [], pluginOutcomes: {} })
}

declare global {
  interface Window {
    __onboarding?: { reset: typeof devResetOnboardingFlow }
  }
}

if (import.meta.env.DEV) {
  window.__onboarding = { reset: devResetOnboardingFlow }
}
