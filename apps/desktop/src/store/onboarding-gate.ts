import { atom, computed } from 'nanostores'

import { isOnboardingEnabled } from '@/lib/onboarding-enabled'
import { readKey, writeKey } from '@/lib/storage'

import { $gateway } from './gateway'
import { DEFAULT_ANSWERS, setOnboardingAnswers } from './onboarding-answers'

const PHASE_KEY = 'hermes-onboarding-phase-v1'

export const ONBOARDING_PHASES = ['idle', 'pending', 'guided', 'skipped', 'handoff', 'done'] as const

export type OnboardingPhase = (typeof ONBOARDING_PHASES)[number]

function isOnboardingPhase(value: string | null): value is OnboardingPhase {
  return ONBOARDING_PHASES.some(phase => phase === value)
}

export interface OnboardingGateState {
  phase: OnboardingPhase
  guideQueued: boolean
  guideKickoff: 'idle' | 'starting' | 'started'
}

type GuideKickoff = { status: 'idle' } | { status: 'starting'; promise: Promise<boolean> } | { status: 'started' }

function loadGate(): OnboardingGateState {
  const saved = readKey(PHASE_KEY)

  const phase = isOnboardingEnabled() && isOnboardingPhase(saved) ? saved : 'idle'

  return {
    phase,
    guideQueued: phase === 'pending' || phase === 'guided',
    guideKickoff: 'idle'
  }
}

export const $onboardingGate = atom<OnboardingGateState>(loadGate())

let guideKickoff: GuideKickoff = { status: 'idle' }
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
  writeKey(PHASE_KEY, phase === 'idle' ? null : phase)
  $onboardingGate.set({ ...$onboardingGate.get(), phase, guideQueued: false })
}

export function guidedOnboardingActive(): boolean {
  const { phase } = $onboardingGate.get()

  return isOnboardingEnabled() && (phase === 'pending' || phase === 'guided' || phase === 'handoff')
}

export function beginOnboardingFlow(firstRunSkipped: boolean): void {
  if (!isOnboardingEnabled() || firstRunSkipped || $onboardingGate.get().phase !== 'idle') {
    return
  }

  setPhase('pending')
  $onboardingGate.set({ ...$onboardingGate.get(), guideQueued: true })
}

export function runGuideKickoff(kickoff: () => Promise<boolean>): Promise<boolean> {
  if (!isOnboardingEnabled()) {
    return Promise.resolve(false)
  }

  if (guideKickoff.status === 'starting') {
    return guideKickoff.promise
  }

  if (guideKickoff.status === 'started') {
    return Promise.resolve(true)
  }

  if (!$onboardingGate.get().guideQueued) {
    return Promise.resolve(false)
  }

  const promise = Promise.resolve()
    .then(kickoff)
    .then(
      started => {
        setGuideKickoff({ status: started ? 'started' : 'idle' })

        if (started && $onboardingGate.get().phase === 'pending') {
          setPhase('guided')
        }

        return started
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
