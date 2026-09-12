import { atom } from 'nanostores'

import { isOnboardingEnabled } from '@/lib/onboarding-enabled'
import { readKey, writeKey } from '@/lib/storage'

import { hasSeenIntroReveal } from './intro-reveal'
import { DEFAULT_ANSWERS, setOnboardingAnswers } from './onboarding-answers'

const PHASE_KEY = 'hermes-onboarding-phase-v1'

export const ONBOARDING_PHASES = ['idle', 'cinematic', 'guided', 'skipped', 'handoff', 'done'] as const

export type OnboardingPhase = (typeof ONBOARDING_PHASES)[number]

function isOnboardingPhase(value: string | null): value is OnboardingPhase {
  return ONBOARDING_PHASES.some(phase => phase === value)
}

export interface OnboardingGateState {
  phase: OnboardingPhase
  guideQueued: boolean
}

type GuideKickoff = { status: 'idle' } | { status: 'starting'; promise: Promise<boolean> } | { status: 'started' }

function loadGate(): OnboardingGateState {
  const saved = readKey(PHASE_KEY)

  const phase = isOnboardingEnabled() && isOnboardingPhase(saved) ? saved : 'idle'

  // Two phases owe a kickoff at boot. `cinematic` with the film already seen
  // is the film-to-guide seam. `guided` is a relaunch mid-guide: without a
  // kickoff the normal app boots around the persisted solo layout — the
  // connected splash, the stock composer and model picker, a small window
  // whose sidebars cannot open — while the gate still says the guide is on.
  // The kickoff adopts the existing guide chat by title, so nothing is lost.
  return { phase, guideQueued: (phase === 'cinematic' && hasSeenIntroReveal()) || phase === 'guided' }
}

export const $onboardingGate = atom<OnboardingGateState>(loadGate())

let guideKickoff: GuideKickoff = { status: 'idle' }

function setPhase(phase: OnboardingPhase): void {
  writeKey(PHASE_KEY, phase === 'idle' ? null : phase)
  $onboardingGate.set({ phase, guideQueued: false })
}

/** The guided first launch is on screen or mid-handoff. Ambient chrome that
 *  would send the user elsewhere (the provider picker, the free-tier chip)
 *  yields to it: the free tier IS the provider for those phases, and the
 *  guide's ready screen is where sign-in is offered. */
export function guidedOnboardingActive(): boolean {
  const { phase } = $onboardingGate.get()

  return isOnboardingEnabled() && (phase === 'cinematic' || phase === 'guided' || phase === 'handoff')
}

export function beginOnboardingFlow(): void {
  if (isOnboardingEnabled() && $onboardingGate.get().phase === 'idle' && !hasSeenIntroReveal()) {
    setPhase('cinematic')
  }
}

export function queueGuideAfterIntro(): void {
  const state = $onboardingGate.get()

  if (isOnboardingEnabled() && state.phase === 'cinematic' && !state.guideQueued && hasSeenIntroReveal()) {
    $onboardingGate.set({ ...state, guideQueued: true })
  }
}

/** The kickoff returns true only after the guided session's seed is durable. */
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

  // Defer the callback until the shared promise is installed, including for
  // callers that re-enter synchronously while starting the session.
  const promise = Promise.resolve()
    .then(kickoff)
    .then(
      started => {
        guideKickoff = { status: started ? 'started' : 'idle' }

        if (started && $onboardingGate.get().phase === 'cinematic') {
          setPhase('guided')
        }

        return started
      },
      error => {
        guideKickoff = { status: 'idle' }

        throw error
      }
    )

  guideKickoff = { status: 'starting', promise }

  return promise
}

export function beginOnboardingHandoff(): void {
  const { phase } = $onboardingGate.get()

  if (isOnboardingEnabled() && (phase === 'guided' || phase === 'skipped')) {
    setPhase('handoff')
  }
}

/** Called when the handoff receipt is accepted. */
export function completeOnboardingFlow(): void {
  if (isOnboardingEnabled() && $onboardingGate.get().phase === 'handoff') {
    setPhase('done')
  }
}

export function skipGuide(): void {
  const { phase } = $onboardingGate.get()

  if (isOnboardingEnabled() && (phase === 'cinematic' || phase === 'guided')) {
    setPhase('skipped')
  }
}

export function devResetOnboardingFlow(): void {
  if (!import.meta.env.DEV) {
    return
  }

  guideKickoff = { status: 'idle' }
  setPhase('idle')
  setOnboardingAnswers({ ...DEFAULT_ANSWERS, connectors: [...DEFAULT_ANSWERS.connectors] })
}

declare global {
  interface Window {
    __onboarding?: { reset: typeof devResetOnboardingFlow }
  }
}

if (import.meta.env.DEV) {
  window.__onboarding = { reset: devResetOnboardingFlow }
}
